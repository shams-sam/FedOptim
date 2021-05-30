import numpy as np
# from tqdm import tqdm

from common.utils import add_gaussian_noise
from data.loader import get_dataloader
from models.model_op import atomo_approximation, get_model_grads, add_param_list, get_model_size,\
    lbgm_approximation, model_update, set_model_grads, sign_sgd_quantization, topk_sparsify
from models.utils import forward, get_loss_fn, get_optim


def atomo(args, node_model, worker_residuals, device):
    uplink, worker_residuals, error = atomo_approximation(
        args, node_model, worker_residuals, device)

    return uplink, worker_residuals, error


def lbgm(args, node_model, sdirs, worker_residuals, device):
    if args.residual:
        sdirs, worker_residuals, u, avg_rho = lbgm_approximation(
            args, node_model, sdirs, worker_residuals, device)
    else:
        sdirs, _, u, avg_rho = lbgm_approximation(
            args, node_model, sdirs, [], device)
        worker_residuals = []

    return sdirs, worker_residuals, u, avg_rho


def sign_sgd(args, node_model, worker_residuals, device):
    if args.residual and False:  # sign sgd works better w/o in expts
        uplink, worker_residuals, error = sign_sgd_quantization(
            args, node_model, worker_residuals, device
        )
    else:
        uplink, worker_residuals, error = sign_sgd_quantization(
            args, node_model, [], device
        )

    return uplink, worker_residuals, error


def topk(args, node_model, worker_residuals):
    # if kgrads is working then topk is not needed because
    # kgrads is much lesser in cost than topk
    if args.residual:
        num_topk, worker_residuals, error = topk_sparsify(
            node_model, args.topk, worker_residuals)
    else:
        num_topk, _, error = topk_sparsify(
            node_model, args.topk, [])

    return num_topk, worker_residuals, error


def distributed_worker_process(args, model, loss_fn,
                               data, target, worker_num_samples,
                               worker_mbuf, sdirs, worker_residuals, device, epoch):
    # mbuf: momentum buffer

    node_model = model.copy()
    uplink, _ = get_model_size(node_model)
    opt = get_optim(args, node_model)

    if args.loss_type == 'mse':
        target = target.float()
    loss, correct = forward(node_model, data, target,
                            opt, loss_fn, device, args.loss_type)
    opt.zero_grad()
    loss.backward()

    worker_grads = get_model_grads(node_model)
    worker_mbuf = model_update(node_model, worker_grads,
                               args, device, [], epoch)

    error = 0
    if args.paradigm:

        if 'atomo' in args.paradigm:
            uplink, worker_residuals, error = atomo_approximation(
                args, node_model, worker_residuals, device)
        elif 'signsgd' in args.paradigm:
            uplink, worker_residuals, error = sign_sgd(
                args, node_model, worker_residuals, device)
        elif 'topk' in args.paradigm:
            uplink, worker_residuals, error = topk(
                args, node_model, worker_residuals)
        if 'lbgm' in args.paradigm:
            sdirs, worker_residuals, u, avg_rho = lbgm(
                args, node_model, sdirs, worker_residuals, device)
            uplink = min(u, uplink)
            error = 1 - avg_rho
        worker_grads = get_model_grads(node_model)

    if args.noise:
        worker_grads = [add_gaussian_noise(
            args, _.cpu()).to(device) for _ in worker_grads]
    worker_grads = [_ * worker_num_samples/args.num_train
                    for _ in worker_grads]

    return node_model, worker_grads, worker_mbuf, worker_residuals, \
        loss, correct, error, sdirs, uplink


def federated_worker_process(args, model, loss_fn,
                             worker_data, worker_targets, worker_num_samples,
                             worker_mbuf, sdirs, worker_residuals, device, epoch):
    # mbuf: momentum buffer

    node_model = model.copy()
    uplink, _ = get_model_size(node_model)
    opt = get_optim(args, node_model)

    data = worker_data.get()
    target = worker_targets.get()
    dataloader = get_dataloader(data, target, args.batch_size)

    correct = 0
    total_loss = 0

    num_batches = len(dataloader)
    accum_grads = False
    for batch_id, (data, target) in enumerate(dataloader):
        if args.loss_type == 'mse':
            target = target.float()
        loss, c = forward(node_model, data, target,
                          opt, loss_fn, device, args.loss_type)
        correct += c
        opt.zero_grad()
        loss.backward()
        total_loss += loss.item()

        worker_grads = get_model_grads(node_model)
        # accum_grads = add_param_list(accum_grads, worker_grads)
        # opt.step()
        worker_mbuf = model_update(node_model, worker_grads,
                                   args, device, [], epoch)
        # momentum on local updates will increase heterogeneity
        assert batch_id < args.tau - 1

    error = 0
    # set_model_grads(node_model, accum_grads)
    if args.paradigm:

        if 'atomo' in args.paradigm:
            uplink, worker_residuals, error = atomo_approximation(
                args, node_model, worker_residuals, device)
        elif 'signsgd' in args.paradigm:
            uplink, worker_residuals, error = sign_sgd(
                args, node_model, worker_residuals, device)
        elif 'topk' in args.paradigm:
            uplink, worker_residuals, error = topk(
                args, node_model, worker_residuals)
        if 'lbgm' in args.paradigm:
            sdirs, worker_residuals, u, avg_rho = lbgm(
                args, node_model, sdirs, worker_residuals, device)
            uplink = min(u, uplink)
            error = 1 - avg_rho
        worker_grads = get_model_grads(node_model)

    num_batches = len(dataloader)
    del dataloader  # to handle the deadlocks
    if args.noise:
        worker_grads = [add_gaussian_noise(
            args, _.cpu()).to(device) for _ in worker_grads]
    worker_grads = [_ * worker_num_samples/args.num_train
                    for _ in worker_grads]

    return node_model, worker_grads, worker_mbuf, worker_residuals, \
        total_loss / num_batches, correct / worker_num_samples, \
        error, sdirs, uplink


def distributed_train(args, model, nodes, X_trains, y_trains,
                      device, loss_fn, worker_models,
                      worker_mbufs, model_mbuf=[],
                      worker_sdirs={}, worker_residuals={}, epoch=0):
    # worker_mbufs: worker momentum buffer
    # model_mbuf: model momentum buffer

    loss_fn_ = get_loss_fn(loss_fn)

    # uncomment the next line if model.eval() in test
    # model.train()

    worker_loaders = {}
    worker_num_samples = {}
    worker_losses = {}
    worker_accs = {}
    worker_grad_sum = 0

    # send data, model to workers
    workers = [_ for _ in nodes.keys() if 'L0' in _]
    num_workers = len(workers)
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_loaders[w] = get_dataloader(x, y, args.batch_size)
        worker_num_samples[w] = x.shape[0]

    # train workers
    uplink, _ = get_model_size(model)
    total_uplink = 0
    avg_error = 0

    # for w in tqdm(workers, leave=False):  # use if progressbar needed
    num_batches = 0
    for sample_list in zip(*worker_loaders.values()):
        batch_uplink = 0
        correct = 0.0
        total_loss = 0.0
        total = 0
        num_batches += 1
        error = 0
        for w, sample in zip(workers, sample_list):
            data, target = sample
            # approximations in training occur in worker_process
            node_model, node_batch_grads, worker_mbufs[w], worker_residuals[w], loss, c, \
                error_per_worker, worker_sdirs[w], u = distributed_worker_process(
                    args, model, loss_fn_, data,
                    target, worker_num_samples[w],
                    worker_mbufs[w] if w in worker_mbufs else [],
                    worker_sdirs[w] if w in worker_sdirs else [],
                    worker_residuals[w] if w in worker_residuals else [],
                    device, epoch)
            batch_uplink += min(uplink, u)
            error += error_per_worker
            correct += c
            total_loss += loss.item()
            total += len(target)

            worker_grad_sum = add_param_list(worker_grad_sum, node_batch_grads)

        model_mbuf = model_update(
            model, worker_grad_sum, args, device, model_mbuf, epoch)
        total_uplink += batch_uplink / num_workers
        avg_error += error / num_workers

    loss = total_loss / num_workers
    acc = correct / total
    uplink = min(uplink, total_uplink / num_batches)
    avg_error = avg_error / num_batches

    return loss, acc, \
        worker_grad_sum, model_mbuf, uplink, avg_error / num_workers


def fl_train(args, model, nodes, X_trains, y_trains,
             device, loss_fn, worker_models,
             worker_mbufs, model_mbuf=[],
             worker_sdirs={}, worker_residuals={}, epoch=0):
    # worker_mbufs: worker momentum buffer
    # model_mbuf: model momentum buffer

    loss_fn_ = get_loss_fn(loss_fn)

    # uncomment the next line if model.eval() in test
    # model.train()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_losses = {}
    worker_accs = {}
    worker_grad_sum = 0

    # send data, model to workers
    workers = [_ for _ in nodes.keys() if 'L0' in _]
    num_workers = len(workers)
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    # train workers
    uplink, _ = get_model_size(model)
    avg_error = 0
    total_uplink = 0

    # for w in tqdm(workers, leave=False):  # use if progressbar needed
    for w in workers:

        # approximations in training occur in worker_process
        node_model, node_batch_grads, worker_mbufs[w], worker_residuals[w], loss, acc, \
            error_per_worker, worker_sdirs[w], u = federated_worker_process(
                args, model, loss_fn_, worker_data[w],
                worker_targets[w], worker_num_samples[w],
                worker_mbufs[w] if w in worker_mbufs else [],
                worker_sdirs[w] if w in worker_sdirs else [],
                worker_residuals[w] if w in worker_residuals else [],
                device, epoch)
        total_uplink += min(uplink, u)
        avg_error += error_per_worker

        worker_grad_sum = add_param_list(worker_grad_sum, node_batch_grads)
        worker_losses[w] = loss
        worker_accs[w] = acc

    uplink = min(uplink, total_uplink / num_workers)
    model_mbuf = model_update(
        model, worker_grad_sum, args, device, model_mbuf, epoch)

    loss = np.array([_ for dump, _ in worker_losses.items()])
    acc = np.array([_ for dump, _ in worker_accs.items()])
    loss_mean, loss_std = loss.mean(), loss.std()
    acc_mean, acc_std = acc.mean(), acc.std()

    return loss_mean, loss_std, acc_mean, acc_std, \
        worker_grad_sum, model_mbuf, uplink, avg_error / num_workers


def test(model, device, test_loader, loss_fn):
    loss_fn_ = get_loss_fn(loss_fn)
    # eval creates issue for models with batch normalization issue
    # pytorch issue discussed here:
    # https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/45
    # model.eval()
    num_minibatches = 0
    running_loss = 0.0
    running_acc = 0.0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        running_loss += loss_fn_(output, target).item()
        if loss_fn != 'mse':
            # output.argmax(1, keepdim=True)
            pred = output.view(-1, output.size(1)).argmax(1, keepdim=True)
            correcti = pred.eq(target.view_as(pred)).sum().item()
            running_acc += correcti/data.shape[0]
        else:
            running_acc = running_loss
        num_minibatches += 1

    return running_acc/num_minibatches, running_loss/num_minibatches
