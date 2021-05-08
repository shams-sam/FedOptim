import numpy as np

from common.utils import add_gaussian_noise, is_approx
from data.distributor import get_cluster_sizes
from data.loader import get_dataloader
from models.model_op import get_model_grads, add_param_list, get_model_size,\
    gradient_approximation, model_update, topk_sparsify
from models.utils import forward, get_loss_fn, get_optim


def sdirs_prepare(args, sdirs, worker_sdirs, workers, full=False):
    num_sdirs = len(worker_sdirs)
    num_workers = len(workers)
    if not is_approx(args):
        return

    if full:
        for w in workers:
            sdirs[w] = worker_sdirs
    elif num_sdirs > num_workers:
        cs = np.array(get_cluster_sizes(num_sdirs, num_workers))
        ccs = [0] + np.cumsum(cs).tolist()
        for idx, w in zip(range(1, len(ccs)), workers):
            sdirs[w] = worker_sdirs[ccs[idx-1]: ccs[idx]]
    else:
        mul = num_workers // num_sdirs
        rem = num_workers - (num_sdirs * mul)
        sdirsi = worker_sdirs
        if mul > 1:
            sdirsi = worker_sdirs * mul
        if rem > 0:
            sdirsi = sdirsi + worker_sdirs[:rem]
        for sdir, w in zip(sdirsi, workers):
            sdirs[w] = [sdir]


def sdirs_approximation(args, node_model, sdirs, worker_residuals, device):
    if args.residual:
        worker_residuals, error = gradient_approximation(
            node_model, sdirs, device, worker_residuals)
    else:
        _, error = gradient_approximation(
            node_model, sdirs, device, [])

    return worker_residuals, error


def topk(args, node_model, worker_residuals):
    # if kgrads is working then topk is not needed because
    # kgrads is much lesser in cost than topk
    if args.residual:
        num_topk, worker_residuals, error = topk_sparsify(
            node_model, args.topk, worker_residuals)
    else:
        num_topk, _, error = topk_sparsify(
            node_model, args.topk, [])

    return num_topk, error


def worker_process(args, model, loss_fn,
                   worker_data, worker_targets, worker_num_samples,
                   sdirs, worker_residuals, device):
    node_model = model.copy()
    uplink, _ = get_model_size(node_model)
    opt = get_optim(args, node_model)

    data = worker_data.get()
    target = worker_targets.get()
    dataloader = get_dataloader(data, target, args.batch_size)

    correct = 0
    total_loss = 0
    error_per_worker = 0
    worker_batch_grads = 0

    for data, target in dataloader:
        loss, c = forward(node_model, data, target,
                          opt, loss_fn, device)
        correct += c
        loss.backward()
        total_loss += loss.item()

        if args.paradigm:
            error = 0
            if 'topk' in args.paradigm:
                uplink, error = topk(
                    args, node_model, worker_residuals, device)
            elif is_approx(args) and len(sdirs):
                worker_residuals, error = sdirs_approximation(
                    args, node_model, sdirs, worker_residuals, device)
                uplink = len(sdirs)
            error_per_worker += error
        worker_batch_grads = add_param_list(
            worker_batch_grads, get_model_grads(node_model))
    num_batches = len(dataloader)
    worker_grads = [_/num_batches for _ in worker_batch_grads]
    if args.noise:
        worker_grads = [add_gaussian_noise(
            args, _.cpu()).to(device) for _ in worker_grads]
    worker_grads = [_ * worker_num_samples/args.num_train
                    for _ in worker_grads]

    error_per_worker /= num_batches

    return node_model, worker_grads, worker_residuals, \
        total_loss/num_batches, correct/worker_num_samples, \
        error_per_worker, uplink


def fl_train(args, model, nodes, X_trains, y_trains,
             device, loss_fn, worker_models,
             worker_sdirs, sdirs, worker_residuals):
    loss_fn_ = get_loss_fn(loss_fn)
    uplink, _ = get_model_size(model)

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
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    num_workers = len(workers)
    if is_approx(args) and len(worker_sdirs):
        if 'kgrad' in args.paradigm and len(worker_sdirs) < args.kgrads:
            sdirs = {}
        else:
            sdirs_prepare(args, sdirs, worker_sdirs, workers, args.sdir_full)

    cumm_error = 0
    # train workers
    for w in workers:
        # approximations in training occur in worker_process
        node_model, node_batch_grads, residuals, loss, acc, \
            error_per_worker, u = worker_process(
                args, model, loss_fn_, worker_data[w],
                worker_targets[w], worker_num_samples[w],
                sdirs[w] if w in sdirs else [],
                worker_residuals[w] if w in worker_residuals else [],
                device)
        worker_residuals[w] = residuals
        uplink = min(uplink, u)
        # worker_models[w] = node_model.send(nodes[w])
        worker_grad_sum = add_param_list(worker_grad_sum, node_batch_grads)
        worker_losses[w] = loss
        worker_accs[w] = acc
        cumm_error += error_per_worker

    cumm_error /= num_workers
    model_update(model, worker_grad_sum, args.lr)

    if args.paradigm and \
       'kgrad' in args.paradigm and \
       len(worker_sdirs) < args.kgrads:
        worker_sdirs.append(worker_grad_sum)

    loss = np.array([_ for dump, _ in worker_losses.items()])
    acc = np.array([_ for dump, _ in worker_accs.items()])
    loss_mean, loss_std = loss.mean(), loss.std()
    acc_mean, acc_std = acc.mean(), acc.std()

    return loss_mean, loss_std, acc_mean, acc_std, \
        worker_grad_sum, u, \
        sdirs, worker_residuals, cumm_error


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
            pred = output.argmax(1, keepdim=True)
            correcti = pred.eq(target.view_as(pred)).sum().item()
            running_acc += correcti/data.shape[0]
        else:
            running_acc = running_loss
        num_minibatches += 1

    return running_acc/num_minibatches, running_loss/num_minibatches
