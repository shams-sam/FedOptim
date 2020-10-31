import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from conjugate_gradient import ConjugateGradient, fletcher_reeves
from model_op import add_model_weights, get_model_weights
from multi_class_hinge_loss import multiClassHingeLoss
from utils import get_dataloader, norm


def fl_train(args, model, fl_graph, nodes, X_trains, y_trains,
             device, epoch, loss_fn, worker_models,
             worker_sdirs, worker_ograds):
    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    model.train()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_optims = {}
    worker_losses = {}

    # send data, model to workers
    # setup optimizer for each worker

    workers = [_ for _ in nodes.keys() if 'L0' in _]
    for w, x, y in zip(workers, X_trains, y_trains):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    for w in workers:
        worker_models[w] = model.copy().send(nodes[w])
        node_model = worker_models[w].get()
        if args.paradigm == 'sgd':
            worker_optims[w] = optim.SGD(
                params=worker_models[w].parameters(), lr=args.lr)
        elif args.paradigm == 'adam':
            worker_optims[w] = optim.Adam(
                params=worker_models[w].parameters(), lr=args.lr)
        elif args.paradigm == 'conj':
            worker_optims[w] = ConjugateGradient(
                params=worker_models[w].parameters(), lr=args.lr)

        data = worker_data[w].get()
        target = worker_targets[w].get()
        dataloader = get_dataloader(data, target, args.batch_size)

        for data, target in dataloader:
            lambda_i = 1e-3
            lambda_i_hat = 0

            data, target = data.to(device), target.to(device)
            worker_optims[w].zero_grad()
            output = node_model(data)
            loss = loss_fn_(output, target)
            loss.backward()
            if args.paradigm == 'conj':
                if len(worker_ograds) == 0:
                    for p in node_model.parameters():
                        worker_sdirs.append(p.grad.clone())
                        worker_ograds.append(p.grad.clone())
                else:
                    grad_accum = []
                    sdir_accum = []
                    for p, ograd, sdir in zip(
                            node_model.parameters(),
                            worker_ograds, worker_sdirs):
                        grad_accum.append(p.grad.clone())
                        beta = fletcher_reeves(p.grad, ograd)
                        p.grad = p.grad - beta * sdir
                        sdir_accum.append(p.grad.clone())
                    worker_ograds = grad_accum
                    worker_sdirs = sdir_accum

                sigma_i = args.sigma/norm(p.grad)

                curr_grad = p.grad.clone()
                p.grad = sigma_i * p.grad
                worker_optims[w].step()
                worker_optims[w].zero_grad()
                output = node_model(data)
                loss = loss_fn_(output, target)
                loss.backward()
                proj_grad = p.grad.clone()
                p.grad = -p.grad / sigma_i
                worker_optims[w].step()
                p.grad = curr_grad
                s_i = (proj_grad - curr_grad)/sigma_i

                curr_norm = norm(curr_grad)
                delta_i = torch.dot(curr_grad.flatten(), s_i)
                delta_i += (lambda_i - lambda_i_hat) * curr_norm.pow(2)
                if delta_i <= 0:
                    lambda_i_hat = 2*(lambda_i - delta_i / curr_norm.pow(2))
                    delta_i = - delta_i - lambda_i * curr_norm.pow(2)
                    lambda_i = lambda_i_hat

                mu_i =

            else:
                worker_optims[w].step()
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = loss.item()

    agg = 'L1_W0'
    worker_models[agg] = model.copy().send(nodes[agg])
    children = fl_graph[agg]

    for child in children:
        worker_models[child].move(nodes[agg])

    with torch.no_grad():
        weighted_models = [get_model_weights(
            worker_models[_],
            worker_num_samples[_]/args.num_train) for _ in children]
        model_sum = weighted_models[0]
        for m in weighted_models[1:]:
            model_sum = add_model_weights(model_sum, m)
        worker_models[agg].load_state_dict(model_sum)

    master = get_model_weights(worker_models[agg].get())
    model.load_state_dict(master)

    if epoch % args.log_interval == 0:
        loss = np.array([_ for dump, _ in worker_losses.items()])
        print('Train Epoch: {} \tLoss: {:.6f} +- {:.6f}'.format(
            epoch, loss.mean(), loss.std()))


def fl_test(args, nodes, X_tests, y_tests,
            device, epoch, loss_fn, worker_models):
    # fog learning with model averaging
    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    worker_data = {}
    worker_targets = {}
    worker_num_samples = {}
    worker_losses = {}
    worker_accs = {}

    workers = [_ for _ in nodes.keys() if 'L0' in _]
    for w, x, y in zip(workers, X_tests, y_tests):
        worker_data[w] = x.send(nodes[w])
        worker_targets[w] = y.send(nodes[w])
        worker_num_samples[w] = x.shape[0]

    for w in workers:
        node_model = worker_models[w].get()
        data = worker_data[w].get()
        target = worker_targets[w].get()
        dataloader = get_dataloader(data, target, worker_num_samples[w])

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = node_model(data)
            loss = loss_fn_(output, target)
            pred = output.argmax(1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
        worker_accs[w] = correct / len(dataloader.dataset)
        worker_models[w] = node_model.send(nodes[w])
        worker_losses[w] = loss.item()

    if epoch % args.log_interval == 0:
        loss = np.array([_ for dump, _ in worker_losses.items()])
        acc = np.array([_ for dump, _ in worker_accs.items()])
        print('Test fog: {}({}) Accuracy: {:.4f} += {:.4f} '
              '\tLoss: {:.6f} +- {:.6f}\n'.format(
                  epoch, len(dataloader), acc.mean(), acc.std(),
                  loss.mean(), loss.std()))

    return loss.mean(), loss.std(), acc.mean(), acc.std()


def test(args, model, device, test_loader, best, epoch=0, loss_fn='nll'):
    if loss_fn == 'nll':
        loss_fn_ = F.nll_loss
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if loss_fn == 'nll':
                test_loss += loss_fn_(output, target, reduction='sum').item()
            elif loss_fn == 'hinge':
                test_loss += loss_fn_(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    if epoch % args.log_interval == 0:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ==> '
              '{:.2f}%'.format(
                  test_loss, correct, len(test_loader.dataset),
                  100.*accuracy, 100.*best))

    return accuracy, test_loss
