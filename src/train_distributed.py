import os
import pickle as pkl
import shutil
import sys

import functools
import syft as sy
import torch
from torch.utils.tensorboard import SummaryWriter

from common.argparser import argparser
from common.arguments import Arguments
from common.utils import get_device, get_paths, init_logger
from data.distributor import get_fl_graph
from data.loader import get_loader
from models.train import distributed_train, test
from models.utils import get_model
from viz.training_plots import training_plots

print = functools.partial(print, flush=True)
torch.set_printoptions(linewidth=120)

# ------------------------------------------------------------------------------
# Setups
# ------------------------------------------------------------------------------

args = Arguments(argparser())
hook = sy.TorchHook(torch)
device = get_device(args)
paths = get_paths(args, distributed=True)
log_file, std_out = init_logger(
    paths.log_file, args.dry_run, args.load_model)
if os.path.exists(paths.tb_path):
    shutil.rmtree(paths.tb_path)
tb = SummaryWriter(paths.tb_path)

print('+' * 80)
print(paths.model_name)
print('+' * 80)

print(args.__dict__)
print('+' * 80)

# prepare graph and data
_, workers = get_fl_graph(hook, args.num_workers)
print('Loading data: {}'.format(paths.data_path))
X_trains, _, y_trains, _, meta = pkl.load(open(paths.data_path, 'rb'))


test_loader = get_loader(args.dataset,
                         args.test_batch_size,
                         train=False,
                         noise=args.noise)

print('+' * 80)

# ------------------------------------------------------------------------------
# Fire the engines
# ------------------------------------------------------------------------------

model, loss_type = get_model(args, ckpt_path=args.load_model)
if args.batch_size == 0:
    args.batch_size = int(meta['batch_size'])
    print("Resetting batch size: {}...".format(args.batch_size))

print('+' * 80)
h_epoch = []
h_acc_test = []
h_acc_train = []
h_acc_train_std = []
h_loss_test = []
h_loss_train = []
h_loss_train_std = []
h_uplink = []
h_grad_agg = []
h_error = []

print('Pre-Training')
# tb_model_summary(model, test_loader, tb, device)
best, i = test(model, device, test_loader, loss_type)
ii, iii = test(model, device, test_loader, loss_type)
print('Acc: {:.4f}'.format(best))
tb.add_scalar('Train_Loss', iii, 0)
tb.add_scalar('Val_Loss', i, 0)
tb.add_scalar('Train_Acc', ii, 0)
tb.add_scalar('Val_Acc', best, 0)

# worker_models: actual models to train
# worker_mbufs: momentum buffer for sgd
# model mbuf: moementum buffer for model
# worker_residuals: for error-feedback during TopK, LBGM, etc.
# worker_sdirs: directions used for approximations
worker_models = {}
worker_mbufs = {}
model_mbuf = []
worker_residuals = {}
worker_sdirs = {}
# sdirs = {}

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

wait = 0
prev_error = 0
print('+' * 80)
print('Training w/ optim:{} and paradigm {}'.format(
    args.optim, ','.join(args.paradigm) if args.paradigm else 'NA'))
print('epoch \t tr loss (acc) \t test loss (acc) \t lbgm+-std')
for epoch in range(args.start_epoch, args.epochs):
    h_epoch.append(epoch)

    train_loss, train_acc, \
        worker_grad_sum, model_mbuf, uplink, avg_error = distributed_train(
            args, model, workers, X_trains, y_trains,
            device, loss_type, worker_models,
            worker_mbufs, model_mbuf, worker_sdirs, worker_residuals,
            0 if not args.scheduler else epoch-1,
        )

    h_acc_train.append(train_acc)
    h_loss_train.append(train_loss)
    h_uplink.append(uplink)
    h_grad_agg.append(worker_grad_sum)
    h_error.append(avg_error)

    acc, loss = test(model, device, test_loader, loss_type)
    h_acc_test.append(acc)
    h_loss_test.append(loss)

    tb.add_scalar('Train_Loss', h_loss_train[-1], epoch)
    tb.add_scalar('Val_Loss', h_loss_test[-1], epoch)
    tb.add_scalar('Train_Acc', h_acc_train[-1], epoch)
    tb.add_scalar('Val_Acc', h_acc_test[-1], epoch)

    if acc > best:
        best = acc
        if args.save_model:
            torch.save(model.state_dict(), paths.best_path)
            best_iter = epoch
    else:
        wait += 1

    if epoch % args.log_intv == 0:
        print('{} \t {:.2f} ({:.2f}) \t {:.5f} ({:.4f}) \t {:.4f} \t {}'.format(
            epoch, train_loss, train_acc,
            loss, acc, avg_error, uplink))
        tb.flush()

    if wait > args.patience:
        if args.early_stopping:
            print('Early stopping after wait = {}...'.format(args.patience))
            break

# ------------------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------------------

tb.close()
if args.save_model:
    print('\nModel best  @ {}, acc {:.4f}: {}'.format(best_iter, best,
                                                      paths.best_path))
    torch.save(model.module.state_dict(), paths.stop_path)
    print('Model stop: {}'.format(paths.stop_path))

pkl.dump((h_epoch, h_acc_test, h_acc_train, h_acc_train_std, h_loss_test,
          h_loss_train, h_loss_train_std, h_uplink, h_grad_agg),
         open(paths.hist_path, 'wb'))
print('Saved: ', paths.hist_path)

# training_plots(
#     {
#         'h_epoch': h_epoch,
#         'h_acc_test': h_acc_test,
#         'h_acc_train': h_acc_train,
#         'h_acc_train_std': h_acc_train_std,
#         'h_loss_test': h_loss_test,
#         'h_loss_train': h_loss_train,
#         'h_loss_train_std': h_loss_train_std,
#         'h_uplink': h_uplink,
#         'h_grad': h_grad_agg,
#         'h_error': h_error,
#     }, args, loss_type, paths.plot_path)

if args.dry_run:
    # print("Remove: ", paths.plot_path)
    # os.remove(paths.plot_path)
    print("Remove: ", paths.hist_path)
    os.remove(paths.hist_path)

print("+" * 38 + "EOF" + "+" * 39)

# ------------------------------------------------------------------------------
# Reset print
# ------------------------------------------------------------------------------

if not args.dry_run:
    log_file.close()
    sys.stdout = std_out
