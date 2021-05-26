import functools
import os
import pickle as pkl
import shutil
import sys

import syft as sy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.argparser import argparser
from common.arguments import Arguments
import common.config as cfg
from common.utils import get_device, get_paths, \
    init_logger, is_approx
from data.loader import get_loader
from models.model_op import add_param_list, get_model_grads
from models.multi_class_hinge_loss import multiClassHingeLoss
from models.utils import get_model
from models.train import test, sdirs_approximation

print = functools.partial(print, flush=True)
torch.set_printoptions(linewidth=120)

# ------------------------------------------------------------------------------
# Setups
# ------------------------------------------------------------------------------

args = Arguments(argparser())
hook = sy.TorchHook(torch)
device = get_device(args)
paths = get_paths(args)
log_file, std_out = init_logger(paths.log_file, args.dry_run)
if os.path.exists(paths.tb_path):
    shutil.rmtree(paths.tb_path)
tb = SummaryWriter(paths.tb_path)

print('+'*80)
print(paths.model_name)
print('+' * 80)

print(args.__dict__)
print('+' * 80)

if args.batch_size == 0:
    args.batch_size = args.num_train
    print("Resetting batch size: {}...".format(args.batch_size))

train_loader = get_loader(args.dataset, args.batch_size,
                          train=True, subset=args.repeat,
                          force_resize=cfg.model_im_size[args.clf])
test_loader = get_loader(args.dataset, args.test_batch_size,
                         train=False, shuffle=False, subset=args.repeat,
                         force_resize=cfg.model_im_size[args.clf])
print('Train size: ', len(train_loader.dataset))
print('Test size: ', len(test_loader.dataset))

print('+' * 80)


# ------------------------------------------------------------------------------
# Fire the engines
# ------------------------------------------------------------------------------

model, loss_type = get_model(args)
agg_type = 'averaging'

if 'sgd' in args.paradigm:
    optimizer = optim.SGD(
        params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
elif 'adam' in args.paradigm:
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
if args.scheduler:
    print('Initializing scheduler...')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

if loss_type == 'hinge':
    loss_fn = multiClassHingeLoss()
elif loss_type == 'ce':
    loss_fn = nn.CrossEntropyLoss().to(device)
elif loss_type == 'mse':
    loss_fn = nn.MSELoss().to(device)
print('+' * 80)

h_epoch = []
h_acc_train = []
h_acc_test = []
h_loss_train = []
h_loss_test = []
h_grads = []

print('Pre-Training')
# tb_model_summary(model, test_loader, tb, device)
best, i = test(model, device, test_loader, loss_type)
ii, iii = test(model, device, test_loader, loss_type)
print('Acc: {:.4f}'.format(best))
tb.add_scalar('Train_Loss', iii, 0)
tb.add_scalar('Val_Loss', i, 0)
tb.add_scalar('Train_Acc', ii, 0)
tb.add_scalar('Val_Acc', best, 0)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

wait = 0
for epoch in range(1, args.epochs + 1):
    gradi = []
    num_minibatches = 0
    running_loss = 0.0
    running_acc = 0.0
    gradi = 0
    for data, target in tqdm(train_loader, total=len(train_loader), leave=False):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        if loss_type == 'mse':
            target = target.float()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        gradi = add_param_list(gradi, get_model_grads(model))
        running_loss += loss.item()
        if loss_type != 'mse':
            predi = output.view(-1, output.size(1)).argmax(1, keepdim=True)
            correcti = predi.eq(target.view_as(predi)).sum().item()
            running_acc += correcti / data.shape[0]
        else:
            running_acc = running_loss
        num_minibatches += 1
    gradi = [_/num_minibatches for _ in gradi]
    if args.scheduler:
        scheduler.step()

    h_grads.append(gradi)
    h_epoch.append(epoch)
    h_acc_train.append(running_acc / num_minibatches)
    h_loss_train.append(running_loss / num_minibatches)

    acc, loss = test(model, device, test_loader, loss_type)
    # for name, weight in model.named_parameters():
    #     tb.add_histogram(name, weight, epoch)
    #     tb.add_histogram(f'{name}.grad', weight.grad, epoch)
    h_acc_test.append(acc)
    h_loss_test.append(loss)

    tb.add_scalar('Train_Loss', h_loss_train[-1], epoch)
    tb.add_scalar('Val_Loss', h_loss_test[-1], epoch)
    tb.add_scalar('Train_Acc', h_acc_train[-1], epoch)
    tb.add_scalar('Val_Acc', h_acc_test[-1], epoch)

    if epoch % args.log_intv == 0:
        print('Train Epoch w\\ {}: {} \tTrain: {:.4f} ({:.2f})'
              ' \t Test: {:.4f} ({:.2f})'.format(
                  args.paradigm, epoch, h_loss_train[-1], h_acc_train[-1],
                  h_loss_test[-1], h_acc_test[-1]))
        tb.flush()

    if acc > best:
        wait = 0
        best = acc
    else:
        wait += 1

    if wait >= args.patience and args.early_stopping:
        print('Early stopping...')
        break


# ------------------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------------------

tb.close()
pkl.dump((h_epoch, h_acc_train, h_acc_test, h_loss_train,
          h_loss_test, h_grads), open(paths.hist_path, 'wb'))
print('Saved: ', paths.hist_path)

if args.dry_run:
    print("Remove: ", paths.hist_path)
    os.remove(paths.hist_path)

print("+" * 38 + "EOF" + "+" * 39)

# ------------------------------------------------------------------------------
# Reset print
# ------------------------------------------------------------------------------

if not args.dry_run:
    log_file.close()
    sys.stdout = std_out
