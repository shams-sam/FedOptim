
"""
Centralized training
observe the gradients over the period of training
"""

import os
import pickle as pkl
import sys

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import syft as sy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from argparser import argparser
from arguments import Arguments
from fcn import FCN
from multi_class_hinge_loss import multiClassHingeLoss
import resnet
from svm import SVM
from train import test
from utils import accumulate_grads_over_epochs, \
    get_testloader, get_trainloader, vec_unit_dot

import functools
print = functools.partial(print, flush=True)

# Setups
args = Arguments(argparser())
hook = sy.TorchHook(torch)
USE_CUDA = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda:{}".format(
    args.device_id[0]) if USE_CUDA else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}
kwargs = {}

ckpt_path = '../ckpts'
folder = '{}_{}'.format(args.dataset, 'centralized')

model_name = 'clf_{}_noise_{}_paradigm_{}' \
    '_lr_{}_decay_{}_batch_{}_distributed_sim'.format(
        args.clf, args.noise, '_'.join(args.paradigm),
        args.lr, args.decay, args.batch_size)

file_ = '{}/{}/logs/{}.log'.format(ckpt_path, folder, model_name)
print("Logging: ", file_)
if not args.dry_run:
    log_file = open(file_, 'w')
    std_out = sys.stdout
    sys.stdout = log_file

print('+'*80)
print(model_name)
print('+'*80)

print(args.__dict__)
print('+'*80)

init_path = '{}/{}/{}_{}.init'.format(ckpt_path, 'init',
                                      args.dataset, args.clf)
best_path = os.path.join(ckpt_path, folder, 'models',  model_name + '.best')
stop_path = os.path.join(ckpt_path, folder, 'models',  model_name + '.stop')

if args.batch_size == 0:
    args.batch_size = args.num_train
    print("Resetting batch size: {}...".format(args.batch_size))

train_loader = get_trainloader(args.dataset, args.batch_size, False)
test_loader = get_testloader(
    args.dataset, args.test_batch_size, noise=args.noise)
print('+'*80)

# Fire the engines
# Fire the engines
if args.clf == 'fcn':
    print('Initializing FCN...')
    model = FCN(args.input_size, args.output_size)
elif args.clf == 'svm':
    print('Initializing SVM...')
    model = SVM(args.input_size, args.output_size)
elif args.clf == 'resnet18':
    print('Initializing ResNet18...')
    model = resnet.resnet18(
        num_channels=args.num_channels,
        num_classes=args.output_size)

model.load_state_dict(torch.load(init_path))
print('Load init: {}'.format(init_path))
model = nn.DataParallel(model.to(device), device_ids=args.device_id)

if 'sgd' in args.paradigm:
    optim = optim.SGD(params=model.parameters(), lr=args.lr)
elif 'adam' in args.paradigm:
    optim = optim.Adam(params=model.parameters(), lr=args.lr)

loss_fn = multiClassHingeLoss() if args.clf == 'svm' else F.nll_loss
loss_type = 'hinge' if args.clf == 'svm' else 'nll'

print('+'*80)

best = 0

x_ax = []
acc_train = []
acc_test = []
l_train = []
l_test = []

print('Pre-Training')
test(args, model, device, test_loader, best, 1, loss_type)

grad = []
grad_angle_prev = []
grad_angle_strt = []
grad_init = []

wait = 0
for epoch in range(1, args.epochs + 1):
    gradi = []
    correcti = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        optim.zero_grad()
        loss.backward()
        gradi.append([p.grad.clone() for p in model.parameters()])
        predi = output.argmax(1, keepdim=True)
        correcti += predi.eq(target.view_as(predi)).sum().item()

    grad_accum = accumulate_grads_over_epochs([gradi], device)
    for p, g in zip(model.parameters(), grad_accum[0]):
        p.grad = g.clone()
    optim.step()

    grad.append([[ii.cpu() for ii in i] for i in gradi])
    x_ax.append(epoch)
    acc_i = correcti/len(train_loader.dataset)
    acc_train.append(acc_i)
    l_train.append(loss.item())

    acc, loss = test(args, model, device, test_loader,
                     best, epoch, loss_type, False)
    acc_test.append(acc)
    l_test.append(loss)

    if epoch % args.log_interval == 0:
        print('Train Epoch w\\ {}: {} \tTrain: {:.4f} ({:.2f})'
              ' \t Test: {:.4f} ({:.2f})'.format(
                  args.paradigm, epoch, l_train[-1], acc_train[-1],
                  l_test[-1], acc_test[-1]))

    if acc > best:
        wait = 0
        best = acc
    else:
        wait += 1

    if wait >= 3 and args.early_stopping:
        print('Early stopping...')
        break

hist_file = '{}/{}/history/{}.pkl'.format(ckpt_path, folder, model_name)
pkl.dump((x_ax, acc_train, acc_test, l_train,
          l_test, grad), open(hist_file, 'wb'))
print('Saved: ', hist_file)

if args.dry_run:
    print("Remove: ", plot_file)
    os.remove(plot_file)
    print("Remove: ", hist_file)
    os.remove(hist_file)


if not args.dry_run:
    log_file.close()
    sys.stdout = std_out
