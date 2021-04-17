"""
Centralized training
use the gradient-component for approximations
pca approximated gradients
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
import torch.optim as optim

from argparser import argparser
from arguments import Arguments
from fcn import FCN
from multi_class_hinge_loss import multiClassHingeLoss
from svm import SVM
from train import test
from utils import accumulate_grads_over_epochs, \
    get_testloader, get_trainloader, gradient_approximation, vec_unit_dot

import functools
print = functools.partial(print, flush=True)

# Setups
args = Arguments(argparser())
hook = sy.TorchHook(torch)
USE_CUDA = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda:{}".format(args.device_id) if USE_CUDA else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}
kwargs = {}

ckpt_path = '../ckpts'
folder = '{}_{}'.format(args.dataset, 'centralized')

model_name = 'clf_{}_noise_{}_paradigm_{}_kgrad_{}' \
    '_lr_{}_decay_{}_batch_{}_distributed_sim'.format(
        args.clf, args.noise, args.paradigm, args.kgrads,
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

train_loader = get_trainloader(args.dataset, args.batch_size)
test_loader = get_testloader(
    args.dataset, args.test_batch_size, noise=args.noise)
print('+'*80)

# Fire the engines
if args.clf == 'fcn':
    print('Initializing FCN...')
    model_class = FCN
elif args.clf == 'svm':
    print('Initializing SVM...')
    model_class = SVM

model = model_class(args.input_size, args.output_size).to(device)

model.load_state_dict(torch.load(init_path))
print('Load init: {}'.format(init_path))

if args.paradigm == 'sgd':
    optim = optim.SGD(params=model.parameters(), lr=args.lr)
elif args.paradigm == 'adam':
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
grad_w, grad_b = [], []

wait = 0
curr_paradigm = args.paradigm
for epoch in range(1, args.epochs + 1):
    gradi = []
    predi = 0
    correcti = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        optim.zero_grad()
        loss.backward()
        if len(grad_w) == args.kgrads:
            curr_paradigm = 'kgrad'
            gradient_approximation(
                model, [grad_w, grad_b], device)
        gradi.append([p.grad.clone() for p in model.parameters()])
        predi = output.argmax(1, keepdim=True)
        correcti += predi.eq(target.view_as(predi)).sum().item()
    grad_accum = accumulate_grads_over_epochs([gradi], device)
    for p, g in zip(model.parameters(), grad_accum[0]):
        p.grad = g.clone()
    optim.step()

    if len(grad_w) < args.kgrads:
        grad_w.append(grad_accum[0][0].flatten())
        grad_b.append(grad_accum[0][1].flatten())
    if len(grad_w) == args.kgrads and type(grad_w) == list:
        grad_w = torch.stack(grad_w, dim=0)
        grad_b = torch.stack(grad_b, dim=0)
    grad.append(gradi)
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
                  curr_paradigm, epoch, l_train[-1], acc_train[-1],
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

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(221)
ax2 = plt.twinx()
l1 = ax1.plot(x_ax, acc_test, 'b', label='test acc')
l1_ = ax1.plot(x_ax, acc_train, 'b.-.', label='train acc')
ax1.set_ylabel('accuracy')

l2 = ax2.plot(x_ax, l_test, 'r', label='test loss'.format(loss_type))
l2_ = ax2.plot(x_ax, l_train, 'r.-.', label='train loss'.format(loss_type))
ax2.set_ylabel('{} loss'.format(loss_type))
ax2.set_xlabel('epochs')
ax1.legend()
ax2.legend()
ax2.grid()

num_params = 0
for p in model.parameters():
    num_params += p.flatten().size()[0]

ax3 = fig.add_subplot(222)
uplink = []
downlink = []
total = []
for i in x_ax:
    if i <= args.kgrads:
        u = num_params * i
    else:
        u = (num_params * args.kgrads) + (args.kgrads * i)
    d = num_params * i
    uplink.append(u)
    downlink.append(d)
    total.append(u+d)
l3 = ax3.plot(x_ax, total, 'r', label='total')
l3 = ax3.plot(x_ax, uplink, 'b', label='uplink')
l3 = ax3.plot(x_ax, downlink, 'g', label='downlink')
ax3.set_ylabel('cumulative num params'.format(loss_type))
ax3.set_xlabel('epochs')
ax3.legend()
ax3.grid()

grad = accumulate_grads_over_epochs(grad, device)
grad_f = grad[0]
vec_angle_f = defaultdict(list)  # vector angle w.r.t to first gradient
vec_angle_name = {}
for g_epoch in grad[1:]:
    for idx, g_layer in enumerate(g_epoch):
        vec_angle_f[idx].append(vec_unit_dot(grad_f[idx], g_layer))
        if idx not in vec_angle_name:
            vec_angle_name[idx] = g_layer.shape
        else:
            assert g_layer.shape == vec_angle_name[idx]

vec_angle_c = defaultdict(list)  # vector angle w.r.t the previous gradient
for idx, g_epoch in enumerate(grad[1:], 1):
    grad_p = grad[idx-1]
    for idx, g_layer in enumerate(g_epoch):
        vec_angle_c[idx].append(vec_unit_dot(grad_p[idx], g_layer))
        if idx not in vec_angle_name:
            vec_angle_name[idx] = g_layer.shape
        else:
            assert g_layer.shape == vec_angle_name[idx]

ax4 = fig.add_subplot(223)
for idx, cos in vec_angle_f.items():
    ax4.plot(x_ax[1:], cos, label=vec_angle_name[idx])
ax4.set_ylabel('dot w.r.t first'.format(loss_type))
ax4.set_xlabel('epoch')
ax4.legend()
ax4.grid()

ax5 = fig.add_subplot(224)
for idx, cos in vec_angle_c.items():
    ax5.plot(x_ax[1:], cos, label=vec_angle_name[idx])
ax5.set_ylabel('dot w.r.t prev'.format(loss_type))
ax5.set_xlabel('epoch')
ax5.legend()
ax5.grid()

fig.subplots_adjust(wspace=0.5)
plot_file = '{}/{}/plots/{}.jpg'.format(ckpt_path, folder, model_name)
plt.savefig(plot_file, bbox_inches='tight', dpi=150)
print('Saved: ', plot_file)
if args.dry_run:
    print("Remove: ", plot_file)
    os.remove(plot_file)
    print("Remove: ", hist_file)
    os.remove(hist_file)


if not args.dry_run:
    log_file.close()
    sys.stdout = std_out
