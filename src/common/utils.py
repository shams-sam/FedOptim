import os
from random import random
import sys

from math import factorial as f
import numpy as np
import torch
import torchvision

import common.config as cfg


class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def add_gaussian_noise(args, tensor):
    noise = AddGaussianNoise(0.0, args.noise)
    return noise(tensor)


def booltype(arg):
    return bool(int(arg))


def decimal_format(num, places=4):
    return round(num, places)


def decor_print(msg):
    print('+'*80)
    print(msg)


def flip(p):
    return True if random() < p else False


def get_device(args):
    USE_CUDA = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    return torch.device("cuda:{}".format(
        args.device_id[0]) if USE_CUDA else "cpu")


def get_paths(args, distributed=False):
    ckpt_path = cfg.ckpt_dir
    tb_path = cfg.tb_dir
    if args.num_workers:
        folder = '{}_{}'.format(args.dataset, args.num_workers)
    else:
        folder = args.dataset
    path = {}

    model_name = 'clf_{}_optim_{}_uniform_{}_non_iid_{}' \
                 '_num_workers_{}_lr_{}_decay_{}_batch_{}'.format(
                     args.clf, args.optim, args.uniform_data,
                     args.non_iid if args.non_iid else 0,
                     args.num_workers if args.num_workers else 0,
                     args.lr, args.decay, args.batch_size
                 )
    pca_ = args.ncomponent if args.ncomponent else args.pca_var
    dga_ = 'dga_{}_bs_{}'.format(args.num_dga, args.dga_bs)
    path['pca_path'] = os.path.join(
        ckpt_path, folder, 'models',  model_name + '.pca_{}'.format(pca_))
    path['dga_path'] = os.path.join(
        ckpt_path, folder, 'models', 'clf_{}_{}_non_iid_{}.pkl'.format(
            args.clf, dga_, args.non_iid
        ))

    if args.dry_run:
        print(model_name)
        model_name = 'debug'
    if args.noise:
        model_name += '_noise_{}'.format(args.noise)
    if args.paradigm:
        if 'atomo' in args.paradigm:
            model_name += '_atomo_{}'.format(args.atomo_r)
        if 'signsgd' in args.paradigm:
            model_name += '_signsgd'
        if 'topk' in args.paradigm:
            model_name += '_topk_{}'.format(args.topk)
        if 'pca' in args.paradigm:
            model_name += '_pca_{}'.format(pca_)
        if 'lbgm' in args.paradigm:
            model_name += '_lbgm_{}'.format(args.error_tol)
        elif 'rp' in args.paradigm:
            model_name += '_rp_{}'.format(
                args.rp_eps if not args.ncomponent else args.ncomponent)
        elif 'kgrad' in args.paradigm:
            model_name += '_kgrad_{}_tol_{}'.format(
                args.kgrads, args.error_tol)
        if args.residual:
            model_name += '_residual'
        if args.sdir_full:
            model_name += '_full'
    if args.start_epoch != 1:
        model_name += '_start_from_{}'.format(args.start_epoch)
    if distributed:
        model_name += '_distributed'

    path['model_name'] = model_name
    path['log_file'] = '{}/{}/logs/{}.log'.format(
        ckpt_path, folder, model_name)
    path['tb_path'] = '{}/{}_{}'.format(
        tb_path, folder, model_name)
    path['init_path'] = '{}/{}/{}_{}.init'.format(ckpt_path, 'init',
                                                  args.dataset, args.clf)
    path['best_path'] = os.path.join(
        ckpt_path, folder, 'models',  model_name + '.best')
    path['stop_path'] = os.path.join(
        ckpt_path, folder, 'models',  model_name + '.stop')
    path['data_path'] = '../ckpts/{}_{}/data/n_classes_per_node_{}' \
                        '_stratify_{}_uniform_{}_repeat_{}.pkl'.format(
                            args.dataset, args.num_workers, args.non_iid,
                            args.stratify, args.uniform_data, args.repeat)
    path['hist_path'] = '{}/{}/history/{}.pkl'.format(
        ckpt_path, folder, model_name)
    path['plot_path'] = '{}/{}/plots/{}.jpg'.format(
        ckpt_path, folder, model_name)

    return Struct(**path)


def get_random_vector(mean, std, size):
    return torch.normal(mean, std, size)


def in_range(elem, upper, lower):
    return (elem >= lower) and (elem <= upper)


def init_logger(log_file, dry_run, append):
    print("Logging: ", log_file)
    std_out = sys.stdout
    if not dry_run:
        log_file = open(
            log_file,
            'a' if os.path.exists(log_file) and append else 'w'
        )
        sys.stdout = log_file

    return log_file, std_out


def is_approx(args):
    return args.paradigm and (
        'kgrad' in args.paradigm or
        'rp' in args.paradigm or
        'pca' in args.paradigm or
        'dga' in args.paradigm
    )


def nCr(n, r):
    return f(n)//f(r)//f(n-r)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def tb_model_summary(model, test_loader, tb, device):
    for data, _ in test_loader:
        grid = torchvision.utils.make_grid(data[:64])
        tb.add_image('test images', grid)
        tb.add_graph(model.module, data.to(device))
        break


def vec_angle(firsts, currents):
    angles = []
    for a, b in zip(firsts, currents):
        dot = torch.dot(a.flatten(), b.flatten())
        a_norm = torch.norm(a)
        b_norm = torch.norm(b)
        cos = dot/(2*a_norm*b_norm)
        angles.append(torch.acos(cos).item() * 180/3.14)
    angles = np.array(angles)

    return angles.mean(), angles.std()


def vec_unit_dot(a, b):
    dot = torch.dot(a.flatten(), b.flatten())
    a_norm = torch.norm(a)
    b_norm = torch.norm(b)
    cos = dot/(a_norm*b_norm)

    return cos.item()
