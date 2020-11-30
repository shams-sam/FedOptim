import config as cfg
from math import factorial as f
import networkx as nx
import numpy as np
from random import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


def booltype(arg):
    return bool(int(arg))


def decimal_format(num, places=4):
    return round(num, places)


def flip(p):
    return True if random() < p else False


def get_dataloader(data, targets, batchsize, shuffle=False):
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batchsize,
                      shuffle=shuffle, num_workers=1)


class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_testloader(dataset, batch_size, shuffle=True, noise=False):
    kwargs = {}
    if dataset == 'mnist':
        if noise:
            return torch.utils.data.DataLoader(
                datasets.MNIST(cfg.data_dir, train=False,
                               download=cfg.download,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                                   AddGaussianNoise(0, 1.0)
                               ])),
                batch_size=batch_size, shuffle=shuffle, **kwargs)
        else:
            return torch.utils.data.DataLoader(
                datasets.MNIST(cfg.data_dir, train=False,
                               download=cfg.download,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                               ])),
                batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'cifar':
        return torch.utils.data.DataLoader(
            datasets.CIFAR10(cfg.data_dir, train=False,
                             download=cfg.download,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'fmnist':
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST(cfg.data_dir, train=False,
                                  download=cfg.download,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.2861,),
                                                           (0.3530,))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_trainloader(dataset, batch_size, shuffle=True):
    kwargs = {}
    if dataset == 'mnist':
        return torch.utils.data.DataLoader(
            datasets.MNIST(cfg.data_dir, train=True,
                           download=cfg.download,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'cifar':
        return torch.utils.data.DataLoader(
            datasets.CIFAR10(cfg.data_dir, train=True,
                             download=cfg.download,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'fmnist':
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST(cfg.data_dir, train=True,
                                  download=cfg.download,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.2861,),
                                                           (0.3530,))])),
            batch_size=batch_size, shuffle=shuffle, **kwargs)


def in_range(elem, upper, lower):
    return (elem >= lower) and (elem <= upper)


def nCr(n, r):
    return f(n)//f(r)//f(n-r)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


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


def accumulate_grads_over_epochs(grad, device):
    g_res = []
    for g_ep in grad:
        g_res_bt = [torch.zeros(_.size()).to(device) for _ in grad[0][0]]
        c = 0
        for g_bt in g_ep:
            for idx, (gi, gii) in enumerate(zip(g_res_bt, g_bt)):
                g_res_bt[idx] = gi + gii
            c += 1
        g_res_bt = [_/c for _ in g_res_bt]
        g_res.append(g_res_bt)

    return g_res


def gradient_approximation(model, grad_dirs, device):
    for p, g in zip(model.parameters(), grad_dirs):
        grad = p.grad.clone()
        size = grad.size()
        res = torch.zeros(size).flatten().to(device)
        for idx in range(g.size()[0]):
            res += calc_projection(grad.flatten(), g[idx]) * g[idx]
        p.grad = res.reshape(size).clone()


def calc_projection(a, b):
    # projection of vector a on vector b
    return (torch.dot(a, b)/torch.dot(b, b))


def weights_init(m, init):
    for param in m.parameters():
        if init == 'normal':
            torch.nn.init.normal(param)
        elif init == 'uniform':
            torch.nn.init.uniform(param)
        elif init == 'xavier':
            torch.nn.init.xavier_uniform(param)


def weights_reset(m, indices):
    for params, idxs in zip(m.parameters(), indices):
        print(params.shape)
        for idx in idxs:
            if len(params.shape) > 1:
                params[:, idx] = torch.normal(mean=torch.Tensor([0.0]),
                                              std=torch.Tensor([1.0]))[0]
            else:
                params[idx] = torch.normal(mean=torch.Tensor([0.0]),
                                           std=torch.Tensor([1.0]))[0]
