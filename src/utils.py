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
