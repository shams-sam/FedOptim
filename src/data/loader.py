import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

import common.config as cfg
from common.utils import AddGaussianNoise


# https://github.com/kuc2477/pytorch-ewc/blob/4afaa6666d6b4f1a91a110caf69e7b77f049dc08/data.py#L4
def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


def _get_transform(mean, std, noise, permutation):
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if noise:
        transform.append(AddGaussianNoise(0, noise))
    if permutation:
        transform.append(transforms.Lambda(
            lambda x: _permutate_image_pixels(x, permutation)))

    return transform


def get_dataloader(data, targets, batchsize, shuffle=False):
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batchsize,
                      shuffle=shuffle, num_workers=1)


def get_loader(dataset, batch_size, train=True,
               shuffle=True, noise=False, permutation=False):
    kwargs = {}
    if dataset == 'mnist':
        transform = _get_transform((0.1307,), (0.3081,), noise, permutation)
        loader = torch.utils.data.DataLoader(
            datasets.MNIST(cfg.data_dir, train=train,
                           download=cfg.download,
                           transform=transforms.Compose(transform)),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'cifar':
        transform = _get_transform(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
            noise, permutation)
        loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(cfg.data_dir, train=train,
                             download=cfg.download,
                             transform=transforms.Compose(transform)),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'fmnist':
        transform = _get_transform((0.2861,), (0.3530,), noise, permutation)
        loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(cfg.data_dir, train=train,
                                  download=cfg.download,
                                  transform=transforms.Compose(transform)),
            batch_size=batch_size, shuffle=shuffle, **kwargs)

    return loader
