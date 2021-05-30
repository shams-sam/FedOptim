from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

import common.config as cfg
from common.utils import AddGaussianNoise
from data.coco import Coco
from datasets import load_dataset


def _get_subset_index(classes, split=0.3):
    _, _, idx_train, idx_valid = train_test_split(
        classes, list(range(len(classes))), test_size=split
    )

    return idx_valid


def _get_transform(params):
    transform = []
    if 'im_size' in params:
        im_size = params['im_size']
        transform.append(transforms.Resize((im_size, im_size)))
    transform += [
        transforms.ToTensor(),
        transforms.Normalize(params['mean'], params['std']),
    ]
    if 'noise' in params:
        transform.append(AddGaussianNoise(0, params['noise']))
    if 'permutation' in params:
        transform.append(transforms.Lambda(
            lambda x: _permutate_image_pixels(
                x, params['permutation'])))

    return transforms.Compose(transform)


# https://github.com/kuc2477/pytorch-ewc/blob/4afaa6666d6b4f1a91a110caf69e7b77f049dc08/data.py#L4
def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


def get_dataloader(data, targets, batchsize, shuffle=False):
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batchsize,
                      shuffle=shuffle, num_workers=1)


def get_loader(dataset, batch_size, train=True,
               shuffle=True, subset=1.0, force_resize=0,
               noise=False, permutation=False):
    kwargs = {
        # 'num_workers': 1,
        # 'pin_memory': False,
    }
    if dataset == 'amazon':
        train = 'train' if train else 'valid'
        dataset = load_dataset('amazon_reviews_multi', 'en',
                               cache_dir=data_dir, split=train)
        return dataset
    elif dataset == 'celeba':
        train = 'train' if train else 'valid'
        params = {
            'mean': (0.5063, 0.4258, 0.3832),
            'std': (0.3106, 0.2904, 0.2897),
            'im_size': cfg.im_size[dataset]
        }
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        dataset = datasets.CelebA(cfg.data_dir, split=train,
                                  download=cfg.download,
                                  transform=transform,
                                  target_type='landmarks')
        if subset < 1:
            indices = _get_subset_index(dataset.attr, subset)
            dataset = Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'cifar':
        params = {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
        }
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(cfg.data_dir, train=train,
                             download=cfg.download,
                             transform=transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'coco':
        train = 'train' if train else 'val'
        params = {
            'mean': (0.4701, 0.4469, 0.4076),
            'std': (0.2463, 0.2424, 0.2596),
            'im_size': cfg.im_size[dataset]
        }
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((cfg.im_size[dataset], cfg.im_size[dataset])),
            transforms.Lambda(lambda x: torch.from_numpy(
                np.array(x).astype(int))),
        ])

        dataset = Coco(
            root='{}/coco/{}2017'.format(cfg.data_dir, train),
            annFile='{}/coco/annotations/instances_{}2017.json'.format(
                cfg.data_dir, train),
            transform=transform,
            target_transform=target_transform,
        )
        if subset < 1:
            indices = _get_subset_index(list(range(len(dataset))), subset)
            dataset = Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'fmnist':
        params = {
            'mean': (0.2861,),
            'std': (0.3530,),
        }
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(cfg.data_dir, train=train,
                                  download=cfg.download,
                                  transform=transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'mnist':
        params = {
            'mean': (0.1307,),
            'std': (0.3081,),
        }
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        loader = torch.utils.data.DataLoader(
            datasets.MNIST(cfg.data_dir, train=train,
                           download=cfg.download,
                           transform=transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'svhn':
        train = 'train' if train else 'test'
        params = {
            'mean': (0.4377, 0.4438, 0.4728),
            'std': (0.1980, 0.2010, 0.1970),
        }
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        loader = torch.utils.data.DataLoader(
            datasets.SVHN('{}/SVHN'.format(cfg.data_dir), split=train,
                          download=cfg.download,
                          transform=transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'voc':
        train = 'train' if train else 'val'
        params = {
            'mean': (0.4568, 0.4432, 0.4083),
            'std': (0.2440, 2414, 2591),
            'im_size': cfg.im_size[dataset]
        }
        if noise:
            params['noise'] = noise
        if force_resize:
            params['im_size'] = force_resize
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)
        target_transform = transforms.Compose([
            transforms.Resize((cfg.im_size[dataset], cfg.im_size[dataset])),
            transforms.Lambda(lambda x: torch.from_numpy(
                np.array(x).astype(int))),
            transforms.Lambda(lambda x: torch.clamp(
                x, max=cfg.output_sizes[dataset] - 1)),
        ])

        loader = torch.utils.data.DataLoader(
            datasets.VOCSegmentation('{}/VOC'.format(cfg.data_dir), image_set=train,
                                     download=cfg.download,
                                     transform=transform,
                                     target_transform=target_transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)

    return loader
