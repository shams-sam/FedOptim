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


def _get_transform_params(dataset, train):
    if dataset == 'celeba':
        params = {
            'mean': (0.5063, 0.4258, 0.3832),
            'std': (0.3106, 0.2904, 0.2897),
            'im_size': cfg.im_size[dataset]
        }
    elif dataset == 'cifar':
        params = {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
        }
    elif dataset == 'cifar100':
        params = {
            'mean': (0.5071, 0.4866, 0.4409),
            'std': (0.2673, 0.2564, 0.2762),
        }
    elif dataset == 'coco':
        params = {
            'mean': (0.4701, 0.4469, 0.4076),
            'std': (0.2463, 0.2424, 0.2596),
            'im_size': cfg.im_size[dataset]
        }
    elif dataset == 'fmnist':
        params = {
            'mean': (0.2861,),
            'std': (0.3530,),
        }
    elif dataset == 'imagenet':
        params = {
            'mean': (0.4811, 0.4575, 0.4078),
            'std': (0.2554, 0.2483, 0.2636),
            'im_size': cfg.im_size[dataset],
        }
    elif dataset == 'miniimagenet':
        params = {
            'mean': (0.4731, 0.4489, 0.4034),
            'std': (0.2592, 0.2512, 0.2670),
            'im_size': cfg.im_size[dataset],
        }
    elif dataset == 'mnist':
        params = {
            'mean': (0.1307,),
            'std': (0.3081,),
        }
    elif dataset == 'svhn':
        params = {
            'mean': (0.4377, 0.4438, 0.4728),
            'std': (0.1980, 0.2010, 0.1970),
        }
    elif dataset == 'voc':
        params = {
            'mean': (0.4568, 0.4432, 0.4083),
            'std': (0.2440, 2414, 2591),
            'im_size': cfg.im_size[dataset]
        }

    if train and dataset in ['cifar', 'cifar100', 'imagenet', 'miniimagenet', 'svhn']:
        params['crop'] = True
        params['flip'] = True
        params['rotate'] = 15

    return params


def _get_transform(params):
    transform = []
    if 'im_size' in params:
        im_size = params['im_size']
        if 'crop' in params:
            transform.append(transforms.RandomCrop(im_size, padding=4))
        else:
            transform.append(transforms.Resize((im_size, im_size)))
    if 'flip' in params:
        transform.append(transforms.RandomHorizontalFlip())
    if 'rotate' in params:
        rotate = params['rotate']
        transform.append(transforms.RandomRotation(rotate))
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


def get_dataloader(data, targets, batchsize, dataset_name, train, shuffle=True):
    dataset = TensorDataset(data, targets)
    if dataset_name:
        dataset.transform = _get_transform(
            _get_transform_params(dataset_name, train)
        )
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
        params = _get_transform_params(dataset, train)
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        train = 'train' if train else 'valid'
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
        params = _get_transform_params(dataset, train)
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
    elif dataset == 'cifar100':
        params = _get_transform_params(dataset, train)
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(cfg.data_dir, train=train,
                              download=cfg.download,
                              transform=transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'coco':
        params = _get_transform_params(dataset, train)
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

        train = 'train' if train else 'val'
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
        params = _get_transform_params(dataset, train)
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
    elif dataset == 'imagenet':
        params = _get_transform_params(dataset, train)
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        train = 'train' if train else 'val'
        dataset = datasets.ImageNet(root='{}/ImageNet'.format(cfg.data_dir),
                                    split=train,
                                    transform=transform)
        if subset < 1:
            indices = _get_subset_index(dataset.targets, subset)
            dataset = Subset(dataset, indices)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'miniimagenet':
        params = _get_transform_params(dataset, train)
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        train = 'train' if train else 'test'
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='{}/MiniImageNet/{}'.format(cfg.data_dir, train),
                                 transform=transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'mnist':
        params = _get_transform_params(dataset, train)
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
        params = _get_transform_params(dataset, train)
        if force_resize:
            params['im_size'] = force_resize
        if noise:
            params['noise'] = noise
        if permutation:
            params['permutation'] = permutation
        transform = _get_transform(params)

        train = 'train' if train else 'test'
        loader = torch.utils.data.DataLoader(
            datasets.SVHN('{}/SVHN'.format(cfg.data_dir), split=train,
                          download=cfg.download,
                          transform=transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif dataset == 'voc':
        params = _get_transform_params(dataset, train)
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

        train = 'train' if train else 'val'
        loader = torch.utils.data.DataLoader(
            datasets.VOCSegmentation('{}/VOC'.format(cfg.data_dir), image_set=train,
                                     download=cfg.download,
                                     transform=transform,
                                     target_transform=target_transform),
            batch_size=batch_size, shuffle=shuffle, **kwargs)

    return loader
