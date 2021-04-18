import numpy as np
import pickle as pkl
from sklearn.random_projection import johnson_lindenstrauss_min_dim as jl_dim
import torch

from common.nb_utils import pca_transform
from common.utils import decor_print, get_device, get_random_vector
from data.loader import get_dataloader
from models.utils import forward, get_loss_fn, get_model, get_optim
from models.model_op import get_layer_size, get_model_grads


def stack_layers(sdirs):
    stacked = []
    for layer_num in range(len(sdirs[0])):
        stacked.append(
            np.hstack(
                [_[layer_num].reshape(-1, 1).cpu().numpy() for _ in sdirs]
            )
        )

    return stacked


def get_dga_sdirs(args, data, labels):
    device = get_device(args)
    sdirs = []
    for x, y in zip(data, labels):
        # dga_bs: dist grad accum. batch size
        dataloader = get_dataloader(x, y, args.dga_bs, shuffle=False)
        count = 0
        for xiter, yiter in dataloader:
            model, loss_type = get_model(args, False)
            loss_fn = get_loss_fn(loss_type)
            opt = get_optim(args, model)

            loss, _ = forward(
                model, xiter, yiter, opt, loss_fn, device)
            loss.backward()
            sdirs.append(get_model_grads(model, flatten=True))
            count += 1
            if count >= args.num_dga:
                break

    stacked = [[] for _ in range(len(sdirs[0]))]

    for l in range(len(sdirs[0])):
        for i in range(len(sdirs)):
            stacked[l].append(sdirs[i][l].flatten())

    sdirs = [[] for _ in range(args.ncomponent)]
    for l, layer in enumerate(stacked):
        layer = torch.stack(layer, dim=0).T.cpu().numpy()
        layer, _ = pca_transform(layer, args.ncomponent)
        for i in range(args.ncomponent):
            sdirs[i].append(layer[:, i].flatten())

    assert len(sdirs) == args.ncomponent

    return sdirs


def get_jl_dim(samples, eps):
    return max([jl_dim(s, eps)[0] for s in samples])


def load_sdirs(path):
    sdirs = pkl.load(open(path, 'rb'))
    sdirs = [[torch.Tensor(l) for l in sdir] for sdir in sdirs]
    decor_print('ncomponents: {}'.format(len(sdirs)))

    return sdirs


def get_rp_dirs(args, model):
    if not args.ncomponent:
        num_sdirs = get_jl_dim(layer_sizes, args.rp_eps)
    else:
        num_sdirs = args.ncomponent
    layer_sizes = get_layer_size(model)
    decor_print('Number of directions for eps {}: {}'.format(
        args.rp_eps, num_sdirs))

    return [
        [
            get_random_vector(1.0, np.sqrt(num_sdirs), s)
            for s in layer_sizes
        ] for _ in range(num_sdirs)
    ]


def get_rp_block(args, model):
    if not args.ncomponent:
        num_sdirs = get_jl_dim(layer_sizes, args.rp_eps)
    else:
        num_sdirs = args.ncomponent
    layer_sizes = get_layer_size(model)
    decor_print('Number of directions for eps {}: {}'.format(
        args.rp_eps, num_sdirs))

    return [
        get_random_vector(1.0, np.sqrt(num_sdirs), (num_sdirs, s[0]))
        for s in layer_sizes
    ]


def get_sdirs(args, model, paths, X, y):
    sdirs = []
    if args.paradigm:
        if 'rp' in args.paradigm:
            sdirs = get_rp_dirs(args, model)
        elif 'pca' in args.paradigm:
            print('Loading: {}'.format(paths.pca_path))
            sdirs = load_sdirs(paths.pca_path)
        elif 'dga' in args.paradigm:
            print('Loading: {}'.format(paths.dga_path))
            sdirs = load_sdirs(paths.dga_path)

    return sdirs
