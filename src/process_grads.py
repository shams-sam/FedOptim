import os
import pickle as pkl

import argparse
import numpy as np
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm

from common.nb_utils import estimate_optimal_ncomponents, pca_transform


ap = argparse.ArgumentParser()
ap.add_argument("--h", required=True, type=str, nargs='+')
ap.add_argument("--m", required=True, type=float)
ap.add_argument("--dry-run", required=True, type=int)

args = vars(ap.parse_args())
histories = args['h']
momentum = args['m']
dry_run = args['dry_run']


def momentum_grad(prev_v, grad, momentum):
    return momentum * prev_v + grad


for history in histories:
    print('Processing: {}'.format(history))
    save_path = '{}_processed_grads.pkl'.format(history[:-4])
    if dry_run:
        print('training : {}'.format(os.path.exists(history)))
        print('processed: {}\n\n'.format(os.path.exists(save_path)))
        continue
    h = pkl.load(open(history, 'rb'))

    epoch, train_acc, test_acc, train_loss, test_loss, grads = h

    mat_99 = np.zeros((len(epoch), len(grads[0])))
    mat_95 = np.zeros((len(epoch), len(grads[0])))
    overlap_99 = []
    overlap_95 = []
    overlap_self = []
    prev_v = [torch.zeros_like(_) for _ in grads[0]]

    layers_processed = 0
    for layer_num in tqdm(range(len(grads[0]))):
        
        grad_stack = []
        for i, igrad in enumerate(grads):
            prev_v[layer_num] = momentum_grad(
                prev_v[layer_num], igrad[layer_num], momentum)
            grad_stack.append(prev_v[layer_num].cpu().flatten().numpy())
            skip_layer = not prev_v[layer_num].sum().item()
            if not skip_layer:
                n_99 = estimate_optimal_ncomponents(
                    np.vstack(grad_stack).T, 0.99)
                n_95 = estimate_optimal_ncomponents(
                    np.vstack(grad_stack).T, 0.95)
                mat_99[i, layer_num] = n_99[0]
                mat_95[i, layer_num] = n_95[0]
            else:
                mat_99[i, layer_num] = 1
                mat_95[i, layer_num] = 1
            assert mat_95[i, layer_num] <= mat_99[i, layer_num]
        if skip_layer:
            print('skipping layer {}...'.format(layer_num))
            continue
        layers_processed += 1
        grad_stack = np.vstack(grad_stack).T

        pca_stack, _ = pca_transform(grad_stack, int(mat_99[i, layer_num]))
        assert _ > 0.985, "explained variance: {}".format(_)
        overlap_99.append(normalize(pca_stack, axis=0).T.dot(
            normalize(grad_stack, axis=0)))

        pca_stack, _ = pca_transform(grad_stack, int(mat_95[i, layer_num]))
        assert _ > 0.945, "explained variance: {}".format(_)
        overlap_95.append(normalize(pca_stack, axis=0).T.dot(
            normalize(grad_stack, axis=0)))

        grad_stack = normalize(grad_stack, axis=0)
        overlap_self.append(grad_stack.T.dot(grad_stack))

    num_params = {}
    for layer_num in range(len(grads[0])):
        num_params[layer_num] = grads[0][layer_num].flatten().size(0)

    print('Layers processed: {}/{}\n'.format(layers_processed, len(grads[0])))
    print('Saving: {}\n\n'.format(save_path))
    pkl.dump({
        'mat_99': mat_99,
        'mat_95': mat_95,
        'overlap_99': overlap_99,
        'overlap_95': overlap_95,
        'overlap_self': overlap_self,
        'num_params': num_params,
    }, open(save_path, 'wb'))
