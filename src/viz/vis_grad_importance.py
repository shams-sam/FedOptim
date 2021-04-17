import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from sklearn.decomposition import PCA
import torch as t

import argparse
from utils import accumulate_grads_over_epochs, Struct

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=True)
ap.add_argument('--paradigms', type=str, nargs='+', required=True)
ap.add_argument('--batch-sizes', type=int, nargs='+', required=True)
ap.add_argument('--dpi', type=int, required=True)
ap.add_argument('--rows', type=int, required=True)
ap.add_argument('--cols', type=int, required=True)
args = vars(ap.parse_args())
args = Struct(**args)

dataset = args.dataset
paradigms = args.paradigms
batches = args.batch_sizes
dpi = args.dpi
rows, cols = args.rows, args.cols
device = t.device("cuda:{}".format(2))

for paradigm in paradigms:
    fig = plt.figure(figsize=(5*cols, (4*rows)-1))
    c = 1
    for batch in batches:
        file = '../ckpts/{}_centralized/history/clf_fcn_noise_None' \
            '_paradigm_{}_lr_0.01_decay_1e-05_batch_{}.pkl'.format(
                dataset, paradigm, batch)
        x_ax, acc_train, acc_test, l_train, l_test, grad = pkl.load(
            open(file, 'rb'))
        grad = accumulate_grads_over_epochs(grad, device)
        grad0 = t.stack([_[0].flatten() for _ in grad], dim=0).T
        grad1 = t.stack([_[1].flatten() for _ in grad], dim=0).T

        pca = PCA()
        pca.fit(grad0.cpu().numpy())
        exp = pca.explained_variance_ratio_[:10]

        pca.fit(grad1.cpu().numpy())
        exp1 = pca.explained_variance_ratio_[:10]

        ax = fig.add_subplot(100*rows+10*cols+c)
        ax.bar(np.array(list(range(1, len(exp)+1))) -
               0.25, exp, color='b', width=0.5)
        ax.bar(np.array(list(range(1, len(exp)+1))) +
               0.25, exp1, color='y', width=0.5)
        ax.grid()
        ax.set_title('{} w/ batch-size {}'.format(paradigm, batch))
        c += 1
    name = '../ckpts/{}_centralized/plots/vis_grad_importance_{}'\
           '.png'.format(dataset, paradigm)
    print('Saving: {}'.format(name))
    fig.subplots_adjust(wspace=0.3)
    plt.savefig(name, bbox_inches='tight', dpi=args.dpi)
    plt.show()
