import os
import pickle as pkl

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['lines.linewidth'] = 5.0
matplotlib.rcParams['lines.markersize'] = 7


ap = argparse.ArgumentParser()
ap.add_argument("--h", required=True, type=str, nargs='+')
ap.add_argument("--loss-type", required=True, type=str, nargs='+')
ap.add_argument("--ylim1", required=True, type=, nargs='+')
ap.add_argument("--ylim2", required=True, type=int, nargs='+')
ap.add_argument("--final", required=False, type=int, default=0)
ap.add_argument("--save", required=True, type=str)
ap.add_argument("--dry-run", required=True, type=int)
ap.add_argument("--models", required=True, type=str, nargs='+')

args = vars(ap.parse_args())
histories = args['h']
loss_types = args['loss_type']
ylim1 = args['ylim1']
ylim2 = args['ylim2']
final = args['final']
save_path = args['save']
dry_run = args['dry_run']
models = args['models']
assert len(histories) == len(loss_types)

cols = len(histories)
fig = plt.figure(figsize=(5 * cols, 8))
plot_idx = 1
for j, history in enumerate(histories):
    hist_file = history
    meta_file = history[:-4] + '_processed_grads.pkl'
    print(hist_file)

    if dry_run:
        assert os.path.exists(hist_file)
        assert os.path.exists(meta_file)
        continue

    h = pkl.load(open(hist_file, 'rb'))
    meta = pkl.load(open(meta_file, 'rb'))
    mat_99 = meta['mat_99']
    mat_95 = meta['mat_95']
    l_params = meta['num_params']
    num_layers = len(l_params)
    epoch, train_acc, test_acc, train_loss, test_loss, grads = h

    mu_99 = mat_99.mean(axis=1)
    std_99 = mat_99.std(axis=1)
    mu_95 = mat_95.mean(axis=1)
    std_95 = mat_95.std(axis=1)
    n_epochs = mat_99.shape[0]
    assert len(epoch) == mat_99.shape[0]

    ax1 = fig.add_subplot(2, cols, plot_idx)
    ax2 = fig.add_subplot(2, cols, plot_idx + cols)
    plot_idx += 1

    ylabel = 'MSE loss' if loss_types[j] == 'mse' else 'accuracy'

    ln1 = ax1.plot(epoch, mu_99, 'b', label='N99-PCA')
    ax1.fill_between(epoch, mu_99 - std_99,
                     mu_99 + std_99, color='b', alpha=0.1)
    ln2 = ax1.plot(epoch, mu_95, 'r', label='N95-PCA')
    ax1.fill_between(epoch, mu_95 - std_95,
                     mu_95 + std_95, color='r', alpha=0.1)
    ln3 = ax2.plot(epoch, np.array(test_acc), 'k', label='accuracy/loss')

    if plot_idx == 2:
        ax1.set_ylabel('#components', fontsize=30)
        ax2.set_ylabel(ylabel, fontsize=30)
    ax1.set_title(models[j].replace(":", "\n"), fontsize=30)
    ax2.set_xlabel('epoch', fontsize=30)
    ax1.set_xticks(list(range(0, n_epochs, n_epochs // 4)))
    # ax1.set_xticklabels([])
    ax2.set_xticks(list(range(0, n_epochs, n_epochs // 4)))
    ax1.grid()
    ax2.grid()
    ax1.set_xlim(0, n_epochs - 1)
    ax2.set_xlim(0, n_epochs - 1)
    ax1.set_ylim(0, ylim1[j])
    ax2.set_ylim(0, ylim2[j])
    if plot_idx != 5:
        continue
    lns = ln1 + ln2  # + ln3
    labs = [lab.get_label() for lab in lns]
    ax1.legend(lns, labs, loc=2, fontsize=25
               # , ncol=3, bbox_to_anchor=(0, 1.15, 5.5, .25),
               # mode='expand', frameon=True
               )

plt.subplots_adjust(hspace=0.25, wspace=0.3)
if not final and not dry_run:
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=100)
else:
    plt.savefig(save_path + '.pdf', bbox_inches='tight', dpi=300)
