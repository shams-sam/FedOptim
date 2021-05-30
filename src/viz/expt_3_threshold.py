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
ap.add_argument("--history", required=True, type=str, nargs='+')
ap.add_argument("--break-pt", required=True, type=int, nargs='+')
ap.add_argument("--loss-type", required=True, type=str, nargs='+')
ap.add_argument("--m-int", required=True, type=float, nargs='+')  # multiplier
ap.add_argument("--m-str", required=True, type=str, nargs='+')
ap.add_argument("--u-int", required=True, type=float, nargs='+')  # multiplier
ap.add_argument("--u-str", required=True, type=str, nargs='+')
ap.add_argument("--ylim1", required=True, type=float, nargs='+')
ap.add_argument("--ylim2", required=True, type=int, nargs='+')
ap.add_argument("--final", required=False, type=int, default=0)
ap.add_argument("--save", required=True, type=str)
ap.add_argument("--dry-run", required=True, type=int)
ap.add_argument("--models", required=True, type=str, nargs='+')
ap.add_argument("--labels", required=True, type=str, nargs='+')

args = vars(ap.parse_args())
histories = args['history']
bp = args['break_pt']
loss_types = args['loss_type']
m_int = args['m_int']
m_str = args['m_str']
u_int = args['u_int']
u_str = args['u_str']
ylim1 = args['ylim1']
ylim2 = args['ylim2']
final = args['final']
save_path = args['save']
dry_run = args['dry_run']
models = args['models']
labels = args['labels']
colors = ['k', 'r', 'g', 'b', 'y', 'c']


groups = [histories[bp[i - 1]: bp[i]] for i in range(1, len(bp))]
cols = len(groups)
fig = plt.figure(figsize=(5 * cols, 8))
plot_idx = 1
mse_flag = False
for j, group in enumerate(groups):
    ax1 = fig.add_subplot(2, cols, plot_idx)
    ax2 = fig.add_subplot(2, cols, plot_idx + cols)
    plot_idx += 1
    ylabel = 'MSE loss' if loss_types[j] == 'mse' else 'accuracy'
    lns = None
    for i, h in enumerate(group):

        if dry_run:
            assert os.path.exists(h)
            print(h)
            continue

        b_ep, b_acc, _, _, b_loss, _, _, b_up, _ = pkl.load(open(h, 'rb'))
        b_up = np.cumsum(b_up) / m_int[j]
        n_epochs = len(b_ep)
        label = r'$\rho_k^{{(threshold)}}={}$'.format(labels[i])
        if labels[i] == 'na':
            label = 'Vanilla FL'
        if loss_types[j] == 'ce':
            ln = ax1.plot(b_ep, b_acc, 'b', label=label, color=colors[i])
        else:
            b_loss = np.array(b_loss)/u_int[j]
            ln = ax1.plot(b_ep, b_loss, 'b', label=label, color=colors[i])
        lns = ln if not lns else lns+ln
        ax2.plot(b_ep, b_up, 'b', label=label, color=colors[i])

    if plot_idx == 2 or (loss_types[j] == 'mse' and not mse_flag):
        if loss_types[j] == 'mse' and not mse_flag:
            ylabel = r'{} ($\times {}$)'.format(ylabel, u_str[j])
            mse_flag = True
        ax1.set_ylabel(ylabel, fontsize=30)
        if plot_idx == 2:
            ylabel = 'agg. #params\n shared'
            if m_str != 'na':
                ylabel = r'{} ($\times {}$)'.format(ylabel, m_str[j])
            ax2.set_ylabel(ylabel, fontsize=30)
    ax1.set_title(models[j].replace(":", "\n"), fontsize=30, pad=20)
    ax2.set_xlabel('t', fontsize=30)
    ax1.set_xticks(list(range(0, n_epochs+1, n_epochs // 4)))
    # ax1.set_xticklabels([])
    ax2.set_xticks(list(range(0, n_epochs+1, n_epochs // 4)))
    ax1.grid()
    ax2.grid()
    ax1.set_xlim(0, n_epochs)
    ax2.set_xlim(0, n_epochs)
    ax1.set_ylim(0, ylim1[j])
    ax2.set_ylim(0, ylim2[j])
    if plot_idx != 2:
        continue

    labs = [lab.get_label() for lab in lns]
    ax1.legend(lns, labs, loc=2, fontsize=30,
               ncol=3, bbox_to_anchor=(-0.05, 1.15, 5.2, 1.0),
               mode='expand', frameon=False
               )

plt.subplots_adjust(hspace=0.25, wspace=0.35)
if not final and not dry_run:
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=100)
else:
    plt.savefig(save_path + '.pdf', bbox_inches='tight', dpi=300)
