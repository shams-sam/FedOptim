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
ap.add_argument("--baseline", required=True, type=str, nargs='+')
ap.add_argument("--ours", required=True, type=str, nargs='+')
ap.add_argument("--loss-type", required=True, type=str, nargs='+')
ap.add_argument("--m-int", required=True, type=float, nargs='+')  # multiplier
ap.add_argument("--m-str", required=True, type=str, nargs='+')
ap.add_argument("--u-int", required=True, type=float, nargs='+')  # multiplier
ap.add_argument("--u-str", required=True, type=str, nargs='+')
ap.add_argument("--ylim1", required=True, type=float, nargs='+')
ap.add_argument("--ylim2", required=True, type=int, nargs='+')
ap.add_argument("--xlim", required=True, type=int, nargs='+')
ap.add_argument("--wspace", required=False, type=float, default=0.35)
ap.add_argument("--final", required=False, type=int, default=0)
ap.add_argument("--save", required=True, type=str)
ap.add_argument("--dry-run", required=True, type=int)
ap.add_argument("--models", required=True, type=str, nargs='+')

args = vars(ap.parse_args())
baselines = args['baseline']
ours = args['ours']
loss_types = args['loss_type']
m_int = args['m_int']
m_str = args['m_str']
u_int = args['u_int']
u_str = args['u_str']
ylim1 = args['ylim1']
ylim2 = args['ylim2']
xlim = args['xlim']
wspace = args['wspace']
final = args['final']
save_path = args['save']
dry_run = args['dry_run']
models = args['models']

cols = len(baselines)
fig = plt.figure(figsize=(5 * cols, 12))
plot_idx = 1
mse_flag = False
for j, (h_baseline, h_ours) in enumerate(zip(baselines, ours)):

    if dry_run:
        print(h_baseline)
        assert os.path.exists(h_baseline)
        print(h_ours)
        assert os.path.exists(h_ours)
        continue

    b_ep, b_acc, _, _, b_loss, _, _, b_up, _ = pkl.load(open(h_baseline, 'rb'))
    o_ep, o_acc, _, _, o_loss, _, _, o_up, _ = pkl.load(open(h_ours, 'rb'))
    b_up, o_up = np.cumsum(b_up) / m_int[j], np.cumsum(o_up) / m_int[j]
    n_epochs = len(b_ep)

    ax1 = fig.add_subplot(3, cols, plot_idx)
    ax2 = fig.add_subplot(3, cols, plot_idx + cols)
    ax3 = fig.add_subplot(3, cols, plot_idx + 2*cols)
    plot_idx += 1

    ylabel = 'loss' if loss_types[j] == 'mse' else 'accuracy'

    if loss_types[j] == 'ce':
        ln1b = ax1.plot(b_ep, b_acc, 'k', label='Vanilla FL')
        ln1o = ax1.plot(o_ep, o_acc, 'r', label='LBGM')
        ln3b = ax3.plot(b_up, b_acc, 'k', label='Vanilla-FL')
        ln3b = ax3.plot(o_up, o_acc, 'r', label='LBGM-FL')
    else:
        b_loss = np.array(b_loss)/u_int[j]
        o_loss = np.array(o_loss)/u_int[j]
        ln1b = ax1.plot(b_ep, b_loss, 'k', label='Vanilla-FL')
        ln1o = ax1.plot(o_ep, o_loss, 'r', label='LBGM-FL')
        ln3b = ax3.plot(b_up, b_loss, 'k', label='Vanilla-FL')
        ln3b = ax3.plot(o_up, o_loss, 'r', label='LBGM-FL')
    ln2b = ax2.plot(b_ep, b_up, 'k', label='Vanilla-FL')
    ln2b = ax2.plot(o_ep, o_up, 'r', label='LBGM-FL')

    if plot_idx == 2 or (loss_types[j] == 'mse' and not mse_flag):
        if loss_types[j] == 'mse' and not mse_flag and u_str[j] != 'na':
            ylabel = r'{} ($\times {}$)'.format(ylabel, u_str[j])
            mse_flag = True
        ax1.set_ylabel(ylabel, fontsize=30)
        ax3.set_ylabel(ylabel, fontsize=30)
        if plot_idx == 2:
            ylabel = '#params\n shared'
            if m_str != 'na':
                ylabel = r'{} ($\times {}$)'.format(ylabel, m_str[j])
            ax2.set_ylabel(ylabel, fontsize=30)
    ax1.set_title(models[j].replace(":", "\n"), fontsize=30, pad=20)
    ax1.set_xlabel('t', fontsize=30)
    ax2.set_xlabel('t', fontsize=30)
    ax3.set_xlabel('#params\n shared' + r'($\times {}$)'.format(
        m_str[j]), fontsize=30)
    ax_epoch = n_epochs if xlim[j] == 0 else xlim[j]
    ax1.set_xticks(list(range(0, ax_epoch+1, ax_epoch // 4)))
    # ax1.set_xticklabels([])
    ax2.set_xticks(list(range(0, ax_epoch+1, ax_epoch // 4)))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.set_xlim(0, ax_epoch)
    ax2.set_xlim(0, ax_epoch)
    ax1.set_ylim(0, ylim1[j])
    ax2.set_ylim(0, ylim2[j])
    ax3.set_xlim(0, ylim2[j])
    ax3.set_ylim(0, ylim1[j])
    if plot_idx != 2:
        continue
    lns = ln1b + ln1o  # + ln3
    labs = [lab.get_label() for lab in lns]
    ax1.legend(lns, labs, loc='lower right', fontsize=30, ncol=1, borderpad=0.75,
               bbox_to_anchor=(-0.15, 1.4, 1.35, 1.6),
               mode='expand', frameon=False
               )

plt.subplots_adjust(hspace=0.35, wspace=wspace)
if not final and not dry_run:
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=100)
else:
    plt.savefig(save_path + '.pdf', bbox_inches='tight', dpi=300)
