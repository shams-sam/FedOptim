import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import argparse
from common.utils import Struct


# matplotlib.rcParams.update({'font.size': 37})
# matplotlib.rcParams['lines.linewidth'] = 2.0
# matplotlib.rcParams['lines.markersize'] = 8

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=False)
ap.add_argument('--num-nodes', type=int, required=False)
ap.add_argument('--histories', type=str, nargs='+', required=True)
ap.add_argument('--labels', type=str, nargs='+', required=True)
ap.add_argument('--name', type=str, required=True)
ap.add_argument('--ncols', type=int, required=True)
ap.add_argument('--dpi', type=int, required=True)
ap.add_argument('--colors', type=str, nargs='+', required=False, default=[])
ap.add_argument('--epochs', type=int, required=False)
ap.add_argument('--infer-folder', type=int, required=True)
args = vars(ap.parse_args())
args = Struct(**args)

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:']
if len(args.colors) > 0:
    colors = args.colors

for idx, history in enumerate(args.histories):
    if args.infer_folder == 1:
        history = '../ckpts/{}_{}/history/{}'.format(
            args.dataset, args.num_nodes, history)
    h_epoch, h_acc, _, _, h_loss, _, _, h_uplink, _, _ = pkl.load(
        open(history, 'rb'))
    h_epoch = h_epoch[:args.epochs]
    h_acc = h_acc[:args.epochs]
    h_loss = h_loss[:args.epochs]
    h_uplink = h_uplink[:args.epochs]
    ax1.plot(h_epoch, h_acc, colors[idx], label=args.labels[idx])
    ax2.plot(h_epoch, np.cumsum(h_uplink), colors[idx], label=args.labels[idx])

ax1.set_xlabel('epochs')
ax1.set_ylabel('accuracy')
ax2.set_xlabel('epochs')
ax2.set_ylabel('# uplink')
ax1.grid(ls='-.', lw=0.25)
ax2.grid(ls='-.', lw=0.25)
ax2.legend(loc='upper right', ncol=args.ncols,
           bbox_to_anchor=(-1.33, 1.1, 2.35, .1),
           mode='expand', frameon=False)
print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.3)
if args.infer_folder == 1:
    args.name = '../ckpts/{}_{}/plots/{}'.format(
        args.dataset, args.num_nodes, args.name)
plt.savefig(args.name,
            bbox_inches='tight', dpi=args.dpi)
