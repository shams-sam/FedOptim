import os
import pickle as pkl

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['lines.markersize'] = 7


ap = argparse.ArgumentParser()
ap.add_argument("--h", required=True, type=str)
ap.add_argument("--dataset", required=True, type=str)
ap.add_argument("--final", required=False, type=int, default=0)
ap.add_argument("--save", required=True, type=str)
ap.add_argument("--dry-run", required=True, type=int)
ap.add_argument("--cols", required=False, type=int, default=4)
ap.add_argument("--rows", required=False, type=int, default=2)

args = vars(ap.parse_args())
history = args['h']
dataset = args['dataset']
final = args['final']
save_path = args['save']
dry_run = args['dry_run']
cols = args['cols']
rows = args['rows']

fig = plt.figure(figsize=(5 * cols, 4 * rows))
plot_idx = 1

hist_file = history
meta_file = history[:-4] + '_processed_grads.pkl'
print(hist_file)

if dry_run:
    assert os.path.exists(hist_file)
    assert os.path.exists(meta_file)
    exit()

meta = pkl.load(open(meta_file, 'rb'))
mat_99 = meta['mat_99']
overlap_pca = meta['overlap_95']
l_params = meta['num_params']
num_layers = len(l_params)
n_epochs = mat_99.shape[0]

slrsi = num_layers % cols # second_last_row_start_index
for layer_num, olap in tqdm(enumerate(overlap_pca, 1), total=len(overlap_pca)):
    curr_row = layer_num // (cols)
    curr_col = layer_num % cols
    curr_row = curr_row if curr_col else curr_row-1
    curr_col = curr_col if curr_col else cols

    ax = fig.add_subplot(rows, cols, plot_idx)
    im = sns.heatmap(np.abs(olap), ax=ax, vmin=0.0, vmax=1.0, cbar=False,
                     xticklabels=n_epochs // 4)
    if curr_row == 0 or rows == 1:
        ax.set_title('{}, L#{}\n#elem: {}'.format(
            dataset, layer_num, l_params[layer_num - 1]), fontsize=30)
    else:
        ax.set_title('L#{}\n#elem: {}'.format(
            layer_num, l_params[layer_num - 1]), fontsize=30)
    if curr_row == rows-1 or (curr_row == rows-2 and curr_col > slrsi and slrsi):
        ax.set_xlabel('epoch gradients', fontsize=30)
    if curr_col == 1:
        ax.set_ylabel('pca gradients', fontsize=30)

    plot_idx += 1

cbar = fig.colorbar(ax.collections[0],  # ticks=[0.8, 0.9, 1],
                    cax=fig.add_axes([0.92, 0.12, 0.02, 0.755]))
# cbar.ax.set_yticklabels(['0.8', '0.9', '1.0'])

plt.subplots_adjust(hspace=0.6, wspace=0.3)
if not final:
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=100)
else:
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=300)
