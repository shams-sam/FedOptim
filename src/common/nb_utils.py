from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as cosine
import torch as t


from models.model_op import accumulate_grads_over_epochs


CPU = t.device('cpu')


def estimate_optimal_ncomponents(mat, exp_var):
    pca = PCA()
    pca.fit(mat)
    total_var = 0
    n_comp = 0
    for var in pca.explained_variance_ratio_:
        total_var += var
        n_comp += 1
        if total_var >= exp_var:
            return n_comp, pca.explained_variance_ratio_


def pca_transform(mat, n_comp):
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(mat), sum(pca.explained_variance_ratio_)


def cosine_sim(u, v):
    return 1-cosine(u, v)


def file_select(file, exclude_list, include_list):
    for ex in exclude_list:
        if ex in file:
            return False
    for inc in include_list:
        if inc not in file:
            return False
    return True


def decor_print(string):
    print('+'*80)
    print(string)
    print('+'*80)


def construct_grad_mat(grads):
    device = grads[0][0][0].get_device()
    device = device if device >= 0 else CPU
    if type(grads[0][0]) == list:
        grads = accumulate_grads_over_epochs(grads, device)
    layer_mats = defaultdict(list)
    for ep_grads in grads:
        for idx, layer in enumerate(ep_grads):
            layer_mats[idx].append(layer.flatten().cpu().reshape(1, -1))
    for idx, mat in layer_mats.items():
        layer_mats[idx] = np.vstack(mat)
    return layer_mats


def init_plots():
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    return ax1, ax2


def plot_row_vectors(mat, ax, t=1.5, skip=10, title=False):
    x_min = min(0, min(mat[:, 0]))
    x_max = max(0, max(mat[:, 0]))
    y_min = min(0, min(mat[:, 1]))
    y_max = max(0, max(mat[:, 1]))
    h = min(y_max, x_max)
    for r in range(len(mat)):
        vec = mat[r]
        ax.arrow(0, 0, vec[0], vec[1], length_includes_head=True,
                 width=h/1000, head_length=h/5, head_width=h/10)
        if r % skip == 0:
            ax.text(vec[0], vec[1], str(r))
        ax.set_xlim(left=t*x_min, right=t*x_max)
        ax.set_ylim(top=t*y_max, bottom=t*y_min)
    if title:
        ax.set_title(title)
