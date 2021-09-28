from common.argparser import argparser
from common.arguments import Arguments
from common.utils import get_paths
from common.nb_utils import estimate_optimal_ncomponents, pca_transform
import pickle as pkl
import torch

args = Arguments(argparser())
paths = get_paths(args)

h = pkl.load(open(paths.hist_path, 'rb'))
grads = h[-2]
stack = [[] for _ in range(len(grads[0]))]
for layer in range(len(grads[0])):
    for epoch in range(len(grads)):
        stack[layer].append(grads[epoch][layer].flatten())

dim = []
for idx, layer in enumerate(stack):
    layer = torch.stack(layer, dim=0).T.cpu().numpy()
    stack[idx] = layer
    nc, exp_var = estimate_optimal_ncomponents(layer, args.pca_var)
    print('nc={} for exp_var={}'.format(nc, args.pca_var))
    if args.ncomponent:
        print('Resetting from nc={} to nc={}'.format(nc, args.ncomponent))
        nc = args.ncomponent
    dim.append(nc)

dim = min(dim)
print('PCA transform using nc:', dim)
sdirs = [[] for _ in range(dim)]

for idx, layer in enumerate(stack):
    layer, _ = pca_transform(layer, dim)
    assert layer.shape[1] == dim
    for idx in range(dim):
        sdirs[idx].append(torch.Tensor(layer[:, idx].flatten()))

print('Saving:', paths.pca_path)
if not args.dry_run:
    pkl.dump(sdirs, open(paths.pca_path, 'wb'))
