import argparse
import config as cfg
from distributor import get_distributed_data
import numpy as np
import pickle as pkl
import torch
from utils import booltype, \
    get_testloader, get_trainloader, Struct


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, type=str)
ap.add_argument("--num-nodes", required=True, type=int)
ap.add_argument("--non-iid", required=True, type=int)
ap.add_argument("--repeat", required=False, type=int, default=True)
ap.add_argument("--shuffle", required=False, type=booltype, default=True)
ap.add_argument("--stratify", required=False, type=booltype, default=True)
ap.add_argument("--uniform", required=False, type=booltype, default=False)
ap.add_argument("--dry-run", required=False, type=booltype, default=False)

args = vars(ap.parse_args())
args = Struct(**args)

num_train = cfg.num_trains[args.dataset]
num_test = cfg.num_tests[args.dataset]
num_classes = cfg.output_sizes[args.dataset]

kwargs = {}

train_loader = get_trainloader(args.dataset, num_train)
test_loader = get_testloader(args.dataset, num_test)

for data, target in train_loader:
    X_train = data
    y_train = target

for data, target in test_loader:
    X_test = data
    y_test = target


def repeat_data(data, repeat):
    rep = [data for _ in range(repeat)]
    rep = torch.cat(rep, dim=0)

    return rep


X_train, y_train = repeat_data(X_train, args.repeat), \
                   repeat_data(y_train, args.repeat)

print('X_train: {}'.format(X_train.shape))
print('y_train: {}'.format(y_train.shape))

print('X_test: {}'.format(X_test.shape))
print('y_test: {}'.format(y_test.shape))

X_trains, y_trains, class_map = get_distributed_data(X_train, y_train,
                                                     args.num_nodes,
                                                     stratify=args.stratify,
                                                     uniform=args.uniform,
                                                     shuffle=args.shuffle,
                                                     non_iid=args.non_iid)

X_tests, y_tests, _ = get_distributed_data(X_test, y_test, args.num_nodes,
                                        stratify=args.stratify,
                                        uniform=args.uniform,
                                        shuffle=args.shuffle,
                                        non_iid=args.non_iid,
                                        class_map=class_map)

bincounts_train = []
for _ in y_trains:
    bincounts_train.append(np.bincount(_, minlength=num_classes))

bincounts_test = []
for _ in y_tests:
    bincounts_test.append(np.bincount(_, minlength=num_classes))

bincounts_train = np.concatenate(bincounts_train).reshape(-1, num_classes)
bincounts_test = np.concatenate(bincounts_test).reshape(-1, num_classes)
print("Max Train:", bincounts_train.max(axis=0))
print("Min Train:", bincounts_train.min(axis=0))
print("Max Test:", bincounts_test.max(axis=0))
print("Min Test:", bincounts_test.min(axis=0))
meta = args.__dict__
meta['batch_size'] = int(max(bincounts_train.sum(axis=1)))
meta['test_batch_size'] = int(max(bincounts_test.sum(axis=1)))
print(meta)

name = ['{}_{}'.format(args.dataset, args.num_nodes), 'data',
        'n_classes_per_node_{}_stratify_{}_uniform_{}_repeat_{}'.format(
            args.non_iid, args.stratify, args.uniform, args.repeat
        )]

filename = '../ckpts/' + '/'.join(name) + '.pkl'
print('Saving: {}'.format(filename))
if not args.dry_run:
    pkl.dump((X_trains, X_tests, y_trains, y_tests, meta), open(filename, 'wb'))
