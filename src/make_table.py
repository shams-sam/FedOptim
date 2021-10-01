import os
import pickle as pkl

import argparse
from terminaltables import AsciiTable

from common.config import ckpt_dir
from common.utils import Struct


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, type=str)
ap.add_argument("--paradigm", required=True, type=str)
args = vars(ap.parse_args())
args = Struct(**args)


DEFAULT_VAL = 'X'


datasets = {
    'cifar': {},
    'cifar100': {},
    'celeba': {},
    'fmnist': {},
    'mnist': {},
    'miniimagenet': {}
}

archs = ['cnn', 'fcn', 'resnet18', 'vgg19']
iids = [0, 3, 10, 100]
workers = [0, 10, 100]
lrs = ['1e-05', '2e-05', 0.1, 0.01, 0.03, 0.001, 0.0001]
batches = [0, 64, 128, 256, 512]
paradigms = ['central', 'federated', 'distributed',
             'lbgm', 'atomo', 'atomo_lbgm', 'signsgd', 'signsgd_lbgm', 'topk', 'topk_lbgm']
lbgms = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
header = ['dataset', 'iid', 'arch', 'workers',
          'paradigm', 'lr', 'batch', 'performance', 'log_file']

data = []

k = args.dataset
p = args.paradigm


def get_row_data(file_name, acc_id, loss_id):

    history = pkl.load(open(file_name, 'rb'))
    acc = history[acc_id]
    loss = history[loss_id]
    log_file = file_name.replace('history', 'logs').replace('pkl', 'log')
    row = [k, iid, arch, w, p, lr, batch, min(
        loss) if 'celeba' in k else max(acc), log_file]
    assert len(row) == len(header)
    return row


for iid in iids:
    for arch in archs:
        for w in workers:
            for lr in lrs:
                for batch in batches:
                    if p == 'central':
                        file_name = f'{ckpt_dir}/{k}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}.pkl'
                        acc_id, loss_id = 2, 4
                        if os.path.exists(file_name):
                            data.append(get_row_data(
                                file_name, acc_id, loss_id))
                    elif p == 'federated':
                        file_name = f'{ckpt_dir}/{k}_{w}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}.pkl'
                        acc_id, loss_id = 1, 4
                        if os.path.exists(file_name):
                            data.append(get_row_data(
                                file_name, acc_id, loss_id))
                    elif p == 'atomo':
                        file_name = f'{ckpt_dir}/{k}_{w}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}_atomo_2.pkl'
                        acc_id, loss_id = 1, 4
                        if os.path.exists(file_name):
                            data.append(get_row_data(
                                file_name, acc_id, loss_id))
                    elif p == 'topk':
                        file_name = f'{ckpt_dir}/{k}_{w}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}_topk_0.1_residual.pkl'
                        acc_id, loss_id = 1, 4
                        if os.path.exists(file_name):
                            data.append(get_row_data(
                                file_name, acc_id, loss_id))
                    elif p == 'signsgd':
                        file_name = f'{ckpt_dir}/{k}_{w}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}_signsgd.pkl'
                        acc_id, loss_id = 1, 4
                        if os.path.exists(file_name):
                            data.append(get_row_data(
                                file_name, acc_id, loss_id))
                    elif p == 'lbgm':
                        for lbgm in lbgms:
                            file_name = f'{ckpt_dir}/{k}_{w}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}_lbgm_{lbgm}.pkl'
                            acc_id, loss_id = 1, 4
                            if os.path.exists(file_name):
                                data.append(get_row_data(
                                    file_name, acc_id, loss_id))
                    elif p == 'atomo_lbgm':
                        for lbgm in lbgms:
                            file_name = f'{ckpt_dir}/{k}_{w}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}_atomo_2_lbgm_{lbgm}.pkl'
                            acc_id, loss_id = 1, 4
                            if os.path.exists(file_name):
                                data.append(get_row_data(
                                    file_name, acc_id, loss_id))
                    elif p == 'signsgd_lbgm':
                        for lbgm in lbgms:
                            file_name = f'{ckpt_dir}/{k}_{w}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}_signsgd_lbgm_{lbgm}.pkl'
                            acc_id, loss_id = 1, 4
                            if os.path.exists(file_name):
                                data.append(get_row_data(
                                    file_name, acc_id, loss_id))
                    elif p == 'topk_lbgm':
                        for lbgm in lbgms:
                            file_name = f'{ckpt_dir}/{k}_{w}/history/clf_{arch}_optim_sgd_uniform_True_non_iid_{iid}_num_workers_{w}_lr_{lr}_decay_1e-05_batch_{batch}_topk_0.1_lbgm_{lbgm}_residual.pkl'
                            acc_id, loss_id = 1, 4
                            if os.path.exists(file_name):
                                data.append(get_row_data(
                                    file_name, acc_id, loss_id))


table = AsciiTable([header] + data)
table.title = f'{k}/{p}'

print(table.table)
