import sys
sys.path.append('../')
from data.loader import get_loader  # noqa

datasets = [
    # 'celeba',  # regression
    # 'cifar',  # classification
    # 'coco', 'voc',  # semantic segmentation
    'fmnist', 'mnist', 'svhn'  # classification
]
test_subset = [
    'celeba', 'coco'
]

for dataset in datasets:
    for split in [True, False]:
        print('-' * 80)
        print('dataset: {}, train:{}'.format(dataset, split))
        loader = get_loader(
            dataset, batch_size=16, train=split, shuffle=True)
        print('\tdata_size:', len(loader.dataset))
        for data, label in loader:
            print('\tbatch_size:', data.shape, label.shape)
            break

        if dataset not in test_subset:
            continue
        loader = get_loader(
            dataset, batch_size=16, train=split, shuffle=True, subset=0.5)
        print('\tdata_size:', len(loader.dataset))
        for data, label in loader:
            print('\tbatch_size:', data.shape, label.shape)
            break
