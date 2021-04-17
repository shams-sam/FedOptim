import argparse
import common.config as cfg
from common.utils import Struct
from models.fcn import FCN
import models.resnet as resnet
from models.svm import SVM
import torch


ap = argparse.ArgumentParser()
ap.add_argument('--models', required=True, type=str, nargs="+")
ap.add_argument('--dataset', required=True, type=str)
args = vars(ap.parse_args())
args = Struct(**args)

if 'fcn' in args.models:
    print("Initializing FCN...")
    model = FCN(cfg.input_sizes[args.dataset],
                cfg.output_sizes[args.dataset])
    print('input_size: {}, output_size: {}'.format(
        model.input_size, model.output_size))
    init_path = '../ckpts/init/{}_fcn.init'.format(args.dataset)
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))

if 'svm' in args.models:
    print("Initializing SVM...")
    model = SVM(cfg.input_sizes[args.dataset],
                cfg.output_sizes[args.dataset])
    print('input_size: {}, output_size: {}'.format(
        model.n_feature, model.n_class))
    init_path = '../ckpts/init/{}_svm.init'.format(args.dataset)
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))

if 'resnet18' in args.models:
    print("Initializing SVM...")
    model = resnet.resnet18(num_channels=cfg.num_channels[args.dataset],
                            num_classes=cfg.output_sizes[args.dataset])
    print('num_channels: {}, output_size: {}'.format(
        model.num_channels, model.num_classes))
    init_path = '../ckpts/init/{}_resnet18.init'.format(args.dataset)
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))
