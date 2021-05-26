import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import get_device
from models.cnn import CNN
from models.fcn import FCN, MLP
from models.multi_class_hinge_loss import multiClassHingeLoss
from models.model_op import get_model_size
import models.resnet as resnet
from models.svm import SVM
from models.unet import UNet
from models.vgg import VGG


def forward(model, data, target, opt, loss_fn, device, loss_type):
    data, target = data.to(device), target.to(device)
    opt.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    if loss_type != 'mse':
        pred = output.argmax(1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
    else:
        correct = 0

    return loss, correct


def get_loss_fn(loss_fn):
    if loss_fn == 'ce':
        # loss_fn_ = F.nll_loss
        loss_fn_ = nn.CrossEntropyLoss()
    elif loss_fn == 'mse':
        loss_fn_ = nn.MSELoss()
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    return loss_fn_


def get_model(args, parallel=True, ckpt_path=False):
    if args.clf == 'fcn':
        print('Initializing FCN...')
        model = FCN(args.input_size, args.output_size)
    elif args.clf == 'mlp':
        print('Initializing MLP...')
        model = MLP(args.input_size, args.output_size)
    elif args.clf == 'svm':
        print('Initializing SVM...')
        model = SVM(args.input_size, args.output_size)
    elif args.clf == 'cnn':
        print('Initializing CNN...')
        model = CNN(nc=args.num_channels,
                    fs=args.cnn_view)
    elif args.clf == 'resnet18':
        print('Initializing ResNet18...')
        model = resnet.resnet18(
            num_channels=args.num_channels,
            num_classes=args.output_size)
    elif args.clf == 'vgg19':
        print('Initializing VGG19...')
        model = VGG(
            vgg_name=args.clf,
            num_channels=args.num_channels,
            num_classes=args.output_size)
    elif args.clf == 'unet':
        print('Initializing UNet...')
        model = UNet(
            in_channels=args.num_channels,
            out_channels=args.output_size)

    num_params, num_layers = get_model_size(model)
    print("# params: {}\n# layers: {}".format(num_params, num_layers))

    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
        print('Load init: {}'.format(ckpt_path))

    if parallel:
        model = nn.DataParallel(
            model.to(get_device(args)), device_ids=args.device_id)
    else:
        model = model.to(get_device(args))

    loss_type = 'hinge' if args.clf == 'svm' else args.loss_type
    print("Loss: {}".format(loss_type))

    return model, loss_type


def get_optim(args, model):
    if args.optim == 'sgd':
        opt = optim.SGD(
            params=model.parameters(), lr=args.lr,
            momentum=args.momentum)
    elif args.optim == 'adam':
        opt = optim.Adam(
            params=model.parameters(), lr=args.lr)

    return opt
