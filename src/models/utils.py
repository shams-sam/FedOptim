import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common.utils import get_device
from models.fcn import FCN, MLP
from models.multi_class_hinge_loss import multiClassHingeLoss
from models.model_op import get_model_size
import models.resnet as resnet
from models.svm import SVM


def forward(model, data, target, opt, loss_fn, device):
    data, target = data.to(device), target.to(device)
    opt.zero_grad()
    output = model(data)
    pred = output.argmax(1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    loss = loss_fn(output, target)

    return loss, correct


def get_loss_fn(loss_fn):
    if loss_fn == 'nll':
        # loss_fn_ = F.nll_loss
        loss_fn_ = nn.CrossEntropyLoss()
    elif loss_fn == 'hinge':
        loss_fn_ = multiClassHingeLoss()

    return loss_fn_


def get_model(args, ckpt_path=False):
    if args.clf == 'fcn':
        print('Initializing FCN...')
        model = FCN(args.input_size, args.output_size)
    elif args.clf == 'mlp':
        print('Initializing MLP...')
        model = MLP(args.input_size, args.output_size)
    elif args.clf == 'svm':
        print('Initializing SVM...')
        model = SVM(args.input_size, args.output_size)
    elif args.clf == 'resnet18':
        print('Initializing ResNet18...')
        model = resnet.resnet18(
            num_channels=args.num_channels,
            num_classes=args.output_size)

    num_params, num_layers = get_model_size(model)
    print("# params: {}\n# layers: {}".format(num_params, num_layers))

    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
        print('Load init: {}'.format(ckpt_path))

    model = nn.DataParallel(
        model.to(get_device(args)), device_ids=args.device_id)

    loss_type = 'hinge' if args.clf == 'svm' else 'nll'
    print("Loss: {}".format(loss_type))

    return model, loss_type


def get_optim(args, model):
    if args.optim == 'sgd':
        opt = optim.SGD(
            params=model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        opt = optim.Adam(
            params=model.parameters(), lr=args.lr)

    return opt
