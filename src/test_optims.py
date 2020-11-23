import torch
from directed_gradient import conj_beta


def add_gaussian(p, g_std):
    if g_std > 0:
        size = p.size()
        p += torch.normal(0, g_std, size=size)
    return p


def sgd_step(param, optimizer, lr, g_std=1.0):
    for p in param:
        p.grad = add_gaussian(p.grad, g_std)
    optimizer.step(lr)


def conj_step(param, ograds, sdirs, optimizer, lr, dev, g_std=1.0):
    if len(ograds) == 0:
        for p in param:
            grad = add_gaussian(p.grad.clone(), g_std)
            ograds.append(grad)
            sdirs.append(grad)
    else:
        grad_accum = []
        sdir_accum = []
        for p, ograd, sdir in zip(param, ograds, sdirs):
            grad = add_gaussian(p.grad.clone(), g_std)
            grad_accum.append(grad.clone())
            beta = conj_beta(grad, ograd, dev)
            p.grad = grad - beta * sdir
            sdir_accum.append(p.grad.clone())
        ograds = grad_accum
        sdirs = sdir_accum

    optimizer.step(lr)

    return ograds, sdirs


def directed_step(param, sdirs, opt, lr, g_std=1.0):
    for p, sdir in zip(param, sdirs):
        dim = p.size()
        p.grad = ((
            torch.dot(p.grad, sdir)/torch.dot(sdir, sdir)
        ) * sdir).reshape(dim)

#     opt.step(lr)
#     return param
