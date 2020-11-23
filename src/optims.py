from directed_gradient import conj_beta
import torch


def gradient_retriever(model):
    grads = []
    grad_norm = 0
    for p in model.parameters():
        grads.append(p.grad.clone())
        grad_norm += torch.norm(p.grad)

    return grad_norm, grads


def default_step(model, optimizer):
    optimizer.step()

    return gradient_retriever(model)


def conj_step(model, ograds, sdirs, optimizer, lr, dev):
    if len(ograds) == 0:
        for p in model.parameters():
            ograds.append(p.grad.clone())
            sdirs.append(p.grad.clone())
    else:
        grad_accum = []
        sdir_accum = []
        for p, ograd, sdir in zip(model.parameters(), ograds, sdirs):
            grad_accum.append(p.grad.clone())
            beta = conj_beta(p.grad, ograd, dev)
            p.grad = p.grad - beta * sdir
            sdir_accum.append(p.grad.clone())
        ograds = grad_accum
        sdirs = sdir_accum

    optimizer.step(lr)

    return gradient_retriever(model)


def directed_step(model, sdirs, optimizer, lr):
    true_grad = []
    for p, sdir in zip(model.parameters(), sdirs):
        dim = p.size()
        scale = torch.norm(p.grad)
        true_grad.append(p.grad.clone())
        p.grad = ((
            torch.dot(p.grad.flatten(), sdir)/torch.dot(sdir, sdir)
        ) * sdir).reshape(dim)
        scale = scale/torch.norm(p.grad)
        p.grad = scale*p.grad

    optimizer.step()

    return gradient_retriever(model), true_grad
