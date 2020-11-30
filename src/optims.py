from directed_gradient import conj_beta
import torch


def gradient_retriever(model):
    """fn to retrieve gradients from model

    :param model: torch model
    :returns: gradient norm, gradient tensor
    :rtype: (float, tensor)

    """
    grads = []
    grad_norm = 0
    for p in model.parameters():
        grads.append(p.grad.clone())
        grad_norm += torch.norm(p.grad)

    return grad_norm, grads


def default_step(model, optimizer):
    """same as conventional sgd step

    :param model: torch model
    :param optimizer: model optimizer
    :returns: gradient norm, gradient tensor
    :rtype: (float, tensor)

    """
    optimizer.step()

    return gradient_retriever(model)


def conj_step(model, ograds, sdirs, optimizer, lr, dev):
    """conjugate gradient step using orthogonal gradient
    and calculated search directions using conjugate methods

    :param model: torch model
    :param ograds: orthogonal gradients
    :param sdirs: search directions
    :param optimizer: model optimizer
    :param lr: learning rate
    :param dev: conjugate gradient method (fr, pr)
    :returns: gradient norm, gradient tensor
    :rtype: (float, tensor)

    """
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
    """takes step in a give direction
    if the passed directions are the orthogonal directions
    it amounts to orthogonal gradient descent

    :param model: torch model
    :param sdirs: search directions
    :param optimizer: model optimizer
    :param lr: learning rate
    :returns: gradient norm, gradient tensor, true gradient
    :rtype: ((float, tensor), tensor)

    """
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
