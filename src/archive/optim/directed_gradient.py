import torch
import torch.optim as optim


class DirectedGradient(optim.Optimizer):
    def __init__(self, params, lr, line_search=False):
        defaults = dict(lr=lr, line_search=line_search)
        super(DirectedGradient, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=1):
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                d_p = p.grad
                p.add_(d_p, alpha=-1*closure)

        return loss


def _fletcher_reeves(grad, ograd):
    num = torch.sum(grad * grad)
    den = torch.sum(ograd * ograd)
    beta = num/den

    return beta


def _polak_ribiere(grad, ograd):
    num = torch.sum(grad * (grad-ograd))
    den = torch.sum(ograd * ograd)
    beta = num/den

    return beta


def conj_beta(grad, ograd, dev='fr'):
    if dev == 'fr':
        beta = _fletcher_reeves(grad, ograd)
    elif dev == 'pr':
        beta = _polak_ribiere(grad, ograd)

    return beta


def rp_directions(grads, n, device):
    rp_sdirs = []
    for _ in range(n):
        rp_sdirs_i = []
        for grad in grads:
            grad_flat = grad.clone().flatten()
            dim = grad_flat.size()[0]
            sdir = torch.normal(
                mean=0.0, std=1.0, size=(dim,)).to(device)
            while torch.dot(sdir, grad_flat) < 0:
                sdir = torch.normal(
                    mean=0.0, std=1.0, size=(dim,)).to(device)
            rp_sdirs_i.append(sdir)
        rp_sdirs.append(rp_sdirs_i)
    return rp_sdirs
