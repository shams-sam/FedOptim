import torch
import torch.optim as optim


class ConjugateGradient(optim.Optimizer):
    def __init__(self, params, lr, line_search=False):
        defaults = dict(lr=lr, line_search=line_search)
        super(ConjugateGradient, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=1):
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                # if d_p is None:
                #     continue
                d_p = p.grad
                p.add_(d_p, alpha=-1*closure)

        return loss
