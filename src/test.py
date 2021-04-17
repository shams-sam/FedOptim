import torch
from models.model_op import calc_projection


def gradient_approximation(orig, sdirs):
    true_grads = []
    idx = 0
    for p in orig:
        true_grads.append(p.clone().flatten())
        p.copy_(torch.zeros(p.size()))
        idx += 1
    for sdir in sdirs:
        for p, s, g in zip(orig, sdir, true_grads):
            size = p.size()
            grad_flat = p.clone().flatten()
            true_grad = g.clone()
            grad_flat = grad_flat + \
                calc_projection(true_grad, s.flatten(), torch.device('cpu')) * \
                s.flatten()
            p.copy_(grad_flat.reshape(size).clone())
    for p in orig:
        print(p)


orig = [torch.ones((10,)), torch.zeros((5,))]
s1 = [(torch.zeros((10,)), torch.ones((5,))) for _ in range(5)]
for idx, (s, ss) in enumerate(s1):
    s[idx] = 1
    ss[idx] = 0


gradient_approximation(orig, s1)
