from collections import defaultdict
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch


def accumulate_grads_over_epochs(grad, device):
    g_res = []
    for g_ep in grad:
        g_res_bt = [torch.zeros(_.size()).to(device) for _ in grad[0][0]]
        c = 0
        for g_bt in g_ep:
            for idx, (gi, gii) in enumerate(zip(g_res_bt, g_bt)):
                g_res_bt[idx] = gi + gii
            c += 1
        g_res_bt = [_/c for _ in g_res_bt]
        g_res.append(g_res_bt)

    return g_res


def accumulate_topk_grads_over_epochs(grad, k, device):
    g_res = []
    for g_ep in grad:
        g_res_bt = [torch.zeros(_.size()).to(device) for _ in grad[0][0]]
        c = [torch.zeros(_.size()).to(device) for _ in grad[0][0]]
        for g_bt in g_ep:
            for idx, (gi, gii) in enumerate(zip(g_res_bt, g_bt)):
                size = gi.size()
                gi = gi.flatten()
                gii = gii.flatten()
                topk = torch.argsort(gii)[-k:]
                for ki in topk:
                    gi[ki] += gii[ki]
                g_res_bt[idx] = gi.reshape(size)
                c[idx] = c[idx].flatten()
                for ki in topk:
                    c[idx][ki] += 1
                c[idx] = c[idx].reshape(size)
        for idx in range(len(c)):
            size = c[idx].size()
            ci = c[idx].flatten()
            zerok = ci == 0
            for ki in zerok:
                ci[ki] = 1
            ci = ci.reshape(size)
            c[idx] = ci
        g_res_bt = [torch.div(_, ci) for _, ci in zip(g_res_bt, c)]
        g_res.append(g_res_bt)

    return g_res


def add_param_list(param1, param2):
    if not param1:
        return param2

    assert len(param1) == len(param2)
    for idx in range(len(param1)):
        param1[idx] = param1[idx].cpu() + param2[idx].cpu()

    return param1


def add_param_dict(param1, param2):
    for key, val in param2.items():
        param1[key] += val

    return param1


def set_model_grads(model, grads):
    for i, p in enumerate(model.parameters()):
        p.grad.copy_(grads[i].clone())


def low_rank_approrximation(mat, r, device):
    u, s, v = torch.svd(mat)
    matr = torch.zeros(len(u), len(v)).to(device)
    for i in range(r):
        if u.size(0) < v.size(0):
            matr += s[i]*(u[i].view(-1, 1).mm(v.T[i].view(1, -1)))
        else:
            matr += s[i]*(u.T[i].view(-1, 1).mm(v[i].view(1, -1)))

    return matr


def truncated_svd_approrximation(mat, r, device):
    mat = mat.cpu().numpy()
    svd = TruncatedSVD(n_components=r)
    mat = svd.fit_transform(mat)
    mat = svd.inverse_transform(mat)
    mat = torch.Tensor(mat).to(device)

    return mat


def atomo_approximation(args, model, residuals, device):
    accum_res = []
    accum_error = 0.0
    total_param = 0.0
    for i, p in enumerate(model.parameters()):
        size = p.size()
        if p.ndim <= 1 or (p.ndim == 2 and min(size) == 1):
            total_param += max(size)
            continue
        grad2d = p.grad.clone().reshape(size[0], -1)
        rank = min(*list(grad2d.size()), args.atomo_r)
        # res_2d = residuals[i] if len(residuals) else \
        #     torch.zeros_like(grad_2d)
        # grad_2d += res_2d
        total_param += rank * sum(grad2d.size())
        grad_lr = truncated_svd_approrximation(grad2d, rank, device)
        accum_res.append((grad2d - grad_lr).reshape(size))
        accum_error += torch.norm(accum_res[-1]) / torch.norm(grad2d)
        p.grad.copy_(grad_lr.reshape(size))

    return total_param, accum_res, accum_error.item() / (i + 1)


def get_param_list_sum(tensors):
    tensor_sum = [_.clone() for _ in tensors[0]]
    for t in tensors[1:]:
        tensor_sum = add_param_list(tensor_sum, t)

    return tensor_sum


def calc_projection(a, b, device):
    # projection of vector a on vector b
    a = a.clone().to(device)
    b = b.clone().to(device)
    p = (torch.dot(a, b) / torch.dot(b, b)).item()
    if p < 0.0 or p > 1.0:
        p = 0.0

    return p


def get_layer_size(model, flatten=True):
    layer_size = []
    for p in model.parameters():
        size = p.flatten().size() if flatten else p.size()
        layer_size.append(size)

    return layer_size


def get_model_grads(model, scaling_factor=1, flatten=False, numpy=False):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.clone()*scaling_factor)

    if flatten:
        grads = [_.flatten().cpu().numpy() if numpy else _.flatten()
                 for _ in grads]
    return grads


def get_random_grads(model, std, device, flatten=False, numpy=False):
    # used for generating grads for random projection
    # std = sqrt(k) where k is the number of dimensions used for projection
    grads = []
    for param in model.parameters():
        grads.append(torch.normal(0, std, size=param.shape).to(device))

    if flatten:
        grads = [_.flatten().cpu().numpy() if numpy else _.flatten()
                 for _ in grads]

    return grads


def get_model_size(model):
    num_params = 0
    num_layers = 0
    for p in model.parameters():
        num_layers += 1
        num_params += p.flatten().size()[0]

    return num_params, num_layers/2.0


def get_model_weights(model, scaling_factor=1):

    if scaling_factor == 1:
        return model.state_dict()

    else:
        weights = model.state_dict()
        for key, val in weights.items():
            weights[key] = val*scaling_factor
        return weights


def get_scheduled_lr(args, epoch):
    epoch = epoch // 15
    return 0 + 0.5 * (args.lr - 0) * (1 + np.cos(epoch * np.pi / args.epochs))


def gradient_approximation(model, sdirs, device, residuals):
    true_grads = []
    idx = 0
    for p in model.parameters():
        if len(residuals):
            true_grads.append(p.grad.clone().flatten() + residuals[idx])
        else:
            true_grads.append(p.grad.clone().flatten())
        p.grad.copy_(torch.zeros(p.size()).to(device))
        idx += 1
    for sdir in sdirs:
        for p, s, g in zip(model.parameters(), sdir, true_grads):
            size = p.grad.size()
            grad_flat = p.grad.clone().flatten()
            true_grad = g.clone()
            grad_flat = grad_flat + \
                calc_projection(true_grad, s.flatten(), device) * \
                s.flatten().clone().to(device)
            p.grad.copy_(grad_flat.reshape(size).clone())
    residuals = []
    error_norm = 0
    num_layers = 0
    for p, g in zip(model.parameters(), true_grads):
        residuals.append(g-p.grad.flatten())
        error_norm += torch.norm(residuals[-1]).item()/torch.norm(g).item()
        num_layers += 1
    return residuals, error_norm/num_layers


# fb: feedback
def lbgm_approximation(args, model, lbgs, residuals, device):
    accum_rho = 0.0
    accum_lbgs = []
    accum_residuals = []
    uplink = 0
    for i, p in enumerate(model.parameters()):
        size = p.grad.size()
        grad_flat = p.grad.clone().flatten()
        grad_res = residuals[i] if len(
            residuals) else torch.zeros_like(grad_flat)
        grad_flat = grad_flat + grad_res

        if len(lbgs):
            rho = calc_projection(
                grad_flat, lbgs[i], device)
        else:
            rho = 0.0

        if rho >= args.error_tol:
            update = rho * lbgs[i]
            accum_residuals.append(grad_flat - update)
            accum_rho += rho
            accum_lbgs.append(lbgs[i])
            uplink += 1
        else:
            update = grad_flat
            accum_lbgs.append(update.clone())
            accum_residuals.append(torch.zeros_like(update))
            uplink += len(update)
        p.grad.copy_(update.reshape(size))

    # number of layers = (i+1) and not i because its zero-indexed
    return accum_lbgs, accum_residuals, uplink, accum_rho / (i + 1)


def load_model_grad(model, grads):
    for param, g in zip(model.parameters(), grads):
        param.grad = g


def model_gradient(model1, model2, lr):
    grads = defaultdict(list)
    for key, val in model1.items():
        grads[key.split('.')[-1]] = weight_gradient(
            model1[key], model2[key], lr)

    return grads


def momentum_grad(prev_v, grad, momentum, device):
    return momentum * prev_v.to(device) + grad


def model_update(model, grads, args, device, prev_v=[], epoch=0):
    idx = 0
    accum_v = []
    for param in model.parameters():
        d = grads[idx]

        if args.momentum:
            if len(prev_v):
                d = args.momentum * prev_v[idx] + d
                accum_v.append(d)
            else:
                accum_v.append(d)

        with torch.no_grad():
            lr = args.lr if not args.scheduler else get_scheduled_lr(
                args, epoch)
            param.copy_(param.add(-args.lr, d.to(device)))
        idx += 1
    return accum_v


def norm(weight):
    return torch.norm(weight.copy().flatten()).item()


def perturb_model(model, epoch, device):
    for p in model.parameters():
        p = p + torch.normal(0.0, 1/epoch, size=p.size()).to(device)


def powersgd_approximation(args, model, residuals, rank, device):

    uplink = 0.0
    for i, p in enumerate(model.parameters()):
        size = p.size()
        if p.ndim <= 1:  # powersgd skips 1d tensors
            uplink += sum(size)
            continue
        grad2d = p.grad.clone().reshape(size[0], -1)
        size2d = grad2d.size()
        r = min(*list(size2d), rank)
        uplink += r * sum(size2d)


def scale_model_weights(weights, factor):
    for key, val in weights.items():
        weights[key] = val*factor

    return weights


def sign_sgd_quantization(args, model, residuals, device):
    accum_res = []
    accum_error = 0.0
    uplink = 0.0
    for i, p in enumerate(model.parameters()):
        size = p.size()
        grad = p.grad.clone()
        res = residuals[i] if len(residuals) else torch.zeros_like(p)
        grad += res
        grad_sign = torch.where(grad >= 0,
                                torch.ones(grad.size()).to(device),
                                -1*torch.ones(grad.size()).to(device))
        accum_res.append(grad - grad_sign)
        accum_error += torch.norm(accum_res[-1]) / torch.norm(grad)
        p.grad.copy_(grad_sign)
        uplink += grad_sign.view(-1).shape[0]

    return uplink, accum_res, accum_error.item() / (i + 1)


def topk_sparsify(model, k, residuals):
    num_selected = 0
    idx = 0
    residuals_tmp = []
    error_norm = 0
    for param in model.parameters():
        size = param.grad.size()
        if len(residuals):
            grad = param.grad.flatten() + residuals[idx]
        else:
            grad = param.grad.flatten()
        num_params = grad.size()[0]
        ki = int(k*num_params)
        num_selected += ki
        ki_bar = num_params - ki
        sorted_indices = torch.argsort(torch.abs(grad))
        bottom_k = sorted_indices[:ki_bar]
        top_k = sorted_indices[ki_bar:]
        assert len(sorted_indices) == len(top_k) + len(bottom_k)
        residuals_tmp.append(grad.clone().index_fill_(0, top_k, 0))
        error_norm += torch.norm(
            grad - residuals_tmp[-1]).item()/torch.norm(grad).item()
        grad = grad.index_fill_(0, bottom_k, 0)
        grad = grad.reshape(size)
        param.grad = grad
        idx += 1

    return num_selected, residuals_tmp, error_norm / idx


def ein_normal(param):
    chars = ['j', 'k', 'l']
    size = param.size()
    ein_in, ein_out = 'i', 'i'
    ein_list = [torch.normal(1.0, 1.0, size=(size[0],))]
    for i, dim in enumerate(size[1:]):
        ein_in = '{},{}'.format(ein_in, chars[i])
        ein_out = '{}{}'.format(ein_out, chars[i])
        ein_list.append(torch.normal(1.0, 1.0, size=(dim,)))

    return torch.einsum('{}->{}'.format(ein_in, ein_out), *ein_list)


def weights_init(m, init):
    with torch.no_grad():
        for param in m.parameters():
            if init == 'low_rank':
                param.copy_(ein_normal(param))
            elif init == 'normal':
                torch.nn.init.normal(param)
            elif init == 'uniform':
                torch.nn.init.uniform(param)
            elif init == 'xavier':
                torch.nn.init.xavier_uniform(param)


def weight_gradient(w1, w2, lr):
    return torch.norm((w1.flatten()-w2.flatten())/lr).item()


def weights_reset(m, indices):
    for params, idxs in zip(m.parameters(), indices):
        for idx in idxs:
            if len(params.shape) > 1:
                params[:, idx] = torch.normal(mean=torch.Tensor([0.0]),
                                              std=torch.Tensor([1.0]))[0]
            else:
                params[idx] = torch.normal(mean=torch.Tensor([0.0]),
                                           std=torch.Tensor([1.0]))[0]
