from collections import defaultdict
import torch


def get_model_weights(model, scaling_factor=1):

    if scaling_factor == 1:
        return model.state_dict()

    else:
        weights = model.state_dict()
        for key, val in weights.items():
            weights[key] = val*scaling_factor
        return weights


def scale_model_weights(weights, factor):
    for key, val in weights.items():
        weights[key] = val*factor

    return weights


def add_model_weights(weights1, weights2):
    for key, val in weights2.items():
        weights1[key] += val

    return weights1


def weight_gradient(w1, w2, lr):
    return torch.norm((w1.flatten()-w2.flatten())/lr).item()


def model_gradient(model1, model2, lr):
    grads = defaultdict(list)
    for key, val in model1.items():
        grads[key.split('.')[-1]] = weight_gradient(
            model1[key], model2[key], lr)

    return grads


def get_num_params(model):
    return sum([_.flatten().size()[0] for _ in model.parameters()])


def norm(weight):
    return torch.norm(weight.copy().flatten())
