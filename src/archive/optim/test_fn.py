# Reference:
# https://github.com/jettify/pytorch-optimizer/blob/master/examples/viz_optimizers.py

import math
import torch as t


def booth(x):
    tensor = x[0]
    x = tensor[0]
    y = tensor[1]
    return t.pow(x+2*y-7, 2) +\
        t.pow(2*x+y-5, 2)


def himmelblau(x):
    tensor = x[0]
    x = tensor[0]
    y = tensor[1]
    return t.pow((x**2)+y-11, 2) + \
        t.pow(x+(y**2)-7, 2)


def square(x):
    tensor = x[0]
    return t.pow(t.norm(tensor), 2)


def rastrigin(x):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    tensor = x[0]
    n = len(tensor)
    A = 10
    f = A*n
    for i in range(n):
        x = tensor[i]
        f += (x ** 2 - A * t.cos(x * math.pi * 2))

    return f


def rosenbrock(x):
    tensor = x[0]
    n = len(tensor)
    f = 0
    for i in range(n-1):
        x_n = tensor[i]
        x_n_plus_1 = tensor[i+1]
        f += 100 * \
            t.pow((x_n_plus_1 - x_n**2), 2) + \
            t.pow(1-x_n, 2)

    return f
