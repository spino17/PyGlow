import torch
from torch import nn


def gaussian_kernel(x, y, sigma):
    m = x.shape[0]
    vec_dim = x.shape[1]
    x = x.view(m, 1, vec_dim)
    y = y.view(m, vec_dim)
    z = (x - y).float()
    return torch.exp((-1 / (2 * (sigma ** 2))) * (torch.norm(z, dim=2) ** 2))


def get(kernel):
    if kernel == "gaussian":
        return gaussian_kernel
