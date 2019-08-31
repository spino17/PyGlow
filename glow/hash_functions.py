import torch


def floor_hash(x, epsilon, b):
    return torch.floor((1 / epsilon) * (x + b))


def get(identifier):
    if identifier == "floor_hash":
        return floor_hash
