import torch.nn.functional as F


def relu(x):
    return F.relu(x)


def sigmoid(x):
    return F.sigmoid(x)


def tanh(x):
    return F.tanh(x)


def softmax(x, dim=1):
    return F.softmax(x, dim)


def log_softmax(x, dim=1):
    return F.log_softmax(x, dim)


def get(identifier):
    if identifier == "relu":
        return relu
    elif identifier == "sigmoid":
        return sigmoid
    elif identifier == "tanh":
        return tanh
    elif identifier == "softmax":
        return softmax
    elif identifier == "log_softmax":
        return log_softmax
    else:
        raise ValueError("Could not interpret " "activation identifier:", identifier)
