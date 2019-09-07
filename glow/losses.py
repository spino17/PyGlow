from torch import nn
from glow.information_bottelneck import HSIC
from torch.nn.functional import one_hot


def cross_entropy(y_pred, y_true):
    return nn.CrossEntropyLoss()(y_pred, y_true)


def NLLLoss(y_pred, y_true):
    return nn.NLLLoss()(y_pred, y_true)


def HSICLoss(z, x, y, sigma, regularize_coeff, gpu):
    estimator = HSIC(sigma, gpu)
    y = one_hot(y, num_classes=-1).float()
    loss_1 = estimator.criterion(z, x)
    loss_2 = estimator.criterion(z, y)
    return loss_1 - regularize_coeff * loss_2


def get(identifier):
    if identifier == "cross_entropy":
        return cross_entropy
    elif identifier == "NLLLoss":
        return NLLLoss
    elif identifier == "HSIC_loss":
        return HSICLoss
    else:
        raise ValueError("Could not interpret " "loss function identifier:", identifier)
