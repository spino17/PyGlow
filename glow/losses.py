from torch import nn
from glow.Information_bottelneck import HSIC


def cross_entropy(y_pred, y_true):
    return nn.CrossEntropyLoss()(y_pred, y_true)


def NLLLoss(y_pred, y_true):
    return nn.NLLLoss()(y_pred, y_true)


def HSICLoss(z, x, y, sigma, regularize_coeff):
    estimator = HSIC(sigma)
    loss_1 = estimator.HS_Criterion(z, x)
    loss_2 = estimator.HS_Criterion(z, y)
    return loss_1 - regularize_coeff * loss_2


def get(loss):
    if loss == "cross_entropy":
        return cross_entropy
    elif loss == "NLLLoss":
        return NLLLoss
    elif loss == "HSIC_loss":
        return HSICLoss
