from torch import nn
from glow.information_bottleneck import HSIC
from torch.nn.functional import one_hot


def cross_entropy(y_pred, y_true):
    return nn.CrossEntropyLoss()(y_pred, y_true)


def NLLLoss(y_pred, y_true):
    return nn.NLLLoss()(y_pred, y_true)


def HSICLoss(z, x, y, gpu, sigma, regularize_coeff):
    estimator = HSIC(sigma, gpu)
    y = one_hot(y, num_classes=-1).float()
    loss_1 = estimator.criterion(z, x)
    loss_2 = estimator.criterion(z, y)
    return loss_1 - regularize_coeff * loss_2


def get(identifier, **kwargs):
    if identifier == "cross_entropy":
        return cross_entropy
    elif identifier == "NLLLoss":
        return NLLLoss
    elif identifier == "HSIC_loss":

        def curry_func(z, x, y, gpu):
            if "sigma" in kwargs.keys():
                sigma = kwargs["sigma"]
            else:
                raise Exception("Cannot find sigma value for HSIC loss function")
            if "regularize_coeff" in kwargs.keys():
                regularize_coeff = kwargs["regularize_coeff"]
            else:
                raise Exception(
                    "Cannot find regularization coefficient for HSIC loss function"
                )
            return HSICLoss(z, x, y, gpu, sigma, regularize_coeff)

        return curry_func
    else:
        raise ValueError("Could not interpret " "loss function identifier:", identifier)
