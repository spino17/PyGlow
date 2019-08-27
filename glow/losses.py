from torch import nn

"""
class Losses:
    def loss_function(name):
        if name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif name == "NLLLoss":
            return nn.NLLLoss()
"""


def cross_entropy(y_pred, y_true):
    return nn.CrossEntropyLoss()(y_pred, y_true)


def NLLLoss(y_pred, y_true):
    return nn.NLLLoss()(y_pred, y_true)


def get(loss):
    if loss == "cross_entropy":
        return cross_entropy
    elif loss == "NLLLoss":
        return NLLLoss
