import torch
from torch import nn
from glow.Information_bottelneck.estimators import HSIC


class HSIC(nn.Module):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, x):
        # TODO

    def fit(self, ):
        # TODO

    def predict(self, x):
        # TODO