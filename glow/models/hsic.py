import torch
from torch import nn
from .network import _Network


class HSIC(nn.Module, _Network):
    """
    The HSIC Bottelneck: Deep Learning without backpropagation

    """

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, x):
        # TODO

    def add(self, layer_obj):
        # TODO

    def compile(self):
        # TODO

    def training_loop(self):
        # TODO