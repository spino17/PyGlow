from torch import nn
import glow.activations as activation_module
from glow.layers.core import Dense


class HSICoutput(Dense):
    """
    Class for HSIC sigma network output layer.

    """

    def __init__(self, output_dim, activation=None):
        super().__init__(output_dim, activation)
