from torch import nn
from glow.utils import Activations as A


class HSICoutput(nn.Module):
    """
    Output layer of HSIC architechure for post training purposes.

    """

    def __init__(self, output_dim, loss, activation="softmax"):
        self.output_shape = (output_dim, 1)  # output_dim = number of classes
        self.loss = loss
        self.activation = activation

    def set_input(self, input_shape):
        self.input_shape = input_shape
        self.weights = nn.Linear(self.input_shape[0], self.output_shape[0])

    def forward(self, x):
        return A.activation_function(self.weights(x), self.activation)
