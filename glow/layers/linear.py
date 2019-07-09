from torch import nn
from utils import activations as A


class Linear(nn.Module):
    """
    class for dense layer

    """
    def __init__(self, output_dim, activation):
        super(linear, self).__init__()
        self.output_dim = output_dim
        self.activation = activation

    # set the input attribute from previous layers
    def set_input(self, input_dim):
        self.input_dim = input_dim
        # components of the NN are defined here
        self.weights = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = A.activation_function(self.weights(x), self.activation)
        return x
