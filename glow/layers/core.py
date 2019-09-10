from torch import nn
import glow.activations as activation_module
from glow.layer import Layer


class Dense(Layer):
    """
    Class for full connected dense layer.


    Arguments:
        output_dim (int): output dimension of the dense layer
        activation (str): activation function to be used for the layer (default: None)

    """

    def __init__(self, output_dim, activation=None):
        super(Dense, self).__init__()
        self.args = [output_dim, activation]
        self.output_shape = (output_dim, 1)
        self.activation = activation

    # set the input attribute from previous layers
    def set_input(self, input_shape):
        self.input_shape = input_shape
        # components of the NN are defined here
        self.weights = nn.Linear(self.input_shape[0], self.output_shape[0])

    def forward(self, x):
        x = activation_module.get(self.activation)(self.weights(x))
        return x


class Dropout(Layer):
    """
     Class for dropout layer - regularization using noise stablity of output.


     Arguments:
         prob (float): probability with which neurons in the previous layer is dropped

    """

    def __init__(self, prob):
        super(Dropout, self).__init__()
        self.args = [prob]
        self.prob = prob

    def set_input(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.dropout_layer = nn.Dropout(self.prob)

    def forward(self, x):
        return self.dropout_layer(x)


class Flatten(Layer):
    """
    Class for flattening the input shape.

    """

    def __init__(self):
        super().__init__()
        self.args = []

    def set_input(self, input_shape):
        self.input_dim = input_shape
        output_dim = 1
        for axis_value in input_shape:
            output_dim = output_dim * axis_value
        self.output_shape = (output_dim, 1)

    def forward(self, x):
        return x.view(x.size(0), -1)
