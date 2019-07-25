from torch import nn


class Dropout(nn.Module):
    """
     class for dropout layer - Regularization

    """

    def __init__(self, prob):
        super(Dropout, self).__init__()
        self.prob = prob

    def set_input(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.dropout_layer = nn.Dropout(self.prob)

    def forward(self, x):
        return self.dropout_layer(x)
