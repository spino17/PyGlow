from torch import nn
from glow.layer import Layer


class _BatchNorm(Layer):
    """
    Base class for all batch normalization layers.

    """

    def __init__(self, dim, eps, momentum):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

    def set_input(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape  # same as input
        if self.dim == 1:
            self.norm_layer = nn.BatchNorm1d(
                num_features=self.input_shape[0], eps=self.eps, momentum=self.momentum
            )
        elif self.dim == 2:
            self.norm_layer = nn.BatchNorm2d(
                num_features=self.input_shape[0], eps=self.eps, momentum=self.momentum
            )
        elif self.dim == 3:
            self.norm_layer = nn.BatchNorm3d(
                num_features=self.input_shape[0], eps=self.eps, momentum=self.momentum
            )

    def forward(self, x):
        return self.norm_layer(x)


class BatchNorm1d(_BatchNorm):
    """
    1-D batch normalization layer.

    See https://arxiv.org/abs/1502.03167 for more information on batch
    normalization.


    Arguments:
        eps (float):  a value added to the denominator for numerical stability (default: 1e-5)
        momentum (float): the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average) (default: 0.1)

    """

    def __init__(self, eps=1e-05, momentum=0.1):
        super().__init__(1, eps, momentum)
        self.args = [eps, momentum]

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class BatchNorm2d(_BatchNorm):
    """
    2-D batch normalization layer.


    Arguments:
        eps (float):  a value added to the denominator for numerical stability (default: 1e-5)
        momentum (float): the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average) (default: 0.1)

    """

    def __init__(self, eps=1e-05, momentum=0.1):
        super().__init__(2, eps, momentum)
        self.args = [eps, momentum]

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class BatchNorm3d(_BatchNorm):
    """
    3-D batch normalization layer.


    Arguments:
        eps (float):  a value added to the denominator for numerical stability (default: 1e-5)
        momentum (float): the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average) (default: 0.1)

    """

    def __init__(self, eps=1e-05, momentum=0.1):
        super().__init__(3, eps, momentum)
        self.args = [eps, momentum]

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)
