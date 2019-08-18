from torch import nn
from glow.utils import Activations as A


class _BatchNorm(nn.Module):
    def __init__(self):
        super(_BatchNorm, self).__init__()
        # TODO

    def set_input(self, input_shape):
        # TODO

    def forward(self, x):
        # TODO


class BatchNorm1d(_BatchNorm):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class BatchNorm2d(_BatchNorm):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class BatchNorm3d(_BatchNorm):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)
