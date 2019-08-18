from torch import nn
from glow.utils import Activations as A
import math


class _Pooling1d(nn.Module):
    def __init__(self, ):
        # TODO

    def set_input(self, input_shape):
        # TODO

    def forward(self, x):
        # TODO


class MaxPool1d(_Pooling1d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class MaxUnpool1d(_Pooling1d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class AvgPool1d(_Pooling1d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class _Pooling2d(nn.Module):
    def __init__(self, ):
        # TODO

    def set_input(self, input_shape):
        # TODO

    def forward(self, x):
        # TODO


class MaxPool2d(_Pooling2d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class MaxUnpool2d(_Pooling2d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class AvgPool2d(_Pooling2d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class _Pooling3d(nn.Module):
    def __init__(self, ):
        # TODO

    def set_input(self, input_shape):
        # TODO

    def forward(self, x):
        # TODO


class MaxPool3d(_Pooling3d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class MaxUnpool3d(_Pooling3d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class AvgPool3d(_Pooling3d):
    def __init__(self, ):
        super().__init__()
        # TODO

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)