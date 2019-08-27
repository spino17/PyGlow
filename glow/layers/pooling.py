from torch import nn
from glow.utils import Activations as A
import math


class _Pooling1d(nn.Module):
    """
    Base abstract class for 1d pooling layer

    """

    def __init__(self, pooling_type, kernel_size, stride, padding, dilation):
        super().__init__()
        self.pooling_type = pooling_type  # Max or Avg
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def set_input(self, input_shape):
        self.input_shape = input_shape
        L_in = self.input_shape[1]
        L_out = math.floor(
            (L_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1
        )
        self.output_shape = (self.input_shape[0], L_out)
        if self.pooling_type == "Max":
            self.pooling_layer = nn.MaxPool1d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        elif self.pooling_type == "Avg":
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
            )

    def forward(self, x):
        return self.pooling_layer(x)


class MaxPool1d(_Pooling1d):
    def __init__(self, kernel_size, stride, padding, dilation=1):
        super().__init__("Max", kernel_size, stride, padding, dilation)

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class AvgPool1d(_Pooling1d):
    def __init__(self, kernel_size, stride, padding):
        super().__init__("Avg", kernel_size, stride, padding, 1)

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class _Pooling2d(nn.Module):
    def __init__(self, pooling_type, kernel_size, stride, padding, dilation):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def set_input(self, input_shape):
        self.input_shape = input_shape
        H_in = self.input_shape[1]
        W_in = self.input_shape[2]
        H_out = math.floor(
            (
                H_in
                + 2 * self.padding[0]
                - self.dilation[0] * (self.kernel_size[0] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )
        W_out = math.floor(
            (
                W_in
                + 2 * self.padding[1]
                - self.dilation[1] * (self.kernel_size[1] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )
        self.output_shape = (self.input_shape[0], H_out, W_out)
        if self.pooling_type == "Max":
            self.pooling_layer = nn.MaxPool2d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        elif self.pooling_layer == "Avg":
            self.pooling_layer = nn.AvgPool2d(
                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
            )

    def forward(self, x):
        return self.pooling_layer(x)


class MaxPool2d(_Pooling2d):
    def __init__(self, kernel_size, stride, padding, dilation=1):
        super().__init__("Max", kernel_size, stride, padding, dilation)

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class AvgPool2d(_Pooling2d):
    def __init__(self, kernel_size, stride, padding):
        super().__init__("Avg", kernel_size, stride, padding, (1, 1))

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class _Pooling3d(nn.Module):
    def __init__(self, pooling_type, kernel_size, stride, padding, dilation):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def set_input(self, input_shape):
        self.input_shape = input_shape
        D_in = self.input_shape[1]
        H_in = self.input_shape[2]
        W_in = self.input_shape[3]
        D_out = math.floor(
            (
                D_in
                + 2 * self.padding[0]
                - self.dilation[0] * (self.kernel_size[0] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )
        H_out = math.floor(
            (
                H_in
                + 2 * self.padding[1]
                - self.dilation[1] * (self.kernel_size[1] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )
        W_out = math.floor(
            (
                W_in
                + 2 * self.padding[2]
                - self.dilation[2] * (self.kernel_size[2] - 1)
                - 1
            )
            / self.stride[2]
            + 1
        )
        self.output_shape = (self.input_shape[0], D_out, H_out, W_out)
        if self.pooling_type == "Max":
            self.pooling_layer = nn.MaxPool3d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        elif self.pooling_layer == "Avg":
            self.pooling_layer = nn.AvgPool3d(
                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
            )

    def forward(self, x):
        return self.pooling_layer(x)


class MaxPool3d(_Pooling3d):
    def __init__(self, kernel_size, stride, padding, dilation=1):
        super().__init__("Max", kernel_size, stride, padding, dilation)

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class AvgPool3d(_Pooling3d):
    def __init__(self, kernel_size, stride, padding):
        super().__init__("Avg", kernel_size, stride, padding, (1, 1, 1))

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)
