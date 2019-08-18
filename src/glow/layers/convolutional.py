from torch import nn
from glow.utils import Activations as A
import math


class _Conv(nn.Module):
    """
    Base abstract class for convolution layers of all rank.

    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        stride,
        padding,
        dilation,
        data_format,
        activation,
        **kwargs
    ):
        super().__init__()
        self.rank = rank
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.data_format = data_format
        if len(kwargs) <= 1:
            self.input_shape_tuple = kwargs
        else:
            print("'input_shape' argument got more values than expected")
        self.activation = activation

    def set_input(self, input_shape):
        """
        if len(self.input_shape_tuple) == 0:
            self.input_shape = input_shape
        else:
            self.input_shape = self.input_shape_tuple
        """
        self.input_shape = input_shape
        self.in_channels = self.input_shape[0]  # according to PyTorch convention

        # defines the layer according to rank from PyTorch Conv layer
        if self.rank == 1:
            self.conv_layer = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
            L_in = self.input_shape[1]
            C_out = self.filters
            L_out = math.floor(
                (L_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
                + 1
            )
            self.output_shape = (C_out, L_out)
        elif self.rank == 2:
            self.conv_layer = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
            H_in = self.input_shape[1]
            W_in = self.input_shape[2]
            C_out = self.filters
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
            self.output_shape = (C_out, H_out, W_out)
        else:
            self.conv_layer = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
            D_in = self.input_shape[1]
            H_in = self.input_shape[2]
            W_in = self.input_shape[3]
            C_out = self.filters
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
            self.output_shape = (C_out, D_out, H_out, W_out)

    def forward(self, x):
        return A.activation_function(self.conv_layer(x), self.activation)


class Conv1d(_Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        stride,
        padding,
        dilation,
        data_format,
        activation,
        **kwargs
    ):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            data_format=data_format,
            activation=activation,
            **kwargs
        )

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class Conv2d(_Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        stride,
        padding,
        dilation,
        data_format,
        activation,
        **kwargs
    ):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            data_format=data_format,
            activation=activation,
            **kwargs
        )

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class Conv3d(_Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        stride,
        padding,
        dilation,
        data_format,
        activation,
        **kwargs
    ):
        super().__init__(
            rank=3,
            filters=filters,
            kerne_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            data_format=data_format,
            activation=activation,
            **kwargs
        )

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)
