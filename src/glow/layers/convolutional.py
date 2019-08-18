from torch import nn
from glow.utils import Activations as A


class _Conv(nn.Module):
    def __init__(
        self, rank, filters, kernel_size, strides, padding, data_format, **kwargs
    ):
        super(_Conv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        if len(kwargs) <= 1:
            self.input_shape_tuple = kwargs
        else:
            print("'input_shape' argument got more values than expected")
        self.output_dim = filters

    def set_input(self, input_dim):
        if len(self.input_shape_tuple) == 0:
            self.in_channels = input_dim
        elif len(self.input_shape_tuple) == 1:
            if self.data_format == "channel_first":
                channel_axis = 1
            else:
                channel_axis = -1
            self.in_channels = self.input_shape_tuple["input_shape"][channel_axis]

        # defines the layer according to rank from PyTorch Conv layer
        if self.rank == 0:
            self.conv_layer = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
        elif self.rank == 1:
            self.conv_layer = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
        else:
            self.conv_layer = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )

    def forward(self, x):
        return self.conv_layer(x)


class Conv1d(_Conv):
    def __init__(self, filters, kernel_size, strides, padding, data_format, **kwargs):
        super(Conv1d, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )

    def set_input(self, input_dim):
        super(Conv1d, self).set_input

    def forward(self, x):
        return super(Conv1d, self).forward(x)


class Conv2d(_Conv):
    def __init__(self, filters, kernel_size, strides, padding, data_format, **kwargs):
        super(Conv2d, self).__init__(
            rank=2,
            filters=filters,
            kerne_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )

    def set_input(self, input_dim):
        super(Conv2d, self).set_input

    def forward(self, x):
        return super(Conv2d, self).forward(x)


class Conv3d(_Conv):
    def __init__(self, filters, kernel_size, strides, padding, data_format, **kwargs):
        super(Conv2d, self).__init__(
            rank=3,
            filters=filters,
            kerne_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )

    def set_input(self, input_dim):
        super(Conv3d, self).set_input

    def forward(self, x):
        return super(Conv3d, self).forward(x)
