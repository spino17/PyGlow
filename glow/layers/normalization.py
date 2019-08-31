from torch import nn


class _BatchNorm(nn.Module):
    def __init__(self, rank, eps, momentum):
        super().__init__()
        self.rank = rank
        self.eps = eps
        self.momentum = momentum

    def set_input(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape  # same as input
        if self.rank == 1:
            self.norm_layer = nn.BatchNorm1d(
                num_features=self.input_shape[0], eps=self.eps, momentum=self.momentum
            )
        elif self.rank == 2:
            self.norm_layer = nn.BatchNorm2d(
                num_features=self.input_shape[0], eps=self.eps, momentum=self.momentum
            )
        else:
            self.norm_layer = nn.BatchNorm3d(
                num_features=self.input_shape[0], eps=self.eps, momentum=self.momentum
            )

    def forward(self, x):
        return self.norm_layer(x)


class BatchNorm1d(_BatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__(1, num_features, eps, momentum)
        self.args = [num_features, eps, momentum]

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class BatchNorm2d(_BatchNorm):
    def __init__(self, eps=1e-05, momentum=0.1):
        super().__init__(2, eps, momentum)
        self.args = [eps, momentum]

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)


class BatchNorm3d(_BatchNorm):
    def __init__(self, eps=1e-05, momentum=0.1):
        super().__init__(3, eps, momentum)
        self.args = [eps, momentum]

    def set_input(self, input_shape):
        super().set_input(input_shape)

    def forward(self, x):
        return super().forward(x)
