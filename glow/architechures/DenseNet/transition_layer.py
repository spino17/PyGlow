from torch import nn


class TransitionLayer(nn.Module):
    def __init__(self, output_dim):
        super(TransitionLayer, self).__init__()
        self.output_dim = output_dim

    def set_input(self, input_shape):
        self.input_shape = input_shape
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=self.output_dim)
        self.conv = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=self.output_dim,
            kernel_size=1,
            bias=False,
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.output_shape = (
            self.output_dim,
            int(self.input_shape[1] / 2),
            int(self.input_shape[2] / 2),
        )

    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)

        return out
