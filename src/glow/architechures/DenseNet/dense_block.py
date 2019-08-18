from torch import nn
from glow.utils import Activations as A
import torch


class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()

    def set_input(self, input_dim):
        self.input_dim = input_dim
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=input_dim)
        self.conv1 = nn.Conv2d(
            in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.output_dim = 160  # conv1 + conv2 + conv3 + conv4 + conv5

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense
