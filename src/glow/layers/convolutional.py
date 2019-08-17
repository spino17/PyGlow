from torch import nn
from glow.utils import Activations as A


class _Conv(nn.Module):
    def __init__(self)