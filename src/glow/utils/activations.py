import torch.nn.functional as F


class Activations:
    def activation_function(x, name):
        if name is None:
            return x
        elif name == "relu":
            return F.relu(x)
        elif name == "sigmoid":
            return F.sigmoid(x)
        elif name == "tanh":
            return F.tanh(x)
        elif name == "softmax":
            return F.softmax(x, dim=1)
        elif name == "none":
            return x
        elif name == "log_softmax":
            return F.log_softmax(x, dim=1)
