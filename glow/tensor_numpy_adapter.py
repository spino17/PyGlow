import torch


class TensorNumpyAdapter:
    """
    class for adapter interface between numpy array
    type and Tensor objects in PyTorch

    """

    def to_tensor(self, x):
        return torch.from_numpy(x).float()

    def to_numpy(self, x):
        return x.numpy()


def get():
    return TensorNumpyAdapter()
