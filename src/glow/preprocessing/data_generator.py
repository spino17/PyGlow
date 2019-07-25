import numpy as np
from glow.tensor_numpy_adapter import TensorNumpyAdapter
from torch.utils.data import Dataset, DataLoader

class DataGenerator(Dataset):
    """
    class for implementing data batch generators

    """
    def __init__(self):
        # TODO
