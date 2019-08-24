import numpy as np
from glow.tensor_numpy_adapter import TensorNumpyAdapter
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.nn.functional import one_hot


class DataGenerator(Dataset):
    """
    class for implementing data batch generators

    """

    def __init__(self):
        super().__init__()

    def set_dataset(self, X, y, batch_size, validation_split=0.2):
        y = y.long().view(-1)
        self.dataset = TensorDataset(X, y)
        self.batch_size = batch_size
        training_length = int(len(self.dataset) * (1 - validation_split))
        lengths = [training_length, len(self.dataset) - training_length]
        self.train_dataset, self.validation_dataset = random_split(
            self.dataset, lengths
        )

    def get_trainloader(self):
        return DataLoader(self.train_dataset, self.batch_size)

    def get_validationloader(self):
        return DataLoader(self.validation_dataset, self.batch_size)
