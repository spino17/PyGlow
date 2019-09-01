import numpy as np
import glow.tensor_numpy_adapter as tensor_numpy_adapter
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.nn.functional import one_hot


class DataGenerator(Dataset):
    """
    class for implementing data batch generators

    """

    def __init__(self):
        super().__init__()
        self.adapter_obj = tensor_numpy_adapter.get()

    def set_dataset(self, X, y, batch_size, validation_split=0.2):
        y = y.long().view(-1)
        self.dataset = TensorDataset(X, y)
        self.batch_size = batch_size
        training_length = int(len(self.dataset) * (1 - validation_split))
        lengths = [training_length, len(self.dataset) - training_length]
        train_dataset, validation_dataset = random_split(self.dataset, lengths)
        return train_dataset, validation_dataset

    def get_trainloader(self, train_dataset, batch_size):
        return DataLoader(train_dataset, batch_size)

    def get_validationloader(self, validation_dataset, batch_size):
        return DataLoader(validation_dataset, batch_size)

    def prepare_numpy_data(self, x_train, y_train, batch_size, validation_split):
        x_train, y_train = (
            self.adapter_obj.to_tensor(x_train),
            self.adapter_obj.to_tensor(y_train),
        )
        train_dataset, validation_dataset = self.set_dataset(
            x_train, y_train, batch_size, validation_split
        )  # tensorise the dataset elements for further processing in pytorch nn module
        train_loader = self.get_trainloader(train_dataset, batch_size)
        val_loader = self.get_validationloader(validation_dataset, batch_size)
        return train_loader, val_loader


"""
    def make_trainloader(self, data_path):
        # TODO

    def make_validationloader(self, data_path):
        # TODO
"""
