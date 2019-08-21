from torch import nn
import torch
from glow.utils import Losses as L
from glow.utils import Optimizers as O
from glow.tensor_numpy_adapter import TensorNumpyAdapter
from glow.preprocessing import DataGenerator
import matplotlib.pyplot as plt


class Network(nn.Module):
    """
    Sequential models implementation on
    PyTorch

    """

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape  # input dimensions
        self.layer_list = nn.ModuleList([])  # list of module type layers
        self.num_layers = 0  # number of layers in the architecture
        self.adapter_obj = TensorNumpyAdapter()
        if torch.cuda.is_available():
            print("running on cuda enabled GPU")
        else:
            print("running on CPU environment")

    def add(self, layer_obj):
        if self.num_layers == 0:
            prev_input_shape = self.input_shape
        else:
            prev_input_shape = self.layer_list[self.num_layers - 1][-1].output_shape
        layer_obj.set_input(prev_input_shape)
        self.layer_list.append(self._make_layer_unit(layer_obj))
        self.num_layers = self.num_layers + 1

    def _make_layer_unit(self, layer_obj):
        layers = []
        layers.append(layer_obj)
        return nn.Sequential(*layers)

    def forward(self, x):
        layers = self.layer_list
        h = x
        iter_num = 0
        # iterate over the layers in the NN
        for layer in layers:
            h = layer(h)
            iter_num += 1
        return h

    def compile(
        self,
        optimizer_name="SGD",
        loss="cross_entropy",
        learning_rate=0.001,
        momentum=0.95,
    ):
        self.criterion = L.loss_function(loss)
        self.optimizer = O.optimizer(
            self.parameters(), learning_rate, momentum, optimizer_name
        )

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        num_epochs,
        validation_split=0.2,
        show_plot=True,
    ):
        train_losses, val_losses, epochs = [], [], []
        x_train, y_train = (
            self.adapter_obj.to_tensor(x_train),
            self.adapter_obj.to_tensor(y_train),
        )
        data_processor = DataGenerator()
        data_processor.set_dataset(
            x_train, y_train, batch_size, validation_split
        )  # tensorise the dataset elements for further processing in pytorch nn module
        TrainLoader = data_processor.get_trainloader()
        ValLoader = data_processor.get_validationloader()
        for epoch in range(num_epochs):
            print("epoch no. ", epoch + 1)
            # training loop
            train_loss = 0
            self.model.train()
            for batch_ndx, sample in enumerate(TrainLoader):
                self.optimizer.zero_grad()
                x = sample[0]
                y = sample[1]
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            else:
                # validation loop
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    # scope of no gradient calculations
                    for batch_ndx, sample in enumerate(ValLoader):
                        x = sample[0]
                        y = sample[1]
                        y_pred = self.model(x)
                        val_loss += self.criterion(y_pred, y).item()
                train_losses.append(train_loss / len(TrainLoader))
                val_losses.append(val_loss / len(ValLoader))
                epochs.append(epoch + 1)

        # plot the loss vs epoch graphs
        if show_plot:
            plt.plot(epochs, train_losses, color="red")
            plt.plot(epochs, val_losses, color="blue")
            plt.show()

    """
    def fit_generator(self, x_train, y_train, batch_size, num_epochs, validation_split=0.2, show_plot=True, data_generator):
        # TODO
    """

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            x = self.adapter_obj.to_tensor(x)
            return self.adapter_obj.to_numpy(self.forward(x))
