from torch import nn
import torch
from glow.utils import Losses as L
from glow.utils import Optimizers as O
from .. import tensor_numpy_adapter
from glow.preprocessing import DataGenerator
import matplotlib.pyplot as plt
from .. import coordinates
from .. import metrics as metric_module
from tqdm import tqdm


class _Network(nn.Module):
    """
    Base class for Sequential models implementation on
    PyTorch.

    """

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape  # input dimensions
        self.layer_list = nn.ModuleList([])  # list of module type layers
        self.num_layers = 0  # number of layers in the architecture
        self.adapter_obj = tensor_numpy_adapter.get()
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
        optimizer="SGD",
        loss="cross_entropy",
        metrics=[],
        learning_rate=0.001,
        momentum=0.95,
    ):
        self.criterion = L.loss_function(loss)
        self.optimizer = O.optimizer(
            self.parameters(), learning_rate, momentum, optimizer
        )
        self.metrics = metrics

    def prepare_numpy_data(self, x_train, y_train, batch_size, validation_split):
        x_train, y_train = (
            self.adapter_obj.to_tensor(x_train),
            self.adapter_obj.to_tensor(y_train),
        )
        data_processor = DataGenerator()
        data_processor.set_dataset(
            x_train, y_train, batch_size, validation_split
        )  # tensorise the dataset elements for further processing in pytorch nn module
        train_loader = data_processor.get_trainloader()
        val_loader = data_processor.get_validationloader()
        return train_loader, val_loader

    def handle_metrics(self, metrics):
        metric_dict = {}
        for metric in metrics:
            if metric == "accuracy":
                metric_fn = metric_module.categorical_accuracy  # returns the function

            metric_dict[metric] = metric_fn

        return metric_dict

    def training_loop(self, num_epochs, train_loader, val_loader, show_plot=True):
        train_losses, val_losses, epochs = [], [], []
        train_len = len(train_loader)
        val_len = len(val_loader)
        metric_dict = self.handle_metrics(self.metrics)
        for epoch in range(num_epochs):
            # training loop
            print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            train_loss = 0
            self.train()
            print("Training loop: ")
            for x, y in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            else:
                # validation loop
                metric_values = []
                for key in metric_dict:
                    metric_values.append(metric_dict[key](y, y_pred))
                print("\n")
                print(
                    "loss: %.2f - acc: %.2f"
                    % (train_loss / train_len, metric_values[0])
                )
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    # scope of no gradient calculations
                    print("Validation loop: ")
                    for x, y in val_loader:
                        y_pred = self.forward(x)
                        val_loss += self.criterion(y_pred, y).item()
                    metric_values = []
                    for key in metric_dict:
                        metric_values.append(metric_dict[key](y, y_pred))
                    print("\n")
                    print(
                        "loss: %.2f - acc: %.2f"
                        % (val_loss / val_len, metric_values[0])
                    )
                train_losses.append(train_loss / train_len)
                val_losses.append(val_loss / val_len)
                epochs.append(epoch + 1)

        # plot the loss vs epoch graphs
        if show_plot:
            plt.plot(epochs, train_losses, color="red")
            plt.plot(epochs, val_losses, color="blue")
            plt.show()

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        num_epochs,
        validation_split=0.2,
        show_plot=True,
    ):
        train_loader, val_loader = self.prepare_numpy_data(
            x_train, y_train, batch_size, validation_split
        )
        self.training_loop(num_epochs, train_loader, val_loader, show_plot=show_plot)

    def fit_generator(self, train_loader, val_loader, num_epochs, show_plot=True):
        self.training_loop(num_epochs, train_loader, val_loader, show_plot)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.adapter_obj.to_tensor(x)
            return self.adapter_obj.to_numpy(self.forward(x))


class Sequential(_Network):
    """
    Keras like Sequential model.

    """

    def __init__(self, input_shape):
        super().__init__(input_shape)


class SequentialIB(_Network):
    """
    Class that attaches Information Bottleneck functionalities
    with the model to analyses the dynamics of training.

    """

    def __init__(self, input_shape, estimator="EDGE", params=None):
        super().__init__(input_shape)
        self.estimator = estimator
        self.params = params

    def forward(self, x):
        layers = self.layer_list
        layer_output = [x.detach()]
        h = x
        iter_num = 0
        # iterate over the layers in the NN
        for layer in layers:
            h = layer(h)
            with torch.no_grad():
                t = h.detach()
                layer_output.append(t)

            iter_num += 1
        return h, layer_output

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        num_epochs,
        validation_split=0.2,
        show_plot=True,
    ):
        train_loader, val_loader = self.prepare_numpy_data(
            x_train, y_train, batch_size, validation_split
        )
        train_losses, val_losses, epochs = [], [], []
        epoch_output = []
        for epoch in range(num_epochs):
            print("epoch no. ", epoch + 1)
            # training loop
            train_loss = 0
            self.train()
            batch_output = []
            for x, y in train_loader:
                self.optimizer.zero_grad()
                y_pred, layer_output = self.forward(x)
                batch_output.append(layer_output)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            else:
                # validation loop
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    # scope of no gradient calculations
                    for x, y in enumerate(val_loader):
                        y_pred, layer_output = self.forward(x)
                        val_loss += self.criterion(y_pred, y).item()
                train_losses.append(train_loss / len(train_loader))
                val_losses.append(val_loss / len(val_loader))
                epochs.append(epoch + 1)
                epoch_output.append(batch_output)

        print("Information plane calculations starting...")
        self.ipc = coordinates.get(
            epoch_output, self.estimator, self.params, self.num_layers
        )
        print("finished")

        # plot the loss vs epoch graphs
        if show_plot:
            plt.plot(epochs, train_losses, color="red")
            plt.plot(epochs, val_losses, color="blue")
            plt.show()

    def IP_plot(self):
        x_axis, y_axis = self.ipc.unpack()
        plt.scatter(x_axis, y_axis)
        plt.show()


"""
class HSIC(_Network):
    def __init__(self, input_shape):
        super().__init__()

    def training_loop(self, num_epochs, train_loader, val_loader, show_plot=True):
        # TODO
"""
