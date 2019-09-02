from torch import nn
import torch
import glow.losses as losses_module
from glow.utils import Optimizers as O
from glow.preprocessing import DataGenerator
import glow.tensor_numpy_adapter as tensor_numpy_adapter
import matplotlib.pyplot as plt
import glow.dynamics as dynamics_module
import glow.metrics as metric_module
from tqdm import tqdm


class _Network(nn.Module):
    """
    Base class for Sequential models implementation on
    PyTorch.

    """

    def __init__(self, input_shape, device, gpu, track_dynamics=False):
        super().__init__()
        self.input_shape = input_shape  # input dimensions
        self.layer_list = nn.ModuleList([])  # list of module type layers
        self.num_layers = 0  # number of layers in the architecture
        self.is_gpu = gpu
        self.device = device
        self.track_dynamics = track_dynamics

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
        hidden_outputs = []
        layers = self.layer_list
        h = x
        iter_num = 0
        # iterate over the layers in the NN
        for layer in layers:
            h = layer(h)
            with torch.no_grad():
                if self.track_dynamics:
                    t = h.detach()
                    hidden_outputs.append(t)
            iter_num += 1
        return h, hidden_outputs

    def compile(
        self,
        optimizer="SGD",
        loss="cross_entropy",
        metrics=[],
        learning_rate=0.001,
        momentum=0.95,
    ):
        self.criterion = losses_module.get(loss)
        self.optimizer = O.optimizer(
            self.parameters(), learning_rate, momentum, optimizer
        )
        self.metrics = metrics

    def handle_metrics(self, metrics):
        metric_dict = {}
        for metric in metrics:
            metric_fn = metric_module.get(metric)  # returns the function
            metric_dict[metric] = metric_fn

        return metric_dict

    def plot_loss(self, epochs, train_losses, val_losses):
        plt.title("Epoch vs Loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.grid(True)
        plt.plot(epochs, train_losses, color="red", label="training loss")
        plt.plot(epochs, val_losses, color="blue", label="validation loss")
        plt.show()

    def attach_evaluator(self, evaluator_obj):
        if self.track_dynamics is True:
            self.evaluator_list.append(evaluator_obj)
        else:
            raise Exception("Cannot attach for track_dyanmics=False")

    def evaluate_dynamics(self):
        evaluators = self.evaluator_list
        evaluated_dynamics = []
        # ** NOTE - This can be done in parallel !
        for evaluator in evaluators:
            evaluated_dynamics.append(self.dynamics_handler.evaluate(evaluator))
        return evaluated_dynamics

    def training_loop(self, num_epochs, train_loader, val_loader, show_plot=True):
        self.to(self.device)
        train_losses, val_losses, epochs = [], [], []
        train_len = len(train_loader)
        val_len = len(val_loader)
        metric_dict = self.handle_metrics(self.metrics)
        epoch_collector = []
        for epoch in range(num_epochs):
            # training loop
            print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            train_loss = 0
            print("Training loop: ")
            pbar = tqdm(total=train_len)
            batch_collector = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred, dynamics_segment = self.forward(x)
                dynamics_segment = [x] + dynamics_segment
                if self.track_dynamics and len(self.evaluator_list) > 0:
                    self.dynamics_handler = dynamics_module.get(dynamics_segment)
                    # batch_collector.append([x, *hidden_outputs, y])
                    evaluated_dynamics_segment = self.evaluate_dynamics()
                    batch_collector.append(evaluated_dynamics_segment)

                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                pbar.update(1)
            else:
                # validation loop
                pbar.close()
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
                    pbar = tqdm(total=val_len)
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_pred, _ = self.forward(x)
                        val_loss += self.criterion(y_pred, y).item()
                        pbar.update(1)
                    pbar.close()
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
                self.train()
            if self.track_dynamics:
                epoch_collector.append(batch_collector)

        if self.track_dynamics:
            self.evaluated_dynamics = dynamics_module.get(epoch_collector)

        # plot the loss vs epoch graphs
        if show_plot:
            self.plot_loss(epochs, train_losses, val_losses)

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        num_epochs,
        validation_split=0.2,
        show_plot=True,
    ):
        data_obj = DataGenerator()
        train_loader, val_loader = data_obj.prepare_numpy_data(
            x_train, y_train, batch_size, validation_split
        )
        self.training_loop(num_epochs, train_loader, val_loader, show_plot=show_plot)

    def fit_generator(self, train_loader, val_loader, num_epochs, show_plot=True):
        self.training_loop(num_epochs, train_loader, val_loader, show_plot)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.adapter_obj.to_tensor(x)
            y_pred, _ = self.forward(x)
            return self.adapter_obj.to_numpy(y_pred)


class Sequential(_Network):
    """
    Keras like Sequential model.

    """

    def __init__(self, input_shape, gpu):
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Running on CUDA enabled device !")
            else:
                raise Exception("No CUDA enabled GPU device found")
        else:
            device = torch.device("cpu")
            print("Running on CPU device !")

        # This paradigm does not support dynamics tracking
        super().__init__(input_shape, device, gpu, False)


class IBSequential(_Network):
    """
    Class that attaches Information Bottleneck functionalities
    with the model to analyses the dynamics of training.

    """

    def __init__(self, input_shape, gpu, track_dynamics=False):
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Running on CUDA enabled device !")
            else:
                raise Exception("No CUDA enabled GPU device found")
        else:
            device = torch.device("cpu")
            print("Running on CPU device !")
        super().__init__(input_shape, device, gpu, track_dynamics)
        self.evaluator_list = []  # collect all the evaluators

    """
    def attach_evaluator(self, evaluator_obj):
        if self.track_dynamics is True:
            self.evaluator_list.append(evaluator_obj)
        else:
            raise Exception("Cannot attach for track_dyanmics=False")

    def evaluate_dynamics(self):
        evaluators = self.evaluator_list
        evaluated_dynamics = []
        # ** NOTE - This can be done in parallel !
        for evaluator in evaluators:
            evaluated_dynamics.append(self.dynamics_handler.evaluate(evaluator))
        return evaluated_dynamics
    """

    def plot_dynamics(self, evaluated_dynamics, plot_show):
        for idx, flag in enumerate(plot_show):
            if flag:
                self.dynamics_handler.plot_dynamics(evaluated_dynamics[idx])
