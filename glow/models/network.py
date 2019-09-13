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
import numpy as np


class Network(nn.Module):
    """
    Base class for all neural network modules.

    Your network should also subclass this class.


    Arguments:
        input_shape (tuple): input tensor shape
        device (torch.device or int): `GPU` or `CPU` for training purposes
        gpu (bool): true if `GPU` is enabled on the system, false otherwise
        track_dynamics (bool): tracks the NN dynamics during training (stores input-output for every intermediate layer)

    Attributes:
        input_shape (tuple): input tensor shape
        layer_list (iterable): an iterable of pytorch :class:`torch.nn.modules.container.Sequential` type layers
        num_layers (int): number of layers in the model
        is_gpu (bool): true if `GPU` is enabled on the system, false otherwise
        device (torch.device or int): `GPU` or `CPU` for training purposes
        track_dynamics (bool): tracks the NN dynamics during training (stores input-output for every intermediate layer)
        criterion (callable): loss function for the model
        optimizer (torch.optim.Optimizer): optimizer for training the model
        metrics (str): metric to be used for evaluating performance of the model

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
        """
        Adds specified (by the `layer_obj` argument) layer to the model.


        Arguments:
            layer_obj (glow.Layer): object of specific layer to be added

        """
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
        """
        Method for defining forward pass through the model.

        This method needs to be overridden by your implementation contain logic
        of the forward pass through your model.


        Arguments:
            x (torch.Tensor): input tensor to the model

        Returns:
            (tuple): tuple containing:
                (torch.Tensor): output tensor of the network
                (iterable): list of hidden layer outputs for dynamics tracking purposes

        """
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
        **kwargs
    ):
        """
        Compile the model with attaching optimizer and loss function to the
        model.


        Arguments:
            optimizer (torch.optim.Optimizer): optimizer to be used during training process
            loss (loss): loss function for back-propagation
            metrics (list): list of all performance metric which needs to be evaluated in validation pass
            learning_rate (float, optional): learning rate for gradient descent step (default: 0.001)
            momentum (float, optional): momentum for different variants of optimizers (default: 0.95)

        """
        if callable(loss):
            self.criterion = loss
        elif isinstance(loss, str):
            self.criterion = losses_module.get(loss, **kwargs)
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
        """
        Attaches an evaluator with the model which will get evaluated at every
        pass of batch and obtain information plane coordinates according to
        defined criterion in the 'evaluator_obj'.

        It appends the 'evaluator_obj'
        to the list `evaluator_list` which contains all the attached evaluators
        with the model.


        Arguments:
            evaluator_obj (glow.information_bottleneck.Estimator): evaluator object with has criterion defined which will get evaluated for the dynamics of the training process

        """
        if self.track_dynamics is True:
            self.evaluator_list.append(evaluator_obj)
        else:
            raise Exception("Cannot attach for track_dynamics=False")

    def evaluate_dynamics(self):
        evaluators = self.evaluator_list
        evaluated_dynamics = []
        # ** NOTE - This can be done in parallel !
        for evaluator in evaluators:
            evaluated_dynamics.append(self.dynamics_handler.evaluate(evaluator))
        return evaluated_dynamics

    def training_loop(self, num_epochs, train_loader, val_loader, show_plot):
        self.to(self.device)
        train_losses, val_losses, epochs = [], [], []
        train_len = len(train_loader)
        val_len = len(val_loader)
        metric_dict = self.handle_metrics(self.metrics)
        epoch_collector = []
        for epoch in range(num_epochs):
            # training loop
            print("\n")
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
            self.evaluated_dynamics = np.array(epoch_collector)

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
        show_plot=False,
    ):
        """
        Fits the dataset passed as numpy array (Keras like pipeline) in the arguments.


        Arguments:
            x_train (numpy.ndarray): training input dataset
            y_train (numpy.ndarray): training ground-truth labels
            batch_size (int): batch size of one batch
            num_epochs (int): number of epochs for training
            validation_split (float, optional): proportion of the total dataset to be used for validation (default: 0.2)
            show_plot (bool, optional): if true plots the training loss (red), validation loss (blue) vs epochs (default: True)

        """
        data_obj = DataGenerator()
        train_loader, val_loader = data_obj.prepare_numpy_data(
            x_train, y_train, batch_size, validation_split
        )
        self.training_loop(num_epochs, train_loader, val_loader, show_plot=show_plot)

    def fit_generator(self, train_loader, val_loader, num_epochs, show_plot=False):
        """
        Fits the dataset by taking data-loader as argument.


        Arguments:
            num_epochs (int): number of epochs for training
            train_loader (torch.utils.data.DataLoader): training dataset (with already processed batches)
            val_loader (torch.utils.data.DataLoader): validation dataset (with already processed batches)
            show_plot (bool, optional): if true plots the training loss (red), validation loss (blue) vs epochs (default: True)

        """
        self.training_loop(num_epochs, train_loader, val_loader, show_plot)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.adapter_obj.to_tensor(x)
            y_pred, _ = self.forward(x)
            return self.adapter_obj.to_numpy(y_pred)


class Sequential(Network):
    """
    Keras like Sequential model.

    Arguments:
        input_shape (tuple): input tensor shape
        gpu (bool, optional): if true then PyGlow will attempt to use `GPU`, for false `CPU` will be used (default: False)
    """

    def __init__(self, input_shape, gpu=False):
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


class IBSequential(Network):
    """
    Keras like Sequential model with extended more sophisticated Information
    Bottleneck functionalities for analysing the dynamics of training.


    Arguments:
        input_shape (tuple): input tensor shape
        gpu (bool, optional): if true then PyGlow will attempt to use `GPU`, for false `CPU` will be used (default: False)
        track_dynamics (bool): if true then will track the input-hidden-output dynamics segment and will allow evaluator to attach to the model, for false no track for dynamics is kept
        save_dynamics (bool, optional): if true then saves the whole training process dynamics into a distributed file (for efficiency)

    Attributes:
        evaluator_list (iterable): list of :class:`glow.information_bottleneck.Estimator` instances which stores the evaluators for the model
        evaluated_dynamics (iterable): list of evaluated dynamics segment information coordinates for intermediate layer for each evaluator averaged over batch for each epoch

    Shape:
        evaluator_list has shape (N, E, L, 2) where:
            - N: Number of epochs
            - E: Number of evaluators
            - L: Number of layers with parameters (Flatten and Dropout excluded)

        and last dimension is equal to 2 which stores 2-D information plane coordinates

    """

    def __init__(
        self, input_shape, gpu=False, track_dynamics=False, save_dynamics=False
    ):
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
