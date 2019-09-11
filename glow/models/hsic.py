import torch
from torch import nn
from glow.models.network import Network
import matplotlib.pyplot as plt
import glow.losses as losses_module
from glow.utils import Optimizers as O
from glow.layers.core import Flatten, Dropout
from glow.layers import HSICoutput
from tqdm import tqdm
from glow.preprocessing import DataGenerator
from glow.information_bottleneck import Estimator


class HSIC(Network):
    """
    The HSIC Bottelneck: Deep Learning without backpropagation

    Arguments:
        input_shape (tuple): input tensor shape
        device (torch.device, optional):
        gpu (bool): true if `GPU` is enabled on the system, false otherwise

    """

    def __init__(self, input_shape, device, gpu, **kwargs):
        super().__init__(input_shape, device, gpu)
        self.loss_dict = {}  # stores the losses of the individual layers

    def add(self, layer_obj, loss_criterion=None, regularize_coeff=0):
        """
        Adds specified layer and loss criterion to the model which will be used
        for measuring objective between layer's current representation and
        optimal representation (representation with minimum IB-based objective).


        Arguments:
            layer_obj (glow.Layer): object of specific layer to be added
            loss_criterion (glow.information_bottleneck.Estimator): loss function for the layer which is added
            regularize_coeff (float): regularization coefficient between generalization and compression of IB type objective

        """
        if self.num_layers == 0:
            prev_input_shape = self.input_shape
        else:
            prev_input_shape = self.layer_list[self.num_layers - 1][-1].output_shape
        layer_obj.set_input(prev_input_shape)
        self.layer_list.append(self._make_layer_unit(layer_obj))
        self.num_layers = self.num_layers + 1
        if loss_criterion is not None:
            if isinstance(loss_criterion, Estimator):
                self.loss_dict[self.num_layers - 1] = [loss_criterion, regularize_coeff]
            else:
                raise Exception("loss criterion expects a instance of 'Estimator'")

    def forward(self, x):
        """
        Method for defining forward pass through the model.

        This method needs to be overridden by your implementation which contains
        the logic of the forward pass through your model.


        Arguments:
            x (torch.Tensor): input tensor to the model

        Returns:
            (iterable): list of hidden layer outputs (objects of type :class:`torch.Tensor`) which are detached from their previous layer's gradients

        """
        layers = self.layer_list
        t = x
        iter_num = 0
        hidden_outputs = []
        for layer_idx, layer in enumerate(layers):
            h = layer(t)
            hidden_outputs.append(h)
            t = h.detach()  # detached vector to cut of the previous gradients
            iter_num += 1
        return hidden_outputs

    def sequential_forward(self, x):
        """
        Sequentially calculate the output taking HSIC network as sequential
        feedforward neural network and is equivalent to forward pass in standard
        models.


        Arguments:
            x (torch.Tensor): input tensor to the network

        Returns:
            (torch.Tensor): output of the sequential feedforward network

        """
        layers = self.layer_list
        h = x
        for layer_idx, layer in enumerate(layers):
            h = layer(h)
        return h

    def make_layer_optimizers(self, optimizer, learning_rate, momentum):
        optimizer_list = []
        for layer_idx, layer in enumerate(self.layer_list):
            if isinstance(layer[0], Flatten) or isinstance(layer[0], Dropout):
                optimizer_list.append(None)
            else:
                optimizer_list.append(
                    O.optimizer(layer.parameters(), learning_rate, momentum, optimizer)
                )
        return optimizer_list

    def compile(
        self,
        loss_criterion,
        optimizer='SGD',
        regularize_coeff=100,
        learning_rate=0.001,
        momentum=0.95,
        **kwargs
    ):
        """
        Compile the HSIC network with loss criterion (criterion objective used
        as loss function for intermediate representations).

        All the layers which did not have any criterion passed as argument at
        the time of '.add' of layer automatically takes this loss criterion.


        Arguments:
            loss_criterion (glow.information_bottleneck.Estimator): criterion function which is an instance of `glow.information_bottleneck.Estimator`
            optimizer (torch.optim.Optimizer): optimizer to be used during training process for all the layers
            regularize_coeff (float): trade-off parameter between generalization and compression according to IB-based theory
            learning_rate (float, optional): learning rate for gradient descent step (default: 0.001)
            momentum (float, optional): momentum for different variants of optimizers (default: 0.95)

        """
        if isinstance(loss_criterion, Estimator):
            self.global_criterion = [loss_criterion, regularize_coeff]
        else:
            raise Exception("loss criterion expects a instance of 'Estimator'")
        self.layer_optimizers = self.make_layer_optimizers(
            optimizer, learning_rate, momentum
        )

    def pre_training_loop(self, num_epochs, train_loader, val_loader):
        """
        Pre training phase in which hidden representations are learned using
        HSIC training paradigm.

        Arguments:
            num_epochs (int): number of epochs for pre-training phase
            train_loader (torch.utils.data.DataLoader): training dataset (with already processed batches)
            val_loader (torch.utils.data.DataLoader): validation dataset (with already processed batches)

        """
        self.to(self.device)
        train_len = len(train_loader)
        for epoch in range(num_epochs):
            # pre-training loop
            print("\n")
            print("Pre-Train-Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            pbar = tqdm(total=train_len)
            i = 0
            for x, y in train_loader:
                # contains the hidden representation from forward pass
                x, y = x.to(self.device), y.to(self.device)
                hidden_outputs = self.forward(x)
                # ** NOTE - This can be done in parallel !
                for idx, z in enumerate(hidden_outputs):
                    if self.layer_optimizers[idx] is not None:
                        self.layer_optimizers[idx].zero_grad()
                        if idx in self.loss_dict.keys():
                            criterion = self.loss_dict[idx]
                        else:
                            criterion = self.global_criterion

                        loss_criterion = criterion[0]
                        regularize_coeff = criterion[1]
                        loss = losses_module.get("HSIC_loss")(
                            z,
                            x.view(x.shape[0], -1),
                            y,
                            loss_criterion,
                            regularize_coeff,
                        )
                        loss.backward()
                        self.layer_optimizers[idx].step()
                pbar.update(1)
            pbar.close()

    def freeze_hidden_grads(self):
        for layer_idx, layer in enumerate(self.layer_list):
            for params in layer.parameters():
                params.requires_grad = False


class HSICSequential(HSIC):
    """
    Base implementation for HSIC networks.

    This class forms instances for multi-model sigma network as given in the paper
    https://arxiv.org/abs/1908.01580 .


    Arguments:
        input_shape (tuple): input tensor shape
        gpu (bool, optional): if true then PyGlow will attempt to use `GPU`, for false `CPU` will be used (default: False)

    """

    def __init__(self, input_shape, gpu=False, **kwargs):
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Running on CUDA enabled device !")
            else:
                raise Exception("No CUDA enabled GPU device found")
        else:
            device = torch.device("cpu")
            print("Running on CPU device !")
        super().__init__(input_shape, device, gpu, **kwargs)


"""
class HSICSigma(HSIC):

    Ensemble various values of sigma to capture dependence at various scales
    and give aggregate output - Sigma Network

    for more information refer to the paper: https://arxiv.org/abs/1908.01580


    def __init__(self, input_shape, gpu=True, **kwargs):
        if gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Running on CUDA enabled device !")
            else:
                raise Exception("No CUDA enabled GPU device found")
        else:
            device = torch.device("cpu")
            print("Running on CPU device !")
        super().__init__()
        self.model_list = nn.ModuleList([])
        self.num_models = 0
        # ** NOTE - This can be done in parallel !
        for sigma in sigma_set:
            self.num_models += 1
            self.model_list.append(HSIC(input_shape, device, gpu))
        self.device = device
        self.num_layers = 0  # number of layers including output layer

    def add(self, layer_obj, loss=None, **kwargs):
        # ** NOTE - This can be done in parallel !
        if isinstance(layer_obj, HSICoutput):
            prev_input_shape = self.model_list[0].layer_list[-1][-1].output_shape
            layer_obj.set_input(prev_input_shape)
            self.model_list.append(nn.Sequential(layer_obj))
            self.output_criterion = None
            if loss is not None:
                if not callable(loss):
                    loss = losses_module.get(loss, **kwargs)
                self.output_criterion = loss
        else:
            for model in self.model_list:
                if len(layer_obj.args) == 0:
                    init_layer_obj = layer_obj.__class__()
                else:
                    init_layer_obj = layer_obj.__class__(*layer_obj.args)
                model.add(init_layer_obj, loss, **kwargs)
        self.num_layers += 1

    def compile_ensemble(
        self, optimizer, ensemble_loss, learning_rate, momentum, **kwargs
    ):
        # ** NOTE - This can be done in parallel !
        for model_idx, model in enumerate(self.model_list):
            if model_idx != self.num_models:
                model.compile(
                    optimizer, ensemble_loss, learning_rate, momentum, **kwargs
                )

    def compile_output(
        self, optimizer, output_loss, learning_rate, momentum, metrics, **kwargs
    ):
        model = self.model_list[self.num_models]
        if not callable(output_loss):
            output_loss = losses_module.get(output_loss, **kwargs)
        if self.output_criterion is None:
            self.output_criterion = output_loss
        self.output_optimizer = O.optimizer(
            model.parameters(), learning_rate, momentum, optimizer
        )
        self.metrics = metrics

    def compile(
        self,
        optimizer,
        ensemble_loss,
        output_loss,
        metrics,
        learning_rate=0.001,
        momentum=0.95,
        **kwargs
    ):
        # ** NOTE - This can be done in parallel !
        for model_idx, model in enumerate(self.model_list):
            if model_idx == self.num_models:
                if not callable(loss):
                    output_loss = losses_module.get(output_loss, **kwargs)
                if self.output_criterion is None:
                    self.output_criterion = output_loss
                self.metrics = metrics
            else:
                model.compile(optimizer, loss, learning_rate, momentum, **kwargs)

    def training_loop(
        self, train_loader, val_loader, pre_num_epochs, post_num_epochs, show_plot
    ):
        print("\n")
        print("Pre Training phase starting ...")
        # ** NOTE - This can be done in parallel !
        for model_idx, model in enumerate(self.model_list):
            if model_idx < self.num_models:
                print("\n")
                print("Pre-Training HSIC segment model - (" + str(model.sigma) + ")")
                model.pre_training_loop(pre_num_epochs, train_loader, val_loader)
                model.freeze_hidden_grads()  # freeze the grads of all the HSIC segments
        print("\n")
        print("Post Training phase starting ...")
        train_losses, val_losses, epochs = [], [], []
        train_len = len(train_loader)
        val_len = len(val_loader)
        metric_dict = self.model_list[0].handle_metrics(self.metrics)
        for epoch in range(post_num_epochs):
            print("Epoch " + str(epoch + 1) + "/" + str(post_num_epochs))
            train_loss = 0
            print("Training loop: ")
            pbar = tqdm(total=train_len)
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.output_optimizer.zero_grad()
                dim_0 = x.shape[0]
                dim_1 = self.model_list[0].layer_list[-1][-1].output_shape[0]
                h = torch.zeros(dim_0, dim_1).to(self.device)
                # ** NOTE - This can be done in parallel !
                for model_idx, model in enumerate(self.model_list):
                    # print(model.is_cuda)
                    model.to(self.device)
                    if model_idx == self.num_models:
                        y_pred = model(h)
                    else:
                        h += model.sequential_forward(
                            x
                        )  # aggregating all the representations from HSIC segments
                loss = self.output_criterion(y_pred, y)
                loss.backward()
                self.output_optimizer.step()
                train_loss += loss.item()
                pbar.update(1)
            else:
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
                    print("Validation loop: ")
                    pbar = tqdm(total=val_len)
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        dim_0 = x.shape[0]
                        dim_1 = self.model_list[0].layer_list[-1][0].output_shape[0]
                        h = torch.zeros(dim_0, dim_1).to(self.device)
                        # ** NOTE - This can be done in parallel !
                        for model_idx, model in enumerate(self.model_list):
                            model.to(self.device)
                            if model_idx == self.num_models:
                                y_pred = model(h)
                            else:
                                h += model.sequential_forward(
                                    x
                                )  # aggregating all the representations from HSIC segments
                        val_loss += self.output_criterion(y_pred, y).item()
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

        # plot the loss vs epoch graphs
        if show_plot:
            self.model_list[0].plot_loss(epochs, train_losses, val_losses)

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        pre_num_epochs,
        post_num_epochs,
        validation_split=0.2,
        show_plot=True,
    ):
        data_obj = DataGenerator()
        train_loader, val_loader = data_obj.prepare_numpy_data(
            x_train, y_train, batch_size, validation_split
        )
        self.training_loop(
            train_loader,
            val_loader,
            pre_num_epochs,
            post_num_epochs,
            show_plot=show_plot,
        )

    def fit_generator(
        self, train_loader, val_loader, pre_num_epochs, post_num_epochs, show_plot=True
    ):
        self.training_loop(
            train_loader, val_loader, pre_num_epochs, post_num_epochs, show_plot
        )
"""
