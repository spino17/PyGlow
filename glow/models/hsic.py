import torch
from torch import nn
from glow.models.network import _Network
import matplotlib.pyplot as plt
import glow.losses as losses_module
from glow.utils import Optimizers as O
from glow.layers.core import Flatten, Dropout
from tqdm import tqdm


class _HSIC(_Network):
    """
    The HSIC Bottelneck: Deep Learning without backpropagation

    """

    def __init__(self, input_shape, sigma, regularize_coeff, gpu=True):
        super().__init__(input_shape)
        self.sigma = sigma
        self.regularize_coeff = regularize_coeff
        self.is_gpu = gpu

    def forward(self, x):
        layers = self.layer_list
        t = x
        iter_num = 0
        hidden_outputs = []
        for layer_idx, layer in enumerate(layers):
            if layer_idx == self.num_layers - 1:
                break
            h = layer(t)
            hidden_outputs.append(h)
            t = h.detach()  # detached vector to cut of the previous gradients
            iter_num += 1
        return hidden_outputs

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
        optimizer="SGD",
        loss="HSIC_loss",
        metrics=[],
        learning_rate=0.001,
        momentum=0.95,
    ):
        # raise exception
        self.criterion = losses_module.get(loss)
        self.layer_optimizers = self.make_layer_optimizers(
            optimizer, learning_rate, momentum
        )
        self.metrics = metrics

    def training_loop(
        self, pre_num_epochs, train_loader, val_loader, show_plot=True
    ):
        post_num_epochs = pre_num_epochs
        self.to(self.device)
        train_losses, val_losses, epochs = [], [], []
        train_len = len(train_loader)
        val_len = len(val_loader)
        metric_dict = self.handle_metrics(self.metrics)
        print("pre-training phase starting ...")
        for epoch in range(pre_num_epochs):
            # training loop
            print("Pre-Train-Epoch " + str(epoch + 1) + "/" + str(pre_num_epochs))
            self.train()
            print("Training loop: ")
            pbar = tqdm(total=train_len)
            for x, y in train_loader:
                # contains the hidden representation from forward pass
                x, y = x.to(self.device), y.to(self.device)
                hidden_outputs = self.forward(x)
                # ** NOTE - This can be done in parallel !
                for idx, z in enumerate(hidden_outputs):
                    if self.layer_optimizers[idx] is not None:
                        self.layer_optimizers[idx].zero_grad()
                        loss = self.criterion(
                            z,
                            x.view(x.shape[0], -1),
                            y,
                            self.sigma,
                            self.regularize_coeff,
                            self.is_gpu,
                        )
                        loss.backward()
                        self.layer_optimizers[idx].step()
                pbar.update(1)
            """
            else:
                # validation loop
                pbar.close()
                val_loss = 0
                print("\n")
                with torch.no_grad():
                    # scope of no gradient calculations
                    print("Validation loop: ")
                    pbar = tqdm(total=val_len)
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        hidden_outputs = self.forward(x)
                        # ** NOTE - This can be done in parallel !
                        for idx, z in enumerate(hidden_outputs):
                            val_loss += self.criterion(
                                z,
                                x.view(x.shape[0], -1),
                                y,
                                self.sigma,
                                self.regularize_coeff,
                                self.is_gpu,
                            ).item()
                        pbar.update(1)
                    pbar.close()
                    print("\n")
                train_losses.append(train_loss / train_len)
                val_losses.append(val_loss / val_len)
                epochs.append(epoch + 1)
            """

        print("post-training phase starting ...")
        criterion = losses_module.get(self.layer_list[-1][0].loss)
        # freeze the pre-trained grads
        for layer_idx, layer in enumerate(self.layer_list):
            if layer_idx == self.num_layers - 1:
                break
            for params in layer.parameters():
                params.require_grad = False

        for epoch in range(post_num_epochs):
            train_loss = 0
            # training loop
            print("Post-Train-Epoch " + str(epoch + 1) + "/" + str(post_num_epochs))
            train_loss = 0
            print("Training loop: ")
            pbar = tqdm(total=train_len)
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.layer_optimizers[-1].zero_grad()
                h = x
                for layer in self.layer_list:
                    h = layer(h)
                y_pred = h
                loss = criterion(y_pred, y)
                loss.backward()
                self.layer_optimizers[-1].step()
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
                        h = x
                        for layer in self.layer_list:
                            h = layer(h)
                        y_pred = h
                        val_loss += criterion(y_pred, y).item()
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
            plt.title("Epoch vs Loss")
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.grid(True)
            plt.plot(epochs, train_losses, color="red", label="training loss")
            plt.plot(epochs, val_losses, color="blue", label="validation loss")
            plt.show()


class HSICSequential(_HSIC):
    def __init__(self, input_shape, sigma, regularize_coeff, gpu):
        super().__init__(input_shape, sigma, regularize_coeff, gpu)


"""
class HSICSigma(_HSIC):
    def __init__(self, input_shape, regularize_coeff):
        # TODO
        return 0
"""
