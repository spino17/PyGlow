import torch
from torch import nn
from glow.models.network import _Network
import matplotlib.pyplot as plt
import glow.losses as losses_module
from glow.utils import Optimizers as O


class _HSIC(_Network):
    """
    The HSIC Bottelneck: Deep Learning without backpropagation

    """

    def __init__(self, input_shape, sigma, regularize_coeff):
        super().__init__(input_shape)
        self.sigma = sigma
        self.regularize_coeff = regularize_coeff

    def forward(self, x):
        layers = self.layer_list
        t = x
        iter_num = 0
        hidden_outputs = []
        for layer in layers:
            h = layer(t)
            hidden_outputs.append(h)
            t = h.detach()  # detached vector to cut of the previous gradients
            iter_num += 1
        return hidden_outputs

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
        self.optimizer = O.optimizer(
            self.parameters(), learning_rate, momentum, optimizer
        )
        self.metrics = metrics

    def training_loop(self, num_epochs, train_loader, val_loader, show_plot=True):
        train_losses, val_losses, epochs = [], [], []
        train_len = len(train_loader)
        val_len = len(val_loader)
        for epoch in range(num_epochs):
            # training loop
            print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            train_loss = 0
            self.train()
            print("Training loop: ")
            for x, y in train_loader:
                # contains the hidden representation from forward pass
                hidden_outputs = self.forward(x)
                # ** NOTE - This can be done in parallel !
                for z in hidden_outputs:
                    loss = self.criterion(z, x, y, self.sigma, self.regularize_coeff)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                else:
                    val_loss = 0
                    with torch.no_grad():
                        for x, y in val_loader:
                            hidden_outputs = self.forward(x)
                            for z in hidden_outputs:
                                val_loss += self.criterion(
                                    z, x, y, self.sigma, self.regularize_coeff
                                ).item()
                    train_losses.append(train_loss / train_len)
                    val_losses.append(val_loss / val_len)
                    epochs.append(epoch + 1)

        # plot the loss vs epoch graphs
        if show_plot:
            plt.title("Epoch vs Loss")
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.grid(True)
            plt.plot(epochs, train_losses, color="red", label="training loss")
            plt.plot(epochs, val_losses, color="blue", label="validation loss")
            plt.show()


class SequentialHSIC(_HSIC):
    def __init__(self, input_shape, sigma, regularize_coeff):
        super().__init__(input_shape, sigma, regularize_coeff)

class SigmaNetwork(_HSIC):
    def __init__(self, input_shape, regularize_coeff):
        # TODO
        return 0
