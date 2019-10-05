from torch import nn
import torch
from tqdm import tqdm
from glow.models import Network
import random


class VIB(nn.Module):
    """
    Class for Deep Variational Information Bottleneck encoders.


    Arguments:
        hidden_dim (int): dimension of the hidden layer on which decoder operates
        encoder (glow.models.Network): encoder model of :class:`glow.models.Network`
        decoder (glow.models.Network): decoder model of :class:`glow.models.Network`

    Attributes:
        is_gpu (bool): true if `GPU` is enabled on the system, false otherwise
        device (torch.device or int): `GPU` or `CPU` for training purposes

    """

    def __init__(self, hidden_dim, encoder, decoder):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.decoder = decoder
        if self.encoder.is_gpu and self.decoder.is_gpu:
            self.is_gpu = True
            self.device = torch.device("cuda")
            print("Running on CUDA enabled device !")
        else:
            self.is_gpu = False
            self.device = torch.device("cpu")
            print("Running on CPU device !")

    def training_loop(self, num_epochs, data_loader, trade_off, num_samples, show_plot):
        self.to(self.device)
        train_losses, epochs = [], [], []
        data_len = len(data_loader)
        for epoch in range(num_epochs):
            # training loop
            print("\n")
            print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            train_loss = 0
            print("Training loop: ")
            pbar = tqdm(total=data_len)
            for x, y in data_loader:
                x = x.to(self.device)
                self.optimizer.zero_grad()
                loss = 0
                # for each vector in batch
                for input_x in x:
                    input_x = input_x.view(1, -1)
                    encoder_output = self.encoder(input_x)
                    mean = encoder_output[0][: self.hidden_dim].view(1, -1)
                    std = encoder_output[0][self.hidden_dim :].view(1, -1)
                    for sample_idx in range(num_samples):
                        norm = (
                            torch.distributions.normal.Normal(
                                torch.tensor([0.0]), torch.tensor([1.0])
                            )
                            .sample(sample_shape=(self.hidden_dim, 1))
                            .view(1, -1)
                        )
                        # reparameterization trick
                        decoder_input = mean + std * norm
                        decoder_output = self.decoder(decoder_input)
                        class_prob = decoder_output[y-1]
                        torch.log(class_prob)

                loss.backward(retain_graph=True)
                self.optimizer.step()
                train_loss += loss.item()
                pbar.update(1)

        # plot the loss vs epoch graphs
        if show_plot:
            self.plot_loss(epochs, train_losses)

    def generate(self, num_samples):
        """
        Samples from the distribution and decodes the hidden representations to
        generator input data like samples.


        Arguments:
            num_samples (int): number of samples to be generated

        """

        # TODO
