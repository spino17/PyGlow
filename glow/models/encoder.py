from torch import nn
import torch
from tqdm import tqdm
from glow.models import Network
import random
from glow.preprocessing import DataGenerator
from glow.utils import Optimizers as O
from torch.nn.functional import one_hot


class VIB(Network):
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

    def compile(
        self, optimizer="SGD", metrics=[], learning_rate=0.001, momentum=0.95, **kwargs
    ):
        """
        Compile the model with attaching optimizer to the model.


        Arguments:
            optimizer (torch.optim.Optimizer): optimizer to be used during training process
            learning_rate (float, optional): learning rate for gradient descent step (default: 0.001)
            momentum (float, optional): momentum for different variants of optimizers (default: 0.95)

        """

        self.optimizer = O.optimizer(
            self.parameters(), learning_rate, momentum, optimizer
        )
        self.metrics = metrics

    def stochastic_prediction(y, batch_size, num_samples):
        # TODO
        return 0

    def training_loop(
        self,
        num_epochs,
        train_loader,
        val_loader,
        trade_off,
        num_samples,
        cov_type,
        show_plot,
    ):
        self.to(self.device)
        train_losses, val_losses, epochs = [], [], []
        train_len = len(train_loader)
        val_len = len(val_loader)
        metric_dict = self.handle_metrics(self.metrics)
        for epoch in range(num_epochs):
            # training loop
            print("\n")
            print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            train_loss = 0
            print("Training loop: ")
            pbar = tqdm(total=train_len)
            for x, y in train_loader:
                batch_size = x.shape[0]
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = 0
                encoder_output, _ = self.encoder(x)
                output_shape = encoder_output.shape[1]
                if output_shape < self.hidden_dim:
                    raise Exception(
                        "Encoder output shape not sufficient for forming hidden representation of shape "
                        + str(self.hidden_dim)
                    )
                else:
                    mean = encoder_output[:, : self.hidden_dim]
                    cov_shape = output_shape - self.hidden_dim

                if cov_type == "diag":
                    if cov_shape != self.hidden_dim:
                        raise ValueError(
                            "Encoder output shape not sufficient for forming diagonal covariance matrix of diagonal dimension "
                            + str(self.hidden_dim)
                        )
                    else:
                        cov_mat = torch.diag_embed(encoder_output[:, self.hidden_dim :])
                elif cov_type == "full":
                    num_elements = (1 / 2) * (self.hidden_dim) * (self.hidden_dim + 1)
                    if cov_shape != num_elements:
                        raise ValueError(
                            "Encoder output shape not sufficient for forming full covariance matrix under cholesky decomposition"
                        )
                    else:
                        upper_vec = encoder_output[:, self.hidden_dim : ]
                        upper_matrix = torch.zeros(
                            (batch_size, self.hidden_dim, self.hidden_dim)
                        ).to(self.device)
                        triu_indices = torch.triu_indices(
                            row=self.hidden_dim, col=self.hidden_dim, offset=0
                        ).to(self.device)
                        upper_matrix[:, triu_indices[0], triu_indices[1]] = upper_vec
                        cov_mat = torch.matmul(
                            torch.transpose(upper_matrix, 1, 2), upper_matrix
                        )

                p = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean, cov_mat
                )  # conditional probability distribution p(z|x) taken to be gaussian
                y_vec = y.repeat(num_samples, 1).view(-1).long()
                index_vec = torch.arange(0, num_samples * batch_size).long()
                # rsample internally uses reparameterize the random variable
                z_samples = p.rsample(
                    (num_samples,)
                )  # stochastic node in computational graph

                loss_term_1 = (
                    -1 * trade_off / (batch_size * num_samples)
                ) * p.log_prob(z_samples).sum()
                z_samples = z_samples.view(-1, self.hidden_dim)
                decoder_output, _ = self.decoder(z_samples)
                loss_term_2 = (1 / (batch_size * num_samples)) * torch.log(
                    decoder_output[index_vec, y_vec]
                ).sum()
                """
                y_pred = self.stochastic_prediction(
                    decoder_output, batch_size, num_samples
                )
                total = y_vec.size(0)
                y_pred = torch.argmax(y_pred, dim=1).long().view(-1)
                correct = (y_pred == y_vec).sum().item()
                acc = correct / total
                """
                r_mean = torch.zeros(self.hidden_dim).to(self.device)
                r_cov = torch.eye(self.hidden_dim).to(self.device)
                r = torch.distributions.multivariate_normal.MultivariateNormal(
                    r_mean, r_cov
                )  # total probability distribution r(z) taken to be spherical gaussian
                loss_term_3 = (1 * trade_off / (batch_size * num_samples)) * r.log_prob(
                    z_samples
                ).sum()

                loss = -1 * (loss_term_1 + loss_term_2 + loss_term_3)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                train_loss += loss.item()
                #print("loss: %.2f, acc: %.2" % ((train_loss / train_len), acc))
                pbar.update(1)
            pbar.close()
            """
            else:
                print("\n")
                print("loss: %.2f" % (train_loss / train_len))
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    # scope of no gradient calculations
                    print("Validation loop: ")
                    pbar = tqdm(total=val_len)
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_pred = []
                        for input_x in x:
                            input_x = input_x.view(1, -1)
                            encoder_output, _ = self.encoder.forward(input_x)
                            mean = encoder_output[0][: self.hidden_dim]
                            std = encoder_output[0][self.hidden_dim :]
                            norm = (
                                torch.distributions.normal.Normal(
                                    torch.tensor([0.0]), torch.tensor([1.0])
                                )
                                .sample(sample_shape=(self.hidden_dim, 1))
                                .view(1, -1)
                            ).to(self.device)
                            decoder_input = mean + std * norm
                            decoder_output, _ = self.decoder.forward(decoder_input)
                            y_pred.append(decoder_output.view(-1).tolist())
                        pbar.update(1)
                    pbar.close()
                    metric_values = []
                    for key in metric_dict:
                        y_pred = torch.tensor(y_pred)
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
            self.plot_loss(epochs, train_losses, val_losses)
        """

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        num_epochs,
        trade_off,
        num_samples,
        cov_type,
        show_plot=False,
    ):
        """
        Fits the dataset passed as numpy array (Keras like pipeline) in the arguments.


        Arguments:
            x_train (numpy.ndarray): training input dataset
            y_train (numpy.ndarray): training ground-truth labels
            batch_size (int): batch size of one batch
            num_epochs (int): number of epochs for training
            trade_off (float): trade-off parameter in information bottleneck theory (trade-off betwenen compression and generalization term)
            num_samples (int): number of samples to be used for stochastic averaging in the loss function
            cov_type (str): type of covariance matrix to be used for estimating the distribution (default: diag)
            validation_split (float, optional): proportion of the total dataset to be used for validation (default: 0.2)
            show_plot (bool, optional): if true plots the training loss (red), validation loss (blue) vs epochs (default: True)

        """

        data_obj = DataGenerator()
        train_loader, val_loader = data_obj.prepare_numpy_data(
            x_train, y_train, batch_size, validation_split=0
        )
        self.training_loop(
            num_epochs,
            train_loader,
            val_loader,
            trade_off,
            num_samples,
            cov_type,
            show_plot,
        )

    def fit_generator(
        self,
        train_loader,
        val_loader,
        num_epochs,
        trade_off,
        num_samples,
        cov_type,
        show_plot=False,
    ):
        """
        Fits the dataset by taking data-loader as argument.


        Arguments:
            train_loader (torch.utils.data.DataLoader): training dataset (with already processed batches)
            val_loader (torch.utils.data.DataLoader): validation dataset (with already processed batches)
            num_epochs (int): number of epochs for training
            trade_off (float): trade-off parameter in information bottleneck theory (trade-off betwenen compression and generalization term)
            num_samples (int): number of samples to be used for stochastic averaging in the loss function
            cov_type (str): type of covariance matrix to be used for estimating the distribution (default: diag)
            show_plot (bool, optional): if true plots the training loss (red), validation loss (blue) vs epochs (default: True)

        """

        self.training_loop(
            num_epochs,
            train_loader,
            val_loader,
            trade_off,
            num_samples,
            cov_type,
            show_plot,
        )

    def generate(self, num_samples):
        """
        Samples from the distribution and decodes the hidden representations to
        generator input data like samples.


        Arguments:
            num_samples (int): number of samples to be generated

        """
        # TODO
        return 0
