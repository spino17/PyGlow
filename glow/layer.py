from torch import nn


class Layer(nn.Module):
    """
    Base class for all layer modules.

    Your layer should also subclass this class.


    """
    def __init__(self, *args):
        super().__init__()
        # self.args = args

    def set_input(self, input_shape):
        """
        Takes input_shape and demands user to define a variable self.output_shape
        which stores the output shape of the custom layer.


        Arguments:
            input_shape (tuple): input shape of the tensor which the layer expects to receive

        """
        pass

    def forward(self, x):
        """
        Forward method overrides PyTorch forward method and contains the logic
        for the forward pass through the custom layer defined.


        Arguments:
            x (torch.Tensor): input tensor to the layer

        """
        pass