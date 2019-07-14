from torch import nn
import torch
from glow.utils import Losses as L
from glow.utils import Optimizers as O


class Network(nn.Module):
    """

    Sequential models implementation on
    PyTorch

    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim  # input dimensions
        self.layer_list = nn.ModuleList([])  # list of module type layers
        self.num_layers = 0  # number of layers in the architecture
        if(torch.cuda.is_available()):
            print("running on cuda enabled GPU")
        else:
            print("running on CPU environment")

    def add(self, layer_obj):
        if(self.num_layers == 0):
            prev_input_dim = self.input_dim
        else:
            prev_input_dim = self.layer_list[self.num_layers-1][-1].output_dim
        layer_obj.set_input(prev_input_dim)
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
        for layer in layers:
            h = layer(h)
            iter_num += 1
        return h

    def compile(self, optimizer_name='SGD', loss='cross_entropy', learning_rate=0.001, momentum=0.95):
        self.criterion = L.loss_function(loss)
        self.optimizer = O.optimizer(self.parameters(), learning_rate, momentum, optimizer_name)

    def fit(self, x_train, y_train, batch_size=1, num_epochs=50, num_classes=10):
        print("starting training process")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for index, x in enumerate(x_train):
                target = y_train[index]
                self.optimizer.zero_grad()
                output = self.forward(x)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if index % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, index + 1, running_loss / 2000))
                    running_loss = 0.0
        print("Training finished !")

    def predict(self, x):
        return self.forward(x)


