import dit
import random
from glow.utils import Hash as H


class _Estimators(Network):
    """
    Class for implementing functionalities to estimate distributions P(X, Y),
    P(T|X), P(Y|T) and correspondingly Information plane coordinate (I(X;T), I(T;Y))
    of all the layers at all the epochs

    """
    def __init__(self, model):
        super().__init__(model.input_shape)  # initialize a Network object
        self.layer_list = model.layer_list
        self.num_layers = model.num_layers
        self.adapter_obj = TensorNumpyAdapter()
        self.criterion = model.criterion
        self.optimizer = model.optimizer

    # returns mutual information between x and y random variable
    def mutual_information(self, x, y):
        pass

    # forward pass with appropiate estimation model
    def forward(self, x):
        layers = self.layer_list
        I_XT = []
        I_YT = []
        t = x
        iter_num = 0
        for layer in layers:
            t = layer(t)
            with torch.no_grad():
                I_XT.append(mutual_information(x, t))
                I_YT.append()
            iter_num += 1
        return t

"""
class KDE(_Estimators):
    # TODO


class KSG(_Estimators):
    # TODO


class KNN(_Estimators):
    # TODO

"""


class EDGE(_Estimators):
    def __init__(self, model, hash_function='floor_hash', epsilon):
        super().__init__(model)
        self.hash_function = hash_function
        self.epsilon
        self.b = random.uniform(0, epsilon) # fixed random number in [0, epsilon]

    def mutual_information(self, x, y):
        h = H.hash_function(self.hash_function, self.epsilon, self.b) # hash function
        N = torch.zeros(F, 1)
        M = torch.zeros(F, 1)
        L = torch.zeros(F, F)
        for k, x_k in enumerate(x):
            y_k = y[k]
            i = h(x_k)
            j = h(y_k)
            N[i] = N[i] + 1
            M[j] = M[j] + 1
            L[i][j] = L[i][j] + 1

        n = (1 / k) * N
        m = (1 / k) * M
        l = # TODO


