import dit
import random
from glow.utils import Hash as H


class _Estimators:
    """
    Class for implementing functionalities to estimate distributions P(X, Y),
    P(T|X), P(Y|T) and correspondingly Information plane coordinate (I(X;T), I(T;Y))
    of all the layers at all the epochs

    """
    def __init__(self):
        # TODO

    # returns mutual information between x and y random variable
    def mutual_information(self, x, y):
        pass

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


