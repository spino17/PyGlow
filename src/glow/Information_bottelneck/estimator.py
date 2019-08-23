import dit
import random
from glow.utils import Hash as H


class _Estimator:
    """
    Class for implementing functionalities to estimate different dependence
    criterion in information theory like mutual information etc. These methods
    are further used in analysing training dyanmics of different architechures.

    """

    def __init__(self, params):
        self.params = params  # input parameters for the estimator

    # returns mutual information between x and y random variable
    def mutual_information(self, x, y):
        pass


class Binned(_Estimator):
    """
    Basic technique for estimating mutual information

    """

    def __init__(self, num_bins):
        super().__init__([num_bins])

    def mutual_information(self, x, y):
        # TODO

"""
class KDE(_Estimator):
    # TODO


class KSG(_Estimator):
    # TODO


class KNN(_Estimator):
    # TODO


class MINE(_Estimator):
    # TODO

"""


class EDGE(_Estimator):
    """
    Mutual information technique propsed in the paper
    'SCALABLE MUTUAL INFORMATION ESTIMATION USING DEPENDENCE GRAPHS'

    """
    def __init__(self, hash_function='floor_hash', epsilon, alpha=1):
        b = random.uniform(0, epsilon)
        super().__init__([hash_function, epsilon, b, alpha])

    def mutual_information(self, x, y):
        h = H.hash_function(self.params[0], self.params[1], self.parmas[2]) # hash function
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


class HSIC(_Estimator):
    """
    Class for estimating Hilbert-Schmidt Independence Criterion as done in
    paper "The HSIC Bottleneck: Deep Learning without Back-Propotion"

    """

    def __init__(self, ):
        super().__init__([])

    def HS_Criterion(self, x, y):
        m = x.shape[0]
        d_x = x.shape[1]
        d_y = y.shape[1]
        K_x =
        K_y =
        H = torch.eye(m, m) - (1 / m) * torch.ones(m, m)
        matrix_x = torch.mm(K_x, H)
        matrix_y = torch.mm(K_y, H)
        return (1 / (m - 1)) * torch.trace(torch.mm(matrix_x, matrix_y))

