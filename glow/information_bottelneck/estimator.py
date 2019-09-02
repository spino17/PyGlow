import random
import glow.hash_functions as hash_module
import math
import torch
import glow.utils.hsic_utils as kernel_module


class _Estimator:
    """
    Class for implementing functionalities to estimate different dependence
    criterion in information theory like mutual information etc. These methods
    are further used in analysing training dyanmics of different architechures.

    """

    def __init__(self, params, gpu):
        self.params = params  # input parameters for the estimator
        if gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise Exception("No CUDA enabled GPU device found")
        else:
            self.device = torch.device("cpu")

    # returns mutual information between x and y random variable
    def criterion(self, x, y):
        pass

    # logic to process the smallest segment of the dynamics
    def eval_dynamics_segment(self, dynamics_segment):
        pass


"""
class Binned(_Estimator):
    # TODO


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

    def __init__(self, hash_function, epsilon, alpha, gpu):
        b = random.uniform(0, epsilon)
        super().__init__([epsilon, b, alpha])
        self.hash_function = hash_function

    def g(self, x):
        return x * torch.log(x) * (1 / math.log(10))

    # mutual information
    def criterion(self, x, y):
        h = hash_module.get(self.hash_function)
        num_sample = x.shape[0]
        F = self.params[3] * num_sample
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

        n = (1 / num_sample) * N
        m = (1 / num_sample) * M
        temp_matrix = torch.mm(N, torch.transpose(M, 0, 1))
        zero_matrix = torch.zeros(F, F)
        w = torch.addcdiv(zero_matrix, N, L, temp_matrix)
        temp_matrix = torch.mm(n, torch.transpose(m, 0, 1))
        mut_info = torch.sum(temp_matrix * g(w))

        return mut_info


class HSIC(_Estimator):
    """
    Class for estimating Hilbert-Schmidt Independence Criterion as done in
    paper "The HSIC Bottleneck: Deep Learning without Back-Propotion"

    """

    def __init__(self, sigma, gpu=True):
        super().__init__([sigma], gpu)

    def criterion(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        sigma = self.params[0]
        m = x.shape[0]
        K_x = kernel_module.get("gaussian")(x, x, sigma)
        K_y = kernel_module.get("gaussian")(y, y, sigma)
        H = torch.eye(m, m) - (1 / m) * torch.ones(m, m)
        H = H.to(self.device)
        matrix_x = torch.mm(K_x, H)
        matrix_y = torch.mm(K_y, H)
        return (1 / (m - 1)) * torch.trace(torch.mm(matrix_x, matrix_y))

    def eval_dynamics_segment(self, dynamics_segment):
        segment_size = len(dynamics_segment)
        x = dynamics_segment[0]
        m = x.shape[0]
        x = x.view(m, -1)
        y = dynamics_segment[segment_size - 1].view(m, -1)
        output_segment = []
        for idx in range(1, segment_size - 1):
            h = dynamics_segment[idx].view(m, -1)
            output_segment.append([self.criterion(h, x), self.criterion(h, y)])
        return output_segment
