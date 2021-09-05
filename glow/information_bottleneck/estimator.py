import random
import glow.hash_functions as hash_module
import math
import torch
import glow.utils.hsic_utils as kernel_module
import numpy as np


class Estimator:
    """
    Base class for all the estimator modules.

    Your estimator should also subclass this class.

    This Class is for implementing functionalities to estimate different dependence
    criterion in information theory like mutual information etc. These methods
    are further used in analysing training dyanmics of different architechures.


    Arguments:
        gpu (bool): if true then all the computation is carried on `GPU` else on `CPU`
        **kwargs: the keyword that stores parameters for the estimators

    """

    def __init__(self, gpu, **kwargs):
        self.params_dict = kwargs  # input parameters for the estimator
        if gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise Exception("No CUDA enabled GPU device found")
        else:
            self.device = torch.device("cpu")

    def criterion(self, x, y):
        """
        Defines the criterion of the estimator for example EDGE algorithm have
        mutual information as its criterion. Generally criterion is some kind
        of dependence or independence measure between `x` and `y`. In the context
        of information theory most widely used criterion is mutual information
        between the two arguments.


        Arguments:
            x (torch.Tensor): first random variable
            y (torch.Tensor): second random variable

        Returns:
            (torch.Tensor): calculated criterion of the two random variables 'x' and 'y'

        """
        pass

    def eval_dynamics_segment(self, dynamics_segment):
        """
        Process smallest segment of dynamics and calculate coordinates using the
        defined criterion.


        Arguments:
            dynamics_segment (iterable): smallest segment of the dynamics of a batch containing input, hidden layer output and label in form of :class:`torch.Tensor` objects

        Returns:
            (iterable): list of calculated coordinates according to the criterion with length equal to 'len(dynamics_segment)-2'

        """
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


class EDGE(Estimator):
    """
    Mutual information technique propsed in the paper
    'SCALABLE MUTUAL INFORMATION ESTIMATION USING DEPENDENCE GRAPHS'


    Arguments:
        hash_function (callable or str): hash function which is used to obtain mapping from data to dependency graph nodes as described in EDGE algorithm
        gpu (bool, optional): if true then all the computation is carried on `GPU` else on `CPU`
        **kwargs: the keyword that stores parameters for EDGE algorithm mutual information criterion

    """

    def __init__(self, hash_function, U=10, gpu=True, **kwargs):
        super().__init__(gpu, **kwargs)
        self.hash_function = hash_function
        self.U = U

    def g(self, x):
        return x * math.log(x) * (1 / math.log(2))

    def criterion(self, x, y):
        """
        Defines the criterion of the EDGE estimator algorithm which have
        mutual information as its criterion.

        """
        h = hash_module.get(self.hash_function, self.params_dict)
        num_samples = x.shape[0]

        edge_list = []
        N = {}
        M = {}
        L = {}

        for k, x_k in enumerate(x):
            y_k = y[k]
            i = h(x_k)
            j = h(y_k)
            if list(x_k.size()) == []:
                i = i.item()
                j = j.item()
            else:
                i = tuple(i.tolist())
                j = tuple(j.tolist())
            
            N[i] = (N[i] + 1.0) if i in N else 1.0
            M[j] = (M[j] + 1.0) if j in M else 1.0
            L[i,j] = (L[i,j] + 1.0) if (i,j) in L else 1.0
            edge_list.append((i, j))

        mut_info = 0.0

        for i, j in edge_list:
            wi = 1.0 * N[i] / num_samples
            wj = 1.0 * M[j] / num_samples
            wij = min(self.U, 1.0 * L[i,j] * num_samples / (N[i]*M[j]))
            mut_info += wi * wj * self.g(wij)
        
        return mut_info


class HSIC(Estimator):
    """
    Class for estimating Hilbert-Schmidt Independence Criterion as done in
    paper "The HSIC Bottleneck: Deep Learning without Back-Propagation".


    Arguments:
        kernel (str): kernel which is used for calculating K matrix in HSIC criterion
        gpu (bool): if true then all the computation is carried on `GPU` else on `CPU`
        **kwargs: the keyword that stores parameters for HSIC criterion

    """

    def __init__(self, kernel, gpu=True, **kwargs):
        super().__init__(gpu, **kwargs)
        self.kernel = kernel

    # Hilbert-Schmid Independence Criterion
    def criterion(self, x, y):
        """
        Defines the HSIC criterion.

        """
        x, y = x.to(self.device), y.to(self.device)
        m = x.shape[0]
        K_x = kernel_module.get(self.kernel)(x, x, self.params_dict)
        K_y = kernel_module.get(self.kernel)(y, y, self.params_dict)
        H = torch.eye(m, m) - (1 / m) * torch.ones(m, m)
        H = H.to(self.device)
        matrix_x = torch.mm(K_x, H)
        matrix_y = torch.mm(K_y, H)
        return (1 / (m - 1)) * torch.trace(torch.mm(matrix_x, matrix_y))
