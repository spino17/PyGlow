import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class Dynamics:
    """
    Class for evaluating all the dynamics related criterion coordinates.

    Arguments:
        dynamics_segment (iterable): smallest segment of the dynamics of a batch
        containing input, hidden layer output and label in form of
        :class:`torch.Tensor` objects

    """

    def __init__(self, dynamics_segment):
        self.dynamics_segment = dynamics_segment

    def evaluate(self, evaluator_obj):
        """
        Evaluate dynamics based on `criterion` method in `evaluator_obj`

        Arguments:
            evaluator_obj (glow.information_bottleneck.Estimator): object that
            defines `criterion` method using which we obtain coordinates

        """
        evaluated_segment = evaluator_obj.eval_dynamics_segment(self.dynamics_segment)
        return evaluated_segment


def get(identifier):
    return Dynamics(identifier)
