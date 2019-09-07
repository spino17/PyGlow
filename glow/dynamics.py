import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class Dynamics:
    """
    Class for evaluating all the dynamics related coordinates

    """

    def __init__(self, dynamics_segment):
        self.dynamics_segment = dynamics_segment

    def evaluate(self, evaluator_obj):
        evaluated_segment = evaluator_obj.eval_dynamics_segment(self.dynamics_segment)
        return evaluated_segment


def get(identifier):
    return Dynamics(identifier)
