import numpy as np
import torch
from torch import nn
from glow.utils import estimators


class IP_Coordinates:
    """
    Class for calculating information plane coordinates

    """

    def __init__(self, data_values, estimator, params, num_layers):
        self.data_values = data_values
        self.estimator = estimator
        self.params = params
        for epoch_output in data_values:
            for batch_output in epoch_output:
                # TODO
                x = batch_output[0]
                y = batch_output[num_layers]
                for i in range(1, num_layers):
                    t = batch_output[i]
                    # TODO

        self.coordinates = # TODO


    def unpack(self):
        return I_x, I_y