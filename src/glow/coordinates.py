import numpy as np
import torch
from torch import nn
# from glow.utils import estimators as E


class IP_Coordinates:
    """
    Class for calculating information plane coordinates

    """

    def __init__(self, data_values, estimator, params, num_layers):
        self.data_values = data_values
        self.estimator = estimator
        self.params = params
        epoch_coord = []
        for epoch_output in data_values:
            batch_coord = []
            for batch_output in epoch_output:
                x = batch_output[0]
                y = batch_output[num_layers]
                layer_coord = []
                for i in range(1, num_layers):
                    t = batch_output[i]
                    # I_x = E.criterion(estimator, params)(t, x)
                    # I_y = E.criterion(estimator, params)(t, y)
                    I_x = 0
                    I_y = 0
                    layer_coord.append([I_x, I_y])
                batch_coord.append(layer_coord)
        epoch_coord.append(batch_coord)
        self.coordinates = epoch_coord


    def unpack(self):
        I_x = []
        I_y = []
        for epoch in self.coordinates:
            for batch in epoch:
                for layer in batch:
                    I_x.append(layer[0])
                    I_y.append(layer[1])
        return I_x, I_y