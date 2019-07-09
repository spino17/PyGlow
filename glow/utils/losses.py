from torch import nn


class Losses:
    def loss_function(name):
        if(name == 'cross_entropy'):
            return nn.CrossEntropyLoss()
        elif(name == 'NLLLoss'):
            return nn.NLLLoss()
