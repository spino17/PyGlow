import dit


class _Estimators(Network):
    """
    Class for implementing functionalities to estimate distributions P(X, Y),
    P(T|X), P(Y|T) and correspondingly Information plane coordinate (I(X;T), I(T;Y))
    of all the layers at all the epochs

    """
    def __init__(self, model):
        super().__init__(model.input_shape)

    def estimation_function(self, x, t):
        pass

    def forward(self, x):
        # TODO

"""
class KDE(_Estimators):
    # TODO


class KSG(_Estimators):
    # TODO


class KNN(_Estimators):
    # TODO

"""

class EDGE(_Estimators):
    # TODO
    def __init__(self):
        super().__init__()

    def estimation_function(self, x, t):
        # TODO