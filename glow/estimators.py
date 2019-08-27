from glow.Information_bottelneck.estimator import Binned, EDGE, HSIC

"""
class Estimator:
    def criterion(name, params):
        if name == "binned":
            return Binned(*params).mutual_information
        elif name == "EDGE":
            return EDGE(*params).mutual_information
        elif name == "HSIC":
            return HSIC(*params).HS_Criterion
"""


def get(estimator, params):
    if estimator == "binned":
        return Binned(*params).mutual_information
    elif estimator == "EDGE":
        return EDGE(*params).mutual_information
    elif estimator == "HSIC":
        return HSIC(*params).HS_Criterion
