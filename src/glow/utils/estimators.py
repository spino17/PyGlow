from glow.Information_bottelneck.estimator import Binned, EDGE, HSIC


class Estimator:
    def criterion(name, params):
        if name == "binned":
            return Binned(*params).mutual_information
        elif name == "EDGE":
            return EDGE(*params).mutual_information
        elif name == "HSIC":
            return HSIC(*params).HS_Criterion
