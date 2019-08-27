from glow.Information_bottelneck.estimator import Binned, EDGE, HSIC


def get(estimator, params):
    if estimator == "binned":
        return Binned(*params).mutual_information
    elif estimator == "EDGE":
        return EDGE(*params).mutual_information
    elif estimator == "HSIC":
        return HSIC(*params).HS_Criterion
