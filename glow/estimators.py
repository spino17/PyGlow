from glow.Information_bottelneck.estimator import EDGE, HSIC


def get(estimator, params):
    if estimator == "binned":
        return 0
    elif estimator == "EDGE":
        return EDGE(*params).mutual_information
    elif estimator == "HSIC":
        return HSIC(*params).HS_Criterion
