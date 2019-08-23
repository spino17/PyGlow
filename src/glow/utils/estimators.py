from glow.Information_bottelneck.estimator import Binned, EDGE, HSIC


class Estimator:
    def estimation_obj(name, params):
        if name == 'binned':
            return Binned(params)