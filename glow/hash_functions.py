import torch


def floor_hash(x, epsilon, b):
    return torch.floor((1 / epsilon) * (x + b))


def get(identifier, params_dict):
    if identifier == "floor_hash":

        def curry_func(x):
            if "epsilon" in params_dict.keys():
                epsilon = params_dict["epsilon"]
            else:
                raise Exception(
                    "Cannot find argument epsilon for hash function floor_hash"
                )

            if "b" in params_dict.keys():
                b = params_dict["b"]
            else:
                raise Exception("Cannot find argument b for hash function floor_hash")
            return floor_hash(x, epsilon, b)

        return curry_func
    else:
        raise ValueError("Could not interpret " "hash function identifier:", identifier)
