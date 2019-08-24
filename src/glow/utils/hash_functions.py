import torch


class Hash:
    def hash_function(name, epsilon, b):
        if name == "floor_hash":
            # floor hash function
            def floor_hash(x):
                return torch.floor((1 / epsilon) * (x + b))

            return floor_hash
