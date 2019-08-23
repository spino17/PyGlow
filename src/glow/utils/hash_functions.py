import torch


class Hash:
    def hash_function(name, epsilon, b):
        if name == "floor_hash":
            # floor hash function
            def floor_hash(x):
                return torch.floor((x + b) / epsilon)

            return floor_hash
