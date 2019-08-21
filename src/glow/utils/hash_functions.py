import math


class Hash:
    def hash_function(name, epsilon, b):
        if name == 'floor_hash':
            # floor hash function
            def floor_hash(x):
                return math.floor((x + b) / epsilon)

            return floor_hash
