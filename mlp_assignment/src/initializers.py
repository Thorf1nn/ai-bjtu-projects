import numpy as np

class Initializer:
    def __call__(self, shape):
        raise NotImplementedError

class RandomNormal(Initializer):
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        return np.random.normal(self.mean, self.std, size=shape)

class RandomUniform(Initializer):
    def __init__(self, low=-0.05, high=0.05):
        self.low = low
        self.high = high

    def __call__(self, shape):
        return np.random.uniform(self.low, self.high, size=shape)

class XavierNormal(Initializer):
    def __call__(self, shape):
        fan_in, fan_out = shape
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, size=shape)

class XavierUniform(Initializer):
    def __call__(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)

class HeNormal(Initializer):
    def __call__(self, shape):
        fan_in, _ = shape
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, size=shape)

class HeUniform(Initializer):
    def __call__(self, shape):
        fan_in, _ = shape
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, size=shape)

def get_initializer(name):
    initializers = {
        'random_normal': RandomNormal,
        'random_uniform': RandomUniform,
        'xavier_normal': XavierNormal,
        'xavier_uniform': XavierUniform,
        'he_normal': HeNormal,
        'he_uniform': HeUniform,
    }
    if name not in initializers:
        raise ValueError(f"Unknown initializer: {name}")
    return initializers[name]()
