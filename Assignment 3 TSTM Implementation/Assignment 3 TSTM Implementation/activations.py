import numpy as np

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_activation(name):
    if name == "tanh":
        return tanh
    elif name == "leaky_relu":
        return leaky_relu
    elif name == "elu":
        return elu
    elif name == "sigmoid":
        return sigmoid
    else:
        raise ValueError(f"Unknown activation: {name}") 