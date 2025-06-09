import numpy as np

class Activation:
    def __call__(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self(x)
        return s * (1 - s)

class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2

class ReLU(Activation):
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

class Softmax(Activation):
    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def derivative(self, x):
        # The derivative of softmax is typically combined with the cross-entropy loss.
        # For a standalone derivative, we would compute the Jacobian matrix,
        # which can be computationally expensive.
        # Here, we'll assume the derivative is handled in the loss function's gradient calculation
        # for simplicity and stability, which is common practice.
        # This method will return 1 as a placeholder, as it will be multiplied by the gradient of the loss
        # w.r.t the output of softmax, which is (y_pred - y_true)
        return 1

class Linear(Activation):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)

def get_activation(name):
    activations = {
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'softmax': Softmax,
        'linear': Linear,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations[name]()
