import numpy as np

class Loss:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError

class MeanSquaredError(Loss):
    def __call__(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class CategoricalCrossentropy(Loss):
    def __call__(self, y_true, y_pred):
        # Add a small epsilon to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def derivative(self, y_true, y_pred):
        # This derivative is calculated assuming the output layer is softmax
        return y_pred - y_true

def get_loss(name):
    losses = {
        'mse': MeanSquaredError,
        'categorical_crossentropy': CategoricalCrossentropy,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss function: {name}")
    return losses[name]()

class Regularizer:
    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, weights):
        loss = 0
        if self.l1 > 0:
            loss += self.l1 * np.sum(np.abs(weights))
        if self.l2 > 0:
            loss += self.l2 * 0.5 * np.sum(np.square(weights))
        return loss

    def derivative(self, weights):
        grad = np.zeros_like(weights)
        if self.l1 > 0:
            grad += self.l1 * np.sign(weights)
        if self.l2 > 0:
            grad += self.l2 * weights
        return grad

def get_regularizer(l1=0, l2=0):
    if l1 > 0 and l2 > 0:
        return Regularizer(l1, l2) # Elastic Net
    elif l1 > 0:
        return Regularizer(l1, 0) # L1
    elif l2 > 0:
        return Regularizer(0, l2) # L2
    return None
