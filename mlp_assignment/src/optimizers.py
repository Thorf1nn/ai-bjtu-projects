import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    def update(self, layer):
        layer.weights -= self.learning_rate * layer.grad_weights
        layer.biases -= self.learning_rate * layer.grad_biases

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_w = {}
        self.v_b = {}

    def update(self, layer):
        if id(layer) not in self.v_w:
            self.v_w[id(layer)] = np.zeros_like(layer.weights)
            self.v_b[id(layer)] = np.zeros_like(layer.biases)

        # Update momentum
        self.v_w[id(layer)] = self.beta * self.v_w[id(layer)] + (1 - self.beta) * layer.grad_weights
        self.v_b[id(layer)] = self.beta * self.v_b[id(layer)] + (1 - self.beta) * layer.grad_biases

        # Update parameters
        layer.weights -= self.learning_rate * self.v_w[id(layer)]
        layer.biases -= self.learning_rate * self.v_b[id(layer)]

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.s_w = {}
        self.s_b = {}

    def update(self, layer):
        if id(layer) not in self.s_w:
            self.s_w[id(layer)] = np.zeros_like(layer.weights)
            self.s_b[id(layer)] = np.zeros_like(layer.biases)

        # Update squared gradients
        self.s_w[id(layer)] = self.beta * self.s_w[id(layer)] + (1 - self.beta) * np.square(layer.grad_weights)
        self.s_b[id(layer)] = self.beta * self.s_b[id(layer)] + (1 - self.beta) * np.square(layer.grad_biases)

        # Update parameters
        layer.weights -= self.learning_rate * layer.grad_weights / (np.sqrt(self.s_w[id(layer)]) + self.epsilon)
        layer.biases -= self.learning_rate * layer.grad_biases / (np.sqrt(self.s_b[id(layer)]) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.t = 0

    def update(self, layer):
        self.t += 1
        if id(layer) not in self.m_w:
            self.m_w[id(layer)] = np.zeros_like(layer.weights)
            self.v_w[id(layer)] = np.zeros_like(layer.weights)
            self.m_b[id(layer)] = np.zeros_like(layer.biases)
            self.v_b[id(layer)] = np.zeros_like(layer.biases)

        # Update momentum
        self.m_w[id(layer)] = self.beta1 * self.m_w[id(layer)] + (1 - self.beta1) * layer.grad_weights
        self.m_b[id(layer)] = self.beta1 * self.m_b[id(layer)] + (1 - self.beta1) * layer.grad_biases

        # Bias correction for momentum
        m_w_corr = self.m_w[id(layer)] / (1 - self.beta1 ** self.t)
        m_b_corr = self.m_b[id(layer)] / (1 - self.beta1 ** self.t)

        # Update squared gradients
        self.v_w[id(layer)] = self.beta2 * self.v_w[id(layer)] + (1 - self.beta2) * np.square(layer.grad_weights)
        self.v_b[id(layer)] = self.beta2 * self.v_b[id(layer)] + (1 - self.beta2) * np.square(layer.grad_biases)

        # Bias correction for squared gradients
        v_w_corr = self.v_w[id(layer)] / (1 - self.beta2 ** self.t)
        v_b_corr = self.v_b[id(layer)] / (1 - self.beta2 ** self.t)

        # Update parameters
        layer.weights -= self.learning_rate * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
        layer.biases -= self.learning_rate * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

def get_optimizer(name, learning_rate=0.01):
    optimizers = {
        'sgd': SGD,
        'momentum': Momentum,
        'rmsprop': RMSProp,
        'adam': Adam,
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    return optimizers[name](learning_rate=learning_rate)
