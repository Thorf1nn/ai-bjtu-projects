import numpy as np
from activations import get_activation, sigmoid

class GRUCell:
    def __init__(self, input_dim, hidden_dim, activation="tanh"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = get_activation(activation)
        self.W_z = np.random.randn(hidden_dim, input_dim + hidden_dim) / np.sqrt(input_dim + hidden_dim)
        self.b_z = np.zeros((hidden_dim, 1))
        self.W_r = np.random.randn(hidden_dim, input_dim + hidden_dim) / np.sqrt(input_dim + hidden_dim)
        self.b_r = np.zeros((hidden_dim, 1))
        self.W_h = np.random.randn(hidden_dim, input_dim + hidden_dim) / np.sqrt(input_dim + hidden_dim)
        self.b_h = np.zeros((hidden_dim, 1))

    def forward(self, x, h_prev):
        concat = np.vstack((h_prev, x))
        z = sigmoid(np.dot(self.W_z, concat) + self.b_z)
        r = sigmoid(np.dot(self.W_r, concat) + self.b_r)
        concat_reset = np.vstack((r * h_prev, x))
        h_hat = self.activation(np.dot(self.W_h, concat_reset) + self.b_h)
        h = (1 - z) * h_prev + z * h_hat
        return h 