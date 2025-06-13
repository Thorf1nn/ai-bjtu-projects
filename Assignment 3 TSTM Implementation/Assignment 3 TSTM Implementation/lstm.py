import numpy as np
from activations import get_activation, sigmoid

class LSTMCell:
    def __init__(self, input_dim, hidden_dim, activation="tanh"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = get_activation(activation)
        # Xavier initialization
        self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim) / np.sqrt(input_dim + hidden_dim)
        self.b_f = np.zeros((hidden_dim, 1))
        self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim) / np.sqrt(input_dim + hidden_dim)
        self.b_i = np.zeros((hidden_dim, 1))
        self.W_c = np.random.randn(hidden_dim, input_dim + hidden_dim) / np.sqrt(input_dim + hidden_dim)
        self.b_c = np.zeros((hidden_dim, 1))
        self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim) / np.sqrt(input_dim + hidden_dim)
        self.b_o = np.zeros((hidden_dim, 1))
        self.cache = None  # Initialize cache to avoid AttributeError

    def forward(self, x, h_prev, c_prev):
        concat = np.vstack((h_prev, x))
        f = sigmoid(np.dot(self.W_f, concat) + self.b_f)
        i = sigmoid(np.dot(self.W_i, concat) + self.b_i)
        c_bar = self.activation(np.dot(self.W_c, concat) + self.b_c)
        c = f * c_prev + i * c_bar
        o = sigmoid(np.dot(self.W_o, concat) + self.b_o)
        h = o * self.activation(c)
        self.cache = (x, h_prev, c_prev, f, i, c_bar, c, o, h, concat)
        return h, c

    def backward(self, dh_next, dc_next):
        # Retrieve cached values
        x, h_prev, c_prev, f, i, c_bar, c, o, h, concat = self.cache

        # Gradients of output
        do = dh_next * self.activation(c)
        dactivation_c = dh_next * o
        dc = dactivation_c * (1 - self.activation(c) ** 2) + dc_next
        df = dc * c_prev
        di = dc * c_bar
        dc_bar = dc * i
        dc_prev = dc * f

        # Gradients of gates (sigmoid and tanh derivatives)
        do_pre = do * o * (1 - o)
        df_pre = df * f * (1 - f)
        di_pre = di * i * (1 - i)
        dc_bar_pre = dc_bar * (1 - c_bar ** 2)

        # Gradients w.r.t. weights and biases
        dW_o = np.dot(do_pre, concat.T)
        dW_f = np.dot(df_pre, concat.T)
        dW_i = np.dot(di_pre, concat.T)
        dW_c = np.dot(dc_bar_pre, concat.T)
        db_o = do_pre
        db_f = df_pre
        db_i = di_pre
        db_c = dc_bar_pre

        # Gradient w.r.t. concat input
        dconcat = (
            np.dot(self.W_f.T, df_pre) +
            np.dot(self.W_i.T, di_pre) +
            np.dot(self.W_c.T, dc_bar_pre) +
            np.dot(self.W_o.T, do_pre)
        )

        dh_prev = dconcat[:self.hidden_dim, :]
        dx = dconcat[self.hidden_dim:, :]

        # Return gradients
        grads = {
            'dW_f': dW_f, 'db_f': db_f,
            'dW_i': dW_i, 'db_i': db_i,
            'dW_c': dW_c, 'db_c': db_c,
            'dW_o': dW_o, 'db_o': db_o
        }
        return dx, dh_prev, dc_prev, grads

    def update_params(self, lr):
        self.W_f -= lr * self.dW_f
        self.W_i -= lr * self.dW_i
        self.W_c -= lr * self.dW_c
        self.W_o -= lr * self.dW_o
        self.b_f -= lr * self.db_f
        self.b_i -= lr * self.db_i
        self.b_c -= lr * self.db_c
        self.b_o -= lr * self.db_o

class LSTMLayer:
    def __init__(self, input_dim, hidden_dim, activation="tanh"):
        self.cell = LSTMCell(input_dim, hidden_dim, activation)
        self.caches = []

    def forward(self, X):
        h, c = np.zeros((self.cell.hidden_dim, 1)), np.zeros((self.cell.hidden_dim, 1))
        outputs = []
        caches = []
        for x in X:
            h, c = self.cell.forward(x, h, c)
            outputs.append(h)
            caches.append(self.cell.cache)
        self.caches = caches  # Store for backward
        return np.stack(outputs)

    def backward(self, dhs):
        # dhs: list of gradients w.r.t. output h at each time step
        dW_f = np.zeros_like(self.cell.W_f)
        dW_i = np.zeros_like(self.cell.W_i)
        dW_c = np.zeros_like(self.cell.W_c)
        dW_o = np.zeros_like(self.cell.W_o)
        db_f = np.zeros_like(self.cell.b_f)
        db_i = np.zeros_like(self.cell.b_i)
        db_c = np.zeros_like(self.cell.b_c)
        db_o = np.zeros_like(self.cell.b_o)
        dh_next = np.zeros((self.cell.hidden_dim, 1))
        dc_next = np.zeros((self.cell.hidden_dim, 1))
        for t in reversed(range(len(self.caches))):
            self.cell.cache = self.caches[t]
            dx, dh_next, dc_next, grads = self.cell.backward(dh_next + dhs[t], dc_next)
            dW_f += grads['dW_f']
            dW_i += grads['dW_i']
            dW_c += grads['dW_c']
            dW_o += grads['dW_o']
            db_f += grads['db_f']
            db_i += grads['db_i']
            db_c += grads['db_c']
            db_o += grads['db_o']
        # Store gradients for parameter update
        self.cell.dW_f = dW_f
        self.cell.dW_i = dW_i
        self.cell.dW_c = dW_c
        self.cell.dW_o = dW_o
        self.cell.db_f = db_f
        self.cell.db_i = db_i
        self.cell.db_c = db_c
        self.cell.db_o = db_o

    def update_params(self, lr):
        self.cell.W_f -= lr * self.cell.dW_f
        self.cell.W_i -= lr * self.cell.dW_i
        self.cell.W_c -= lr * self.cell.dW_c
        self.cell.W_o -= lr * self.cell.dW_o
        self.cell.b_f -= lr * self.cell.db_f
        self.cell.b_i -= lr * self.cell.db_i
        self.cell.b_c -= lr * self.cell.db_c
        self.cell.b_o -= lr * self.cell.db_o 