import numpy as np
from lstm import LSTMLayer

class Encoder:
    def __init__(self, input_dim, hidden_dim):
        self.lstm = LSTMLayer(input_dim, hidden_dim)

    def forward(self, X):
        return self.lstm.forward(X)

class Decoder:
    def __init__(self, input_dim, hidden_dim, vocab_size):
        self.lstm = LSTMLayer(input_dim, hidden_dim)
        self.W_y = np.random.randn(vocab_size, hidden_dim) / np.sqrt(hidden_dim)
        self.b_y = np.zeros((vocab_size, 1))

    def forward(self, X, encoder_hidden):
        h, c = encoder_hidden, np.zeros_like(encoder_hidden)
        outputs = []
        hiddens = []
        for x in X:
            h, c = self.lstm.cell.forward(x, h, c)
            y = np.dot(self.W_y, h) + self.b_y
            outputs.append(y)
            hiddens.append(h)
        return np.stack(outputs), np.stack(hiddens)

    def backward_output_layer(self, probs, target_indices, hiddens):
        seq_len, vocab_size = probs.shape
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)
        dhs = []
        for t in range(seq_len):
            y_true = np.zeros((vocab_size, 1))
            y_true[target_indices[t]] = 1
            dy = probs[t].reshape(-1, 1) - y_true  # (vocab_size, 1)
            dW_y += np.dot(dy, hiddens[t].T)
            db_y += dy
            dh = np.dot(self.W_y.T, dy)
            dhs.append(dh)
        return dW_y, db_y, dhs 