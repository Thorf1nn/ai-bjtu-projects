import numpy as np

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, indices):
        # indices: (seq_len,) or (batch_size, seq_len)
        return np.array([self.embeddings[idx] for idx in indices]) 