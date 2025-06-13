import numpy as np

# Bahdanau Attention (Additive)
def bahdanau_attention(encoder_outputs, decoder_hidden):
    # encoder_outputs: (seq_len, hidden_dim, 1)
    # decoder_hidden: (hidden_dim, 1)
    # Placeholder for demonstration
    # Implement actual attention mechanism as needed
    attn_weights = np.ones((encoder_outputs.shape[0], 1)) / encoder_outputs.shape[0]
    context = np.sum(encoder_outputs * attn_weights[:, None, :], axis=0)
    return context, attn_weights

# Luong Attention (Multiplicative)
def luong_attention(encoder_outputs, decoder_hidden):
    # encoder_outputs: (seq_len, hidden_dim, 1)
    # decoder_hidden: (hidden_dim, 1)
    # Placeholder for demonstration
    scores = np.dot(encoder_outputs.reshape(encoder_outputs.shape[0], -1), decoder_hidden.reshape(-1, 1))
    attn_weights = np.exp(scores) / np.sum(np.exp(scores))
    context = np.sum(encoder_outputs * attn_weights[:, None, :], axis=0)
    return context, attn_weights 