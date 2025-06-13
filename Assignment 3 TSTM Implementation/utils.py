import numpy as np

def cross_entropy_loss(logits, target_indices, pad_idx=None):
    print("cross_entropy_loss called with pad_idx =", pad_idx)
    loss = 0.0
    probs = []
    for i, logit in enumerate(logits):
        prob = np.exp(logit - np.max(logit))  # softmax
        prob = prob / np.sum(prob)
        probs.append(prob)
        target = target_indices[i]
        if pad_idx is None or target != pad_idx:
            loss -= np.log(prob[target] + 1e-9)  # avoid log(0)
    return loss / len(target_indices), probs


def compute_bleu(reference, candidate, n=4):
    # Simple BLEU implementation for illustration
    def n_gram(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    bleu = 1.0
    for i in range(1, n+1):
        ref_ngrams = set(n_gram(reference, i))
        cand_ngrams = set(n_gram(candidate, i))
        overlap = len(ref_ngrams & cand_ngrams)
        total = max(len(cand_ngrams), 1)
        bleu *= overlap / total
    bleu = bleu ** (1/n)
    return bleu




