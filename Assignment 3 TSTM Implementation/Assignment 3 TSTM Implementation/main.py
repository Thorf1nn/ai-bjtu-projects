from seq2seq import Encoder, Decoder
from utils import compute_bleu, cross_entropy_loss
import numpy as np
from train import train
from data_loader import read_tmx_parallel_corpus, build_vocab, sentence_to_indices, pad_sequence
from embedding import Embedding
from activations import get_activation, sigmoid
import xml.etree.ElementTree as ET
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
nltk.download('punkt')

# === Compute BLEU using NLTK ===
def compute_bleu_nltk(reference, candidate):
    chencherry = SmoothingFunction()
    return sentence_bleu([reference], candidate, smoothing_function=chencherry.method1)

# === Load TMX Dataset ===
tmx_path = 'database/EN-ES_Website_corpus/output_data_en-es.tmx'
src_lang = 'en'
tgt_lang = 'es'
num_samples = 1000
eng_sentences, es_sentences = read_tmx_parallel_corpus(tmx_path, src_lang, tgt_lang, num_samples)

# === Build Vocab and Index Sentences ===
eng_vocab = build_vocab(eng_sentences)
es_vocab = build_vocab(es_sentences)
max_len = max(max(len(s.split()) for s in eng_sentences), max(len(s.split()) for s in es_sentences))
eng_indices = [pad_sequence([eng_vocab['<SOS>']] + sentence_to_indices(s, eng_vocab) + [eng_vocab['<EOS>']], max_len+2) for s in eng_sentences]
es_indices = [pad_sequence([es_vocab['<SOS>']] + sentence_to_indices(s, es_vocab) + [es_vocab['<EOS>']], max_len+2) for s in es_sentences]

# === Embeddings ===
embedding_dim = 16
eng_embedding = Embedding(len(eng_vocab), embedding_dim)
es_embedding = Embedding(len(es_vocab), embedding_dim)

# === Prepare Training Data ===
X_data, Y_data = [], []
for eng_idx, es_idx in zip(eng_indices, es_indices):
    eng_embedded = eng_embedding.forward(eng_idx)
    es_embedded = es_embedding.forward(es_idx)
    X = [x.reshape(-1, 1) for x in eng_embedded]
    Y = [y.reshape(-1, 1) for y in es_embedded]
    X_data.append(X)
    Y_data.append(Y)
data = list(zip(X_data, Y_data, es_indices))

# === Split Train/Test ===
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# === Model ===
encoder = Encoder(input_dim=embedding_dim, hidden_dim=20)
decoder = Decoder(input_dim=embedding_dim, hidden_dim=20, vocab_size=len(es_vocab))

# === Training Loop ===
epochs = 5
lr = 0.01
for epoch in range(epochs):
    total_loss = 0
    for X, Y, target_indices in train_data:
        encoder_hidden = encoder.forward(X)
        h, c = encoder_hidden[-1], np.zeros_like(encoder_hidden[-1])
        logits, hiddens = [], []
        for idx in target_indices[:-1]:
            emb = es_embedding.forward([idx])[0].reshape(-1, 1)
            h, c = decoder.lstm.cell.forward(emb, h, c)
            logit = np.dot(decoder.W_y, h) + decoder.b_y
            logits.append(logit)
            hiddens.append(h)
        loss, probs_list = cross_entropy_loss(logits, target_indices, pad_idx=es_vocab['<PAD>'])
        probs = np.hstack(probs_list).T
        total_loss += loss
        dW_y, db_y, dhs = decoder.backward_output_layer(probs, target_indices, hiddens)
        decoder.W_y -= lr * dW_y
        decoder.b_y -= lr * db_y
        decoder.lstm.backward(dhs)
        decoder.lstm.update_params(lr)
    print(f"Epoch {epoch+1}, Loss: {total_loss.item() / len(train_data):.4f}")

# === Inference: Greedy Decode ===
inv_es_vocab = {idx: word for word, idx in es_vocab.items()}
def greedy_decode(encoder, decoder, eng_embedding, es_embedding, eng_vocab, es_vocab, input_sentence, max_len=20):
    input_indices = [eng_vocab.get(word, eng_vocab['<UNK>']) for word in input_sentence.split()]
    input_indices = [eng_vocab['<SOS>']] + input_indices + [eng_vocab['<EOS>']]
    input_indices = pad_sequence(input_indices, max_len)
    eng_embedded = eng_embedding.forward(input_indices)
    X = [x.reshape(-1, 1) for x in eng_embedded]
    encoder_hidden = encoder.forward(X)[-1]
    h, c = encoder_hidden, np.zeros_like(encoder_hidden)
    current_token = es_vocab['<SOS>']
    decoded_indices = []
    for _ in range(max_len):
        token_embedded = es_embedding.forward([current_token])[0].reshape(-1, 1)
        h, c = decoder.lstm.cell.forward(token_embedded, h, c)
        logit = np.dot(decoder.W_y, h) + decoder.b_y
        probs = get_activation("tanh")(logit)
        current_token = np.argmax(probs)
        decoded_indices.append(current_token)
    return [inv_es_vocab.get(idx, inv_es_vocab.get('<UNK>', '<UNK>')) for idx in decoded_indices]

print("Sample translations and BLEU score after training:")

for i in range(3):  # Show 3 sample translations
    src = eng_sentences[i].split()
    tgt = es_sentences[i].split()
    pred = greedy_decode(encoder, decoder, eng_embedding, es_embedding, eng_vocab, es_vocab, " ".join(src))
    print("SRC:", " ".join(src))
    print("REF:", " ".join(tgt))
    print("PRED:", " ".join(pred))
    print()

# Calculate BLEU score
all_preds = [greedy_decode(encoder, decoder, eng_embedding, es_embedding, eng_vocab, es_vocab, " ".join(src)) for src in eng_sentences]
bleu_scores = [compute_bleu(ref.split(), pred) for ref, pred in zip(es_sentences, all_preds)]
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU score:", avg_bleu)