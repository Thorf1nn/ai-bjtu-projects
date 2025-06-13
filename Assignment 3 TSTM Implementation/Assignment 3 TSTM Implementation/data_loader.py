import os
from collections import Counter
import xml.etree.ElementTree as ET

def read_parallel_corpus(eng_path, urdu_path, num_samples=None):
    with open(eng_path, encoding='utf-8') as f_eng, open(urdu_path, encoding='utf-8') as f_urdu:
        eng_lines = f_eng.readlines()
        urdu_lines = f_urdu.readlines()
    if num_samples:
        eng_lines = eng_lines[:num_samples]
        urdu_lines = urdu_lines[:num_samples]
    return [l.strip() for l in eng_lines], [l.strip() for l in urdu_lines]

def read_tmx_parallel_corpus(tmx_path, src_lang='en', tgt_lang='es', num_samples=None):
    src_sentences = []
    tgt_sentences = []
    tree = ET.parse(tmx_path)
    root = tree.getroot()
    for tu in root.iter('tu'):
        segs = {tuv.attrib['{http://www.w3.org/XML/1998/namespace}lang']: tuv.find('seg').text
                for tuv in tu.findall('tuv')}
        if src_lang in segs and tgt_lang in segs:
            src_sentences.append(segs[src_lang])
            tgt_sentences.append(segs[tgt_lang])
            if num_samples and len(src_sentences) >= num_samples:
                break
    return src_sentences, tgt_sentences

def build_vocab(sentences, min_freq=1):
    vocab = {}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<SOS>'] = 2
    vocab['<EOS>'] = 3
    counter = Counter(word for sent in sentences for word in sent.split())
    idx = 4
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

def sentence_to_indices(sentence, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in sentence.split()]

def pad_sequence(seq, max_len, pad_value=0):
    return seq + [pad_value] * (max_len - len(seq))