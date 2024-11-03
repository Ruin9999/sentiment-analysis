# embeddings/embedding_preparation.py

from gensim.models import Word2Vec
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader

UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'

def train_word2vec(tokenized_sentences, vector_size=100, window=3, min_count=2, workers=4, sg=1, epochs=5):
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs
    )
    return model

def build_vocab(tokenized_sentences, word2vec_vocab):
    final_vocab = set()
    for sentence in tokenized_sentences:
        for word in sentence:
            final_vocab.add(word)
    final_vocab.add(UNK_TOKEN)
    final_vocab.add(PAD_TOKEN)
    word_to_index = {word: i for i, word in enumerate(final_vocab)}
    return word_to_index

def handle_oov(tokenized_sentences, word2vec_vocab):
    for i, sentence in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [word if word in word2vec_vocab else UNK_TOKEN for word in sentence]
    return tokenized_sentences

def create_embedding_matrix(word_to_index, word2vec_model, embedding_dim):
    embedding_matrix = np.zeros((len(word_to_index), embedding_dim))
    for word, i in word_to_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

def words_to_indices(sentences, word_to_index):
    return [[word_to_index.get(word, word_to_index[UNK_TOKEN]) for word in sentence] for sentence in sentences]

def create_dataloader(X, y, word_to_index, batch_size=16, shuffle=True):
    X_tensor = [torch.tensor(seq, dtype=torch.long) for seq in X]
    X_padded = pad_sequence(X_tensor, batch_first=True, padding_value=word_to_index[PAD_TOKEN])
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_padded, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
