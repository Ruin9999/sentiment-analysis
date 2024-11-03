import numpy as np

UNK_TOKEN = '<UNK>'

def replace_oov_words(tokenized_sentences, vocab):
    return [[word if word in vocab else UNK_TOKEN for word in sentence] for sentence in tokenized_sentences]


def replace_oov_with_avg(tokenized_sentences, word2vec_model, word_to_index, embedding_dim):
    avg_vector = np.mean(word2vec_model.wv.vectors, axis=0)
    for i, sentence in enumerate(tokenized_sentences):
        processed_sentence = []
        for word in sentence:
            if word in word_to_index:
                processed_sentence.append(word)
            else:
                processed_sentence.append(UNK_TOKEN)
                # Alternatively, handle differently based on the strategy
        tokenized_sentences[i] = processed_sentence
    return tokenized_sentences
