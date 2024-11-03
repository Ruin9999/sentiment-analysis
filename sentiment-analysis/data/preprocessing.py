
import re
import nltk
from nltk.tokenize import word_tokenize

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\'\!\?\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_sentences(dataset):
    tokenized_sentences = []
    for text in dataset['text']:
        tokenized_sentences.append(word_tokenize(text))
    return tokenized_sentences

def apply_preprocessing(dataset):
    dataset = dataset.map(lambda x: {'text': clean_text(x['text'])})
    return dataset
