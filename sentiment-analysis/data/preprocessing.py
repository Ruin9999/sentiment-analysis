
import re
import nltk
from nltk.tokenize import word_tokenize
import os
from contextlib import redirect_stdout

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\'\!\?\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_sentences(dataset):
    
    print("Downloading nltk tokenizer...")
    repo_root = os.getcwd()  
    nltk_data_path = os.path.join(repo_root, 'nltk')

    nltk.data.path.append(nltk_data_path)
    required_resource = 'tokenizers/punkt'

    try:
        nltk.data.find(required_resource)
        print(f"'{required_resource}' already exists.")
    except LookupError:
        print(f"Downloading '{required_resource}'...")
        nltk.download('all', download_dir=nltk_data_path)

    tokenized_sentences = []
    for text in dataset['text']:
        tokenized_sentences.append(word_tokenize(text))
    return tokenized_sentences

def apply_preprocessing(dataset):
    dataset = dataset.map(lambda x: {'text': clean_text(x['text'])})
    return dataset
