# main.py

import numpy as np
import torch
from data.data_loader import load_rotten_tomatoes_dataset
from data.preprocessing import apply_preprocessing, tokenize_sentences
from embeddings.embedding_preparation import (
    train_word2vec, 
    build_vocab, 
    handle_oov, 
    build_vocab, 
    create_embedding_matrix, 
    words_to_indices, 
    create_dataloader
)
from models.rnn import SentimentRNN
from models.bilstm import SentimentBiLSTM
from models.bigru import SentimentBiGRU
from models.cnn import SentimentCNN
from models.improved_cnn import ImprovedSentimentCNN
from training.trainer import SentimentClassifier
from training.hyperparameter_tuning import tune_hyperparameters
from utils.oov_handler import replace_oov_words
from utils.visualization import plot_accuracy, plot_loss

def main():
    # Step 1: Load Datasets
    train_dataset, validation_dataset, test_dataset = load_rotten_tomatoes_dataset()
    
    # Step 2: Preprocess Datasets
    train_dataset = apply_preprocessing(train_dataset)
    validation_dataset = apply_preprocessing(validation_dataset)
    test_dataset = apply_preprocessing(test_dataset)
    
    # Step 3: Tokenize
    tokenized_train = tokenize_sentences(train_dataset)
    tokenized_val = tokenize_sentences(validation_dataset)
    tokenized_test = tokenize_sentences(test_dataset)
    
    # Step 4: Train Word2Vec
    word2vec_model = train_word2vec(tokenized_train)
    
    # Step 5: Handle OOV
    original_vocab = set(word for sentence in tokenized_train for word in sentence)
    word2vec_vocab = set(word2vec_model.wv.key_to_index)
    tokenized_train = handle_oov(tokenized_train, word2vec_vocab)
    tokenized_val = handle_oov(tokenized_val, word2vec_vocab)
    tokenized_test = handle_oov(tokenized_test, word2vec_vocab)
    
    # Step 6: Build Vocabulary
    word_to_index = build_vocab(tokenized_train, word2vec_vocab)
    
    # Step 7: Create Embedding Matrix
    embedding_matrix = create_embedding_matrix(word_to_index, word2vec_model, embedding_dim=100)
    
    # Step 8: Convert Words to Indices
    train_X = words_to_indices(tokenized_train, word_to_index)
    val_X = words_to_indices(tokenized_val, word_to_index)
    test_X = words_to_indices(tokenized_test, word_to_index)
    
    train_y = train_dataset['label']
    val_y = validation_dataset['label']
    test_y = test_dataset['label']
    
    # Step 9: Create DataLoaders
    train_dataloader = create_dataloader(train_X, train_y, word_to_index, batch_size=32, shuffle=True)
    val_dataloader = create_dataloader(val_X, val_y, word_to_index, batch_size=32, shuffle=False)
    test_dataloader = create_dataloader(test_X, test_y, word_to_index, batch_size=32, shuffle=False)
    
    # Step 10: Initialize Model
    model = SentimentRNN(
        vocab_size=len(word_to_index),
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,
        pad_idx=word_to_index.get('<PAD>', 0),
        embedding_matrix=embedding_matrix,
        freeze_embeddings=True,
        aggregation_method='max_pooling',
        dropout_rate=0.2
    )
    
    # Step 11: Wrap in SentimentClassifier
    classifier = SentimentClassifier(model=model, learning_rate=1e-3)
    
    # Step 12: Define Trainer
    trainer = pl.Trainer(max_epochs=20, 
                         callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min'),
                                    pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)])
    
    # Step 13: Train Model
    trainer.fit(classifier, train_dataloader, val_dataloader)
    
    # Step 14: Test Model
    trainer.test(classifier, test_dataloader)
    
    
if __name__ == "__main__":
    main()
