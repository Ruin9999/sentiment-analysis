
import numpy as np
import torch
from data.data_loader import load_rotten_tomatoes_dataset
from data.preprocessing import apply_preprocessing, tokenize_sentences
from embeddings.embedding_preparation import (
    train_word2vec, 
    build_vocab, 
    handle_oov, 
    create_embedding_matrix, 
    words_to_indices, 
    create_dataloader
)
from models.rnn import SentimentRNN
from training.hyperparameter_tuning import tune_hyperparameters
import os
from training.trainer import SentimentClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import inspect 

# /p/scratch/ccstdl/xu17/jz/code/sentiment-analysis/sentiment-analysis/main_hyperparam_tuning.py

import numpy as np
import torch
from data.data_loader import load_rotten_tomatoes_dataset
from data.preprocessing import apply_preprocessing, tokenize_sentences
from embeddings.embedding_preparation import (
    train_word2vec, 
    build_vocab, 
    handle_oov, 
    create_embedding_matrix, 
    words_to_indices, 
    create_dataloader
)
from models.rnn import SentimentRNN
from training.hyperparameter_tuning import tune_hyperparameters
import os

def main():
    # Step 1: Load Datasets
    print("Loading datasets...")
    train_dataset, validation_dataset, test_dataset = load_rotten_tomatoes_dataset()
    
    # Step 2: Preprocess Datasets
    print("Preprocessing datasets...")
    train_dataset = apply_preprocessing(train_dataset)
    validation_dataset = apply_preprocessing(validation_dataset)
    test_dataset = apply_preprocessing(test_dataset)
    
    # Step 3: Tokenize
    print("Tokenizing datasets...")
    tokenized_train = tokenize_sentences(train_dataset)
    tokenized_val = tokenize_sentences(validation_dataset)
    tokenized_test = tokenize_sentences(test_dataset)
    
    # Step 4: Train Word2Vec
    print("Training Word2Vec model...")
    word2vec_model = train_word2vec(tokenized_train)
    
    # Step 5: Handle OOV
    print("Handling OOV words...")
    original_vocab = set(word for sentence in tokenized_train for word in sentence)
    word2vec_vocab = set(word2vec_model.wv.key_to_index)
    tokenized_train = handle_oov(tokenized_train, word2vec_vocab)
    tokenized_val = handle_oov(tokenized_val, word2vec_vocab)
    tokenized_test = handle_oov(tokenized_test, word2vec_vocab)
    
    # Step 6: Build Vocabulary
    print("Building vocabulary...")
    word_to_index = build_vocab(tokenized_train, word2vec_vocab)
    
    # Step 7: Create Embedding Matrix
    print("Creating embedding matrix...")
    embedding_matrix = create_embedding_matrix(word_to_index, word2vec_model, embedding_dim=100)
    
    # Step 8: Convert Words to Indices
    print("Converting words to indices...")
    train_X = words_to_indices(tokenized_train, word_to_index)
    val_X = words_to_indices(tokenized_val, word_to_index)
    test_X = words_to_indices(tokenized_test, word_to_index)
    
    train_y = train_dataset['label']
    val_y = validation_dataset['label']
    test_y = test_dataset['label']
    
    # Step 9: Create DataLoaders
    print("Creating DataLoaders...")
    train_dataloader = create_dataloader(train_X, train_y, word_to_index, batch_size=512, shuffle=True)
    val_dataloader = create_dataloader(val_X, val_y, word_to_index, batch_size=512, shuffle=False)
    test_dataloader = create_dataloader(test_X, test_y, word_to_index, batch_size=512, shuffle=False)
    
    # Step 10: Define Model Class
    print("Defining model class for hyperparameter tuning...")
    model_class = SentimentRNN  

    # Step 11: Define Configuration Directory
    config_dir = 'configs/hyperparam_tuning'
    os.makedirs(config_dir, exist_ok=True)
    
    # Step 12: Perform Hyperparameter Tuning
    print("Starting hyperparameter tuning...")
    best_params, best_val_acc = tune_hyperparameters(
        model_class=model_class,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        word_to_index=word_to_index,
        word2vec_model=word2vec_model,
        embedding_matrix=embedding_matrix,
        n_trials=20,  
        config_dir=config_dir
    )
    
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Step 13: Train Final Model with Best Hyperparameters
    print("Training final model with best hyperparameters...")

    learning_rate = best_params['learning_rate']
    dropout_rate = best_params['dropout_rate']
    hidden_dim = best_params['hidden_dim']
    aggregation_method = best_params['aggregation_method']
    
    final_model_kwargs = {
        'vocab_size': len(word_to_index),
        'embedding_dim': word2vec_model.vector_size,
        'hidden_dim': hidden_dim,
        'output_dim': 2,
        'pad_idx': word_to_index.get('<PAD>', 0),
        'embedding_matrix': embedding_matrix,
        'freeze_embeddings': True,
        'aggregation_method': aggregation_method,
        'dropout_rate': dropout_rate
    }
    
    signature = inspect.signature(model_class.__init__)
    if 'num_layers' in signature.parameters:
        final_model_kwargs['num_layers'] = best_params.get('num_layers', 1) 


    final_model = model_class(**final_model_kwargs)
    
    final_classifier = SentimentClassifier(model=final_model, learning_rate=learning_rate, config_path=os.path.join(config_dir, 'final_model_config.json'))
    
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/final_model',
        filename=f'{model_class.__name__}-FinalModel-val_acc{{val_acc:.2f}}',
        save_top_k=1,
        mode='max'
    )
    
    logger = TensorBoardLogger("logs/final_model", name="sentiment_analysis_final")
    
    trainer = pl.Trainer(
        max_epochs=20,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator='cuda',
        devices=[0]
    )
    
    trainer.fit(final_classifier, train_dataloader, val_dataloader)
    
    print("Testing final model...")
    trainer.test(final_classifier, test_dataloader)
    
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path} for inference...")
        best_classifier = SentimentClassifier.load_from_checkpoint(best_model_path)
        trainer.test(best_classifier, test_dataloader)

if __name__ == "__main__":
    main()
