
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
from models.bilstm import SentimentBiLSTM
from models.bigru import SentimentBiGRU
from models.cnn import SentimentCNN
from models.improved_cnn import ImprovedSentimentCNN
from training.trainer import SentimentClassifier
from training.hyperparameter_tuning import tune_hyperparameters
from utils.oov_handler import replace_oov_words
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os
from utils.get_model import get_model_class, get_weights_path, get_model_config
from sklearn.metrics import classification_report
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description="Run sentiment analysis model tuning.")
    parser.add_argument('--model', type=str, default='rnn', help="Specify the model to tune (e.g., 'rnn', 'bilstm', 'bigru', 'cnn', 'improved_cnn')")
    args = parser.parse_args()

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

    # Step 9: Create DataLoader for Test Set
    print("Creating DataLoader for test set...")
    test_loader = create_dataloader(test_X, test_y, word_to_index, batch_size=256, shuffle=False)

    # Step 10: Define Available Models
    available_models = ['rnn', 'rnn_freeze', 'bilstm', 'bigru', 'cnn']
    reports = []

    # Step 11: Iterate Through All Available Models
    for model_name in available_models:
        print(100*"=")
        
        print(f"\nEvaluating model: {model_name}")

        # Load Model Class
        MODEL_CLASS = get_model_class(model_name)
        if MODEL_CLASS is None:
            print(f"Model class for '{model_name}' not found. Skipping...")
            continue

        # Load Weights Path
        weight_path = get_weights_path(model_name)
        if weight_path is None or not os.path.exists(weight_path):
            print(f"Weights for '{model_name}' not found at '{weight_path}'. Skipping...")
            continue

        # Load Model Configuration
        try:
            model_config = get_model_config(model_name)
        except Exception as e:
            print(f"Error loading config for '{model_name}': {e}. Skipping...")
            continue


        # Initialize the Model
        try:
            # Only pass parameters that exist in the model's __init__ method and ignore 'vocab_size' if not required.
            valid_params = {k: v for k, v in model_config.items() if k in MODEL_CLASS.__init__.__code__.co_varnames}
            
            # Check if 'vocab_size' is actually required by each model before adding it
            if 'vocab_size' in MODEL_CLASS.__init__.__code__.co_varnames:
                valid_params['vocab_size'] = len(word_to_index)
                model = MODEL_CLASS(
                    embedding_matrix=embedding_matrix,
                    **valid_params
                )
            else:
                model = MODEL_CLASS(
                    embedding_matrix=embedding_matrix,
                    **valid_params
                )
                
        except Exception as e:
            print(f"Error initializing model '{model_name}': {e}. Skipping...")
            continue

        # Load the Trained Model from Checkpoint
        try:
            classifier = SentimentClassifier.load_from_checkpoint(
                checkpoint_path=weight_path,
                model=model,
                learning_rate=model_config.get('learning_rate', 1e-3)
            )
            classifier.eval()
            classifier.freeze()
        except Exception as e:
            print(f"Error loading checkpoint for '{model_name}': {e}. Skipping...")
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier.to(device)

        all_preds = []
        all_true = []

        print(100*"=")
        print(f"Running evaluation for '{model_name}' on test set...")
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = classifier(batch_X)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())

        # Generate Classification Report
        report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        print(report_df)
        report_df['model'] = model_name
        reports.append(report_df)

        print(f"Completed evaluation for '{model_name}'.")

    # Step 12: Aggregate All Reports
    if reports:
       if reports:
        print(100*"=")
        print("\nAggregating all classification reports...")
        all_reports_df = pd.concat(reports, ignore_index=True)
        
        # Reorder columns to move 'model' to the first column
        cols = all_reports_df.columns.tolist()
        cols = ['model'] + [col for col in cols if col != 'model']
        all_reports_df = all_reports_df[cols]
        
        # Save the complete report to CSV
        output_csv = 'classification_reports.csv'
        all_reports_df.to_csv(output_csv, index=False)
        print(f"All classification reports have been saved to '{output_csv}'.")
    else:
        print("No classification reports to save.")


    # Step 13: Load an Example mode for Inference
    print(100*"=")
    print("Loading Example RNN model ...")
    MODEL_CLASS = get_model_class('rnn')
    weight_path = get_weights_path('rnn')
    model_config = get_model_config('rnn')
    model_config.pop('vocab_size', None) 

    model = MODEL_CLASS(
        vocab_size=len(word_to_index),
        embedding_matrix=embedding_matrix,
        # pad_idx=if word_to_index.get('<PAD>', 0),
        **{k: v for k, v in model_config.items() if k in MODEL_CLASS.__init__.__code__.co_varnames if k != 'vocab_size'}
    )
    
    # Load the Trained Model from Checkpoint
    classifier = SentimentClassifier.load_from_checkpoint(
        checkpoint_path=weight_path,
        model=model,
        learning_rate=model_config.get('learning_rate', 1e-3)
    )
    classifier.eval()

    # Prepare a Single Text for Inference
    single_test_input_text = test_dataset['text'][0]
    single_test_input = torch.tensor(test_X[0]).unsqueeze(0).to(classifier.device)  

    # Run Inference
    print("Running inference on a single test input...")
    print(f"Test Input: {single_test_input_text}")
    print(f"Test Input: {single_test_input}")
    with torch.no_grad():
        prediction = classifier(single_test_input)
        predicted_label = torch.argmax(prediction, dim=1).item()
    print(f"Predicted Label: {predicted_label}")


if __name__ == "__main__":
    main()
