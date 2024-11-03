# training/hyperparameter_tuning.py

import optuna
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true.numpy(), y_pred.numpy())

def objective(trial, model_class, train_dataloader, val_dataloader, word_to_index, word2vec_model):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    aggregation_method = trial.suggest_categorical('aggregation_method', ['last_hidden', 'last_output', 'mean_pooling', 'max_pooling', 'attention'])
    
    # Initialize model
    model = model_class(
        vocab_size=len(word_to_index),
        embedding_dim=word2vec_model.vector_size,
        hidden_dim=hidden_dim,
        output_dim=2,
        pad_idx=word_to_index.get('<PAD>', 0),
        embedding_matrix=None,  # Provide the embedding matrix as needed
        freeze_embeddings=True,
        aggregation_method=aggregation_method,
        dropout_rate=dropout_rate
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(5):  # Limited epochs for speed
        model.train()
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_acc += calculate_accuracy(labels, predicted)
    val_acc /= len(val_dataloader)
    
    return val_acc

def tune_hyperparameters(model_class, train_dataloader, val_dataloader, word_to_index, word2vec_model, n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_class, train_dataloader, val_dataloader, word_to_index, word2vec_model), n_trials=n_trials)
    return study.best_params, study.best_value
