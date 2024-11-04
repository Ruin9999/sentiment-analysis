
import optuna
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import json

from training.trainer import SentimentClassifier 
import inspect 

def objective(trial, model_class, train_dataloader, val_dataloader, word_to_index, word2vec_model, embedding_matrix, config_dir='configs'):
   
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
    aggregation_method = trial.suggest_categorical('aggregation_method', ['last_hidden', 'last_output', 'mean_pooling', 'max_pooling', 'attention'])

    model_kwargs = {
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
        num_layers = trial.suggest_int('num_layers', 1, 3)
        model_kwargs['num_layers'] = num_layers
    
    try:
        model = model_class(**model_kwargs)
    except TypeError as e:
        trial.report(-float('inf'), step=0)
        return -float('inf')

    model = model_class(**model_kwargs)
    
    classifier = SentimentClassifier(model=model, learning_rate=learning_rate, config_path=None)
    
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=False)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/hyperparam_tuning',
        filename=f'{model_class.__name__}_trial_{trial.number}-val_acc{{val_acc:.2f}}',
        save_top_k=1,
        mode='max'
    )
    
    logger = TensorBoardLogger("logs/hyperparam_tuning", name=f"{model_class.__name__}_trial_{trial.number}")
    
    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=False,  
        log_every_n_steps=10,
        accelerator='cuda',
        devices=[0]
    )
    

    trainer.fit(classifier, train_dataloader, val_dataloader)
    
    val_result = trainer.validate(classifier, val_dataloader, verbose=False)
    val_acc = val_result[0]['val_acc']
    
    trial_config_dir = os.path.join(config_dir, f'trial_{trial.number}')
    os.makedirs(trial_config_dir, exist_ok=True)
    config_path = os.path.join(trial_config_dir, 'config.json')
    
    config = {
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'hidden_dim': hidden_dim,
        'aggregation_method': aggregation_method
    }
    
    if 'num_layers' in model_kwargs:
        config['num_layers'] = model_kwargs['num_layers']
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return val_acc

def tune_hyperparameters(model_class, train_dataloader, val_dataloader, word_to_index, word2vec_model, embedding_matrix, n_trials=50, config_dir='configs'):

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(
            trial, 
            model_class, 
            train_dataloader, 
            val_dataloader, 
            word_to_index, 
            word2vec_model, 
            embedding_matrix, 
            config_dir
        ),
        n_trials=n_trials
    )
    return study.best_params, study.best_value
