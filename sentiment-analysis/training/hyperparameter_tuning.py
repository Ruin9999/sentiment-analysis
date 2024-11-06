
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
   
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 2e-3)
    
    model_kwargs = {
        'embedding_dim': word2vec_model.vector_size,
        'pad_idx': word_to_index.get('<PAD>', 0),
        'embedding_matrix': embedding_matrix,
        'freeze_embeddings': False,
    }

    signature = inspect.signature(model_class.__init__)
    if 'dropout_rate' in signature.parameters:
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.3)
        model_kwargs['dropout_rate'] = dropout_rate

    if 'output_dim' in signature.parameters: 
        model_kwargs['output_dim'] = 2

    if  'hidden_dim' in signature.parameters:
        hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
        model_kwargs['hidden_dim'] = hidden_dim

    if 'num_layers' in signature.parameters:
        num_layers = trial.suggest_int('num_layers', 1, 4)
        model_kwargs['num_layers'] = num_layers

    if 'aggregation_method' in signature.parameters:
        aggregation_method = trial.suggest_categorical('aggregation_method', ['last_hidden', 'last_output', 'mean_pooling', 'max_pooling', 'attention'])
        model_kwargs['aggregation_method'] = aggregation_method

    if 'vocab_size' in signature.parameters:
        model_kwargs['vocab_size'] = len(word_to_index)

    if 'num_filters' in signature.parameters:
        num_filters = trial.suggest_int('num_filters', 100, 300, step=2) 
        model_kwargs['num_filters'] = num_filters

    if 'filter_sizes' in signature.parameters:
        filter_sizes = trial.suggest_categorical('filter_sizes', [[2, 3, 4], [3, 4, 5], [4, 5, 6]])
        model_kwargs['filter_sizes'] = filter_sizes

    if 'hidden_dim1' in signature.parameters:
        hidden_dim1 = trial.suggest_int('hidden_dim1', 64, 256)
        model_kwargs['hidden_dim1'] = hidden_dim1

    if 'hidden_dim2' in signature.parameters:
        hidden_dim2 = trial.suggest_int('hidden_dim2', 32, 128)
        model_kwargs['hidden_dim2'] = hidden_dim2

    if 'hidden_dim3' in signature.parameters:
        hidden_dim3 = trial.suggest_int('hidden_dim3', 16, 64)
        model_kwargs['hidden_dim3'] = hidden_dim3

    if 'num_classes' in signature.parameters:
        model_kwargs['num_classes'] = 2

    
    try:
        model = model_class(**model_kwargs)
    except TypeError as e:
        trial.report(-float('inf'), step=0)
        return -float('inf')

    model = model_class(**model_kwargs)
    
    classifier = SentimentClassifier(model=model, learning_rate=learning_rate, config_path=None)
    
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=False)
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
        log_every_n_steps=3,
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
    }
    
    if 'num_layers' in model_kwargs:
        config['num_layers'] = model_kwargs['num_layers']

    if 'dropout_rate' in model_kwargs:
        config['dropout_rate'] = model_kwargs['dropout_rate']

    if 'output_dim' in model_kwargs:
        config['output_dim'] = model_kwargs['output_dim']

    if 'aggregation_method' in model_kwargs:
        config['aggregation_method'] = model_kwargs['aggregation_method']
    
    if 'vocab_size' in model_kwargs:
        config['vocab_size'] = model_kwargs['vocab_size']

    if 'num_filters' in signature.parameters:
        config['num_filters'] = model_kwargs['num_filters']

    if 'filter_sizes' in model_kwargs:
        config['filter_sizes'] = model_kwargs['filter_sizes']

    if 'num_classes' in model_kwargs:
        config['num_classes'] = model_kwargs['num_classes']

    if 'hidden_dim1' in model_kwargs:
        config['hidden_dim1'] = model_kwargs['hidden_dim1']

    if 'hidden_dim2' in model_kwargs:
        config['hidden_dim2'] = model_kwargs['hidden_dim2']
    
    if 'hidden_dim3' in model_kwargs:
        config['hidden_dim3'] = model_kwargs['hidden_dim3']

    if 'hidden_dim' in model_kwargs:
        config['hidden_dim'] = model_kwargs['hidden_dim']
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return val_acc

def tune_hyperparameters(model_class, train_dataloader, val_dataloader, word_to_index, word2vec_model, embedding_matrix, n_trials=50, config_dir='configs'):

    study = optuna.create_study(direction='maximize', 
                                study_name=f'{model_class.__name__}_tuning',  
                                sampler=optuna.samplers.TPESampler(seed=42),
                                )

    print(f'Study name: {study.study_name}')

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
