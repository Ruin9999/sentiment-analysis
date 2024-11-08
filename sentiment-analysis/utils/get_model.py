from models.rnn import SentimentRNN
from models.bilstm import SentimentBiLSTM
from models.bigru import SentimentBiGRU
from models.improved_cnn import ImprovedSentimentCNN
import json 

def get_model_class(model_name):
    model_mapping = {
        'rnn': SentimentRNN,
        'rnn_freeze': SentimentRNN,
        'bilstm': SentimentBiLSTM,
        'bigru': SentimentBiGRU,
        'cnn': ImprovedSentimentCNN,
    }
    return model_mapping.get(model_name.lower(), SentimentRNN)  # Default to SentimentRNN if not found


def get_weights_path(model_name):
    weight_mapping = {
        'rnn': '/p/scratch/ccstdl/xu17/jz/code/sentiment-analysis/checkpoints/final_model/SentimentRNN-FinalModel-val_accval_acc=0.78.ckpt',
        'rnn_freeze': '/p/scratch/ccstdl/xu17/jz/code/sentiment-analysis/checkpoints/final_model/SentimentRNN-Freeze-FinalModel-val_accval_acc=0.62.ckpt',
        'bilstm': '/p/scratch/ccstdl/xu17/jz/code/sentiment-analysis/checkpoints/final_model/SentimentBiLSTM-FinalModel-val_accval_acc=0.76-v1.ckpt',
        'bigru': '/p/scratch/ccstdl/xu17/jz/code/sentiment-analysis/checkpoints/final_model/SentimentBiGRU-FinalModel-val_accval_acc=0.76-v1.ckpt',
        # 'cnn': ImprovedSentimentCNN,
    }
    return weight_mapping.get(model_name.lower()) 


def get_model_config(model_name):

    config_mapping = {
        'rnn': '/p/scratch/ccstdl/xu17/jz/code/sentiment-analysis/configs/SentimentRNN_unfreeze_tuning/SentimentRNN_final_config.json',
        'rnn_freeze': '/p/scratch/ccstdl/xu17/jz/code/sentiment-analysis/configs/SentimentRNN_freeze_tuning/SentimentRNN-Freeze_final_config.json',
        'bilstm': '/configs/SentimentBiLSTM_final_config.json',
        'bigru': '/configs/SentimentBiGRU_final_config.json',
        # 'cnn': '/configs/ImprovedSentimentCNN_final_config.json',
    }

    with open(config_mapping.get(model_name.lower()), 'r') as f:
        return json.load(f)
  