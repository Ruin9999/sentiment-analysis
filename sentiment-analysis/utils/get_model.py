from models.rnn import SentimentRNN
from models.bilstm import SentimentBiLSTM
from models.bigru import SentimentBiGRU
from models.improved_cnn import ImprovedSentimentCNN
import json


def get_model_class(model_name):
    model_mapping = {
        "rnn": SentimentRNN,
        "rnn_freeze": SentimentRNN,
        "bilstm": SentimentBiLSTM,
        "bigru": SentimentBiGRU,
        "cnn": ImprovedSentimentCNN,
    }
    return model_mapping.get(
        model_name.lower(), SentimentRNN
    )  # Default to SentimentRNN if not found


def get_weights_path(model_name):
    weight_mapping = {
        "rnn": "../best_models/final_model/SentimentRNN-FinalModel-val_accval_acc=0.78.ckpt",
        "rnn_freeze": "../best_models/final_model/SentimentRNN-Freeze-FinalModel-val_accval_acc=0.62.ckpt",
        "bilstm": "../best_models/final_model/SentimentBiLSTM-FinalModel-val_accval_acc=0.76-v1.ckpt",
        "bigru": "../best_models/final_model/SentimentBiGRU-FinalModel-val_accval_acc=0.76-v1.ckpt",
        "cnn": "../best_models/final_model/cnn-epoch=08-val_acc=0.76.ckpt",
    }
    return weight_mapping.get(model_name.lower())


def get_model_config(model_name):

    config_mapping = {
        "rnn": "../best_models/final_config/SentimentRNN_final_config.json",
        "rnn_freeze": "../best_models/final_config/SentimentRNN-Freeze_final_config.json",
        "bilstm": "../best_models/final_config/SentimentBiLSTM_final_config.json",
        "bigru": "../final_config/SentimentBiGRU_final_config.json",
        "cnn": "../best_models/final_config/ImprovedSentimentCNN_final_config.json",
    }

    with open(config_mapping.get(model_name.lower()), "r") as f:
        return json.load(f)
