from models.rnn import SentimentRNN
from models.bilstm import SentimentBiLSTM
from models.bigru import SentimentBiGRU
from models.improved_cnn import ImprovedSentimentCNN

def get_model_class(model_name):
    model_mapping = {
        'rnn': SentimentRNN,
        'bilstm': SentimentBiLSTM,
        'bigru': SentimentBiGRU,
        'cnn': ImprovedSentimentCNN,
    }
    return model_mapping.get(model_name.lower(), SentimentRNN)  # Default to SentimentRNN if not found
