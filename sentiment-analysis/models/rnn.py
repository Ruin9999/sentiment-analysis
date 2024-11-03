
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, embedding_matrix, 
                 freeze_embeddings=True, aggregation_method='max_pooling', dropout_rate=0.2):
        super(SentimentRNN, self).__init__()
        embedding_tensor = torch.FloatTensor(embedding_matrix)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, padding_idx=pad_idx, freeze=freeze_embeddings)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.Linear(hidden_dim, 1)
        self.aggregation_method = aggregation_method

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        
        if self.aggregation_method == 'last_hidden':
            sentence_repr = hidden.squeeze(0)
        elif self.aggregation_method == 'last_output':
            sentence_repr = output[:, -1, :]
        elif self.aggregation_method == 'mean_pooling':
            sentence_repr = torch.mean(output, dim=1)
        elif self.aggregation_method == 'max_pooling':
            sentence_repr, _ = torch.max(output, dim=1)
        elif self.aggregation_method == 'attention':
            attention_weights = F.softmax(self.attention(output), dim=1)
            sentence_repr = torch.sum(attention_weights * output, dim=1)
        
        sentence_repr = self.dropout(sentence_repr)
        return self.fc(sentence_repr)
