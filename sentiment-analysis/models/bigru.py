
import torch
import torch.nn as nn

class SentimentBiGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, pad_idx, embedding_matrix, 
                 freeze_embeddings=True, dropout_rate=0.5, num_layers=1):
        super(SentimentBiGRU, self).__init__()
        embedding_tensor = torch.FloatTensor(embedding_matrix)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, padding_idx=pad_idx, freeze=freeze_embeddings)
        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          bidirectional=True, 
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        gru_output, hidden = self.gru(embedded)
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden_cat = self.dropout(hidden_cat)
        output = self.fc(hidden_cat)
        return output
