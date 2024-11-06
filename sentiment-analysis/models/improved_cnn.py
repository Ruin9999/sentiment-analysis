
import torch
import torch.nn as nn

class ImprovedSentimentCNN(nn.Module):
    def __init__(self, embedding_dim, 
                 embedding_matrix, 
                 pad_idx, 
                 num_classes, 
                 num_filters, 
                 filter_sizes, 
                 dropout=0.5, 
                 freeze_embeddings=True, 
                 hidden_dim1=128, 
                 hidden_dim2=64, 
                 hidden_dim3=32):
        
        super(ImprovedSentimentCNN, self).__init__()
        
        embedding_tensor = torch.FloatTensor(embedding_matrix)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, padding_idx=pad_idx, freeze=freeze_embeddings)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.bn = nn.BatchNorm1d(num_filters)
        self.attention = nn.MultiheadAttention(embed_dim=num_filters * len(filter_sizes), num_heads=2, dropout=dropout)  
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)     # [batch_size, 1, seq_len, embedding_dim]
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x)).squeeze(3)  # [batch_size, num_filters, seq_len - filter_size +1]
            pool_out = torch.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)  # [batch_size, num_filters]
            conv_outputs.append(pool_out)
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
