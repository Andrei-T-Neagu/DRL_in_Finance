import torch
import torch.nn as nn

class FFNN(nn.Module):
    
    def __init__(self, in_features, out_features, num_layers, hidden_size, dropout=0.0):
        
        super().__init__()
        
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.linears = nn.ModuleList([nn.Linear(in_features=hidden_size, out_features=hidden_size) for i in range(num_layers-1)])
        self.final_linear = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.final_linear(x)
        return x