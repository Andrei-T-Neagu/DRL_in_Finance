import torch
import torch.nn as nn

class FFNN(nn.Module):
    
    def __init__(self, in_features, out_features, num_layers, hidden_size, dueling=False, log_predicted=False, dropout=0.0):
        
        super().__init__()
        self.num_layers = num_layers
        self.dueling = dueling
        self.log_predicted = log_predicted
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        if num_layers > 2:
            self.linears = nn.ModuleList([nn.Linear(in_features=hidden_size, out_features=hidden_size) for i in range(num_layers-2)])
        if self.dueling:
            self.V = nn.Linear(in_features=hidden_size, out_features=1)
            self.A = nn.Linear(in_features=hidden_size, out_features=out_features)
        else:
            self.final_linear = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.num_layers > 2:
            for linear in self.linears:
                x = linear(x)
                x = self.relu(x)
                x = self.dropout(x)
        if self.dueling:
            V = self.V(x)
            A = self.A(x)
            Q = V + (A - A.mean(dim=1, keepdim=True))
            return Q
        else:
            x = self.final_linear(x)
            if self.log_predicted:
                x = torch.clamp(x, -20, 20)
            return x