import torch
import torch.nn as nn

class GRU_multilayer_cell(nn.Module):
    
    def __init__(self, batch_size, input_size, hidden_size, num_layers, device, dropout=0.5):
        
        super().__init__()

        self.first_gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        self.gru_cells = nn.ModuleList([nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size) for i in range(num_layers-1)])
        
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

        self.dropout = nn.Dropout(dropout)

        self.batch_size = batch_size
        self.nbs_units = hidden_size
        self.device = device
        self.num_layers = num_layers

    def forward(self, x, hs):
        
        hs[0] = self.first_gru_cell(x, hs[0])
        hs[0] = self.dropout(hs[0])
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                hs[i+1] = self.gru_cells[i](hs[i], hs[i+1])
                hs[i+1] = self.dropout(hs[i+1])
            x = self.linear(hs[i+1])
        else:
            x = self.linear(hs[0])
        return x, hs