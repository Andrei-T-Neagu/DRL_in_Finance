import torch
import torch.nn as nn

class LSTM_multilayer_cell(nn.Module):
    
    def __init__(self, batch_size, input_size, hidden_size, num_layers, device, dropout=0.0):
        
        super().__init__()

        self.first_lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size) for i in range(num_layers-1)])
        
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

        self.dropout = nn.Dropout(dropout)

        self.batch_size = batch_size
        self.nbs_units = hidden_size
        self.device = device
        self.num_layers = num_layers

    # def init_hidden_state(self):
    #     return [torch.zeros(self.batch_size, self.nbs_units, device=self.device) for i in range(self.num_layers)]
    
    # def init_cell_state(self):
    #     return [torch.zeros(self.batch_size, self.nbs_units, device=self.device) for i in range(self.num_layers)]

    def forward(self, x, hs, cs):
        
        hs[0], cs[0] = self.first_lstm_cell(x, (hs[0], cs[0]))
        hs[0], cs[0] = self.dropout(hs[0]), self.dropout(cs[0])
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                hs[i+1], cs[i+1] = self.lstm_cells[i](hs[i], (hs[i+1], cs[i+1]))
                hs[i+1], cs[i+1] = self.dropout(hs[i+1]), self.dropout(cs[i+1])
            x = self.linear(hs[i+1])
        else:
            x = self.linear(hs[0])
        return x, hs, cs