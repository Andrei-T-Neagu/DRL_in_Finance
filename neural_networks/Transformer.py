import torch
import torch.nn as nn

class Transformer(nn.Module):
    
    def __init__(self, in_features, seq_length, d_model, dim_feedforward, n_heads, num_layers, dropout=0.0):
        
        super().__init__()

        self.linear1 = nn.Linear(in_features=in_features, out_features=d_model)
        self.positional_encoding = nn.Parameter(torch.rand(seq_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.final_linear = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, x, src_key_padding_mask):
        
        x = self.linear1(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_linear(x)
        x = torch.mean(x, dim=1)
        return x