import torch
import torch.nn as nn
import math
# -- Positional encoding ------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):                             # x:[B, L, D]
        return x + self.pe[:, :x.size(1)]
# -- Attention pooling --------------------------------------------------------
class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, 1)

    def forward(self, x):                             # x:[B, L, D]
        w = torch.softmax(self.q(x).squeeze(-1), dim=1)   # [B, L]
        return torch.sum(w.unsqueeze(-1) * x, dim=1)      # [B, D]
    
class REST(nn.Module):
    def __init__(self, in_feat, n_classes, win_len,
                 d_model=256, nhead=8, nlayers_epoch=4, nlayers_seq=4,
                 ff=512, fc_hidden1=128,fc_hidden2=64, dropout=0.1, use_layernorm=True):
        super().__init__()

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.input_norm = nn.LayerNorm(in_feat)

        # Epoch-level encoding
        self.epoch_in_proj = nn.Linear(in_feat, d_model)
        self.epoch_pos_enc = PositionalEncoding(d_model, max_len=10)
        epoch_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.epoch_transformer = nn.TransformerEncoder(epoch_layer, nlayers_epoch)
        self.epoch_pool = AttnPool(d_model)

        # Sequence-level encoding
        self.seq_pos_enc = PositionalEncoding(d_model, max_len=win_len)
        seq_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.seq_transformer = nn.TransformerEncoder(seq_layer, nlayers_seq)

        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden2, n_classes)
        )

    def forward(self, x):  # x: [B, win_len, frames, feat]
        B, W, F, C = x.shape

        # Flatten batch and window
        x = x.view(B * W, F, C)
        
        if self.use_layernorm:
            x = self.input_norm(x)   # [B*W, F, C] — LayerNorm over last dim (C)

        # Epoch-level transformer
        x = self.epoch_in_proj(x)
        x = self.epoch_pos_enc(x)
        x = self.epoch_transformer(x)
        x = self.epoch_pool(x)  # [B*W, d_model]

        # Reshape back to sequence form
        x = x.view(B, W, -1)

        # Sequence-level transformer
        x = self.seq_pos_enc(x)
        x = self.seq_transformer(x)

        # Classification
        logits = self.fc(x)  # [B, W, n_classes]
        return logits