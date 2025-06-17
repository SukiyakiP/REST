# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import  DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,cohen_kappa_score,accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

# %%
model_path = r''   # Path to the trained model (pth file)
ds_path = r''   # Path to the test dataset (npz file)

# %%
# Parameters
fs = 512  # Sampling frequency
epoch_length = 4  # Epoch length in seconds
nperseg = 256  # Segment length for PSD computation
sequence_length = 90 # Number of epochs in a sequence (for Transformer model) 15 epochs/minute, change to larger number for longer temporal context
window_size=sequence_length
step=30 # Step size for creating sequences
batch_size = 128  # Batch size for training
n_classes = 3   # Number of sleep stages (e.g., Wake, NREM, REM)
f_bin=130 # Frequency bin for PSD computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Reshape features into sequences
def create_sequences(data, labels, window_size, step): # data: [n_samples, n_features], labels: [n_samples]
    X, y = [], []
    max_start = len(data) - window_size + 1 # last start index for a whole window
    for start in range(0, max_start, step): # step through data
        end = start + window_size # end index for current window
        X.append(data[start:end]) # shape [n_windows, window_size, n_features]
        y.append(labels[start:end]) # shape [n_windows, window_size]
    return np.array(X), np.array(y) # shape [n_windows, window_size, n_features], [n_windows, window_size]

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
    
class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, 1)

    def forward(self, x):                             # x:[B, L, D]
        w = torch.softmax(self.q(x).squeeze(-1), dim=1)   # [B, L]
        return torch.sum(w.unsqueeze(-1) * x, dim=1)      # [B, D]

# -- Epoch‑level encoder ------------------------------------------------------
class EpochEncoder(nn.Module):
    def __init__(self, in_feat, d_model=128, nhead=8, nlayers=2, ff=256, dropout=0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_feat, d_model)
        self.pos_enc  = PositionalEncoding(d_model, max_len=10)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc      = nn.TransformerEncoder(encoder_layer, nlayers)
        self.pool     = AttnPool(d_model)

    def forward(self, x):                 # x:[B, frames, feat]
        x = self.in_proj(x)
        x = self.pos_enc(x)
        x = self.enc(x)                   # [B, frames, d_model]
        return self.pool(x)               # [B, d_model]

# -- Sequence‑level model -----------------------------------------------------
class SleepTransformer(nn.Module):
    def __init__(self, in_feat, n_classes, win_len,    # win_len = window_size
                 d_model=256, nhead=8, nlayers_epoch=4, nlayers_seq=4,
                 ff=512, fc_hidden=64, dropout=0.1):
        super().__init__()
        self.epoch_encoder = EpochEncoder(in_feat, d_model, nhead,
                                          nlayers_epoch, ff, dropout)

        self.pos_enc_seq   = PositionalEncoding(d_model, max_len=win_len)
        seq_layer          = nn.TransformerEncoderLayer(d_model, nhead,
                                                        ff, dropout, batch_first=True)
        self.seq_encoder   = nn.TransformerEncoder(seq_layer, nlayers_seq)

        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes)
        )

    def forward(self, x):                 # x:[B, win_len, frames, feat]
        B, W, F, C = x.shape
        x = x.view(B * W, F, C)           # merge batch & window
        epoch_vec = self.epoch_encoder(x) # [B*W, d_model]
        epoch_vec = epoch_vec.view(B, W, -1)

        x = self.pos_enc_seq(epoch_vec)
        x = self.seq_encoder(x)           # [B, W, d_model]

        logits = self.fc(x)               # [B, W, n_classes]
        return logits

# %%
arr = np.load(ds_path)
EEG=arr['EEG']
EMG=arr['EMG']
score=arr['score']-1
score[score > 2] = 0             # collapse stage “?” to Wake
score = score.astype(np.int64)

# EEG = EEG.transpose(0, 2, 1)   # → (151179, 5, 65)
# EMG = EMG.transpose(0, 2, 1)   # → (151179, 5, 65)

# Concatenate along feature dimension  → [n_epochs, frames=5, feat=65*2]
epoch_tensor = np.concatenate([EEG, EMG], axis=-1).astype(np.float32)
labels = score                                              # [n_epochs]

# Build sliding windows exactly like before
X, Y = create_sequences(epoch_tensor, labels, window_size, step)  # X:[n_win, win, 5, feat]
sequences_tensor = torch.tensor(X, dtype=torch.float32).to(device)
sequences_batch=DataLoader(sequences_tensor, batch_size=batch_size, shuffle=False)

# %%
 # Path to the trained model
model = SleepTransformer(f_bin, n_classes,window_size).to(device)
model.load_state_dict(torch.load(model_path))  # Load the trained weights
model.to(device)  # Move the model to the GPU
model.eval()  # Set the model to evaluation mode
all_preds=[]
with torch.no_grad():
    for batch_X in sequences_batch:
        batch_X= batch_X.to(device)
        output = model(batch_X)  # Shape: [batch_size, sequence_length, n_classes]
        predicted = torch.argmax(output.data, 2)  # Shape: [batch_size, sequence_length] 
        first_epoch_preds = predicted[:,:step].cpu().numpy()
        all_preds.append(first_epoch_preds)
predictions = np.concatenate(all_preds, axis=0).flatten() + 1  # Adjust labels if needed


# %%
predicted_score = np.array(predictions)
Reference_score=score+1
Reference_score[Reference_score==4] = 1
reference_score = Reference_score[:len(predicted_score)]

# %%
plt.hist(predicted_score, bins=np.arange(6) - 0.5, edgecolor='black')
plt.xticks(range(5))
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Histogram of Predictions')
plt.show()

plt.hist(reference_score, bins=np.arange(6) - 0.5, edgecolor='black')
plt.xticks(range(5))
plt.xlabel('Reference Class')
plt.ylabel('Frequency')
plt.title('Histogram of Reference')
plt.show()
# Compute the confusion matrix
labels = [1, 2, 3]
cm = confusion_matrix(reference_score, predicted_score, labels=labels)
kappa = cohen_kappa_score(reference_score, predicted_score)

print(f"\nCohen's Kappa: {kappa:.3f}")

# Optional: heatmap plot
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Wake", "NREM", "REM"],
            yticklabels=["Wake", "NREM", "REM"])
plt.xlabel("REST Prediction")
plt.ylabel("Manual Score")
plt.title("Confusion Matrix (Manual vs Transformer)")
plt.tight_layout()
plt.savefig("confusion_matrix_manual_VS_transformer.tiff", dpi=600, format="tiff")
plt.show()
accuracy = accuracy_score(reference_score, predicted_score)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(reference_score, predicted_score, target_names=["Wake", "NREM", "REM"]))


# %%
plt.figure(figsize=(15, 5))
# plt.step(range(1800), predicted_score[:1800], where='mid', label='Predicted Labels')
plt.step(range(len(reference_score)), reference_score, where='mid', label='Reference Labels')
plt.xlabel('Epoch')
plt.ylabel('Reference Label')
plt.title('Reference Labels as Stairs Plot')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(15, 5))
# plt.step(range(1800), predicted_score[:1800], where='mid', label='Predicted Labels')
plt.step(range(len(predicted_score)), predicted_score, where='mid', label='Predicted Labels')
plt.xlabel('Epoch')
plt.ylabel('Predicted Label')
plt.title('Predicted Labels as Stairs Plot')
plt.legend()
plt.show()


