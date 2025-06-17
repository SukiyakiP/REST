# %% [markdown]
# This version is stable, the model_V8 has an accuracy of 92.8%.

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader, TensorDataset,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import math
import gc
from tqdm import tqdm

# %%
# Parameters
fs = 512  # Sampling frequency
epoch_length = 4  # Epoch length in seconds
window_size = 90 # Window size for sliding window
step = 30 # Step size for sliding window
nperseg = 256  # Segment length for PSD computation
batch_size = 128 # Batch size for training
n_epochs = 100  # Number of training epochs
f_bin=130 # Frequency bin for PSD computation
n_classes = 3   # Number of sleep stages (e.g., Wake, NREM, REM)
WeightedLoss = True # Use weighted loss function
ds_path = r'' # Path to the training dataset
model_path = r'' # Path to save the model

arr = np.load(ds_path)
EEG = arr["EEG"]     # shape: [n_epochs,   256 * 4]  (down‑sampled to 64 Hz)
EMG = arr["EMG"]     # shape: [n_epochs, 1024 * 4]  (down‑sampled to 256 Hz)
score = arr["score"] - 1
score[score > 2] = 0             # collapse stage “?” to Wake
score = score.astype(np.int64) #wake=0,NREM=1,REM=2
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

def get_oversampled_indices(Y, rem_label=2, repeat_factor=3):
    """
    Returns a list of indices for oversampling REM-containing sequences.
    
    Arguments:
    - Y: ndarray shape [n_sequences, win_len], your training label matrix.
    - rem_label: int, the REM class label.
    - repeat_factor: int, how many times to upsample REM sequences.
    
    Returns:
    - oversampled_indices: shuffled array of indices (with REM sequences repeated)
    """
    # Find REM-containing sequences
    rem_mask = np.any(Y == rem_label, axis=1)
    rem_indices = np.where(rem_mask)[0]
    non_rem_indices = np.where(~rem_mask)[0]

    # Repeat REM indices only
    repeated_rem_indices = np.tile(rem_indices, repeat_factor)

    # Combine and shuffle
    all_indices = np.concatenate([non_rem_indices, repeated_rem_indices])
    np.random.shuffle(all_indices)

    return all_indices


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        alpha: tensor of shape [n_classes] — class weights (optional)
        gamma: focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Can be class weights (like from compute_class_weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits
        # targets: [B] labels
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')  # [B]
        pt = torch.exp(-ce_loss)  # pt = probability of the correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # [B]

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
# Concatenate along feature dimension  → [n_epochs, frames=5, feat=65*2]
epoch_tensor = np.concatenate([EEG, EMG], axis=-1).astype(np.float32)
labels = score # [n_epochs]

# Build sliding windows exactly like before
X, Y = create_sequences(epoch_tensor, labels, window_size, step)  # X:[n_win, win, 5, feat]
del EEG, EMG ,epoch_tensor

# %%
# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
del X, Y

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_val = torch.tensor(Y_val, dtype=torch.long)

# Create DataLoader for training and validation
train_dataset = TensorDataset(X_train, Y_train)
oversampled_idx = get_oversampled_indices(Y_train.numpy(), repeat_factor=3)
oversampled_dataset = Subset(train_dataset, oversampled_idx)

train_loader = DataLoader(
    oversampled_dataset,
    batch_size=batch_size,
    shuffle=True,  # still shuffle across the repeated indices
)

val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# %%
train_labels_flat = Y_train.view(-1).cpu().numpy()
del X_train, X_val, Y_train, Y_val
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_flat), y=train_labels_flat)
# Convert class weights to a PyTorch tensor and move to the GPU
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Initialize model, loss function, and optimizer
model = SleepTransformer(f_bin, n_classes,window_size).to(device)

if WeightedLoss:
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = FocalLoss(alpha=class_weights, gamma=2)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# %%
# Training loop
best_val_accuracy = 0.0
patientce = 0
counter = 0
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)  # Shape: [batch_size, sequence_length, n_classes]
        loss = criterion(output.view(-1, n_classes), batch_Y.view(-1))  # Flatten for loss computation
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_Y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation"):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            output = model(batch_X)  # Shape: [batch_size, sequence_length, n_classes]
            loss = criterion(output.view(-1, n_classes), batch_Y.view(-1))  # Flatten for loss computation
            val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(output.data, 2)  # Shape: [batch_size, sequence_length] 
            total += batch_Y.size(0) * batch_Y.size(1)  # Total number of predictions
            correct += (predicted == batch_Y).sum().item()  # Correct predictions

    # Print epoch results
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{n_epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), model_path)
        print(f"New best model saved with accuracy {best_val_accuracy:.2f}%")
    else:
        patience += 1
        if patience >= 50:
            print("Early stopping triggered.")
            break
print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")
