# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from RESTCORE import REST
from RESTutils import FocalLoss, create_sequences
from SleepDataset import SleepDataset

# %%
# Parameters
fs = 512          # Raw EDF sampling frequency (used during compilation only)
epoch_length = 4  # Epoch length in seconds
window_size = 90  # Window size for sliding window (epochs)
step = 60         # Step size for sliding window
batch_size = 256  # Batch size for training
n_epochs = 100    # Number of training epochs
f_bin = 130       # Feature bins per frame (65 EEG + 65 EMG)
frames = 5        # STFT time frames per epoch
n_classes = 4     # Number of sleep stages (Wake, NREM, REM, Artifact)
WeightedLoss = False
Use_BatchNorm = True # Use 1D Batch Normalization (must be used in Inference as well)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data and model paths
DATA_DIR   = r"D:\Training data V2.0_tensor"
Model_path = r"M:\Alex\Python\REST V1.5\model_artifact.pth"

model = REST(
    in_feat=f_bin,
    n_classes=n_classes,
    win_len=window_size,
    d_model=256,
    nhead=8,
    nlayers_epoch=4,
    nlayers_seq=4,
    ff=512,
    fc_hidden1=128,
    fc_hidden2=64,
    dropout=0.1,
    use_batchnorm=Use_BatchNorm
).to(device)

# %%
# Build lazy-loading datasets (no full-RAM load)
train_ds = SleepDataset(
    data_dir   = DATA_DIR,
    win_len    = window_size,
    step       = step,
    rem_repeat = 2,
    split      = 'train',
    val_split  = 0.2,
    frames     = frames,
    feat       = f_bin,
)
val_ds = SleepDataset(
    data_dir   = DATA_DIR,
    win_len    = window_size,
    step       = step,
    split      = 'val',
    val_split  = 0.2,
    frames     = frames,
    feat       = f_bin,
)

print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

# %%
# Compute class weights from memory-mapped score array (no RAM spike)
# Stored scores are 1-based (1=Wake, 2=NREM, 3=REM, 4=Artifact); shift to 0-based for class_weight
raw_scores  = train_ds.scores
valid_mask  = raw_scores != -100
valid_labels = raw_scores[valid_mask] - 1   # 0=Wake 1=NREM 2=REM 3=Artifact
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(valid_labels),
    y=valid_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Initialize loss function and optimizer
if WeightedLoss:
    criterion = FocalLoss(alpha=class_weights, gamma=2, ignore_index=-100)
else:
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# %%
# Training loop
best_val_accuracy = 0.0
patientce = 0
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
        # batch_X: [B, win_len, frames, feat]  batch_Y: [B, win_len]
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)   # [B, win_len, n_classes]
        loss = criterion(output.view(-1, n_classes), batch_Y.view(-1))
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
            output = model(batch_X)
            loss = criterion(output.view(-1, n_classes), batch_Y.view(-1))
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 2)
            mask = batch_Y != -100
            total   += mask.sum().item()
            correct += (predicted[mask] == batch_Y[mask]).sum().item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    val_accuracy = 100 * correct / max(total, 1)
    print(f"Epoch {epoch+1}/{n_epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), Model_path)
        print(f"New best model saved with accuracy {best_val_accuracy:.2f}%")
    else:
        patientce += 1
        if patientce >= 50:
            print("Early stopping triggered.")
            break
print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")

# %%
model.load_state_dict(torch.load(Model_path, weights_only=True))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_Y in val_loader:
        batch_X = batch_X.to(device)
        output = model(batch_X)
        preds  = torch.argmax(output, dim=2)
        all_preds.append(preds.cpu().view(-1))
        all_targets.append(batch_Y.view(-1))

all_preds   = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

mask        = all_targets != -100
all_preds   = all_preds[mask]
all_targets = all_targets[mask]

print("\nClassification Report (Validation Set):")
print(classification_report(all_targets, all_preds, target_names=["Wake", "NREM", "REM", "Artifact"]))
print("Confusion Matrix:")
print(confusion_matrix(all_targets, all_preds))
