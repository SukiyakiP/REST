# %%
import os
import math
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
WeightedLoss    = True
Use_LayerNorm   = True
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS   = 5
GRAD_CLIP       = 1.0
NUM_WORKERS     = 0

# Synthetic artifact injection (training split only) — set INJECT_P > 0 to enable
INJECT_P    = 0.0   # disabled: training on real artifacts only
WAKE_SHARE  = 0.8
INJECT_SEED = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data and model paths
DATA_DIR   = r"D:\Training data Extended"
Model_path = r"M:\Alex\Python\REST\model_Updated.pth"

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
    use_layernorm=Use_LayerNorm
).to(device)

# %%
# Build lazy-loading datasets (no full-RAM load)
train_ds = SleepDataset(
    data_dir    = DATA_DIR,
    win_len     = window_size,
    step        = step,
    rem_repeat  = 2,
    split       = 'train',
    val_split   = 0.2,
    frames      = frames,
    feat        = f_bin,
    inject_p    = INJECT_P,
    wake_share  = WAKE_SHARE,
    inject_seed = INJECT_SEED,
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

# Sanity-check injection: pull a few train windows and report class distribution
print(f"\n[Inject] config: p={INJECT_P}, wake_share={WAKE_SHARE}, seed={INJECT_SEED}")
_n_check = 8
_class_counts = np.zeros(4, dtype=np.int64)
_total_eps = 0
for _i in range(_n_check):
    _, _y = train_ds[_i]
    _y_np = _y.numpy()
    for _c in range(4):
        _class_counts[_c] += int((_y_np == _c).sum())
    _total_eps += int((_y_np != -100).sum())
_pct = 100.0 / max(_total_eps, 1)
print(f"[Inject] sample {_n_check} train windows ({_total_eps} valid epochs): "
      f"W={_class_counts[0]} ({_class_counts[0]*_pct:.1f}%), "
      f"N={_class_counts[1]} ({_class_counts[1]*_pct:.1f}%), "
      f"R={_class_counts[2]} ({_class_counts[2]*_pct:.1f}%), "
      f"A={_class_counts[3]} ({_class_counts[3]*_pct:.1f}%)")
print(f"[Inject] expected artifact rate ~{INJECT_P*100:.1f}% + tiny real-artifact baseline\n")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0))
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0))

# %%
# Loss
if WeightedLoss:
    class_weights = torch.tensor([0.6, 1.0, 2.0, 2.0], dtype=torch.float32).to(device)
    FOCAL_GAMMA = 1.0
    print(f"[Loss] FocalLoss(gamma={FOCAL_GAMMA}, label_smoothing={LABEL_SMOOTHING}) "
          f"weights={class_weights.tolist()}")
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA,
                          ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
else:
    print(f"[Loss] CrossEntropyLoss(label_smoothing={LABEL_SMOOTHING})")
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-3)

# LR schedule: linear warmup for WARMUP_EPOCHS, then cosine decay to ~0
def _lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(n_epochs - WARMUP_EPOCHS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

# AMP — no-op when running on CPU
use_amp = device.type == 'cuda'
scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

# %%
# Resume support: full state ('_latest.pth') survives crashes mid-training.
# Best-model file (Model_path) keeps the bare state_dict for Inference.py compatibility.
# Best-art-F1 model (..._artf1.pth) targets artifact detection specifically.
LATEST_PATH = Model_path.replace('.pth', '_latest.pth')
ARTF1_PATH  = Model_path.replace('.pth', '_artf1.pth')
best_val_accuracy = 0.0
best_art_f1       = 0.0
patientce         = 0
start_epoch       = 0

if os.path.exists(LATEST_PATH):
    print(f"[Resume] Loading full checkpoint from {LATEST_PATH}")
    ckpt = torch.load(LATEST_PATH, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch       = ckpt['epoch'] + 1
    best_val_accuracy = ckpt['best_val_acc']
    best_art_f1       = ckpt.get('best_art_f1', 0.0)
    patientce         = ckpt.get('patience', 0)
    if 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    if 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    print(f"[Resume] epoch {start_epoch+1}/{n_epochs}, "
          f"best_val_acc={best_val_accuracy:.2f}%, best_art_f1={best_art_f1:.3f}, "
          f"patience={patientce}")
elif os.path.exists(Model_path):
    print(f"[Resume] Warm-starting weights from {Model_path} (optimizer/scheduler reset)")
    model.load_state_dict(torch.load(Model_path, weights_only=True, map_location=device))

# %%
# Training loop
for epoch in range(start_epoch, n_epochs):
    current_lr = optimizer.param_groups[0]['lr']
    model.train()
    train_loss = 0.0
    for batch_X, batch_Y in tqdm(train_loader,
                                  desc=f"Epoch {epoch+1}/{n_epochs} [lr={current_lr:.2e}]",
                                  mininterval=30, file=__import__('sys').stderr):
        batch_X = batch_X.to(device, non_blocking=True)
        batch_Y = batch_Y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(batch_X)
            loss   = criterion(output.view(-1, n_classes), batch_Y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct  = 0
    total    = 0
    pred_per_class = np.zeros(4, dtype=np.int64)
    true_per_class = np.zeros(4, dtype=np.int64)
    tp_per_class   = np.zeros(4, dtype=np.int64)
    with torch.no_grad():
        for batch_X, batch_Y in tqdm(val_loader,
                                      desc=f"Epoch {epoch+1}/{n_epochs} - Validation",
                                      mininterval=30, file=__import__('sys').stderr):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch_X)
                loss   = criterion(output.view(-1, n_classes), batch_Y.view(-1))
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 2)
            mask    = batch_Y != -100
            total   += mask.sum().item()
            correct += (predicted[mask] == batch_Y[mask]).sum().item()
            p_np = predicted[mask].cpu().numpy()
            y_np = batch_Y[mask].cpu().numpy()
            for _c in range(4):
                pred_per_class[_c] += int((p_np == _c).sum())
                true_per_class[_c] += int((y_np == _c).sum())
                tp_per_class[_c]   += int(((p_np == _c) & (y_np == _c)).sum())

    train_loss  /= len(train_loader)
    val_loss    /= len(val_loader)
    val_accuracy = 100 * correct / max(total, 1)
    print(f"Epoch {epoch+1}/{n_epochs}  lr={current_lr:.2e}  "
          f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_accuracy:.2f}%")
    art_p = art_r = art_f1 = 0.0
    for _c, _name in enumerate(["Wake", "NREM", "REM ", "Art "]):
        _p  = tp_per_class[_c] / max(pred_per_class[_c], 1)
        _r  = tp_per_class[_c] / max(true_per_class[_c], 1)
        _f1 = 2 * _p * _r / max(_p + _r, 1e-8)
        if _c == 3:
            art_p, art_r, art_f1 = _p, _r, _f1
        print(f"  {_name}: pred={pred_per_class[_c]:>7d} true={true_per_class[_c]:>7d} "
              f"TP={tp_per_class[_c]:>7d} P={_p:.3f} R={_r:.3f} F1={_f1:.3f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), Model_path)
        print(f"New best val_acc model saved: {best_val_accuracy:.2f}%")
        patientce = 0
    else:
        patientce += 1
        if patientce >= 50:
            print("Early stopping triggered.")
            break

    if art_f1 > best_art_f1:
        best_art_f1 = art_f1
        torch.save(model.state_dict(), ARTF1_PATH)
        print(f"New best art_F1 model saved: {best_art_f1:.3f} (P={art_p:.3f} R={art_r:.3f})")

    torch.save({
        'epoch':        epoch,
        'model':        model.state_dict(),
        'optimizer':    optimizer.state_dict(),
        'scheduler':    scheduler.state_dict(),
        'scaler':       scaler.state_dict(),
        'best_val_acc': best_val_accuracy,
        'best_art_f1':  best_art_f1,
        'patience':     patientce,
    }, LATEST_PATH)

print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")

# %%
model.load_state_dict(torch.load(Model_path, weights_only=True))
model.eval()

all_preds   = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_Y in val_loader:
        batch_X = batch_X.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(batch_X)
        preds = torch.argmax(output, dim=2)
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
