"""
train_lstm.py  —  LSTM Bias Predictor for RL-Adaptive EKF  (v4 Final)
======================================================================
For: LSTM + RL-Adaptive EKF Localization Project

What changed from v3 (the only two changes)
--------------------------------------------
  v4-FIX 1 — TARGET is now BIAS, not absolute acceleration.
              bias_fwd = gt_accel_fwd_mps2 - ax_corr
              bias_lat = gt_accel_lat_mps2 - ay_corr
              Computed in load_and_clean() after input clipping.

  v4-FIX 2 — output_is_bias=True saved in checkpoint and stats file
              so ekf.py can assert it is loading the right model and
              apply output as: a_fwd = ax_corr + lstm_output.

Why bias is better than absolute acceleration
---------------------------------------------
  Training on absolute acceleration:
      LSTM learns the full motion signal, then substitutes it into EKF.
      When LSTM is uncertain, a bad prediction replaces the IMU entirely.

  Training on bias (gt - imu):
      LSTM learns specifically what the IMU gets wrong.
      When LSTM is uncertain, bias prediction ≈ 0 → IMU dominates safely.
      The IMU reading provides a physical floor that bounds worst-case error.

  Both approaches produce the same result when LSTM is accurate.
  Bias approach is significantly safer when LSTM is uncertain (e.g.
  first steps in a new tunnel before buffer fills).

Application in ekf.py (v4):
    a_fwd = ax_corr + lstm_bias      ← additive correction
    NOT: a_fwd = lstm_output         ← replacement (v3, now removed)

Architecture (unchanged from v3):
    Input(5) → LayerNorm → LSTM(64) → LN → Dropout(0.3)
             → LSTM(32)  → LN → Dropout(0.3) → last step
             → Linear(32→16) → GELU → Linear(16→2)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# =============================================================================
# PATHS
# =============================================================================
DATA_PATH    = r'C:\Users\heman\Music\rl_imu_project\data\town04_dataset.csv'
MODEL_DIR    = r'C:\Users\heman\Music\rl_imu_project\models'
RESULTS_DIR  = r'C:\Users\heman\Music\rl_imu_project\results'
MODEL_PATH   = os.path.join(MODEL_DIR,   'lstm_drift_predictor.pth')
STATS_PATH   = os.path.join(MODEL_DIR,   'lstm_normalisation.npz')
PLOT_PATH    = os.path.join(RESULTS_DIR, 'lstm_training.png')
METRICS_PATH = os.path.join(RESULTS_DIR, 'lstm_metrics.txt')


# =============================================================================
# CONFIGURATION
# =============================================================================
SEQ_LEN  = 40
STRIDE   = 2
FEATURE_COLS = ['ax_corr', 'ay_corr', 'wz', 'gt_speed_mps', 'gps_denied']

# v4-FIX 1: bias targets (computed in DataPreprocessor.load_and_clean)
TARGET_COLS  = ['bias_fwd', 'bias_lat']

TRAIN_RUNS = [0, 1, 2]
VAL_RUNS   = [3]
HIDDEN1    = 64
HIDDEN2    = 32
DROPOUT    = 0.3

BATCH_SIZE    = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 150
PATIENCE      = 20
TUNNEL_WEIGHT = 2.0
GRAD_CLIP     = 1.0
LR_PATIENCE   = 8
LR_FACTOR     = 0.5
LR_MIN        = 1e-6

INPUT_CLIP = {
    'ax_corr':      (-20.0, 20.0),
    'ay_corr':      (-15.0, 15.0),
    'wz':           (-5.0,   5.0),
    'gt_speed_mps': (0.0,   35.0),
    'gps_denied':   (0.0,    1.0),
}
# Bias values are much smaller than raw acceleration — typically ±3-5 m/s²
BIAS_CLIP = (-8.0, 8.0)


# =============================================================================
# DATA PREPROCESSOR
# =============================================================================
class DataPreprocessor:
    def __init__(self):
        self.feat_mean = None
        self.feat_std  = None
        self.tgt_mean  = None
        self.tgt_std   = None

    def load_and_clean(self, path):
        df = pd.read_csv(path)
        n0 = len(df)
        needed = ['ax_corr', 'ay_corr', 'wz', 'gt_speed_mps', 'gps_denied',
                  'gt_accel_fwd_mps2', 'gt_accel_lat_mps2', 'run_id', 'timestamp']
        df = df.dropna(subset=needed).reset_index(drop=True)
        if len(df) < n0:
            print(f"  Dropped {n0 - len(df)} NaN rows")

        for col, (lo, hi) in INPUT_CLIP.items():
            before = df[col].abs().max()
            df[col] = df[col].clip(lo, hi)
            if before > hi * 1.01:
                print(f"  Clipped {col}: ±{before:.2f} → ±{df[col].abs().max():.2f}")

        # v4-FIX 1: compute bias columns after input clipping
        # bias = what the IMU is missing = truth - reported
        df['bias_fwd'] = (df['gt_accel_fwd_mps2'] - df['ax_corr']).clip(*BIAS_CLIP)
        df['bias_lat'] = (df['gt_accel_lat_mps2'] - df['ay_corr']).clip(*BIAS_CLIP)

        print(f"  Bias stats (should be small, ~0 mean):")
        print(f"    fwd: mean={df['bias_fwd'].mean():.4f}  std={df['bias_fwd'].std():.4f}  "
              f"max|bias|={df['bias_fwd'].abs().max():.4f}")
        print(f"    lat: mean={df['bias_lat'].mean():.4f}  std={df['bias_lat'].std():.4f}  "
              f"max|bias|={df['bias_lat'].abs().max():.4f}")
        print(f"  Loaded {len(df):,} rows | tunnel={int(df['gps_denied'].sum()):,} "
              f"({100*df['gps_denied'].mean():.1f}%)")
        return df

    def fit(self, df_train):
        norm_feat_cols = [c for c in FEATURE_COLS if c != 'gps_denied']
        self.feat_mean = df_train[norm_feat_cols].mean().values.astype(np.float32)
        self.feat_std  = df_train[norm_feat_cols].std().values.astype(np.float32)
        self.feat_std  = np.where(self.feat_std < 1e-6, 1.0, self.feat_std)
        self.tgt_mean  = df_train[TARGET_COLS].mean().values.astype(np.float32)
        self.tgt_std   = df_train[TARGET_COLS].std().values.astype(np.float32)
        self.tgt_std   = np.where(self.tgt_std < 1e-6, 1.0, self.tgt_std)
        print(f"  Feature means: {dict(zip([c for c in FEATURE_COLS if c!='gps_denied'], self.feat_mean.round(4)))}")
        print(f"  Bias tgt mean: {dict(zip(TARGET_COLS, self.tgt_mean.round(4)))}")
        print(f"  Bias tgt std : {dict(zip(TARGET_COLS, self.tgt_std.round(4)))}  (much smaller than accel ~2-4)")

    def transform_features(self, df):
        out = df.copy()
        norm_feat_cols = [c for c in FEATURE_COLS if c != 'gps_denied']
        out[norm_feat_cols] = (df[norm_feat_cols].values.astype(np.float32) - self.feat_mean) / self.feat_std
        return out

    def normalise_targets(self, arr):
        return (arr - self.tgt_mean) / self.tgt_std

    def denormalise_targets(self, arr):
        return arr * self.tgt_std + self.tgt_mean

    def save(self, path):
        np.savez(
            path,
            feat_mean      = self.feat_mean,
            feat_std       = self.feat_std,
            tgt_mean       = self.tgt_mean,
            tgt_std        = self.tgt_std,
            feature_cols   = np.array(FEATURE_COLS),
            target_cols    = np.array(TARGET_COLS),
            seq_len        = np.array([SEQ_LEN]),
            seed           = np.array([SEED]),
            output_is_bias = np.array([True]),   # v4 flag: ekf.py uses a_fwd = imu + output
        )
        print(f"  Stats saved → {path}  (output_is_bias=True)")


# =============================================================================
# DATASET  (unchanged from v3 except pulls bias columns as targets)
# =============================================================================
class IMUSequenceDataset(Dataset):
    def __init__(self, df_norm, preprocessor, stride=1):
        features    = df_norm[FEATURE_COLS].values.astype(np.float32)
        targets_raw = df_norm[TARGET_COLS].values.astype(np.float32)   # bias columns
        targets     = preprocessor.normalise_targets(targets_raw).astype(np.float32)
        run_ids     = df_norm['run_id'].values
        gps_denied  = df_norm['gps_denied'].values.astype(np.float32)

        seqs, tgts, wgts = [], [], []
        n = len(features)
        for i in range(0, n - SEQ_LEN, stride):
            end = i + SEQ_LEN
            if end >= n:                         continue
            if run_ids[i] != run_ids[end]:       continue   # boundary check on TARGET step
            x = features[i:end]
            y = targets[end]
            w = TUNNEL_WEIGHT if gps_denied[i:end].any() else 1.0
            seqs.append(x); tgts.append(y); wgts.append(w)

        self.X = np.array(seqs, dtype=np.float32)
        self.Y = np.array(tgts, dtype=np.float32)
        self.W = np.array(wgts, dtype=np.float32)
        n_tun = (self.W > 1.0).sum()
        print(f"    {len(self.X):,} sequences | tunnel: {n_tun:,} ({100*n_tun/max(len(self.X),1):.1f}%)")

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]),
                torch.from_numpy(self.Y[idx]),
                torch.tensor(self.W[idx], dtype=torch.float32))


# =============================================================================
# MODEL  (architecture unchanged from v3 — output semantics changed to bias)
# =============================================================================
class LSTMDriftPredictor(nn.Module):
    """
    Same architecture as v3. Output is now IMU bias (m/s²) not absolute accel.
    EKF applies as: a_fwd = ax_corr + model_output  (additive correction).
    """
    def __init__(self, input_size=5, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm1    = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1      = nn.LayerNorm(h1);  self.drop1 = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2      = nn.LayerNorm(h2);  self.drop2 = nn.Dropout(dropout)
        self.head     = nn.Sequential(nn.Linear(h2,16), nn.GELU(), nn.Linear(16, len(TARGET_COLS)))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:  nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name: nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p); n = p.size(0)
                p.data[n//4:n//2].fill_(1.0)

    def forward(self, x):
        x = self.input_ln(x)
        o1, _ = self.lstm1(x);  o1 = self.ln1(o1);  o1 = self.drop1(o1)
        o2, _ = self.lstm2(o1); o2 = self.ln2(o2);  o2 = self.drop2(o2)
        return self.head(o2[:, -1, :])


# =============================================================================
# LOSS / TRAINING / EVALUATION  (unchanged from v3)
# =============================================================================
class WeightedMSELoss(nn.Module):
    def forward(self, pred, target, weights):
        return (((pred - target)**2).mean(dim=1) * weights).mean()

def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_gnorm = n_batches = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            if training: optimizer.zero_grad()
            pred = model(x); loss = criterion(pred, y, w)
            if training:
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step(); total_gnorm += gnorm.item()
            total_loss += loss.item() * len(x); n_batches += 1
    return total_loss / len(loader.dataset), total_gnorm / max(n_batches, 1)

def evaluate(model, loader, preprocessor, device):
    model.eval()
    all_pred, all_true, all_tun = [], [], []
    with torch.no_grad():
        for x, y, w in loader:
            all_pred.append(model(x.to(device)).cpu().numpy())
            all_true.append(y.numpy())
            all_tun.append(w.numpy() > 1.0)
    pred_n = np.vstack(all_pred); true_n = np.vstack(all_true); is_tun = np.concatenate(all_tun)
    pred_p = preprocessor.denormalise_targets(pred_n)
    true_p = preprocessor.denormalise_targets(true_n)
    def _m(p, t, label):
        return {f'{label}_MAE': float(np.mean(np.abs(p-t))),
                f'{label}_RMSE': float(np.sqrt(np.mean((p-t)**2))),
                f'{label}_R2': float(1 - np.sum((p-t)**2) / (np.sum((t-t.mean())**2)+1e-8))}
    metrics = {}
    for i, col in enumerate(['fwd_bias', 'lat_bias']):
        metrics.update(_m(pred_p[:,i], true_p[:,i], f'{col}_all'))
        if is_tun.sum() > 0:
            metrics.update(_m(pred_p[is_tun,i], true_p[is_tun,i], f'{col}_tunnel'))
    return metrics, pred_p, true_p, is_tun

def plot_results(train_losses, val_losses, pred_p, true_p, is_tun, save_path):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("LSTM Bias Predictor — Training Results (v4)", fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)
    ax_loss = fig.add_subplot(gs[0,0]); ax_fwd = fig.add_subplot(gs[0,1])
    ax_ts   = fig.add_subplot(gs[1,0]); ax_lat = fig.add_subplot(gs[1,1])
    ax_err  = fig.add_subplot(gs[2,0])
    ep = range(1, len(train_losses)+1)
    ax_loss.plot(ep, train_losses, 'b-', lw=1.5, label='Train')
    ax_loss.plot(ep, val_losses,   'r-', lw=1.5, label='Val')
    ax_loss.set_yscale('log'); ax_loss.set_title("Loss (log)"); ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    idx = np.random.choice(len(true_p), size=min(2000,len(true_p)), replace=False)
    tun_idx = idx[is_tun[idx]]; road_idx = idx[~is_tun[idx]]
    lim = max(abs(true_p[idx,0]).max(), abs(pred_p[idx,0]).max())*1.05
    ax_fwd.scatter(true_p[road_idx,0], pred_p[road_idx,0], alpha=0.3, s=2, c='steelblue', label='Road')
    ax_fwd.scatter(true_p[tun_idx,0],  pred_p[tun_idx,0],  alpha=0.5, s=4, c='tomato',    label='Tunnel')
    ax_fwd.plot([-lim,lim],[-lim,lim],'k--',lw=1); ax_fwd.set_title("Fwd Bias: Pred vs True")
    ax_fwd.legend(fontsize=7); ax_fwd.grid(True,alpha=0.3); ax_fwd.set_xlim(-lim,lim); ax_fwd.set_ylim(-lim,lim)
    n = min(500,len(true_p)); ts = np.arange(n)*0.05
    ax_ts.plot(ts, true_p[:n,0],'g-',lw=1,label='True bias'); ax_ts.plot(ts, pred_p[:n,0],'b--',lw=1,label='Pred bias')
    if is_tun[:n].any():
        ax_ts.fill_between(ts, ax_ts.get_ylim()[0], ax_ts.get_ylim()[1], where=is_tun[:n], alpha=0.15, color='red')
    ax_ts.set_title("Fwd Bias Time Series (25s)"); ax_ts.legend(fontsize=7); ax_ts.grid(True,alpha=0.3)
    lim2 = max(abs(true_p[idx,1]).max(), abs(pred_p[idx,1]).max())*1.05
    ax_lat.scatter(true_p[road_idx,1], pred_p[road_idx,1], alpha=0.3, s=2, c='steelblue')
    ax_lat.scatter(true_p[tun_idx,1],  pred_p[tun_idx,1],  alpha=0.5, s=4, c='tomato')
    ax_lat.plot([-lim2,lim2],[-lim2,lim2],'k--',lw=1); ax_lat.set_title("Lat Bias: Pred vs True"); ax_lat.grid(True,alpha=0.3)
    err = np.abs(pred_p[:,0]-true_p[:,0]); bins = np.linspace(0,np.percentile(err,95),40)
    if is_tun.sum() > 0:
        ax_err.hist(err[~is_tun],bins=bins,alpha=0.6,color='steelblue',label=f'Road (n={int((~is_tun).sum())})',density=True)
        ax_err.hist(err[is_tun], bins=bins,alpha=0.6,color='tomato',   label=f'Tunnel (n={int(is_tun.sum())})',density=True)
    ax_err.set_title("Fwd Bias Error Distribution"); ax_err.legend(fontsize=7); ax_err.grid(True,alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.show()
    print(f"  Plot saved → {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*62}")
    print(f"  LSTM Bias Predictor  —  v4 Final")
    print(f"  Device   : {device}  |  Target: BIAS not absolute accel")
    print(f"  EKF use  : a_fwd = ax_corr + lstm_output  (additive)")
    print(f"{'='*62}\n")

    prep = DataPreprocessor()
    df   = prep.load_and_clean(DATA_PATH)
    df_train = df[df['run_id'].isin(TRAIN_RUNS)].reset_index(drop=True)
    df_val_f = df[df['run_id'].isin(VAL_RUNS)].reset_index(drop=True)
    mid      = len(df_val_f) // 2
    df_val   = df_val_f.iloc[:mid].reset_index(drop=True)
    df_test  = df_val_f.iloc[mid:].reset_index(drop=True)
    print(f"  Train:{len(df_train):,}  Val:{len(df_val):,}  Test:{len(df_test):,}")

    prep.fit(df_train); prep.save(STATS_PATH)
    df_train_n = prep.transform_features(df_train)
    df_val_n   = prep.transform_features(df_val)
    df_test_n  = prep.transform_features(df_test)

    print("\nBuilding datasets...")
    print("  Train:"); train_ds = IMUSequenceDataset(df_train_n, prep, stride=STRIDE)
    print("  Val:");   val_ds   = IMUSequenceDataset(df_val_n,   prep, stride=1)
    print("  Test:");  test_ds  = IMUSequenceDataset(df_test_n,  prep, stride=1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LSTMDriftPredictor(input_size=len(FEATURE_COLS), h1=HIDDEN1, h2=HIDDEN2, dropout=DROPOUT).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
    criterion = WeightedMSELoss()

    print(f"\nTraining (max {NUM_EPOCHS} epochs, patience={PATIENCE})...\n")
    print(f"  {'Ep':>4}  {'Train':>10}  {'Val':>10}  {'Best':>10}  {'GNorm':>7}  {'LR':>8}  {'NoImp':>5}")
    print("  " + "-"*62)
    train_losses, val_losses = [], []
    best_val = float('inf'); no_improve = 0; best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, gnorm = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, grad_clip=GRAD_CLIP)
        vl_loss, _     = run_epoch(model, val_loader,   criterion, device)
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        scheduler.step(vl_loss)
        if vl_loss < best_val:
            best_val = vl_loss; best_epoch = epoch; no_improve = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict(), 'val_loss': best_val,
                        'config': {'input_size': len(FEATURE_COLS), 'h1': HIDDEN1, 'h2': HIDDEN2,
                                   'dropout': DROPOUT, 'seq_len': SEQ_LEN,
                                   'feature_cols': FEATURE_COLS, 'target_cols': TARGET_COLS,
                                   'output_is_bias': True, 'seed': SEED}}, MODEL_PATH)
        else:
            no_improve += 1
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  {epoch:>4}  {tr_loss:>10.6f}  {vl_loss:>10.6f}  {best_val:>10.6f}  "
              f"{gnorm:>7.3f}  {lr_now:>8.2e}  {no_improve:>3}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}."); break

    print(f"\n  Best val loss: {best_val:.6f} at epoch {best_epoch}")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    print(f"\n{'='*62}\n  TEST EVALUATION\n{'='*62}")
    metrics, pred_p, true_p, is_tun = evaluate(model, test_loader, prep, device)
    lines = ["LSTM Bias Predictor v4 — Test Metrics", "="*62,
             f"Seed:{SEED}  Best epoch:{best_epoch}  Val loss:{best_val:.6f}",
             "Targets: bias = gt_accel - imu_accel  (EKF: a_fwd = imu + lstm_output)", "",
             f"{'Metric':<35} {'Value':>10}", "-"*47]
    for k, v in metrics.items():
        unit = "(m/s²)" if "MAE" in k or "RMSE" in k else ""
        line = f"  {k:<33} {v:>10.4f}  {unit}"; print(line); lines.append(line)
    with open(METRICS_PATH,'w') as f: f.write("\n".join(lines))
    print(f"\n  Metrics → {METRICS_PATH}")

    plot_results(train_losses, val_losses, pred_p, true_p, is_tun, PLOT_PATH)

    print(f"\n{'='*62}\n  DONE\n  Model → {MODEL_PATH}\n  Stats → {STATS_PATH}")
    print(f"\n  CRITICAL: After training, run ekf.py v4 which applies output as:")
    print(f"    a_fwd = ax_corr + lstm_output   (NOT a_fwd = lstm_output)")
    print(f"{'='*62}\n")

if __name__ == '__main__':
    main()
