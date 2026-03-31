"""
train_lstm.py  —  LSTM Drift Predictor for RL-Adaptive EKF  (v3 Final)
=======================================================================
For: LSTM + RL-Adaptive EKF Localization Project

What This LSTM Learns
---------------------
During GPS denial (tunnel), the EKF has no measurement update and
relies entirely on IMU dead-reckoning, which drifts. The LSTM learns
to predict the TRUE forward and lateral acceleration one step ahead,
given the last 2 seconds of IMU history. This replaces the raw noisy
IMU reading inside the EKF predict() step during tunnel traversal.

  Inputs  (5 features, SEQ_LEN=40 steps = 2.0 seconds of history):
    ax_corr        gravity-corrected forward accel   (m/s²)
    ay_corr        gravity-corrected lateral accel   (m/s²)
    wz             yaw rate from IMU gyroscope       (rad/s)
    gt_speed_mps   vehicle speed                     (m/s)
    gps_denied     GPS denial flag (0 or 1)

  Why gps_denied as input:
    The LSTM modulates its predictions differently in tunnels
    (pure IMU mode) vs open road (GPS-corrected mode). Without
    this flag, the LSTM cannot distinguish these contexts.

  Outputs (2 targets, 1 step ahead = 0.05s):
    gt_accel_fwd_mps2   true forward  accel (zero-phase filtered)
    gt_accel_lat_mps2   true lateral  accel (zero-phase filtered)

Architecture  (v3 — right-sized for 17,960 training sequences)
--------------------------------------------------------------
  Input(5) → LayerNorm → LSTM(64) → LayerNorm → Dropout(0.3)
           → LSTM(32)  → LayerNorm → Dropout(0.3) → last step
           → Linear(32→16) → GELU → Linear(16→2)

  Params: ~31,000   Sequences: ~17,960   Ratio: ~0.57  ← good fit

Fixes from v2
-------------
  FIX 1  — Cross-run target leakage: boundary check now includes the
            target step (run_ids[end] instead of run_ids[end-1]).
            Previously the last input step of Run N and the first row
            of Run N+1 could be paired as (sequence, target), causing
            gradient spikes once per run boundary.
  FIX 2  — LayerNorm / Dropout order swapped: LN now runs on the
            stable LSTM output BEFORE dropout, not after. Dropout
            randomly zeros + rescales activations, making variance
            fluctuate wildly; LN after dropout was normalising a
            corrupted distribution. Correct order: LN → Dropout.

Fixes carried from v2
---------------------
  v2-FIX 1  — gps_denied added as 5th input feature
  v2-FIX 2  — SEQ_LEN 20→40  (1s→2s context)
  v2-FIX 3  — Model 120K→31K params (right-sized for dataset)
  v2-FIX 4  — LayerNorm after each LSTM layer (training stability)
  v2-FIX 5  — AdamW instead of Adam (proper weight decay)
  v2-FIX 6  — Target normalisation inside Dataset (not fragile loop)
  v2-FIX 7  — Tunnel-only evaluation (primary project metric)
  v2-FIX 8  — verbose=True removed from ReduceLROnPlateau (deprecated)
  v2-FIX 9  — torch.manual_seed + np.random.seed (reproducibility)
  v2-FIX 10 — NUM_LAYERS, PRED_HORIZON removed (defined but never used)
  v2-FIX 11 — Tunnel weight 3.0→2.0 (less aggressive, better balance)
  v2-FIX 12 — Print every epoch (full visibility)
  v2-FIX 13 — Scatter plot subsampled to 2000 pts (fast rendering)
  v2-FIX 14 — GELU instead of ReLU in head (smoother for regression)

Output
------
  models/lstm_drift_predictor.pth     trained model weights + metadata
  models/lstm_normalisation.npz       normalisation stats for EKF bridge
  results/lstm_training.png           5-panel results plot
  results/lstm_metrics.txt            full metrics including tunnel-only
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

# ── Sequence ──────────────────────────────────────────────────────────────────
SEQ_LEN = 40    # 40 × 0.05s = 2.0 seconds of context
STRIDE  = 2     # stride-2 prevents excessive overlap, halves dataset redundancy

# ── Features / targets ────────────────────────────────────────────────────────
FEATURE_COLS = ['ax_corr', 'ay_corr', 'wz', 'gt_speed_mps', 'gps_denied']
TARGET_COLS  = ['gt_accel_fwd_mps2', 'gt_accel_lat_mps2']

# ── Data split: by run to prevent leakage ─────────────────────────────────────
TRAIN_RUNS = [0, 1, 2]   # 18,000 rows
VAL_RUNS   = [3]          # 6,000 rows → first 3,000 val, last 3,000 test

# ── Model ─────────────────────────────────────────────────────────────────────
HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 150
PATIENCE      = 20       # early stopping patience
TUNNEL_WEIGHT = 2.0      # loss multiplier for tunnel sequences
GRAD_CLIP     = 1.0
LR_PATIENCE   = 8
LR_FACTOR     = 0.5
LR_MIN        = 1e-6

# ── Input clipping (ax_corr had outlier at -27.4 in the data) ─────────────────
INPUT_CLIP = {
    'ax_corr':      (-20.0, 20.0),
    'ay_corr':      (-15.0, 15.0),
    'wz':           (-5.0,   5.0),
    'gt_speed_mps': (0.0,   35.0),
    'gps_denied':   (0.0,    1.0),   # already 0/1, clip is a safety net
}


# =============================================================================
# DATA PREPROCESSOR
# =============================================================================

class DataPreprocessor:
    """
    Loads, cleans, clips, and normalises the collected CSV data.

    Rules:
      - Normalisation statistics computed on TRAINING data ONLY.
      - Features normalised with z-score (mean=0, std=1).
      - Targets normalised with z-score (so MSE loss is not dominated
        by whichever target has the larger physical range).
      - gps_denied is NOT normalised (already 0/1; z-scoring it would
        hurt the LSTM's ability to use it as a context switch).
    """

    def __init__(self):
        self.feat_mean = None
        self.feat_std  = None
        self.tgt_mean  = None
        self.tgt_std   = None

    def load_and_clean(self, path):
        df = pd.read_csv(path)
        n0 = len(df)

        # Drop NaN rows (from dropped frames during collection)
        needed = FEATURE_COLS + TARGET_COLS + ['run_id', 'timestamp']
        df = df.dropna(subset=needed).reset_index(drop=True)
        if len(df) < n0:
            print(f"  Dropped {n0 - len(df)} NaN rows")

        # Clip outlier inputs; gps_denied is naturally 0/1, skip it
        for col, (lo, hi) in INPUT_CLIP.items():
            before_max = df[col].abs().max()
            df[col] = df[col].clip(lo, hi)
            if before_max > hi * 1.01:
                print(f"  Clipped {col}: was ±{before_max:.2f}, "
                      f"now ±{df[col].abs().max():.2f}")

        print(f"  Loaded {len(df):,} rows | "
              f"tunnel={int(df['gps_denied'].sum()):,} "
              f"({100 * df['gps_denied'].mean():.1f}%)")
        return df

    def fit(self, df_train):
        """Fit normalisation statistics on training data only."""
        norm_feat_cols = [c for c in FEATURE_COLS if c != 'gps_denied']
        self.feat_mean = df_train[norm_feat_cols].mean().values.astype(np.float32)
        self.feat_std  = df_train[norm_feat_cols].std().values.astype(np.float32)
        self.feat_std  = np.where(self.feat_std < 1e-6, 1.0, self.feat_std)

        self.tgt_mean = df_train[TARGET_COLS].mean().values.astype(np.float32)
        self.tgt_std  = df_train[TARGET_COLS].std().values.astype(np.float32)
        self.tgt_std  = np.where(self.tgt_std < 1e-6, 1.0, self.tgt_std)

        print(f"\n  Feature means : {dict(zip([c for c in FEATURE_COLS if c != 'gps_denied'], self.feat_mean.round(4)))}")
        print(f"  Feature stds  : {dict(zip([c for c in FEATURE_COLS if c != 'gps_denied'], self.feat_std.round(4)))}")
        print(f"  Target  means : {dict(zip(TARGET_COLS, self.tgt_mean.round(4)))}")
        print(f"  Target  stds  : {dict(zip(TARGET_COLS, self.tgt_std.round(4)))}")

    def transform_features(self, df):
        """
        Returns a new DataFrame with z-scored features.
        gps_denied is left as 0/1. All other columns unchanged.
        """
        out = df.copy()
        norm_feat_cols = [c for c in FEATURE_COLS if c != 'gps_denied']
        out[norm_feat_cols] = (
            df[norm_feat_cols].values.astype(np.float32) - self.feat_mean
        ) / self.feat_std
        return out

    def normalise_targets(self, arr):
        """arr: (N, 2) numpy array → z-scored array."""
        return (arr - self.tgt_mean) / self.tgt_std

    def denormalise_targets(self, arr):
        """arr: (N, 2) numpy array → physical units (m/s²)."""
        return arr * self.tgt_std + self.tgt_mean

    def save(self, path):
        np.savez(
            path,
            feat_mean    = self.feat_mean,
            feat_std     = self.feat_std,
            tgt_mean     = self.tgt_mean,
            tgt_std      = self.tgt_std,
            feature_cols = np.array(FEATURE_COLS),
            target_cols  = np.array(TARGET_COLS),
            seq_len      = np.array([SEQ_LEN]),
            seed         = np.array([SEED]),
        )
        print(f"  Normalisation stats saved → {path}")


# =============================================================================
# DATASET
# =============================================================================

class IMUSequenceDataset(Dataset):
    """
    Sliding window dataset.

    Each sample:
      x : (SEQ_LEN, 5)  normalised feature sequence  [steps i … i+SEQ_LEN-1]
      y : (2,)           normalised target at step i+SEQ_LEN  (next step)
      w : scalar         sample weight (TUNNEL_WEIGHT if any step in tunnel)

    Boundary exclusion (FIX 1):
      A sequence at index i spans input rows [i, end-1] and target row [end].
      The boundary check now verifies run_ids[i] == run_ids[end] (not end-1).
      Previously, the last input step of Run N and the first row of Run N+1
      could be paired as (input sequence, target), injecting a massive
      false gradient spike once per run transition.

    Targets are normalised here (not in main) so the Dataset is
    self-contained and the normalisation path is unambiguous.
    """

    def __init__(self, df_norm, preprocessor, stride=1):
        """
        df_norm      : DataFrame with normalised features, raw targets,
                       run_id, gps_denied columns.
        preprocessor : DataPreprocessor (used to normalise targets).
        stride       : step between consecutive sequence start indices.
        """
        features    = df_norm[FEATURE_COLS].values.astype(np.float32)
        targets_raw = df_norm[TARGET_COLS].values.astype(np.float32)
        targets     = preprocessor.normalise_targets(targets_raw).astype(np.float32)
        run_ids     = df_norm['run_id'].values
        gps_denied  = df_norm['gps_denied'].values.astype(np.float32)

        seqs, tgts, wgts = [], [], []
        n = len(features)

        for i in range(0, n - SEQ_LEN, stride):
            end = i + SEQ_LEN  # index of the TARGET step

            # Guard: target index must be within bounds
            if end >= n:
                continue

            # FIX 1: boundary check includes the TARGET step (run_ids[end]),
            # not just the last input step (run_ids[end-1]).
            # Without this fix, the target could come from the next run's
            # first row (spawn-point, near-zero speed) while the input
            # sequence is from the end of the previous run — a false label
            # that caused one artificial gradient spike per run boundary.
            if run_ids[i] != run_ids[end]:
                continue

            x = features[i:end]   # (SEQ_LEN, 5)
            y = targets[end]       # (2,)
            w = (TUNNEL_WEIGHT
                 if gps_denied[i:end].any()
                 else 1.0)

            seqs.append(x)
            tgts.append(y)
            wgts.append(w)

        self.X = np.array(seqs, dtype=np.float32)
        self.Y = np.array(tgts, dtype=np.float32)
        self.W = np.array(wgts, dtype=np.float32)

        n_tun = (self.W > 1.0).sum()
        print(f"    {len(self.X):,} sequences | "
              f"tunnel: {n_tun:,} ({100 * n_tun / max(len(self.X), 1):.1f}%)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y[idx]),
            torch.tensor(self.W[idx], dtype=torch.float32),
        )


# =============================================================================
# MODEL
# =============================================================================

class LSTMDriftPredictor(nn.Module):
    """
    Two-layer stacked LSTM with LayerNorm for IMU drift prediction.

    LayerNorm (not BatchNorm) is used because:
      - LayerNorm operates per-sample → stable for variable-length sequences.
      - BatchNorm during inference with batch_size=1 (EKF bridge) collapses.

    FIX 2 — LayerNorm / Dropout order:
      Correct order is:  LSTM → LayerNorm → Dropout
      Previous order was LSTM → Dropout → LayerNorm, which was wrong because:
        1. Dropout randomly zeros activations and rescales the rest, causing
           the layer's variance to swing on every forward pass.
        2. LayerNorm placed after Dropout was normalising a randomly corrupted
           distribution rather than the stable LSTM output — this made the
           training loss curve jittery and slowed convergence.
      Normalising the stable LSTM output first, then applying dropout, gives
      LayerNorm a clean signal and makes the regularisation more principled.

    Architecture:
      InputLN → LSTM(5→64) → LN → Dropout
             → LSTM(64→32) → LN → Dropout
             → last step → Linear(32→16) → GELU → Linear(16→2)

    ~31,000 parameters.
    """

    def __init__(self, input_size=5, h1=64, h2=32, dropout=0.3):
        super().__init__()

        # Normalise raw inputs before LSTM (stabilises early training)
        self.input_ln = nn.LayerNorm(input_size)

        self.lstm1 = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1   = nn.LayerNorm(h1)     # FIX 2: LN before Dropout
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2   = nn.LayerNorm(h2)     # FIX 2: LN before Dropout
        self.drop2 = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(h2, 16),
            nn.GELU(),             # smoother than ReLU for regression
            nn.Linear(16, len(TARGET_COLS)),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Xavier for input-hidden weights, orthogonal for hidden-hidden.
        Forget gate bias initialised to 1.0 (preserves long-term memory).
        PyTorch LSTM bias layout: [input, forget, cell, output] × hidden_size.
        Forget gate occupies slice [hidden_size : 2*hidden_size].
        """
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias = 1

    def forward(self, x):
        """
        x   : (batch, SEQ_LEN, 5)
        out : (batch, 2)
        """
        x = self.input_ln(x)

        # Layer 1: LSTM → LayerNorm → Dropout  (FIX 2 order)
        o1, _ = self.lstm1(x)    # (batch, seq, 64)
        o1    = self.ln1(o1)
        o1    = self.drop1(o1)

        # Layer 2: LSTM → LayerNorm → Dropout  (FIX 2 order)
        o2, _ = self.lstm2(o1)   # (batch, seq, 32)
        o2    = self.ln2(o2)
        o2    = self.drop2(o2)

        last = o2[:, -1, :]      # (batch, 32) — last timestep only
        return self.head(last)   # (batch, 2)


# =============================================================================
# LOSS
# =============================================================================

class WeightedMSELoss(nn.Module):
    """
    Sample-weighted MSE. Tunnel sequences get TUNNEL_WEIGHT × higher loss,
    forcing the model to prioritise GPS-denied prediction accuracy.
    """

    def forward(self, pred, target, weights):
        # Per-sample MSE averaged over output dimensions → (batch,)
        per_sample = ((pred - target) ** 2).mean(dim=1)
        return (per_sample * weights).mean()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def run_epoch(model, loader, criterion, device,
              optimizer=None, grad_clip=None):
    """
    One full pass over `loader`.
    If optimizer is provided: training mode (backprop + weight update).
    Otherwise: evaluation mode (no grad).
    Returns (mean_weighted_mse_loss, mean_grad_norm).
    grad_norm is 0.0 for eval passes.
    """
    training = optimizer is not None
    model.train(training)
    total_loss  = 0.0
    total_gnorm = 0.0
    n_batches   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)

            if training:
                optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y, w)

            if training:
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total_gnorm += gnorm.item()

            total_loss += loss.item() * len(x)
            n_batches  += 1

    mean_loss  = total_loss / len(loader.dataset)
    mean_gnorm = total_gnorm / max(n_batches, 1)
    return mean_loss, mean_gnorm


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, loader, preprocessor, device):
    """
    Returns:
      metrics : dict — MAE, RMSE, R² per target × {all, tunnel}
      pred_p  : (N, 2) predictions in physical units (m/s²)
      true_p  : (N, 2) ground truth in physical units (m/s²)
      is_tun  : (N,) bool — True for tunnel sequences
    """
    model.eval()
    all_pred, all_true, all_tun = [], [], []

    with torch.no_grad():
        for x, y, w in loader:
            pred = model(x.to(device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
            all_tun.append(w.numpy() > 1.0)

    pred_n = np.vstack(all_pred)
    true_n = np.vstack(all_true)
    is_tun = np.concatenate(all_tun)

    pred_p = preprocessor.denormalise_targets(pred_n)
    true_p = preprocessor.denormalise_targets(true_n)

    def _metrics(p, t, label):
        mae  = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        r2   = 1 - np.sum((p - t) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-8)
        return {
            f'{label}_MAE':  mae,
            f'{label}_RMSE': rmse,
            f'{label}_R2':   r2,
        }

    metrics = {}
    for i, col in enumerate(TARGET_COLS):
        tag = col.replace('gt_accel_', '').replace('_mps2', '')
        metrics.update(_metrics(pred_p[:, i], true_p[:, i], f'{tag}_all'))
        if is_tun.sum() > 0:
            metrics.update(
                _metrics(pred_p[is_tun, i], true_p[is_tun, i], f'{tag}_tunnel')
            )

    return metrics, pred_p, true_p, is_tun


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_results(train_losses, val_losses, pred_p, true_p, is_tun, save_path):
    """
    5-panel results figure:
      [0,0] Train/val loss curves (log scale)
      [0,1] Fwd accel scatter  (subsampled to 2000 pts)
      [1,0] Fwd accel time series (first 500 steps = 25s)
      [1,1] Lat accel scatter
      [2,0] Tunnel vs open-road error distribution
    """
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("LSTM Drift Predictor — Training Results  (v3)",
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_fwd  = fig.add_subplot(gs[0, 1])
    ax_ts   = fig.add_subplot(gs[1, 0])
    ax_lat  = fig.add_subplot(gs[1, 1])
    ax_err  = fig.add_subplot(gs[2, 0])

    # ── Loss curves ──────────────────────────────────────────────────────────
    ep = range(1, len(train_losses) + 1)
    ax_loss.plot(ep, train_losses, 'b-', lw=1.5, label='Train')
    ax_loss.plot(ep, val_losses,   'r-', lw=1.5, label='Val')
    ax_loss.set_yscale('log')
    ax_loss.set_title("Weighted MSE Loss (log scale)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # ── Subsampled scatter: forward accel ─────────────────────────────────────
    idx      = np.random.choice(len(true_p), size=min(2000, len(true_p)), replace=False)
    tun_idx  = idx[is_tun[idx]]
    road_idx = idx[~is_tun[idx]]
    lim      = max(abs(true_p[idx, 0]).max(), abs(pred_p[idx, 0]).max()) * 1.05

    ax_fwd.scatter(true_p[road_idx, 0], pred_p[road_idx, 0],
                   alpha=0.3, s=2, c='steelblue', label='Open road')
    ax_fwd.scatter(true_p[tun_idx,  0], pred_p[tun_idx,  0],
                   alpha=0.5, s=4, c='tomato',    label='Tunnel')
    ax_fwd.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='Perfect')
    ax_fwd.set_title("Forward Accel: Pred vs True")
    ax_fwd.set_xlabel("True (m/s²)")
    ax_fwd.set_ylabel("Pred (m/s²)")
    ax_fwd.legend(fontsize=7)
    ax_fwd.grid(True, alpha=0.3)
    ax_fwd.set_xlim(-lim, lim)
    ax_fwd.set_ylim(-lim, lim)

    # ── Time series ───────────────────────────────────────────────────────────
    n  = min(500, len(true_p))
    ts = np.arange(n) * 0.05
    ax_ts.plot(ts, true_p[:n, 0], 'g-',  lw=1, label='True')
    ax_ts.plot(ts, pred_p[:n, 0], 'b--', lw=1, label='LSTM')
    tun_mask = is_tun[:n]
    if tun_mask.any():
        ylims = ax_ts.get_ylim()
        ax_ts.fill_between(ts, ylims[0], ylims[1],
                           where=tun_mask, alpha=0.15, color='red',
                           label='Tunnel zone')
    ax_ts.set_title("Forward Accel Time Series (first 25s)")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Accel (m/s²)")
    ax_ts.legend(fontsize=7)
    ax_ts.grid(True, alpha=0.3)

    # ── Lateral accel scatter ─────────────────────────────────────────────────
    lim2 = max(abs(true_p[idx, 1]).max(), abs(pred_p[idx, 1]).max()) * 1.05
    ax_lat.scatter(true_p[road_idx, 1], pred_p[road_idx, 1],
                   alpha=0.3, s=2, c='steelblue', label='Open road')
    ax_lat.scatter(true_p[tun_idx,  1], pred_p[tun_idx,  1],
                   alpha=0.5, s=4, c='tomato',    label='Tunnel')
    ax_lat.plot([-lim2, lim2], [-lim2, lim2], 'k--', lw=1)
    ax_lat.set_title("Lateral Accel: Pred vs True")
    ax_lat.set_xlabel("True (m/s²)")
    ax_lat.set_ylabel("Pred (m/s²)")
    ax_lat.legend(fontsize=7)
    ax_lat.grid(True, alpha=0.3)
    ax_lat.set_xlim(-lim2, lim2)
    ax_lat.set_ylim(-lim2, lim2)

    # ── Error distribution: tunnel vs open road ───────────────────────────────
    err_fwd = np.abs(pred_p[:, 0] - true_p[:, 0])
    bins    = np.linspace(0, np.percentile(err_fwd, 95), 40)
    if is_tun.sum() > 0:
        ax_err.hist(err_fwd[~is_tun], bins=bins, alpha=0.6,
                    color='steelblue',
                    label=f'Open road (n={int((~is_tun).sum())})',
                    density=True)
        ax_err.hist(err_fwd[is_tun],  bins=bins, alpha=0.6,
                    color='tomato',
                    label=f'Tunnel (n={int(is_tun.sum())})',
                    density=True)
    ax_err.set_title("Forward Accel Absolute Error Distribution")
    ax_err.set_xlabel("|Pred - True|  (m/s²)")
    ax_err.set_ylabel("Density")
    ax_err.legend(fontsize=7)
    ax_err.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Plot saved → {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*62}")
    print(f"  LSTM Drift Predictor  —  v3 Final")
    print(f"  Device  : {device}")
    print(f"  Seed    : {SEED}")
    print(f"  SEQ_LEN : {SEQ_LEN} steps ({SEQ_LEN * 0.05:.1f}s context)")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Targets : {TARGET_COLS}")
    print(f"{'='*62}\n")

    # ── Load & clean ──────────────────────────────────────────────────────────
    print("Loading data...")
    prep = DataPreprocessor()
    df   = prep.load_and_clean(DATA_PATH)

    # ── Split by run (no leakage) ─────────────────────────────────────────────
    df_train    = df[df['run_id'].isin(TRAIN_RUNS)].reset_index(drop=True)
    df_val_full = df[df['run_id'].isin(VAL_RUNS)].reset_index(drop=True)
    mid         = len(df_val_full) // 2
    df_val      = df_val_full.iloc[:mid].reset_index(drop=True)
    df_test     = df_val_full.iloc[mid:].reset_index(drop=True)

    print(f"\nData split:")
    print(f"  Train : {len(df_train):,} rows  runs={TRAIN_RUNS}")
    print(f"  Val   : {len(df_val):,} rows  run {VAL_RUNS} first half")
    print(f"  Test  : {len(df_test):,} rows  run {VAL_RUNS} second half")

    # ── Fit normalisation on training data only ───────────────────────────────
    print("\nFitting normalisation...")
    prep.fit(df_train)
    prep.save(STATS_PATH)

    df_train_n = prep.transform_features(df_train)
    df_val_n   = prep.transform_features(df_val)
    df_test_n  = prep.transform_features(df_test)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nBuilding datasets (targets normalised inside Dataset)...")
    print("  Train:")
    train_ds = IMUSequenceDataset(df_train_n, prep, stride=STRIDE)
    print("  Val:")
    val_ds   = IMUSequenceDataset(df_val_n,   prep, stride=1)
    print("  Test:")
    test_ds  = IMUSequenceDataset(df_test_n,  prep, stride=1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LSTMDriftPredictor(
        input_size=len(FEATURE_COLS),
        h1=HIDDEN1, h2=HIDDEN2, dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} parameters")
    print(f"  LSTM(5→{HIDDEN1}) → LSTM({HIDDEN1}→{HIDDEN2}) → "
          f"Linear({HIDDEN2}→16) → Linear(16→2)")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=LR_FACTOR, patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )
    criterion = WeightedMSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining (max {NUM_EPOCHS} epochs, "
          f"early-stop patience={PATIENCE})...\n")
    print(f"  {'Ep':>4}  {'Train':>10}  {'Val':>10}  "
          f"{'Best':>10}  {'GNorm':>7}  {'LR':>8}  {'NoImprove':>9}")
    print("  " + "-" * 70)

    train_losses, val_losses = [], []
    best_val   = float('inf')
    no_improve = 0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        tr_loss, gnorm = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, grad_clip=GRAD_CLIP,
        )
        vl_loss, _ = run_epoch(
            model, val_loader, criterion, device,
        )

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        scheduler.step(vl_loss)

        if vl_loss < best_val:
            best_val   = vl_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss':    best_val,
                'config': {
                    'input_size':   len(FEATURE_COLS),
                    'h1':           HIDDEN1,
                    'h2':           HIDDEN2,
                    'dropout':      DROPOUT,
                    'seq_len':      SEQ_LEN,
                    'feature_cols': FEATURE_COLS,
                    'target_cols':  TARGET_COLS,
                    'seed':         SEED,
                },
            }, MODEL_PATH)
        else:
            no_improve += 1

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  {epoch:>4}  {tr_loss:>10.6f}  {vl_loss:>10.6f}  "
              f"{best_val:>10.6f}  {gnorm:>7.3f}  {lr_now:>8.2e}  "
              f"{no_improve:>4}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    print(f"\n  Best val loss : {best_val:.6f}  (epoch {best_epoch})")

    # ── Load best checkpoint ──────────────────────────────────────────────────
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"  Loaded best model from epoch {ckpt['epoch']}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  TEST SET EVALUATION")
    print(f"{'='*62}")

    metrics, pred_p, true_p, is_tun = evaluate(
        model, test_loader, prep, device,
    )

    lines = [
        "LSTM Drift Predictor v3 — Test Metrics",
        "=" * 62,
        f"Seed: {SEED}  |  Best epoch: {best_epoch}  |  Val loss: {best_val:.6f}",
        "",
        f"{'Metric':<35} {'Value':>10}",
        "-" * 47,
    ]
    for k, v in metrics.items():
        unit = "(m/s²)" if ("MAE" in k or "RMSE" in k) else ""
        line = f"  {k:<33} {v:>10.4f}  {unit}"
        print(line)
        lines.append(line)

    print(f"\n  Tunnel sequences in test : {is_tun.sum():,} / {len(is_tun):,} "
          f"({100 * is_tun.mean():.1f}%)")

    with open(METRICS_PATH, 'w') as f:
        f.write("\n".join(lines))
    print(f"\n  Metrics saved → {METRICS_PATH}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_results(train_losses, val_losses, pred_p, true_p, is_tun, PLOT_PATH)

    print(f"\n{'='*62}")
    print("  DONE")
    print(f"  Model   → {MODEL_PATH}")
    print(f"  Stats   → {STATS_PATH}")
    print(f"  Metrics → {METRICS_PATH}")
    print(f"  Plot    → {PLOT_PATH}")
    print(f"\n  Expected results:")
    print(f"    fwd_all_R2    > 0.85   (open road + tunnel)")
    print(f"    fwd_tunnel_R2 > 0.75   (GPS denied — harder)")
    print(f"    fwd_all_MAE   < 0.5 m/s²")
    print(f"\n  Next: python lstm_ekf_bridge.py")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    main()
