"""
train_lstm.py  —  LSTM Drift Compensator  (v2)
================================================

PURPOSE
-------
Train an LSTM that predicts vehicle 2-D displacement (Δx, Δy) over a
2.5-second window of IMU + derived features.

During GPS denial (Town04 tunnel) the Kalman Filter loses its measurement
correction step and position error accumulates.  The KF-LSTM fusion module
calls this model every tick to estimate how far the vehicle moved, so the
KF position estimate can be corrected even without GPS.

WHY EACH DESIGN DECISION WAS MADE
-----------------------------------

sin(yaw) / cos(yaw)  [ADDED]
    Raw ax, ay are in the vehicle body frame.  Without heading the network
    must infer the body→world rotation implicitly from the gyro sequence.
    Adding sin/cos of heading gives it that rotation explicitly.
    sin/cos is used instead of raw yaw to avoid the ±π discontinuity.
    During training  : yaw derived from consecutive gt_x, gt_y positions.
    During inference : yaw taken from the KF state estimate.

a_horiz = sqrt(ax²+ay²)  [ADDED]
    Horizontal acceleration magnitude with gravity removed.
    az ≈ 9.8 m/s² always in CARLA (gravity not subtracted by the sensor),
    so a_mag = sqrt(ax²+ay²+az²) ≈ 9.8 always and is nearly uninformative.
    a_horiz isolates the lateral/longitudinal dynamics that actually matter.

gyro_mag = sqrt(wx²+wy²+wz²)  [ADDED]
    Rotation-invariant scalar summary of how fast the vehicle is turning.
    Useful for distinguishing straight segments from corners.

delta_speed  [ADDED]
    speed[t] − speed[t−1].  Longitudinal acceleration proxy without needing
    to integrate IMU.  First sample is set to 0.

SmoothL1Loss (Huber)  [ADDED, replaces MSELoss]
    CARLA IMU spikes on kerb hits, hard braking, and lane transitions.
    Under MSE a single spike dominates the batch gradient (squared penalty).
    Huber is quadratic for |error| < 1 and linear for larger errors,
    so outliers have bounded influence on training.

STRIDE = 10  [CHANGED from 5]
    STRIDE=5 means 90% window overlap → highly correlated consecutive samples
    → validation loss appears smoother than it truly is.
    STRIDE=10 halves the overlap while keeping ≈2400 usable windows from a
    20-minute collection.

Unidirectional LSTM  [KEPT — bidirectional rejected]
    A bidirectional LSTM reads the window both forward and backward before
    producing output.  It therefore cannot emit a prediction until it has seen
    the entire window, i.e. it imposes a mandatory SEQ_LEN-step (2.5 s) delay.
    The KF-LSTM fusion module calls the model every CARLA tick.  A 2.5-second
    frozen correction window defeats the purpose of real-time fusion.

Δt not added  [REJECTED]
    CARLA runs in synchronous mode with fixed_delta_seconds=0.05.  Every tick
    is exactly 0.05 s by the simulator engine — there is no jitter.
    A constant column of 0.05 adds zero information.  Valid for real hardware.

Predict (Δx, Δy)  [KEPT — (distance, heading_change) rejected]
    Converting (distance, heading_change) → (Δx, Δy) at inference requires
    the absolute heading at the start of every window as external decoding
    context, creating a circular dependency with the KF state.
    Direct (Δx, Δy) in the local coordinate frame is already well-bounded
    (≈ ±30 m at highway speed) and the target distribution is smooth.

FEATURE SET  (12 features, INPUT_SIZE = 12)
--------------------------------------------
  0  ax          linear acceleration x           m/s²
  1  ay          linear acceleration y           m/s²
  2  az          linear acceleration z           m/s²  (includes ~9.8 gravity)
  3  wx          angular velocity x              rad/s
  4  wy          angular velocity y              rad/s
  5  wz          angular velocity z              rad/s  (yaw rate)
  6  speed_mps   vehicle speed                   m/s
  7  sin_yaw     sin of heading angle            —
  8  cos_yaw     cos of heading angle            —
  9  a_horiz     sqrt(ax²+ay²)                   m/s²
 10  gyro_mag    sqrt(wx²+wy²+wz²)               rad/s
 11  delta_spd   speed[t] − speed[t−1]           m/s

RUN
---
    cd C:\\Users\\heman\\Music\\rl_imu_project
    carla_env\\Scripts\\activate
    pip install torch numpy pandas matplotlib
    python lstm\\train_lstm.py

OUTPUTS
-------
    models/lstm_drift.pt            model weights + saved hyperparameters
    models/lstm_norm_params.npz     per-feature mean & std  (required at inference)
    models/lstm_training_log.csv    loss / rmse / lr per epoch
    plots/lstm_training_curves.png  Huber loss + val-RMSE over epochs
    plots/lstm_test_predictions.png 4-panel diagnostic on held-out test split

INFERENCE CHECKLIST
-------------------
    At inference time, before feeding a window to the model:

    1. Compute yaw from KF state:
           sin_yaw_i  = sin(kf_yaw)  for each step i in the window
           cos_yaw_i  = cos(kf_yaw)

    2. Compute derived features per step:
           a_horiz_i  = sqrt(ax_i² + ay_i²)
           gyro_mag_i = sqrt(wx_i² + wy_i² + wz_i²)
           delta_spd_i = speed_i − speed_{i−1}  (0 for the first step)

    3. Assemble (SEQ_LEN, 12) float32 array in FEATURE_COLS order.

    4. Load lstm_norm_params.npz, apply:
           x_norm = (x − mean) / std

    5. Call model.forward(torch.from_numpy(x_norm).unsqueeze(0))
       Output is (1, 2) → (Δx, Δy) in metres.
"""

# ── standard library ──────────────────────────────────────────────────────────
import os
import sys
import time
import csv

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# Non-interactive backend must be set before any pyplot import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION  —  edit only this block
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR  = r'C:\Users\heman\Music\rl_imu_project'
DATA_CSV  = os.path.join(BASE_DIR, 'data', 'town04_collected.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR  = os.path.join(BASE_DIR, 'plots')

# Columns that must exist in the raw CSV
RAW_COLS = ['timestamp', 'ax', 'ay', 'az', 'wx', 'wy', 'wz',
            'speed_mps', 'gt_x', 'gt_y', 'gps_denied']

# Feature columns fed to the model — order matters for norm params
FEATURE_COLS = [
    'ax', 'ay', 'az',
    'wx', 'wy', 'wz',
    'speed_mps',
    'sin_yaw', 'cos_yaw',
    'a_horiz',
    'gyro_mag',
    'delta_spd',
]
INPUT_SIZE = len(FEATURE_COLS)   # 12

# Windowing
SEQ_LEN = 50    # window length  : 50 × 0.05 s = 2.5 s
STRIDE  = 10    # window step    : 10 × 0.05 s = 0.5 s  →  80% overlap

# Model
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2

# Training
BATCH_SIZE  = 256
EPOCHS      = 100
LR_INIT     = 1e-3
LR_MIN      = 1e-6
LR_FACTOR   = 0.5   # multiply LR when val loss stops improving
LR_PATIENCE = 8     # epochs before LR drop
ES_PATIENCE = 20    # epochs before early stop

# Data split — time-ordered, no shuffling across boundaries
VAL_FRAC  = 0.15
TEST_FRAC = 0.10

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
#   DATASET
# ══════════════════════════════════════════════════════════════════════════════

class DisplacementDataset(Dataset):
    """
    Wraps normalised (X, y) arrays as a PyTorch Dataset.

    X : float32 ndarray  (N, SEQ_LEN, INPUT_SIZE)
    y : float32 ndarray  (N, 2)   —  (Δgt_x, Δgt_y) in metres
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # Convert once at construction — avoids per-item tensor allocation
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
#   MODEL
# ══════════════════════════════════════════════════════════════════════════════

class LSTMDriftCompensator(nn.Module):
    """
    Unidirectional 2-layer LSTM with a 3-layer MLP regression head.

    Unidirectional by design: the output at the last timestep uses only
    past and present data — no future lookahead — so the model can be called
    every CARLA tick without imposing any latency on the KF correction.

    Input  : (batch, SEQ_LEN, INPUT_SIZE)
    Output : (batch, 2)   —  (Δx, Δy) in metres
    """

    def __init__(self,
                 input_size:  int   = INPUT_SIZE,
                 hidden_size: int   = HIDDEN_SIZE,
                 num_layers:  int   = NUM_LAYERS,
                 dropout:     float = DROPOUT) -> None:
        super(LSTMDriftCompensator, self).__init__()

        # Dropout only applies between stacked LSTM layers, not after the last
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),   # no final activation — output is unbounded real
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x           : (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last        = lstm_out[:, -1, :]   # (batch, hidden_size)
        return self.head(last)             # (batch, 2)


# ══════════════════════════════════════════════════════════════════════════════
#   DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> pd.DataFrame:
    """Load the collected CSV, validate columns, drop NaN rows, report stats."""

    if not os.path.isfile(path):
        sys.exit(
            f"\n[ERROR]  Data file not found:\n         {path}\n"
            f"         Run data_collection/collect_data.py first.\n"
        )

    df = pd.read_csv(path)

    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        sys.exit(
            f"\n[ERROR]  Missing columns: {missing}\n"
            f"         Found: {list(df.columns)}\n"
        )

    n_before = len(df)
    df = df.dropna(subset=RAW_COLS).reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped > 0:
        print(f"[Data]   Dropped {dropped} rows with NaN values")

    min_rows = SEQ_LEN * 20
    if len(df) < min_rows:
        sys.exit(
            f"\n[ERROR]  Only {len(df)} rows — need at least {min_rows}.\n"
            f"         Collect at least 10 minutes of data.\n"
        )

    n        = len(df)
    n_denied = int(df['gps_denied'].sum())
    duration = float(df['timestamp'].iloc[-1]) - float(df['timestamp'].iloc[0])

    print(f"[Data]   Rows       : {n:,}")
    print(f"[Data]   Duration   : {duration:.1f} s  ({duration / 60:.1f} min)")
    print(f"[Data]   GPS denied : {n_denied:,}  ({100 * n_denied / n:.1f} %)")
    print(f"[Data]   Speed      : "
          f"{df['speed_mps'].min():.2f} – {df['speed_mps'].max():.2f} m/s")
    print(f"[Data]   gt_x       : "
          f"{df['gt_x'].min():.1f} – {df['gt_x'].max():.1f} m")
    print(f"[Data]   gt_y       : "
          f"{df['gt_y'].min():.1f} – {df['gt_y'].max():.1f} m")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all derived feature columns to the DataFrame.

    sin_yaw / cos_yaw
        Computed from atan2 of consecutive ground truth displacements.
        This is the vehicle heading in the world frame.
        sin/cos avoids the ±π wraparound discontinuity of raw yaw.

    a_horiz
        sqrt(ax² + ay²) — horizontal acceleration magnitude.
        az is excluded because CARLA IMU leaves gravity in az (az ≈ 9.8 m/s²
        always), so a_mag = sqrt(ax²+ay²+az²) ≈ 9.8 — nearly constant and
        uninformative.  a_horiz captures lateral/longitudinal dynamics only.

    gyro_mag
        sqrt(wx²+wy²+wz²) — total rotation rate.
        Rotation-invariant: distinguishes cornering from straight segments.

    delta_spd
        speed[t] − speed[t−1].  Longitudinal acceleration proxy.
        First row is set to 0.0 (no predecessor available).
    """
    df = df.copy()

    # Heading from consecutive ground truth positions
    dx  = df['gt_x'].diff().fillna(0.0).values.astype(np.float32)
    dy  = df['gt_y'].diff().fillna(0.0).values.astype(np.float32)
    yaw = np.arctan2(dx, dy)
    df['sin_yaw'] = np.sin(yaw).astype(np.float32)
    df['cos_yaw'] = np.cos(yaw).astype(np.float32)

    # Horizontal acceleration (gravity-free)
    df['a_horiz'] = np.sqrt(
        df['ax'].values ** 2 + df['ay'].values ** 2
    ).astype(np.float32)

    # Total rotation rate
    df['gyro_mag'] = np.sqrt(
        df['wx'].values ** 2 + df['wy'].values ** 2 + df['wz'].values ** 2
    ).astype(np.float32)

    # Longitudinal acceleration proxy
    df['delta_spd'] = df['speed_mps'].diff().fillna(0.0).astype(np.float32)

    # Guard: abort if any feature is NaN or Inf after computation
    for col in FEATURE_COLS:
        n_nan = int(df[col].isna().sum())
        n_inf = int(np.isinf(df[col].values).sum())
        if n_nan > 0 or n_inf > 0:
            sys.exit(
                f"\n[ERROR]  Feature '{col}' has {n_nan} NaN and "
                f"{n_inf} Inf values after computation.\n"
            )

    return df


def build_windows(df: pd.DataFrame):
    """
    Sliding-window extraction over the full sequence.

    For window i starting at row index t:
        X[i] = FEATURE_COLS values at rows  [t, t+SEQ_LEN)
        y[i] = gt_pos[t+SEQ_LEN] − gt_pos[t]

    Returns
    -------
    X : float32 ndarray  (N, SEQ_LEN, INPUT_SIZE)
    y : float32 ndarray  (N, 2)
    """
    features = df[FEATURE_COLS].values.astype(np.float32)
    gt_x     = df['gt_x'].values.astype(np.float32)
    gt_y     = df['gt_y'].values.astype(np.float32)
    n_rows   = len(df)

    n_max = (n_rows - SEQ_LEN) // STRIDE
    X = np.empty((n_max, SEQ_LEN, INPUT_SIZE), dtype=np.float32)
    y = np.empty((n_max, 2),                   dtype=np.float32)

    count = 0
    for t in range(0, n_rows - SEQ_LEN, STRIDE):
        end         = t + SEQ_LEN
        X[count]    = features[t:end]
        y[count, 0] = gt_x[end] - gt_x[t]
        y[count, 1] = gt_y[end] - gt_y[t]
        count      += 1

    X = X[:count]
    y = y[:count]

    print(f"[Windows] Count   : {count:,}  "
          f"(seq={SEQ_LEN}×0.05={SEQ_LEN*0.05:.2f}s  "
          f"stride={STRIDE}×0.05={STRIDE*0.05:.2f}s  "
          f"{100*(1-STRIDE/SEQ_LEN):.0f}% overlap)")
    print(f"[Windows] Δx range: {y[:,0].min():.2f} – {y[:,0].max():.2f} m")
    print(f"[Windows] Δy range: {y[:,1].min():.2f} – {y[:,1].max():.2f} m")

    return X, y


def time_split(X: np.ndarray, y: np.ndarray):
    """
    Chronological split.  No shuffling across boundaries — that would leak
    future motion patterns into the training set and inflate validation metrics.

    Layout:  [─── train (75%) ───][── val (15%) ──][─ test (10%) ─]
                                  oldest ──────────────────► newest

    Returns  X_train, y_train, X_val, y_val, X_test, y_test
    """
    n       = len(X)
    n_test  = max(1, int(n * TEST_FRAC))
    n_val   = max(1, int(n * VAL_FRAC))
    n_train = n - n_val - n_test

    if n_train < 200:
        sys.exit(
            f"\n[ERROR]  Only {n_train} training windows after split.\n"
            f"         Collect more data or reduce VAL_FRAC / TEST_FRAC.\n"
        )

    return (
        X[:n_train],              y[:n_train],
        X[n_train:n_train+n_val], y[n_train:n_train+n_val],
        X[n_train+n_val:],        y[n_train+n_val:],
    )


def fit_normaliser(X_train: np.ndarray):
    """
    Compute per-feature mean and std from training windows ONLY.
    Applying training statistics to val and test prevents those splits from
    influencing the normalisation.

    Returns mean, std — both float32 ndarray  shape (INPUT_SIZE,)
    """
    _, _, n_feat = X_train.shape
    flat = X_train.reshape(-1, n_feat)
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-8, 1.0, std)   # guard constant/near-constant features
    return mean, std


def apply_normaliser(X:    np.ndarray,
                     mean: np.ndarray,
                     std:  np.ndarray) -> np.ndarray:
    n, seq, n_feat = X.shape
    flat = X.reshape(-1, n_feat)
    return ((flat - mean) / std).reshape(n, seq, n_feat).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#   TRAINING / EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model:     nn.Module,
                    loader:    DataLoader,
                    optimiser: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device:    torch.device) -> float:
    """One full pass over the training set.  Returns mean loss."""
    model.train()
    total_loss, total_n = 0.0, 0

    for X_b, y_b in loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)

        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()

        # Gradient clipping prevents the well-known LSTM exploding-gradient issue
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        total_loss += loss.item() * len(X_b)
        total_n    += len(X_b)

    return total_loss / total_n


@torch.no_grad()
def evaluate(model:     nn.Module,
             loader:    DataLoader,
             criterion: nn.Module,
             device:    torch.device):
    """
    Full evaluation pass.

    Returns
    -------
    avg_loss : float            mean Huber loss
    rmse_2d  : float            mean Euclidean displacement error (metres)
    preds    : (N, 2) ndarray
    trues    : (N, 2) ndarray
    """
    model.eval()
    total_loss, total_n = 0.0, 0
    preds_list, trues_list = [], []

    for X_b, y_b in loader:
        X_b  = X_b.to(device)
        y_b  = y_b.to(device)
        pred = model(X_b)

        total_loss += criterion(pred, y_b).item() * len(X_b)
        total_n    += len(X_b)
        preds_list.append(pred.cpu().numpy())
        trues_list.append(y_b.cpu().numpy())

    preds   = np.concatenate(preds_list, axis=0)
    trues   = np.concatenate(trues_list, axis=0)
    rmse_2d = float(np.sqrt(((preds - trues) ** 2).sum(axis=1)).mean())

    return total_loss / total_n, rmse_2d, preds, trues


# ══════════════════════════════════════════════════════════════════════════════
#   PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(train_losses: list,
                         val_losses:   list,
                         val_rmses:    list,
                         path:         str) -> None:

    epochs = list(range(1, len(train_losses) + 1))
    best   = min(val_rmses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, train_losses, label='Train', linewidth=1.5)
    ax1.plot(epochs, val_losses,   label='Val',   linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Huber Loss')
    ax1.set_title('Loss curves  (log scale)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_rmses, color='green', linewidth=1.5)
    ax2.axhline(best, color='red', linestyle='--', linewidth=1,
                label=f'Best  {best:.4f} m')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE (m)')
    ax2.set_title('Validation 2D RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('LSTM Drift Compensator — Training', fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot]   Saved: {path}")


def plot_test_predictions(preds: np.ndarray,
                          trues: np.ndarray,
                          path:  str) -> None:
    """4-panel: time-series and scatter for Δx and Δy on the test split."""

    n_show = min(500, len(preds))
    idx    = np.arange(n_show)
    rmse_x = float(np.sqrt(np.mean((preds[:, 0] - trues[:, 0]) ** 2)))
    rmse_y = float(np.sqrt(np.mean((preds[:, 1] - trues[:, 1]) ** 2)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── time series ─────────────────────────────────────────────────────
    for col, label, ax in [(0, 'Δx', axes[0, 0]), (1, 'Δy', axes[0, 1])]:
        ax.plot(idx, trues[:n_show, col],
                label=f'True {label}', alpha=0.8, linewidth=0.9)
        ax.plot(idx, preds[:n_show, col],
                label=f'Pred {label}', alpha=0.8, linewidth=0.9)
        ax.set_title(f'Displacement {label} — test set (first {n_show})')
        ax.set_xlabel('Window index')
        ax.set_ylabel('metres')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ── scatter ──────────────────────────────────────────────────────────
    for col, rmse, color, ax in [
        (0, rmse_x, 'steelblue',  axes[1, 0]),
        (1, rmse_y, 'darkorange', axes[1, 1]),
    ]:
        label = 'Δx' if col == 0 else 'Δy'
        vals  = np.concatenate([trues[:, col], preds[:, col]])
        lim   = float(np.abs(vals).max()) * 1.1
        ax.scatter(trues[:, col], preds[:, col],
                   alpha=0.25, s=4, color=color)
        ax.plot([-lim, lim], [-lim, lim], 'r--',
                linewidth=1, label='Perfect')
        ax.set_title(f'{label}  True vs Predicted   RMSE = {rmse:.4f} m')
        ax.set_xlabel(f'True {label} (m)')
        ax.set_ylabel(f'Predicted {label} (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('LSTM Drift Compensator — Test Predictions', fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot]   Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 68)
    print('  LSTM DRIFT COMPENSATOR — Training  (v2)')
    print('=' * 68)
    print(f'  Device     : {device}')
    print(f'  Data       : {DATA_CSV}')
    print(f'  Features   : {INPUT_SIZE}  →  {FEATURE_COLS}')
    print(f'  Window     : {SEQ_LEN} steps = {SEQ_LEN * 0.05:.2f} s')
    print(f'  Stride     : {STRIDE} steps = {STRIDE * 0.05:.2f} s  '
          f'({100*(1 - STRIDE/SEQ_LEN):.0f}% overlap)')
    print(f'  Loss       : SmoothL1Loss (Huber)')
    print(f'  LSTM       : unidirectional  hidden={HIDDEN_SIZE}  layers={NUM_LAYERS}')
    print('=' * 68 + '\n')

    # ── 1. Load ───────────────────────────────────────────────────────────
    print('[1/7]  Loading data ...')
    df = load_csv(DATA_CSV)

    # ── 2. Derived features ───────────────────────────────────────────────
    print('\n[2/7]  Computing derived features ...')
    df = build_features(df)
    for col in ['sin_yaw', 'cos_yaw', 'a_horiz', 'gyro_mag', 'delta_spd']:
        print(f'       {col:12s}  [{df[col].min():+.4f}, {df[col].max():+.4f}]')

    # ── 3. Windows ────────────────────────────────────────────────────────
    print('\n[3/7]  Building sliding windows ...')
    X, y = build_windows(df)

    # ── 4. Split ──────────────────────────────────────────────────────────
    print('\n[4/7]  Splitting (time-ordered) ...')
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y)
    print(f'       Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}')

    # ── 5. Normalise ──────────────────────────────────────────────────────
    print('\n[5/7]  Fitting normaliser on training set only ...')
    norm_mean, norm_std = fit_normaliser(X_train)

    X_train_n = apply_normaliser(X_train, norm_mean, norm_std)
    X_val_n   = apply_normaliser(X_val,   norm_mean, norm_std)
    X_test_n  = apply_normaliser(X_test,  norm_mean, norm_std)

    chk = X_train_n.reshape(-1, INPUT_SIZE)
    print(f'       Post-norm  mean={chk.mean():.5f}  std={chk.std():.5f}'
          f'  (should be ≈ 0.0, ≈ 1.0)')

    norm_path = os.path.join(MODEL_DIR, 'lstm_norm_params.npz')
    np.savez(norm_path,
             mean         = norm_mean,
             std          = norm_std,
             feature_cols = np.array(FEATURE_COLS))
    print(f'       Saved  : {norm_path}')

    # ── 6. Model + training setup ─────────────────────────────────────────
    print('\n[6/7]  Building model ...')
    train_loader = DataLoader(
        DisplacementDataset(X_train_n, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        DisplacementDataset(X_val_n, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    test_loader = DataLoader(
        DisplacementDataset(X_test_n, y_test),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )

    model = LSTMDriftCompensator(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'       Trainable parameters : {n_params:,}')

    # SmoothL1Loss = Huber: quadratic for |e| < 1, linear beyond — robust to outliers
    criterion = nn.SmoothL1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    # ── 7. Training loop ──────────────────────────────────────────────────
    print('\n[7/7]  Training ...\n')
    hdr = (f"  {'Epoch':>5}  {'Train':>11}  {'Val':>11}  "
           f"{'Val RMSE':>10}  {'LR':>10}  Note")
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))

    model_path = os.path.join(MODEL_DIR, 'lstm_drift.pt')
    log_path   = os.path.join(MODEL_DIR, 'lstm_training_log.csv')

    train_losses, val_losses, val_rmses, log_rows = [], [], [], []
    best_val_loss = float('inf')
    best_epoch    = 1
    es_counter    = 0
    t0            = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_loss              = train_one_epoch(
            model, train_loader, optimiser, criterion, device)
        v_loss, v_rmse, _, _ = evaluate(
            model, val_loader, criterion, device)

        scheduler.step(v_loss)
        lr = float(optimiser.param_groups[0]['lr'])

        train_losses.append(float(t_loss))
        val_losses.append(float(v_loss))
        val_rmses.append(float(v_rmse))
        log_rows.append({'epoch': epoch,
                         'train_loss': round(t_loss, 8),
                         'val_loss':   round(v_loss, 8),
                         'val_rmse_m': round(v_rmse, 6),
                         'lr':         round(lr,     8)})

        is_best = v_loss < best_val_loss
        if is_best:
            best_val_loss = v_loss
            best_epoch    = epoch
            es_counter    = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_loss'   : float(v_loss),
                'val_rmse'   : float(v_rmse),
                'hparams'    : {
                    'input_size'   : INPUT_SIZE,
                    'hidden_size'  : HIDDEN_SIZE,
                    'num_layers'   : NUM_LAYERS,
                    'dropout'      : DROPOUT,
                    'seq_len'      : SEQ_LEN,
                    'feature_cols' : FEATURE_COLS,
                },
            }, model_path)
        else:
            es_counter += 1

        note = '← BEST' if is_best else ''
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"  {epoch:>5}  {t_loss:>11.6f}  {v_loss:>11.6f}  "
                  f"{v_rmse:>9.4f} m  {lr:>10.2e}  {note}")

        if es_counter >= ES_PATIENCE:
            print(f"\n  [Early Stop]  No improvement for {ES_PATIENCE} epochs "
                  f"— stopped at epoch {epoch}.")
            break

    # ── Save log ──────────────────────────────────────────────────────────
    with open(log_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f'\n  Loading best checkpoint (epoch {best_epoch}) ...')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    _, test_rmse, test_preds, test_trues = evaluate(
        model, test_loader, criterion, device)

    rmse_x = float(np.sqrt(np.mean((test_preds[:, 0] - test_trues[:, 0]) ** 2)))
    rmse_y = float(np.sqrt(np.mean((test_preds[:, 1] - test_trues[:, 1]) ** 2)))
    mae_x  = float(np.mean(np.abs(test_preds[:, 0] - test_trues[:, 0])))
    mae_y  = float(np.mean(np.abs(test_preds[:, 1] - test_trues[:, 1])))

    print(f'\n{"=" * 68}')
    print(f'  TEST RESULTS   (best model: epoch {best_epoch})')
    print(f'{"=" * 68}')
    print(f'  2D RMSE       : {test_rmse:.4f} m')
    print(f'  RMSE  Δx      : {rmse_x:.4f} m')
    print(f'  RMSE  Δy      : {rmse_y:.4f} m')
    print(f'  MAE   Δx      : {mae_x:.4f} m')
    print(f'  MAE   Δy      : {mae_y:.4f} m')
    print(f'  Training time : {time.time() - t0:.0f} s')
    print(f'{"=" * 68}')
    print(f'  Saved:')
    print(f'    {model_path}')
    print(f'    {norm_path}')
    print(f'    {log_path}')

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_training_curves(
        train_losses, val_losses, val_rmses,
        os.path.join(PLOT_DIR, 'lstm_training_curves.png'))
    plot_test_predictions(
        test_preds, test_trues,
        os.path.join(PLOT_DIR, 'lstm_test_predictions.png'))

    print('\n  DONE.  Next step → kalman_filter.py  then  fusion.py')


if __name__ == '__main__':
    main()
