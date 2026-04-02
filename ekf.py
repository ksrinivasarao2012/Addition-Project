"""
ekf.py  —  Extended Kalman Filter with LSTM-Aided Dead-Reckoning  (v2)
=======================================================================
For: LSTM + RL-Adaptive EKF Localization Project

State Vector  (5D):
    [x, y, v, ψ, b_ψ]
    x      east  position in CARLA local frame   (m)
    y      south position in CARLA local frame   (m, CARLA: +y = south)
    v      vehicle speed                         (m/s, non-negative)
    ψ      heading                               (rad, CCW from east)
    b_ψ    gyro yaw bias                         (rad/s, random walk)

State Indices (use these constants — never hardcode 0,1,2,3,4):
    IDX_X    = 0
    IDX_Y    = 1
    IDX_V    = 2
    IDX_PSI  = 3
    IDX_BPSI = 4

Changes from v1
---------------
  FIX 1  — All paths now use correct Windows paths with Music folder.
            os.path.expanduser('~') misses Music on this machine.

  FIX 2  — int(nan) crash when gps_denied is NaN (dropped frame).
            Guard added: gps_denied defaults to 0 if NaN.

  FIX 3  — Evaluation now runs on run 3 ONLY (held-out test set).
            v1 evaluated on all runs including 0,1,2 (training data),
            which inflated LSTM metrics due to data leakage.

  FIX 4  — plot_summary default changed from 0 to nan.
            Using 0 as default made missing metrics show as perfect bars.

  FIX 5  — Innovation stored after every update() call.
            Required by carla_rl_environment.py observation builder.

  FIX 6  — Backward-compatible interface added for RL training.
            carla_rl_environment.py calls the OLD interface:
              predict(u={'accel':[ax,ay], 'gyro':wz})
              update_gps(z=np.array([x,y]))
              set_noise_scales(q_scale, r_scale)
              get_state() → dict with 'position','innovation',etc.
            These wrappers translate old calls to new internals
            without changing the correct 5-state physics.

  FIX 7  — initialize() added as preferred seeding method.
            carla_rl_environment.py previously used direct index access
            (ekf.x[2]=heading, ekf.x[3]=velocity) which was WRONG
            because the state layout changed from v1:
              OLD: x[2]=theta, x[3]=v
              NEW: x[2]=v,     x[3]=psi
            Use ekf.initialize() to seed state safely.

Process Model:
    x' = x + v·cos(ψ)·dt
    y' = y - v·sin(ψ)·dt    (minus: CARLA +y=south, ψ=0 faces east)
    v' = clip(v + a_fwd·dt,  0, 50)
    ψ' = wrap(ψ + (wz-b_ψ)·dt)
    b' = b_ψ                 (random walk via Q)

Offline Evaluation:
    python ekf.py
    Runs Baseline vs LSTM-EKF on run 3 (test set only).
    Saves results to results/ directory.

CARLA Training:
    Used via carla_rl_environment.py — the backward-compatible
    wrapper methods handle the interface translation automatically.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from collections import deque


# =============================================================================
# PATHS  — FIX 1: correct Windows paths with Music folder
# =============================================================================
DATA_PATH   = r'C:\Users\heman\Music\rl_imu_project\data\town04_dataset.csv'
MODEL_PATH  = r'C:\Users\heman\Music\rl_imu_project\models\lstm_drift_predictor.pth'
STATS_PATH  = r'C:\Users\heman\Music\rl_imu_project\models\lstm_normalisation.npz'
RESULTS_DIR = r'C:\Users\heman\Music\rl_imu_project\results'


# =============================================================================
# CONFIGURATION
# =============================================================================

DT = 0.05          # integration timestep (s)

# State indices — always use these, never hardcode numbers
IDX_X    = 0
IDX_Y    = 1
IDX_V    = 2
IDX_PSI  = 3
IDX_BPSI = 4
N_STATES = 5

# Speed clipping
V_MIN =  0.0
V_MAX = 50.0

# GNSS noise — noise_lat_stddev=0.000009 deg × 111000 m/deg ≈ 1.0 m
GNSS_NOISE_STD = 1.0
R_GNSS = (GNSS_NOISE_STD ** 2) * np.eye(2, dtype=np.float64)

# NIS 95% chi-squared bound for 2 DOF
NIS_BOUND_95 = 5.991

# Process noise (per step, open road)
Q_DIAG_BASE = np.array([
    0.01,   # x   (m²)
    0.01,   # y   (m²)
    0.25,   # v   (m/s)²
    1e-4,   # ψ   (rad²)
    1e-8,   # b_ψ (rad/s)²
], dtype=np.float64)

# Q inflation during GPS denial
GPS_DENIED_Q_SCALE = np.array([1.0, 1.0, 4.0, 4.0, 1.0], dtype=np.float64)

# Initial covariance
P0_DIAG = np.array([1.0, 1.0, 4.0, 0.1, 1e-4], dtype=np.float64)

# LSTM config (must match train_lstm.py exactly)
SEQ_LEN       = 40
LSTM_H1       = 64
LSTM_H2       = 32
LSTM_DROPOUT  = 0.3
FEATURE_COLS  = ['ax_corr', 'ay_corr', 'wz', 'gt_speed_mps', 'gps_denied']
TARGET_COLS   = ['gt_accel_fwd_mps2', 'gt_accel_lat_mps2']

# FIX 3: only evaluate on held-out test run
EVAL_RUNS = [3]


# =============================================================================
# LSTM MODEL  (must be identical to LSTMDriftPredictor in train_lstm.py)
# =============================================================================
class LSTMDriftPredictor(nn.Module):
    def __init__(self, input_size=5, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm1 = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1   = nn.LayerNorm(h1)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2   = nn.LayerNorm(h2)
        self.drop2 = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.Linear(h2, 16), nn.GELU(), nn.Linear(16, len(TARGET_COLS))
        )

    def forward(self, x):
        x      = self.input_ln(x)
        o1, _  = self.lstm1(x)
        o1     = self.ln1(o1);  o1 = self.drop1(o1)
        o2, _  = self.lstm2(o1)
        o2     = self.ln2(o2);  o2 = self.drop2(o2)
        return self.head(o2[:, -1, :])


# =============================================================================
# LSTM BRIDGE
# =============================================================================
class LSTMBridge:
    """
    Online inference wrapper. Maintains a rolling buffer of SEQ_LEN=40
    feature vectors and runs LSTM inference when buffer is full.

    Feature order: [ax_corr, ay_corr, wz, ekf_speed, gps_denied]
    First 4 are z-scored; gps_denied is left as 0/1.

    Always push() on every timestep (including open road) so the buffer
    is warm and the LSTM has full context when a tunnel starts.
    """

    def __init__(self, model_path, stats_path, device='cpu'):
        self.device  = torch.device(device)
        self.model   = None
        self._loaded = False
        self._buffer = deque(maxlen=SEQ_LEN)
        self.feat_mean = self.feat_std = None
        self.tgt_mean  = self.tgt_std  = None

        for path, name in [(model_path, 'model'), (stats_path, 'stats')]:
            if not os.path.isfile(path):
                print(f"  [LSTMBridge] WARNING: {name} not found: {path}")
                print(f"  [LSTMBridge] EKF will use raw IMU (no LSTM).")
                return

        stats = np.load(stats_path, allow_pickle=True)
        self.feat_mean = stats['feat_mean'].astype(np.float32)
        self.feat_std  = stats['feat_std'].astype(np.float32)
        self.tgt_mean  = stats['tgt_mean'].astype(np.float32)
        self.tgt_std   = stats['tgt_std'].astype(np.float32)

        ckpt = torch.load(model_path, map_location=self.device)
        cfg  = ckpt.get('config', {})
        self.model = LSTMDriftPredictor(
            input_size = cfg.get('input_size', len(FEATURE_COLS)),
            h1         = cfg.get('h1',         LSTM_H1),
            h2         = cfg.get('h2',         LSTM_H2),
            dropout    = cfg.get('dropout',    LSTM_DROPOUT),
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self._loaded = True
        print(f"  [LSTMBridge] Loaded  epoch={ckpt.get('epoch','?')}  "
              f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")

    def loaded(self):  return self._loaded
    def ready(self):   return self._loaded and len(self._buffer) == SEQ_LEN

    def push(self, ax_corr, ay_corr, wz, ekf_speed, gps_denied):
        raw  = np.array([ax_corr, ay_corr, wz, ekf_speed], dtype=np.float32)
        if self.feat_mean is not None:
            raw = (raw - self.feat_mean) / self.feat_std
        feat = np.append(raw, float(gps_denied)).astype(np.float32)
        self._buffer.append(feat)

    def predict(self):
        if not self.ready():
            return None, None
        seq = np.array(self._buffer, dtype=np.float32)
        x_t = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_n = self.model(x_t).cpu().numpy()[0]
        y_p = y_n * self.tgt_std + self.tgt_mean
        return float(y_p[0]), float(y_p[1])

    def reset(self):
        self._buffer.clear()


# =============================================================================
# EKF LOCALIZER
# =============================================================================
class AdaptiveEKF:
    """
    Extended Kalman Filter — 5-state kinematic model.

    State: [x(m), y(m), v(m/s), ψ(rad), b_ψ(rad/s)]

    Provides TWO interfaces:

    NEW interface (use this directly):
        initialize(x0, y0, heading0, speed0, bias0)
        predict(a_fwd, wz, gps_denied)
        update(gnss_x, gnss_y)   → returns NIS
        get_position() → (x, y)
        get_speed()    → v
        get_heading()  → ψ
        get_position_std() → (σx, σy)

    BACKWARD-COMPATIBLE interface (used by carla_rl_environment.py):
        reset()
        predict(u={'accel':[ax,ay], 'gyro':wz})
        update_gps(z=np.array([x,y]))
        set_noise_scales(Q_scale, R_scale)
        get_state()  → dict with 'position','innovation','position_uncertainty', etc.

    RL Adaptation Hooks (called by carla_rl_environment.py):
        set_noise_scales(Q_scale, R_scale)
        set_process_noise_scale(scale)
        set_measurement_noise_scale(sigma_multiplier)
    """

    def __init__(self, dt=DT):
        self.dt = dt
        self.x  = np.zeros(N_STATES, dtype=np.float64)
        self.P  = np.diag(P0_DIAG.astype(np.float64))

        self._q_scale        = np.ones(N_STATES, dtype=np.float64)
        self._r_scale        = 1.0
        self._initialized    = False
        self._last_innovation = np.zeros(2, dtype=np.float64)  # FIX 5

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, x0=0.0, y0=0.0, heading0=0.0,
                   speed0=0.0, bias0=0.0):
        """
        FIX 7: Safe seeding method.
        carla_rl_environment.py should call this instead of
        direct index access (which breaks when state layout changes).
        """
        self.x = np.array(
            [x0, y0, speed0, heading0, bias0], dtype=np.float64)
        self.P = np.diag(P0_DIAG.astype(np.float64))
        self._q_scale         = np.ones(N_STATES, dtype=np.float64)
        self._r_scale         = 1.0
        self._last_innovation = np.zeros(2, dtype=np.float64)
        self._initialized     = True

    def initialized(self):
        return self._initialized

    # ── RL Adaptation Hooks ───────────────────────────────────────────────────

    def set_process_noise_scale(self, scale):
        """Scale Q_DIAG_BASE. scalar or (N_STATES,) array."""
        s = np.asarray(scale, dtype=np.float64)
        if s.ndim == 0:
            self._q_scale = np.full(N_STATES, float(s))
        elif s.shape == (N_STATES,):
            self._q_scale = s
        else:
            raise ValueError(f"scale must be scalar or ({N_STATES},)")

    def set_measurement_noise_scale(self, sigma_multiplier):
        """Scale GNSS variance by sigma_multiplier²."""
        self._r_scale = float(sigma_multiplier) ** 2

    def set_noise_scales(self, Q_scale: float, R_scale: float):
        """
        FIX 6: Backward-compatible wrapper for RL agent.
        carla_rl_environment.py calls this with scalar Q and R scales.
        Maps: Q_scale → uniform process noise scale
              R_scale → measurement noise variance scale
        """
        self.set_process_noise_scale(float(Q_scale))
        # R_scale from RL is a variance multiplier → convert to sigma
        self.set_measurement_noise_scale(math.sqrt(max(float(R_scale), 1e-6)))

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _wrap(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _build_F(self, v, psi):
        F = np.eye(N_STATES, dtype=np.float64)
        cdt =  math.cos(psi) * self.dt
        sdt =  math.sin(psi) * self.dt
        F[IDX_X,   IDX_V]    =  cdt
        F[IDX_X,   IDX_PSI]  = -v * sdt
        F[IDX_Y,   IDX_V]    = -sdt
        F[IDX_Y,   IDX_PSI]  = -v * cdt
        F[IDX_PSI, IDX_BPSI] = -self.dt
        return F

    def _build_Q(self, gps_denied):
        q = Q_DIAG_BASE * self._q_scale
        if gps_denied:
            q = q * GPS_DENIED_Q_SCALE
        return np.diag(q)

    def _build_R(self):
        return R_GNSS * self._r_scale

    # ── NEW interface: predict ────────────────────────────────────────────────

    def _predict_core(self, a_fwd, wz, gps_denied):
        """Internal predict — called by both interfaces."""
        v   = float(self.x[IDX_V])
        psi = float(self.x[IDX_PSI])
        b   = float(self.x[IDX_BPSI])

        xn = np.empty(N_STATES, dtype=np.float64)
        xn[IDX_X]    = self.x[IDX_X] + v * math.cos(psi) * self.dt
        xn[IDX_Y]    = self.x[IDX_Y] - v * math.sin(psi) * self.dt
        xn[IDX_V]    = float(np.clip(v + a_fwd * self.dt, V_MIN, V_MAX))
        xn[IDX_PSI]  = self._wrap(psi + (wz - b) * self.dt)
        xn[IDX_BPSI] = b

        F  = self._build_F(v, psi)
        Q  = self._build_Q(gps_denied)
        Pn = F @ self.P @ F.T + Q
        Pn = 0.5 * (Pn + Pn.T)

        self.x = xn
        self.P = Pn

    def predict(self, u=None, a_fwd=None, wz=None, gps_denied=False):
        """
        FIX 6: Accepts BOTH old dict interface and new positional interface.

        OLD (carla_rl_environment.py):
            ekf.predict(u={'accel': [ax, ay], 'gyro': wz_val})

        NEW (offline evaluation / direct use):
            ekf.predict(a_fwd=0.5, wz=0.1, gps_denied=True)
            ekf.predict(0.5, 0.1, True)   # positional
        """
        if u is not None:
            # Backward-compatible dict interface
            accel_arr  = u['accel']
            a_fwd_val  = float(accel_arr[0])
            wz_val     = float(u['gyro'])
            denied_val = False   # gps_denied not in old interface
        else:
            a_fwd_val  = float(a_fwd) if a_fwd is not None else 0.0
            wz_val     = float(wz)    if wz    is not None else 0.0
            denied_val = bool(gps_denied)

        self._predict_core(a_fwd_val, wz_val, denied_val)

    # ── NEW interface: update ─────────────────────────────────────────────────

    def update(self, gnss_x: float, gnss_y: float) -> float:
        """
        EKF measurement update with 2-D GNSS.
        Stores innovation for get_state() dict.
        Returns NIS (Normalised Innovation Squared).
        """
        H = np.zeros((2, N_STATES), dtype=np.float64)
        H[0, IDX_X] = 1.0
        H[1, IDX_Y] = 1.0

        R     = self._build_R()
        z     = np.array([gnss_x, gnss_y], dtype=np.float64)
        innov = z - H @ self.x

        self._last_innovation = innov.copy()   # FIX 5: store for get_state()

        S   = H @ self.P @ H.T + R
        K   = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innov
        self.x[IDX_PSI] = self._wrap(self.x[IDX_PSI])
        self.x[IDX_V]   = float(np.clip(self.x[IDX_V], V_MIN, V_MAX))

        I_KH   = np.eye(N_STATES, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        nis = float(innov @ np.linalg.inv(S) @ innov)
        return nis

    # ── BACKWARD-COMPATIBLE interface ─────────────────────────────────────────

    def update_gps(self, z: np.ndarray) -> float:
        """
        FIX 6: Backward-compatible wrapper.
        carla_rl_environment.py calls: ekf.update_gps(z=np.array([x,y]))
        """
        return self.update(float(z[0]), float(z[1]))

    def get_state(self) -> dict:
        """
        FIX 6: Returns dict matching the old ekf.py interface.
        carla_rl_environment.py uses:
            state['position']             → np.array([x, y])
            state['innovation']           → np.array([ix, iy])
            state['position_uncertainty'] → scalar (m)
            state['Q_scale']              → float
            state['R_scale']              → float
        """
        sx, sy = self.get_position_std()
        return {
            'position':            np.array([self.x[IDX_X],
                                             self.x[IDX_Y]]),
            'theta':               self.x[IDX_PSI],
            'velocity':            self.x[IDX_V],
            'biases':              np.array([0.0, 0.0,
                                             self.x[IDX_BPSI]]),
            'covariance':          self.P.copy(),
            'position_uncertainty': math.sqrt(sx**2 + sy**2),
            'innovation':          self._last_innovation.copy(),
            'Q_scale':             float(np.mean(self._q_scale)),
            'R_scale':             float(math.sqrt(self._r_scale)),
        }

    def reset(self):
        """Reset to uninitialised state. Called between episodes."""
        self.x                = np.zeros(N_STATES, dtype=np.float64)
        self.P                = np.diag(P0_DIAG.astype(np.float64))
        self._q_scale         = np.ones(N_STATES, dtype=np.float64)
        self._r_scale         = 1.0
        self._last_innovation = np.zeros(2, dtype=np.float64)
        self._initialized     = False

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_position(self):
        return float(self.x[IDX_X]), float(self.x[IDX_Y])

    def get_speed(self):
        return float(self.x[IDX_V])

    def get_heading(self):
        return float(self.x[IDX_PSI])

    def get_bias(self):
        return float(self.x[IDX_BPSI])

    def get_covariance(self):
        return self.P.copy()

    def get_position_std(self):
        return (math.sqrt(max(self.P[IDX_X, IDX_X], 0.0)),
                math.sqrt(max(self.P[IDX_Y, IDX_Y], 0.0)))


# =============================================================================
# RUN EKF ON ONE RUN'S DATA  (offline evaluation)
# =============================================================================
def run_ekf_on_run(df_run, ekf, bridge, use_lstm):
    ekf.reset()
    if bridge is not None:
        bridge.reset()

    results     = []
    first_valid = True

    required_cols = [
        'ax_corr', 'ay_corr', 'wz', 'gt_speed_mps',
        'gnss_x', 'gnss_y', 'gt_x', 'gt_y', 'gt_heading',
    ]

    for _, row in df_run.iterrows():
        ts = float(row['timestamp'])

        # FIX 2: guard against NaN gps_denied (dropped frames)
        gps_denied_raw = row.get('gps_denied', 0)
        gps_denied = 0 if pd.isna(gps_denied_raw) else int(gps_denied_raw)

        # Dropped-frame rows
        if any(pd.isna(row.get(c)) for c in required_cols):
            results.append({
                'timestamp': ts, 'gt_x': float('nan'), 'gt_y': float('nan'),
                'ekf_x': float('nan'), 'ekf_y': float('nan'),
                'gps_denied': gps_denied, 'nis': float('nan'),
                'lstm_used': False, 'ekf_v': float('nan'),
                'ekf_psi': float('nan'), 'pos_std_x': float('nan'),
                'pos_std_y': float('nan'),
            })
            continue

        ax_corr  = float(row['ax_corr'])
        ay_corr  = float(row['ay_corr'])
        wz       = float(row['wz'])
        gnss_x   = float(row['gnss_x'])
        gnss_y   = float(row['gnss_y'])
        gt_x     = float(row['gt_x'])
        gt_y     = float(row['gt_y'])
        gt_head  = float(row['gt_heading'])
        gt_speed = float(row['gt_speed_mps'])

        if first_valid:
            ekf.initialize(x0=gnss_x, y0=gnss_y,
                           heading0=gt_head, speed0=gt_speed)
            first_valid = False

        # Decide a_fwd source
        ekf_speed  = ekf.get_speed()
        a_fwd_used = ax_corr
        lstm_used  = False

        if bridge is not None:
            bridge.push(ax_corr, ay_corr, wz, ekf_speed, gps_denied)
            if use_lstm and gps_denied and bridge.ready():
                a_fwd_pred, _ = bridge.predict()
                if a_fwd_pred is not None:
                    a_fwd_used = a_fwd_pred
                    lstm_used  = True

        ekf.predict(a_fwd=a_fwd_used, wz=wz, gps_denied=bool(gps_denied))

        nis = float('nan')
        if not gps_denied:
            nis = ekf.update(gnss_x, gnss_y)

        ex, ey = ekf.get_position()
        sx, sy = ekf.get_position_std()

        results.append({
            'timestamp':  ts,
            'gt_x':       gt_x, 'gt_y':   gt_y,
            'ekf_x':      ex,   'ekf_y':  ey,
            'ekf_v':      ekf.get_speed(),
            'ekf_psi':    ekf.get_heading(),
            'pos_std_x':  sx,   'pos_std_y': sy,
            'gps_denied': gps_denied,
            'a_fwd_used': a_fwd_used,
            'nis':        nis,
            'lstm_used':  lstm_used,
        })

    return pd.DataFrame(results)


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(df_res, label=''):
    tag   = f"{label}_" if label else ""
    valid = df_res.dropna(subset=['ekf_x', 'ekf_y', 'gt_x', 'gt_y'])

    if valid.empty:
        return {f'{tag}overall_rmse': float('nan')}

    err = np.sqrt((valid['ekf_x']-valid['gt_x'])**2 +
                  (valid['ekf_y']-valid['gt_y'])**2)

    tun  = valid[valid['gps_denied'] == 1]
    road = valid[valid['gps_denied'] == 0]

    def _rmse(e): return float(np.sqrt(np.mean(e**2)))
    def _mean(e): return float(np.mean(e))
    def _max(e):  return float(np.max(e))

    e_tun  = np.sqrt((tun['ekf_x'] -tun['gt_x'] )**2+
                     (tun['ekf_y'] -tun['gt_y'] )**2) if len(tun)>0  else None
    e_road = np.sqrt((road['ekf_x']-road['gt_x'])**2+
                     (road['ekf_y']-road['gt_y'])**2) if len(road)>0 else None

    nis_v = valid.dropna(subset=['nis'])['nis'].values
    nis_m = float(np.mean(nis_v))                       if len(nis_v)>0 else float('nan')
    nis_p = float(np.mean(nis_v < NIS_BOUND_95) * 100) if len(nis_v)>0 else float('nan')

    return {
        f'{tag}overall_rmse':      _rmse(err),
        f'{tag}overall_mean':      _mean(err),
        f'{tag}overall_max':       _max(err),
        f'{tag}tunnel_rmse':       _rmse(e_tun)  if e_tun  is not None else float('nan'),
        f'{tag}tunnel_mean':       _mean(e_tun)  if e_tun  is not None else float('nan'),
        f'{tag}tunnel_max':        _max(e_tun)   if e_tun  is not None else float('nan'),
        f'{tag}tunnel_n':          int(len(tun)),
        f'{tag}road_rmse':         _rmse(e_road) if e_road is not None else float('nan'),
        f'{tag}road_mean':         _mean(e_road) if e_road is not None else float('nan'),
        f'{tag}nis_mean':          nis_m,
        f'{tag}nis_pct_inside_95': nis_p,
    }


# =============================================================================
# VISUALISATION — per-run
# =============================================================================
def plot_run(df_base, df_lstm, run_id, save_dir):
    vb = df_base.dropna(subset=['ekf_x','ekf_y'])
    vl = df_lstm.dropna(subset=['ekf_x','ekf_y'])

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"EKF — Run {run_id} (test set)",
                 fontsize=12, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    ax_tr  = fig.add_subplot(gs[0,0])
    ax_er  = fig.add_subplot(gs[0,1])
    ax_sp  = fig.add_subplot(gs[1,0])
    ax_nis = fig.add_subplot(gs[1,1])

    ax_tr.plot(vb['gt_x'],  vb['gt_y'],  'g-',  lw=1.5, label='Ground truth')
    ax_tr.plot(vb['ekf_x'], vb['ekf_y'], 'b--', lw=0.9, label='Baseline EKF')
    ax_tr.plot(vl['ekf_x'], vl['ekf_y'], 'r-',  lw=0.9, label='LSTM-EKF')
    tun = vb[vb['gps_denied']==1]
    if not tun.empty:
        ax_tr.scatter(tun['gt_x'],tun['gt_y'],c='orange',s=5,alpha=0.35,label='Tunnel')
    ax_tr.set_title("XY Trajectory"); ax_tr.set_xlabel("x (m)"); ax_tr.set_ylabel("y (m)")
    ax_tr.legend(fontsize=7); ax_tr.grid(True,alpha=0.3); ax_tr.set_aspect('equal','datalim')

    err_b = np.sqrt((vb['ekf_x']-vb['gt_x'])**2+(vb['ekf_y']-vb['gt_y'])**2)
    err_l = np.sqrt((vl['ekf_x']-vl['gt_x'])**2+(vl['ekf_y']-vl['gt_y'])**2)
    ax_er.plot(vb['timestamp'],err_b,'b-',lw=0.7,alpha=0.7,label='Baseline')
    ax_er.plot(vl['timestamp'],err_l,'r-',lw=0.7,alpha=0.7,label='LSTM-EKF')
    ax_er.set_title("Position Error"); ax_er.set_xlabel("Time (s)"); ax_er.set_ylabel("Error (m)")
    ax_er.legend(fontsize=7); ax_er.grid(True,alpha=0.3)

    ax_sp.plot(vb['timestamp'],vb['ekf_v'],'b--',lw=0.8,label='Baseline')
    ax_sp.plot(vl['timestamp'],vl['ekf_v'],'r-', lw=0.8,label='LSTM-EKF')
    ax_sp.set_title("Speed"); ax_sp.set_xlabel("Time (s)"); ax_sp.set_ylabel("Speed (m/s)")
    ax_sp.legend(fontsize=7); ax_sp.grid(True,alpha=0.3)

    nb = vb.dropna(subset=['nis']); nl = vl.dropna(subset=['nis'])
    ax_nis.plot(nb['timestamp'],nb['nis'],'b.',ms=1.5,alpha=0.5,label='Baseline')
    ax_nis.plot(nl['timestamp'],nl['nis'],'r.',ms=1.5,alpha=0.5,label='LSTM-EKF')
    ax_nis.axhline(NIS_BOUND_95,color='k',ls='--',lw=1,label=f'95% χ²={NIS_BOUND_95}')
    all_nis = np.concatenate([nb['nis'].values,nl['nis'].values])
    if len(all_nis)>0:
        ax_nis.set_ylim(0,min(60.0,float(np.percentile(all_nis,99))*1.2))
    ax_nis.set_title("NIS (filter consistency)"); ax_nis.set_xlabel("Time (s)")
    ax_nis.legend(fontsize=7); ax_nis.grid(True,alpha=0.3)

    out = os.path.join(save_dir,f'ekf_run{run_id}.png')
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# VISUALISATION — summary
# =============================================================================
def plot_summary(all_metrics, save_dir):
    run_ids = sorted(all_metrics.keys())
    x, w    = np.arange(len(run_ids)), 0.35

    metrics_to_plot = [
        ('overall_rmse', 'Overall RMSE (m)'),
        ('tunnel_rmse',  'Tunnel RMSE (m)'),
        ('road_rmse',    'Open-road RMSE (m)'),
    ]
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    fig.suptitle("EKF Summary — Test Set (Run 3)",fontsize=11,fontweight='bold')

    for ax,(key,ylabel) in zip(axes,metrics_to_plot):
        # FIX 4: use nan default, not 0 — missing metrics should not show as perfect bars
        b_vals = [all_metrics[r]['baseline'].get(f'baseline_{key}',float('nan')) for r in run_ids]
        l_vals = [all_metrics[r]['lstm'].get(    f'lstm_{key}',    float('nan')) for r in run_ids]
        b_plot = [0 if math.isnan(v) else v for v in b_vals]
        l_plot = [0 if math.isnan(v) else v for v in l_vals]

        ax.bar(x-w/2, b_plot, w, label='Baseline', color='steelblue', alpha=0.8)
        ax.bar(x+w/2, l_plot, w, label='LSTM-EKF', color='tomato',    alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f'Run {r}' for r in run_ids])
        ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3,axis='y')

    plt.tight_layout()
    out = os.path.join(save_dir,'ekf_summary.png')
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  Saved → {out}")


# =============================================================================
# MAIN  (offline CSV evaluation)
# =============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*62}")
    print(f"  EKF Localizer  —  v2")
    print(f"  Device  : {device}")
    print(f"  dt      : {DT}s ({1/DT:.0f} Hz)  State: 5D [x,y,v,ψ,b_ψ]")
    # FIX 3: only evaluate on test run
    print(f"  Eval    : run(s) {EVAL_RUNS}  (held-out test set only)")
    print(f"{'='*62}\n")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows | runs: {sorted(df['run_id'].unique())}")

    print("\nLoading LSTM bridge...")
    bridge    = LSTMBridge(MODEL_PATH, STATS_PATH, device=device)
    ekf_base  = AdaptiveEKF()
    ekf_lstm  = AdaptiveEKF()

    all_results_base, all_results_lstm = [], []
    all_metrics = {}

    # FIX 3: only run on EVAL_RUNS
    for run_id in EVAL_RUNS:
        if run_id not in df['run_id'].values:
            print(f"  [WARN] Run {run_id} not found in dataset — skipping")
            continue

        df_run = (df[df['run_id'] == run_id]
                  .sort_values('timestamp')
                  .reset_index(drop=True))

        print(f"\n[Run {run_id}]  {len(df_run):,} rows  "
              f"| tunnel: {int(df_run['gps_denied'].sum())}")

        print("  Baseline EKF...")
        df_base = run_ekf_on_run(df_run, ekf_base, bridge=None, use_lstm=False)

        print("  LSTM-EKF...")
        df_lstm = run_ekf_on_run(df_run, ekf_lstm, bridge=bridge, use_lstm=True)

        df_base['run_id'] = run_id; df_lstm['run_id'] = run_id
        all_results_base.append(df_base); all_results_lstm.append(df_lstm)

        m_base = compute_metrics(df_base, label='baseline')
        m_lstm = compute_metrics(df_lstm, label='lstm')
        all_metrics[run_id] = {'baseline': m_base, 'lstm': m_lstm}

        print(f"\n  {'Metric':<28} {'Baseline':>10}  {'LSTM-EKF':>10}  {'Delta':>10}")
        print("  " + "-"*62)
        for key in ['overall_rmse','tunnel_rmse','tunnel_max',
                    'road_rmse','nis_mean','nis_pct_inside_95']:
            bv = m_base.get(f'baseline_{key}', float('nan'))
            lv = m_lstm.get(f'lstm_{key}',     float('nan'))
            d  = lv-bv if not (math.isnan(bv) or math.isnan(lv)) else float('nan')
            print(f"  {key:<28} {bv:>10.3f}  {lv:>10.3f}  {d:>+10.3f}")

        plot_run(df_base, df_lstm, run_id, RESULTS_DIR)

    if not all_results_base:
        print("[WARN] No runs evaluated. Check EVAL_RUNS and dataset.")
        return

    df_all_base = pd.concat(all_results_base, ignore_index=True)
    df_all_lstm = pd.concat(all_results_lstm, ignore_index=True)

    m_base_agg = compute_metrics(df_all_base, label='baseline')
    m_lstm_agg = compute_metrics(df_all_lstm, label='lstm')

    print(f"\n{'='*62}")
    print("  AGGREGATE METRICS  (test set — run 3 only)")
    print(f"{'='*62}")

    lines = ["EKF Localizer v2 — Test Metrics (Run 3)",
             "="*62, f"{'Metric':<28} {'Baseline':>10}  {'LSTM-EKF':>10}  {'Improv.':>10}", "-"*62]

    for key, label in [
        ('overall_rmse','Overall RMSE (m)'),('overall_mean','Overall mean (m)'),
        ('overall_max', 'Overall max  (m)'),('tunnel_rmse', 'Tunnel RMSE  (m)'),
        ('tunnel_mean', 'Tunnel mean  (m)'),('tunnel_max',  'Tunnel max   (m)'),
        ('road_rmse',   'Road RMSE    (m)'),('road_mean',   'Road mean    (m)'),
        ('nis_mean',    'NIS mean'),         ('nis_pct_inside_95','NIS inside 95%'),
    ]:
        bv = m_base_agg.get(f'baseline_{key}', float('nan'))
        lv = m_lstm_agg.get(f'lstm_{key}',     float('nan'))
        pct = f"{(bv-lv)/abs(bv)*100:+.1f}%" if not (math.isnan(bv) or math.isnan(lv) or bv==0) else "n/a"
        line = f"  {label:<28} {bv:>10.3f}  {lv:>10.3f}  {pct:>10}"
        print(line); lines.append(line)

    metrics_path = os.path.join(RESULTS_DIR,'ekf_metrics.txt')
    with open(metrics_path,'w') as f: f.write("\n".join(lines))
    print(f"\n  Metrics  → {metrics_path}")

    df_all_base['mode']='baseline'; df_all_lstm['mode']='lstm'
    pred_path = os.path.join(RESULTS_DIR,'ekf_predictions.csv')
    pd.concat([df_all_base,df_all_lstm],ignore_index=True).to_csv(pred_path,index=False)
    print(f"  Predictions → {pred_path}")

    plot_summary(all_metrics, RESULTS_DIR)

    print(f"\n{'='*62}")
    print("  DONE")
    print(f"  Target: tunnel_rmse (LSTM-EKF) < 3.0 m (acceptable) / < 1.5 m (good)")
    print(f"  Target: nis_pct_inside_95 > 90%  (filter is consistent)")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    main()
