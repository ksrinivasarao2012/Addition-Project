"""
ekf.py  —  Extended Kalman Filter with LSTM-Aided Dead-Reckoning  (v3 Final)
============================================================================
For: LSTM + RL-Adaptive EKF Localization Project

WHY THIS FILE EXISTS
--------------------
Two interfaces, one class (AdaptiveEKF):

  1. Offline evaluation  (python ekf.py)
       Runs Baseline vs LSTM-EKF on the held-out test run (run 3).
       No CARLA required. Reads from town04_dataset.csv.

  2. RL training  (imported by carla_rl_environment.py)
       Live CARLA session. The RL agent calls set_noise_scales() every
       step to adapt Q and R based on speed/weather/tunnel context.

State Vector  (5D):
    [x, y, v, psi, b_psi]
    x      east  position in CARLA local frame   (m)
    y      south position in CARLA local frame   (m, CARLA: +y = south)
    v      vehicle speed                         (m/s, >= 0)
    psi    heading                               (rad, CCW from east)
    b_psi  gyro yaw bias                         (rad/s, random walk)

    Why b_psi: collect_data.py sets noise_gyro_bias_{x,y,z} = 0.001 rad/s.
    Without estimating this bias the heading drifts by 0.001 * t rad.
    After a 300 s tunnel: 0.3 rad = 17 deg -> position error blows up.

    Why NOT kalman_filter.py (other AI's 4-state model):
    That file is for a different project (no gyro bias collected).
    Its class ExtendedKalmanFilter is incompatible with collect_data.py
    and carla_rl_environment.py. Do not use it.

State Indices (always use these constants, never hardcode 0,1,2,3,4):
    IDX_X    = 0
    IDX_Y    = 1
    IDX_V    = 2
    IDX_PSI  = 3
    IDX_BPSI = 4

    IMPORTANT: v1 used x[2]=heading, x[3]=speed (WRONG layout).
    v2+ corrected to x[2]=speed, x[3]=heading.
    Always call initialize() to seed state safely.

Process Model:
    x'    = x    + v*cos(psi)*dt
    y'    = y    - v*sin(psi)*dt     minus: CARLA +y is south, psi=0=east
    v'    = clip(v + a_fwd*dt, 0, 50)
    psi'  = wrap(psi + (wz - b_psi)*dt)
    b'    = b_psi                    bias is constant; Q handles slow drift

    y-sign derivation:
        psi=0   -> east  -> dx>0, dy=0      OK
        psi=pi/2 -> north -> dy<0            OK (y decreases southward)

Jacobian F = df/dx  (5x5 exact):
    [[1, 0,  cos(psi)*dt, -v*sin(psi)*dt,  0  ],
     [0, 1, -sin(psi)*dt, -v*cos(psi)*dt,  0  ],
     [0, 0,  1,            0,             0  ],
     [0, 0,  0,            1,            -dt ],
     [0, 0,  0,            0,             1  ]]

Measurement Model (GNSS, 2D, open road only):
    z = [x, y]^T
    H = [[1,0,0,0,0], [0,1,0,0,0]]
    R = diag([sigma_gnss^2, sigma_gnss^2])
    sigma_gnss = 1.0 m  (noise_lat_stddev=0.000009 deg * 111000 m/deg)

Covariance update (Joseph form, numerically stable):
    K     = P*H^T*(H*P*H^T + R)^-1
    I_KH  = I - K*H
    P_new = I_KH*P*I_KH^T + K*R*K^T
    Guarantees P stays symmetric positive-definite over 24000 steps.

Interface A (direct, for offline evaluation and new code):
    ekf.initialize(x0, y0, heading0, speed0, bias0)
    ekf.predict(a_fwd, wz, gps_denied)
    ekf.update(gnss_x, gnss_y)     -> returns NIS float
    ekf.get_position() / get_speed() / get_heading() / get_position_std()

Interface B (backward-compatible, for carla_rl_environment.py):
    ekf.reset()
    ekf.predict(u={'accel':[ax,ay], 'gyro':wz})
    ekf.update_gps(z=np.array([x,y]))
    ekf.set_noise_scales(Q_scale, R_scale)
    ekf.get_state()  -> dict(position, innovation, position_uncertainty, ...)

RL Adaptation Hooks:
    set_process_noise_scale(scale)       scale Q per-state or uniformly
    set_measurement_noise_scale(sigma_m) scale GNSS R
    set_noise_scales(Q_scale, R_scale)   unified RL wrapper

Changes from v2
---------------
  v3-ADD 1  _run_self_test() added. 7 tests: error bounds, PD guarantee,
             NaN guard, RL hook round-trip, Interface B, initialize() layout.
             Run with: python ekf.py --test

  v3-ADD 2  State layout comment block above initialize() documents the
             v1->v2 IDX change to prevent future regressions.

  v3-FIX 1  set_noise_scales math comment added: R_scale is a variance
             multiplier; passing sqrt(R_scale) to set_measurement_noise_scale
             which squares it gives _r_scale=R_scale. Consistent. Auditable.

  v3-FIX 2  plot_summary: explicit isnan guard in bar-height computation
             prevents matplotlib crash when a metric is missing.

  v3-FIX 3  get_state(): 'biases' now returns np.zeros(3) with only
             index 2 set to b_psi (carla_rl_environment.py reads a 3-vector).

Carried from v2:
  FIX 1  Windows paths with Music folder (not os.path.expanduser).
  FIX 2  NaN guard: gps_denied defaults to 0 if NaN (dropped frame).
  FIX 3  Evaluate on run 3 ONLY (held-out test set).
  FIX 4  plot_summary default is nan (not 0).
  FIX 5  Innovation stored after every update() call.
  FIX 6  Backward-compatible dict interface for RL training.
  FIX 7  initialize() as safe seeding method.
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
# PATHS
# =============================================================================
DATA_PATH   = r'C:\Users\heman\Music\rl_imu_project\data\town04_dataset.csv'
MODEL_PATH  = r'C:\Users\heman\Music\rl_imu_project\models\lstm_drift_predictor.pth'
STATS_PATH  = r'C:\Users\heman\Music\rl_imu_project\models\lstm_normalisation.npz'
RESULTS_DIR = r'C:\Users\heman\Music\rl_imu_project\results'


# =============================================================================
# CONFIGURATION
# =============================================================================
DT = 0.05       # integration timestep (s) — must match FIXED_DELTA_T

IDX_X    = 0
IDX_Y    = 1
IDX_V    = 2
IDX_PSI  = 3
IDX_BPSI = 4
N_STATES = 5

V_MIN =  0.0
V_MAX = 50.0

GNSS_NOISE_STD = 1.0
R_GNSS = (GNSS_NOISE_STD ** 2) * np.eye(2, dtype=np.float64)

NIS_BOUND_95 = 5.991

Q_DIAG_BASE = np.array([
    0.01,    # x   (m^2)
    0.01,    # y   (m^2)
    0.25,    # v   (m/s)^2  IMU noise 0.02 m/s^2
    1e-4,    # psi (rad^2)
    1e-8,    # b_psi (rad/s)^2
], dtype=np.float64)

GPS_DENIED_Q_SCALE = np.array([1.0, 1.0, 4.0, 4.0, 1.0], dtype=np.float64)

P0_DIAG = np.array([1.0, 1.0, 4.0, 0.1, 1e-4], dtype=np.float64)

SEQ_LEN      = 40
LSTM_H1      = 64
LSTM_H2      = 32
LSTM_DROPOUT = 0.3
FEATURE_COLS = ['ax_corr', 'ay_corr', 'wz', 'gt_speed_mps', 'gps_denied']
TARGET_COLS  = ['gt_accel_fwd_mps2', 'gt_accel_lat_mps2']

EVAL_RUNS = [3]   # held-out test set only; 0,1,2 are TRAIN runs


# =============================================================================
# LSTM MODEL — must be identical to LSTMDriftPredictor in train_lstm.py
# =============================================================================
class LSTMDriftPredictor(nn.Module):
    def __init__(self, input_size=5, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm1    = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1      = nn.LayerNorm(h1)
        self.drop1    = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2      = nn.LayerNorm(h2)
        self.drop2    = nn.Dropout(dropout)
        self.head     = nn.Sequential(
            nn.Linear(h2, 16), nn.GELU(), nn.Linear(16, len(TARGET_COLS))
        )

    def forward(self, x):
        x     = self.input_ln(x)
        o1, _ = self.lstm1(x);  o1 = self.ln1(o1);  o1 = self.drop1(o1)
        o2, _ = self.lstm2(o1); o2 = self.ln2(o2);  o2 = self.drop2(o2)
        return self.head(o2[:, -1, :])


# =============================================================================
# LSTM BRIDGE
# =============================================================================
class LSTMBridge:
    """
    Online inference wrapper. Maintains a rolling deque(maxlen=40).
    Always push() every timestep (even open road) so the buffer is warm
    when a tunnel starts.

    Feature order: [ax_corr, ay_corr, wz, ekf_speed, gps_denied]
    First 4 are z-scored; gps_denied is left as 0/1.
    Fails gracefully if model/stats files are missing.
    """

    def __init__(self, model_path, stats_path, device='cpu'):
        self.device    = torch.device(device)
        self.model     = None
        self._loaded   = False
        self._buffer   = deque(maxlen=SEQ_LEN)
        self.feat_mean = None
        self.feat_std  = None
        self.tgt_mean  = None
        self.tgt_std   = None

        for path, name in [(model_path, 'model'), (stats_path, 'stats')]:
            if not os.path.isfile(path):
                print(f"  [LSTMBridge] WARNING: {name} not found: {path}")
                print(f"  [LSTMBridge] Falling back to raw IMU (no LSTM).")
                return

        stats          = np.load(stats_path, allow_pickle=True)
        self.feat_mean = stats['feat_mean'].astype(np.float32)
        self.feat_std  = stats['feat_std'].astype(np.float32)
        self.tgt_mean  = stats['tgt_mean'].astype(np.float32)
        self.tgt_std   = stats['tgt_std'].astype(np.float32)

        ckpt = torch.load(model_path, map_location=self.device)
        cfg  = ckpt.get('config', {})
        self.model = LSTMDriftPredictor(
            input_size=cfg.get('input_size', len(FEATURE_COLS)),
            h1=cfg.get('h1', LSTM_H1), h2=cfg.get('h2', LSTM_H2),
            dropout=cfg.get('dropout', LSTM_DROPOUT),
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self._loaded = True
        print(f"  [LSTMBridge] Loaded  epoch={ckpt.get('epoch','?')}  "
              f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")

    def loaded(self): return self._loaded
    def ready(self):  return self._loaded and len(self._buffer) == SEQ_LEN

    def push(self, ax_corr, ay_corr, wz, ekf_speed, gps_denied):
        raw  = np.array([ax_corr, ay_corr, wz, ekf_speed], dtype=np.float32)
        if self.feat_mean is not None:
            raw = (raw - self.feat_mean) / self.feat_std
        self._buffer.append(
            np.append(raw, float(gps_denied)).astype(np.float32))

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
# ADAPTIVE EKF
# =============================================================================
class AdaptiveEKF:
    """
    5-state EKF: [x(m), y(m), v(m/s), psi(rad), b_psi(rad/s)]

    Interface A — direct (offline / new code):
        initialize(x0, y0, heading0, speed0, bias0)
        predict(a_fwd, wz, gps_denied)
        update(gnss_x, gnss_y)  -> NIS
        get_position() / get_speed() / get_heading() / get_position_std()

    Interface B — backward-compatible (carla_rl_environment.py):
        reset()
        predict(u={'accel':[ax,ay], 'gyro':wz})
        update_gps(z=np.array([x,y]))
        set_noise_scales(Q_scale, R_scale)
        get_state()  -> dict
    """

    def __init__(self, dt=DT):
        self.dt               = float(dt)
        self.x                = np.zeros(N_STATES, dtype=np.float64)
        self.P                = np.diag(P0_DIAG.astype(np.float64))
        self._q_scale         = np.ones(N_STATES, dtype=np.float64)
        self._r_scale         = 1.0
        self._initialized     = False
        self._last_innovation = np.zeros(2, dtype=np.float64)

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, x0=0.0, y0=0.0, heading0=0.0,
                   speed0=0.0, bias0=0.0):
        """
        Safe state seeding. Always use this instead of x[i] directly.

        State layout (v2+):
            x[IDX_V=2]   = speed    (was heading in v1 — FIXED)
            x[IDX_PSI=3] = heading  (was speed   in v1 — FIXED)
        """
        self.x = np.array([x0, y0, speed0, heading0, bias0],
                           dtype=np.float64)
        self.P                = np.diag(P0_DIAG.astype(np.float64))
        self._q_scale         = np.ones(N_STATES, dtype=np.float64)
        self._r_scale         = 1.0
        self._last_innovation = np.zeros(2, dtype=np.float64)
        self._initialized     = True

    def initialized(self): return self._initialized

    # ── RL Adaptation Hooks ───────────────────────────────────────────────────

    def set_process_noise_scale(self, scale):
        """scale: scalar or (N_STATES,) array — multiplies Q_DIAG_BASE."""
        s = np.asarray(scale, dtype=np.float64)
        if s.ndim == 0:
            self._q_scale = np.full(N_STATES, float(s))
        elif s.shape == (N_STATES,):
            self._q_scale = s.copy()
        else:
            raise ValueError(f"scale must be scalar or ({N_STATES},)")

    def set_measurement_noise_scale(self, sigma_multiplier):
        """R variance = R_GNSS * sigma_multiplier^2. 1.0 = nominal."""
        self._r_scale = float(sigma_multiplier) ** 2

    def set_noise_scales(self, Q_scale: float, R_scale: float):
        """
        Interface B — called by carla_rl_environment.py every step.
        Q_scale, R_scale are variance multipliers in [0.1, 3.0].

        Math for R:
            _r_scale = R_scale (desired)
            set_measurement_noise_scale squares its argument, so pass sqrt:
            set_measurement_noise_scale(sqrt(R_scale)) -> _r_scale = R_scale
        """
        self.set_process_noise_scale(float(Q_scale))
        self.set_measurement_noise_scale(
            math.sqrt(max(float(R_scale), 1e-9)))

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _wrap(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _build_F(self, v: float, psi: float) -> np.ndarray:
        F = np.eye(N_STATES, dtype=np.float64)
        c = math.cos(psi) * self.dt
        s = math.sin(psi) * self.dt
        F[IDX_X,   IDX_V]    =  c
        F[IDX_X,   IDX_PSI]  = -v * s
        F[IDX_Y,   IDX_V]    = -s           # CARLA +y=south
        F[IDX_Y,   IDX_PSI]  = -v * c
        F[IDX_PSI, IDX_BPSI] = -self.dt
        return F

    def _build_Q(self, gps_denied: bool) -> np.ndarray:
        q = Q_DIAG_BASE * self._q_scale
        if gps_denied:
            q = q * GPS_DENIED_Q_SCALE
        return np.diag(q)

    def _build_R(self) -> np.ndarray:
        return R_GNSS * self._r_scale

    def _assert_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                "AdaptiveEKF.initialize() must be called before "
                "predict() or update().")

    # ── EKF Steps ────────────────────────────────────────────────────────────

    def _predict_core(self, a_fwd: float, wz: float,
                      gps_denied: bool) -> None:
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
        self.x = xn
        self.P = 0.5 * (Pn + Pn.T)

    def predict(self, u=None, a_fwd=None, wz=None, gps_denied=False):
        """
        Interface A  (keyword / positional):
            ekf.predict(a_fwd=0.5, wz=0.01, gps_denied=True)
        Interface B  (dict — carla_rl_environment.py):
            ekf.predict(u={'accel':[ax,ay], 'gyro':wz})
        """
        self._assert_initialized()
        if u is not None:
            self._predict_core(float(u['accel'][0]), float(u['gyro']), False)
        else:
            self._predict_core(
                float(a_fwd) if a_fwd is not None else 0.0,
                float(wz)    if wz    is not None else 0.0,
                bool(gps_denied),
            )

    def update(self, gnss_x: float, gnss_y: float) -> float:
        """
        GNSS measurement update (Interface A). Call only when not GPS-denied.
        Returns NIS — follows chi2(2). NIS > 5.991 means filter overconfident.
        """
        self._assert_initialized()
        H = np.zeros((2, N_STATES), dtype=np.float64)
        H[0, IDX_X] = 1.0
        H[1, IDX_Y] = 1.0

        R     = self._build_R()
        z     = np.array([float(gnss_x), float(gnss_y)], dtype=np.float64)
        innov = z - H @ self.x
        self._last_innovation = innov.copy()

        S   = H @ self.P @ H.T + R
        K   = self.P @ H.T @ np.linalg.inv(S)
        self.x       = self.x + K @ innov
        self.x[IDX_PSI] = self._wrap(self.x[IDX_PSI])
        self.x[IDX_V]   = float(np.clip(self.x[IDX_V], V_MIN, V_MAX))

        I_KH   = np.eye(N_STATES, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return float(innov @ np.linalg.inv(S) @ innov)

    # ── Interface B wrappers ─────────────────────────────────────────────────

    def update_gps(self, z: np.ndarray) -> float:
        """Interface B — carla_rl_environment.py calls update_gps(z=[x,y])."""
        return self.update(float(z[0]), float(z[1]))

    def get_state(self) -> dict:
        """
        Interface B — dict consumed by carla_rl_environment.py:
            state['position']             np.array([x, y])
            state['innovation']           np.array([ix, iy])
            state['position_uncertainty'] scalar (m)
            state['theta']                heading (rad)
            state['velocity']             speed (m/s)
            state['biases']               np.array([0, 0, b_psi])  3-vector
            state['covariance']           P (5x5)
            state['Q_scale']              float
            state['R_scale']              float (sigma equivalent)
        """
        sx, sy = self.get_position_std()
        return {
            'position':            np.array([self.x[IDX_X], self.x[IDX_Y]]),
            'theta':               float(self.x[IDX_PSI]),
            'velocity':            float(self.x[IDX_V]),
            'biases':              np.array([0.0, 0.0, self.x[IDX_BPSI]]),
            'covariance':          self.P.copy(),
            'position_uncertainty': math.sqrt(sx ** 2 + sy ** 2),
            'innovation':          self._last_innovation.copy(),
            'Q_scale':             float(np.mean(self._q_scale)),
            'R_scale':             float(math.sqrt(max(self._r_scale, 1e-9))),
        }

    def reset(self):
        """Reset to uninitialised state. Call between RL episodes."""
        self.x                = np.zeros(N_STATES, dtype=np.float64)
        self.P                = np.diag(P0_DIAG.astype(np.float64))
        self._q_scale         = np.ones(N_STATES, dtype=np.float64)
        self._r_scale         = 1.0
        self._last_innovation = np.zeros(2, dtype=np.float64)
        self._initialized     = False

    # ── Interface A accessors ─────────────────────────────────────────────────

    def get_position(self):       return float(self.x[IDX_X]), float(self.x[IDX_Y])
    def get_speed(self):          return float(self.x[IDX_V])
    def get_heading(self):        return float(self.x[IDX_PSI])
    def get_bias(self):           return float(self.x[IDX_BPSI])
    def get_state_vector(self):   return self.x.copy()
    def get_covariance(self):     return self.P.copy()

    def get_position_std(self):
        return (math.sqrt(max(self.P[IDX_X, IDX_X], 0.0)),
                math.sqrt(max(self.P[IDX_Y, IDX_Y], 0.0)))

    @property
    def trace_P(self) -> float:
        """Scalar total uncertainty — useful in RL reward functions."""
        return float(np.trace(self.P))


# =============================================================================
# RUN EKF ON ONE RUN'S DATA  (offline evaluation)
# =============================================================================
def run_ekf_on_run(df_run, ekf, bridge, use_lstm):
    ekf.reset()
    if bridge is not None:
        bridge.reset()

    results     = []
    first_valid = True

    required_cols = ['ax_corr', 'ay_corr', 'wz', 'gt_speed_mps',
                     'gnss_x', 'gnss_y', 'gt_x', 'gt_y', 'gt_heading']

    for _, row in df_run.iterrows():
        ts = float(row['timestamp'])

        # FIX 2: NaN guard for gps_denied (dropped frames write NaN here)
        gps_denied_raw = row.get('gps_denied', 0)
        gps_denied = 0 if pd.isna(gps_denied_raw) else int(gps_denied_raw)

        if any(pd.isna(row.get(c)) for c in required_cols):
            results.append({
                'timestamp': ts, 'gt_x': float('nan'), 'gt_y': float('nan'),
                'ekf_x': float('nan'), 'ekf_y': float('nan'),
                'ekf_v': float('nan'), 'ekf_psi': float('nan'),
                'ekf_bias': float('nan'), 'pos_std_x': float('nan'),
                'pos_std_y': float('nan'), 'gps_denied': gps_denied,
                'a_fwd_used': float('nan'), 'nis': float('nan'),
                'lstm_used': False,
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
                           heading0=gt_head, speed0=gt_speed, bias0=0.0)
            first_valid = False

        a_fwd_used = ax_corr
        lstm_used  = False

        if bridge is not None:
            bridge.push(ax_corr, ay_corr, wz, ekf.get_speed(), gps_denied)
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
            'gt_x': gt_x, 'gt_y': gt_y,
            'ekf_x': ex,  'ekf_y': ey,
            'ekf_v':      ekf.get_speed(),
            'ekf_psi':    ekf.get_heading(),
            'ekf_bias':   ekf.get_bias(),
            'pos_std_x':  sx, 'pos_std_y': sy,
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

    err = np.sqrt((valid['ekf_x'] - valid['gt_x'])**2 +
                  (valid['ekf_y'] - valid['gt_y'])**2)
    tun  = valid[valid['gps_denied'] == 1]
    road = valid[valid['gps_denied'] == 0]

    def _rmse(e): return float(np.sqrt(np.mean(e**2)))
    def _mean(e): return float(np.mean(e))
    def _max(e):  return float(np.max(e))

    e_tun  = np.sqrt((tun['ekf_x'] -tun['gt_x'])**2 +
                     (tun['ekf_y'] -tun['gt_y'])**2) if len(tun)>0  else None
    e_road = np.sqrt((road['ekf_x']-road['gt_x'])**2 +
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
# VISUALISATION
# =============================================================================
def plot_run(df_base, df_lstm, run_id, save_dir):
    vb = df_base.dropna(subset=['ekf_x', 'ekf_y'])
    vl = df_lstm.dropna(subset=['ekf_x', 'ekf_y'])

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"EKF — Run {run_id}  (test set)",
                 fontsize=12, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
    ax_tr  = fig.add_subplot(gs[0, 0])
    ax_er  = fig.add_subplot(gs[0, 1])
    ax_sp  = fig.add_subplot(gs[1, 0])
    ax_nis = fig.add_subplot(gs[1, 1])

    ax_tr.plot(vb['gt_x'],  vb['gt_y'],  'g-',  lw=1.5, label='Ground truth')
    ax_tr.plot(vb['ekf_x'], vb['ekf_y'], 'b--', lw=0.9, label='Baseline EKF')
    ax_tr.plot(vl['ekf_x'], vl['ekf_y'], 'r-',  lw=0.9, label='LSTM-EKF')
    tun = vb[vb['gps_denied'] == 1]
    if not tun.empty:
        ax_tr.scatter(tun['gt_x'], tun['gt_y'], c='orange', s=5,
                      alpha=0.35, label='Tunnel zone')
    ax_tr.set_title("XY Trajectory"); ax_tr.set_xlabel("x (m)"); ax_tr.set_ylabel("y (m)")
    ax_tr.legend(fontsize=7); ax_tr.grid(True, alpha=0.3)
    ax_tr.set_aspect('equal', adjustable='datalim')

    err_b = np.sqrt((vb['ekf_x']-vb['gt_x'])**2+(vb['ekf_y']-vb['gt_y'])**2)
    err_l = np.sqrt((vl['ekf_x']-vl['gt_x'])**2+(vl['ekf_y']-vl['gt_y'])**2)
    ax_er.plot(vb['timestamp'], err_b, 'b-', lw=0.7, alpha=0.7, label='Baseline')
    ax_er.plot(vl['timestamp'], err_l, 'r-', lw=0.7, alpha=0.7, label='LSTM-EKF')
    ax_er.set_title("Position Error"); ax_er.set_xlabel("Time (s)"); ax_er.set_ylabel("Error (m)")
    ax_er.legend(fontsize=7); ax_er.grid(True, alpha=0.3)

    ax_sp.plot(vb['timestamp'], vb['ekf_v'], 'b--', lw=0.8, label='Baseline')
    ax_sp.plot(vl['timestamp'], vl['ekf_v'], 'r-',  lw=0.8, label='LSTM-EKF')
    ax_sp.set_title("Speed Estimate"); ax_sp.set_xlabel("Time (s)"); ax_sp.set_ylabel("Speed (m/s)")
    ax_sp.legend(fontsize=7); ax_sp.grid(True, alpha=0.3)

    nb = vb.dropna(subset=['nis']); nl = vl.dropna(subset=['nis'])
    ax_nis.plot(nb['timestamp'], nb['nis'], 'b.', ms=1.5, alpha=0.5, label='Baseline')
    ax_nis.plot(nl['timestamp'], nl['nis'], 'r.', ms=1.5, alpha=0.5, label='LSTM-EKF')
    ax_nis.axhline(NIS_BOUND_95, color='k', ls='--', lw=1,
                   label=f'95% chi2={NIS_BOUND_95}')
    all_nis = np.concatenate([nb['nis'].values, nl['nis'].values])
    if len(all_nis) > 0:
        ax_nis.set_ylim(0, min(60.0, float(np.percentile(all_nis, 99))*1.2))
    ax_nis.set_title("NIS (filter consistency)"); ax_nis.set_xlabel("Time (s)")
    ax_nis.legend(fontsize=7); ax_nis.grid(True, alpha=0.3)

    out = os.path.join(save_dir, f'ekf_run{run_id}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved -> {out}")


def plot_summary(all_metrics, save_dir):
    run_ids = sorted(all_metrics.keys())
    x, w    = np.arange(len(run_ids)), 0.35
    metrics_to_plot = [('overall_rmse','Overall RMSE (m)'),
                       ('tunnel_rmse', 'Tunnel RMSE (m)'),
                       ('road_rmse',   'Open-road RMSE (m)')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("EKF Summary — Test Set (Run 3)", fontsize=11, fontweight='bold')

    for ax, (key, ylabel) in zip(axes, metrics_to_plot):
        b_raw = [all_metrics[r]['baseline'].get(f'baseline_{key}', float('nan'))
                 for r in run_ids]
        l_raw = [all_metrics[r]['lstm'].get(f'lstm_{key}', float('nan'))
                 for r in run_ids]
        # FIX v3-FIX 2: isnan guard prevents matplotlib crash
        b_plot = [0.0 if (v is None or (isinstance(v,float) and math.isnan(v))) else v for v in b_raw]
        l_plot = [0.0 if (v is None or (isinstance(v,float) and math.isnan(v))) else v for v in l_raw]
        ax.bar(x-w/2, b_plot, w, label='Baseline', color='steelblue', alpha=0.8)
        ax.bar(x+w/2, l_plot, w, label='LSTM-EKF', color='tomato',    alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f'Run {r}' for r in run_ids])
        ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(save_dir, 'ekf_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved -> {out}")


# =============================================================================
# SELF-TEST  (run with: python ekf.py --test)
# 7 tests adapted from kalman_filter.py concept, corrected for 5-state model
# =============================================================================
def _run_self_test():
    print("=" * 62)
    print("  AdaptiveEKF Self-Test  (5-state, CARLA frame)")
    print("=" * 62)

    np.random.seed(0)
    dt      = 0.05
    n_steps = int(60 / dt)
    DENY_START, DENY_END = 400, 600

    ekf = AdaptiveEKF(dt=dt)
    gt_x, gt_y, gt_v, gt_psi = 0.0, 0.0, 10.7, 0.0
    ekf.initialize(x0=gt_x, y0=gt_y, heading0=gt_psi, speed0=gt_v, bias0=0.0)

    errors_road, errors_tunnel, pass_count = [], [], 0

    for k in range(n_steps):
        gt_psi  = AdaptiveEKF._wrap(gt_psi + 0.002)
        gt_x   += gt_v * math.cos(gt_psi) * dt
        gt_y   -= gt_v * math.sin(gt_psi) * dt   # CARLA +y=south

        ax_meas = np.random.normal(0.0, 0.02)
        wz_meas = 0.002 + 0.001 + np.random.normal(0.0, 0.005)
        gps_denied = DENY_START <= k < DENY_END

        ekf.predict(a_fwd=ax_meas, wz=wz_meas, gps_denied=gps_denied)

        if not gps_denied:
            ekf.update(gt_x + np.random.normal(0, GNSS_NOISE_STD),
                       gt_y + np.random.normal(0, GNSS_NOISE_STD))

        ex, ey = ekf.get_position()
        err = math.sqrt((ex - gt_x)**2 + (ey - gt_y)**2)
        (errors_tunnel if gps_denied else errors_road).append(err)

        P    = ekf.get_covariance()
        eigs = np.linalg.eigvalsh(P)
        assert np.all(eigs > 0), f"FAIL: P not PD at step {k}"
        assert np.allclose(P, P.T, atol=1e-10), f"FAIL: P not symmetric at step {k}"

    # Test 1: open-road error
    mr = float(np.mean(errors_road))
    assert mr < 2.0, f"FAIL: road mean {mr:.3f} >= 2.0 m"
    print(f"  PASS  1/7  Open-road mean error : {mr:.4f} m  (< 2.0 m)")
    pass_count += 1

    # Test 2: tunnel dead-reckoning error
    if errors_tunnel:
        ft = errors_tunnel[-1]
        assert ft < 15.0, f"FAIL: tunnel final {ft:.3f} >= 15.0 m"
        print(f"  PASS  2/7  Tunnel final error   : {ft:.4f} m  (< 15.0 m)")
        pass_count += 1

    # Test 3: P symmetric PD every step
    print(f"  PASS  3/7  P symmetric PD at every step")
    pass_count += 1

    # Test 4: NaN gps_denied guard
    try:
        ekf.predict(a_fwd=0.0, wz=0.0, gps_denied=False)
        print(f"  PASS  4/7  NaN gps_denied guard  (no crash)")
        pass_count += 1
    except Exception as e:
        print(f"  FAIL  4/7  NaN guard crashed: {e}")

    # Test 5: RL hook round-trip
    ekf.set_noise_scales(Q_scale=2.0, R_scale=2.0)
    s = ekf.get_state()
    assert abs(s['Q_scale'] - 2.0) < 1e-6, f"FAIL: Q_scale={s['Q_scale']}"
    assert abs(s['R_scale'] - math.sqrt(2.0)) < 1e-5, f"FAIL: R_scale={s['R_scale']}"
    print(f"  PASS  5/7  RL hook round-trip Q=2.0 R=2.0")
    pass_count += 1

    # Test 6: Interface B
    ekf.reset(); ekf.initialize(x0=10.0, y0=5.0, heading0=0.1, speed0=8.0)
    ekf.predict(u={'accel': [0.5, 0.0], 'gyro': 0.02})
    ekf.update_gps(z=np.array([10.3, 4.9]))
    s = ekf.get_state()
    assert 'position' in s and len(s['position']) == 2
    assert 'innovation' in s and len(s['innovation']) == 2
    assert len(s['biases']) == 3, "FAIL: biases must be length 3"
    print(f"  PASS  6/7  Interface B (dict predict / update_gps / get_state)")
    pass_count += 1

    # Test 7: initialize() state layout
    ekf.reset(); ekf.initialize(x0=1., y0=2., heading0=0.5, speed0=12., bias0=0.001)
    assert abs(ekf.x[IDX_V]   - 12.0) < 1e-9, "FAIL: IDX_V seed"
    assert abs(ekf.x[IDX_PSI] -  0.5) < 1e-9, "FAIL: IDX_PSI seed"
    print(f"  PASS  7/7  initialize() seeds IDX_V={IDX_V}, IDX_PSI={IDX_PSI} correctly")
    pass_count += 1

    print(f"\n  {pass_count}/7 tests passed")
    print("=" * 62)
    print("  ALL SELF-TESTS PASSED")
    print("=" * 62 + "\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*62}")
    print(f"  EKF Localizer  —  v3 Final")
    print(f"  Device : {device}  |  dt={DT}s ({1/DT:.0f} Hz)  |  State: 5D")
    print(f"  Eval   : run(s) {EVAL_RUNS}  (held-out test set only)")
    print(f"{'='*62}\n")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows | runs: {sorted(df['run_id'].unique())}")

    bridge   = LSTMBridge(MODEL_PATH, STATS_PATH, device=device)
    ekf_base = AdaptiveEKF()
    ekf_lstm = AdaptiveEKF()

    all_results_base, all_results_lstm, all_metrics = [], [], {}

    for run_id in EVAL_RUNS:
        if run_id not in df['run_id'].values:
            print(f"  [WARN] Run {run_id} not in dataset — skipping.")
            continue

        df_run = (df[df['run_id'] == run_id]
                  .sort_values('timestamp').reset_index(drop=True))
        print(f"\n[Run {run_id}]  {len(df_run):,} rows  "
              f"| tunnel: {int(df_run['gps_denied'].sum())}")

        print("  Baseline EKF...")
        df_base = run_ekf_on_run(df_run, ekf_base, bridge=None, use_lstm=False)
        print("  LSTM-EKF...")
        df_lstm = run_ekf_on_run(df_run, ekf_lstm, bridge=bridge, use_lstm=True)

        df_base['run_id'] = df_lstm['run_id'] = run_id
        all_results_base.append(df_base); all_results_lstm.append(df_lstm)

        m_b = compute_metrics(df_base, 'baseline')
        m_l = compute_metrics(df_lstm, 'lstm')
        all_metrics[run_id] = {'baseline': m_b, 'lstm': m_l}

        print(f"\n  {'Metric':<28} {'Baseline':>10}  {'LSTM-EKF':>10}  {'Delta':>10}")
        print("  " + "-"*62)
        for key in ['overall_rmse','tunnel_rmse','tunnel_max',
                    'road_rmse','nis_mean','nis_pct_inside_95']:
            bv = m_b.get(f'baseline_{key}', float('nan'))
            lv = m_l.get(f'lstm_{key}',     float('nan'))
            d  = (lv-bv) if not (math.isnan(bv) or math.isnan(lv)) else float('nan')
            print(f"  {key:<28} {bv:>10.3f}  {lv:>10.3f}  {d:>+10.3f}")

        plot_run(df_base, df_lstm, run_id, RESULTS_DIR)

    if not all_results_base:
        print("\n[WARN] No runs evaluated."); return

    df_ab = pd.concat(all_results_base, ignore_index=True)
    df_al = pd.concat(all_results_lstm, ignore_index=True)
    m_ab  = compute_metrics(df_ab, 'baseline')
    m_al  = compute_metrics(df_al, 'lstm')

    print(f"\n{'='*62}\n  AGGREGATE METRICS  (test set — run 3 only)\n{'='*62}")
    lines = ["EKF Localizer v3 — Test Metrics (Run 3)", "="*62,
             f"{'Metric':<28} {'Baseline':>10}  {'LSTM-EKF':>10}  {'Improvement':>12}",
             "-"*62]

    for key, label in [
        ('overall_rmse','Overall RMSE (m)'), ('overall_mean','Overall mean (m)'),
        ('overall_max', 'Overall max  (m)'), ('tunnel_rmse', 'Tunnel RMSE  (m)'),
        ('tunnel_mean', 'Tunnel mean  (m)'), ('tunnel_max',  'Tunnel max   (m)'),
        ('road_rmse',   'Road RMSE    (m)'), ('road_mean',   'Road mean    (m)'),
        ('nis_mean',    'NIS mean'),         ('nis_pct_inside_95','NIS inside 95%'),
    ]:
        bv  = m_ab.get(f'baseline_{key}', float('nan'))
        lv  = m_al.get(f'lstm_{key}',     float('nan'))
        pct = (f"{(bv-lv)/abs(bv)*100:+.1f}%"
               if not (math.isnan(bv) or math.isnan(lv) or bv==0) else "n/a")
        line = f"  {label:<28} {bv:>10.3f}  {lv:>10.3f}  {pct:>10}"
        print(line); lines.append(line)

    mpath = os.path.join(RESULTS_DIR, 'ekf_metrics.txt')
    with open(mpath,'w') as f: f.write("\n".join(lines))
    print(f"\n  Metrics     -> {mpath}")

    df_ab['mode']='baseline'; df_al['mode']='lstm'
    ppath = os.path.join(RESULTS_DIR, 'ekf_predictions.csv')
    pd.concat([df_ab, df_al], ignore_index=True).to_csv(ppath, index=False)
    print(f"  Predictions -> {ppath}")

    plot_summary(all_metrics, RESULTS_DIR)

    print(f"\n{'='*62}\n  DONE")
    print(f"  Target: tunnel_rmse (LSTM-EKF) < 3.0 m  (acceptable)")
    print(f"          tunnel_rmse (LSTM-EKF) < 1.5 m  (good)")
    print(f"          nis_pct_inside_95       > 90 %   (consistent)")
    print(f"\n  Next: python rl_train.py  (CARLA must be running)")
    print(f"{'='*62}\n")


# =============================================================================
# ENTRY POINTS
# =============================================================================
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        _run_self_test()
    else:
        main()
