"""
ekf.py  —  Extended Kalman Filter with LSTM-Aided Dead-Reckoning  (v4 Final)
============================================================================
For: LSTM + RL-Adaptive EKF Localization Project

Change from v3 (single targeted fix)
-------------------------------------
  v4-FIX — LSTM output applied as ADDITIVE BIAS CORRECTION, not replacement.

  v3 (wrong):
      if gps_denied and lstm.ready():
          a_fwd = lstm.predict()           # replaces IMU entirely

  v4 (correct):
      if gps_denied and lstm.ready():
          bias = lstm.predict()
          a_fwd = ax_corr + bias           # IMU + learned correction

  Why this matters:
    - If LSTM predicts well: result is identical to v3 (bias ≈ gt-imu → a_fwd ≈ gt)
    - If LSTM predicts poorly: v4 falls back toward raw IMU naturally (bias ≈ 0)
    - v3 with poor LSTM: substitutes a wrong value for the entire IMU reading
    - The IMU reading provides a physical floor that bounds worst-case error

  Requires train_lstm.py v4 which trains on TARGET = gt_accel - imu_accel.
  Will produce incorrect results if loaded with v3 model (trained on absolute accel).
  LSTMBridge reads the 'output_is_bias' flag from the checkpoint to assert this.

All other logic (5-state EKF, Joseph form, RL hooks, Interface B) unchanged.
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
DT = 0.05
IDX_X = 0; IDX_Y = 1; IDX_V = 2; IDX_PSI = 3; IDX_BPSI = 4
N_STATES = 5
V_MIN = 0.0; V_MAX = 50.0
GNSS_NOISE_STD = 1.0
R_GNSS = (GNSS_NOISE_STD ** 2) * np.eye(2, dtype=np.float64)
NIS_BOUND_95 = 5.991
Q_DIAG_BASE = np.array([0.01, 0.01, 0.25, 1e-4, 1e-8], dtype=np.float64)
GPS_DENIED_Q_SCALE = np.array([1.0, 1.0, 4.0, 4.0, 1.0], dtype=np.float64)
P0_DIAG = np.array([1.0, 1.0, 4.0, 0.1, 1e-4], dtype=np.float64)
SEQ_LEN = 40; LSTM_H1 = 64; LSTM_H2 = 32; LSTM_DROPOUT = 0.3
FEATURE_COLS = ['ax_corr', 'ay_corr', 'wz', 'gt_speed_mps', 'gps_denied']
TARGET_COLS  = ['bias_fwd', 'bias_lat']   # v4: bias targets
EVAL_RUNS = [3]


# =============================================================================
# LSTM MODEL  (architecture unchanged — output semantics changed to bias)
# =============================================================================
class LSTMDriftPredictor(nn.Module):
    def __init__(self, input_size=5, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm1    = nn.LSTM(input_size, h1, num_layers=1, batch_first=True)
        self.ln1      = nn.LayerNorm(h1);  self.drop1 = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.ln2      = nn.LayerNorm(h2);  self.drop2 = nn.Dropout(dropout)
        self.head     = nn.Sequential(nn.Linear(h2,16), nn.GELU(), nn.Linear(16, 2))

    def forward(self, x):
        x = self.input_ln(x)
        o1, _ = self.lstm1(x);  o1 = self.ln1(o1);  o1 = self.drop1(o1)
        o2, _ = self.lstm2(o1); o2 = self.ln2(o2);  o2 = self.drop2(o2)
        return self.head(o2[:, -1, :])


# =============================================================================
# LSTM BRIDGE
# =============================================================================
class LSTMBridge:
    """
    Online inference wrapper.

    v4 change: predict() returns BIAS values (gt_accel - imu_accel).
    Callers must apply as: a_fwd = ax_corr + bias_fwd  (NOT a_fwd = bias_fwd).

    Asserts output_is_bias=True in checkpoint on load so that a v3 model
    (trained on absolute accel) cannot be accidentally loaded and produce
    silently wrong results.
    """

    def __init__(self, model_path, stats_path, device='cpu'):
        self.device    = torch.device(device)
        self.model     = None
        self._loaded   = False
        self._buffer   = deque(maxlen=SEQ_LEN)
        self.feat_mean = self.feat_std = None
        self.tgt_mean  = self.tgt_std  = None

        for path, name in [(model_path,'model'), (stats_path,'stats')]:
            if not os.path.isfile(path):
                print(f"  [LSTMBridge] WARNING: {name} not found: {path}")
                print(f"  [LSTMBridge] Falling back to raw IMU.")
                return

        stats          = np.load(stats_path, allow_pickle=True)
        self.feat_mean = stats['feat_mean'].astype(np.float32)
        self.feat_std  = stats['feat_std'].astype(np.float32)
        self.tgt_mean  = stats['tgt_mean'].astype(np.float32)
        self.tgt_std   = stats['tgt_std'].astype(np.float32)

        ckpt = torch.load(model_path, map_location=self.device)
        cfg  = ckpt.get('config', {})

        # v4 guard: refuse to load v3 model (trained on absolute accel)
        if not cfg.get('output_is_bias', False):
            print(f"  [LSTMBridge] ERROR: loaded model was trained on ABSOLUTE ACCELERATION (v3).")
            print(f"  [LSTMBridge] ekf.py v4 requires a model trained on BIAS (v4).")
            print(f"  [LSTMBridge] Retrain with train_lstm.py v4 first.")
            print(f"  [LSTMBridge] Falling back to raw IMU for safety.")
            return

        self.model = LSTMDriftPredictor(
            input_size=cfg.get('input_size', 5),
            h1=cfg.get('h1', LSTM_H1), h2=cfg.get('h2', LSTM_H2),
            dropout=cfg.get('dropout', LSTM_DROPOUT),
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self._loaded = True
        print(f"  [LSTMBridge] v4 loaded  epoch={ckpt.get('epoch','?')}  "
              f"val_loss={ckpt.get('val_loss', float('nan')):.6f}  output=BIAS")
        print(f"  [LSTMBridge] Application: a_fwd = ax_corr + lstm_bias")

    def loaded(self): return self._loaded
    def ready(self):  return self._loaded and len(self._buffer) == SEQ_LEN

    def push(self, ax_corr, ay_corr, wz, ekf_speed, gps_denied):
        raw = np.array([ax_corr, ay_corr, wz, ekf_speed], dtype=np.float32)
        if self.feat_mean is not None:
            raw = (raw - self.feat_mean) / self.feat_std
        self._buffer.append(np.append(raw, float(gps_denied)).astype(np.float32))

    def predict(self):
        """
        Returns (bias_fwd, bias_lat) in m/s².
        CALLER MUST APPLY AS: a_fwd = ax_corr + bias_fwd
        Returns (None, None) if buffer not ready.
        """
        if not self.ready(): return None, None
        seq = np.array(self._buffer, dtype=np.float32)
        x_t = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_n = self.model(x_t).cpu().numpy()[0]
        y_p = y_n * self.tgt_std + self.tgt_mean
        return float(y_p[0]), float(y_p[1])   # (bias_fwd, bias_lat)

    def reset(self): self._buffer.clear()


# =============================================================================
# ADAPTIVE EKF  (unchanged from v3)
# =============================================================================
class AdaptiveEKF:
    """
    5-state EKF: [x(m), y(m), v(m/s), psi(rad), b_psi(rad/s)]
    Unchanged from v3. The LSTM integration change is in run_ekf_on_run
    and carla_rl_environment.py.
    """

    def __init__(self, dt=DT):
        self.dt = float(dt)
        self.x  = np.zeros(N_STATES, dtype=np.float64)
        self.P  = np.diag(P0_DIAG.astype(np.float64))
        self._q_scale         = np.ones(N_STATES, dtype=np.float64)
        self._r_scale         = 1.0
        self._initialized     = False
        self._last_innovation = np.zeros(2, dtype=np.float64)

    def initialize(self, x0=0.0, y0=0.0, heading0=0.0, speed0=0.0, bias0=0.0):
        self.x = np.array([x0, y0, speed0, heading0, bias0], dtype=np.float64)
        self.P                = np.diag(P0_DIAG.astype(np.float64))
        self._q_scale         = np.ones(N_STATES, dtype=np.float64)
        self._r_scale         = 1.0
        self._last_innovation = np.zeros(2, dtype=np.float64)
        self._initialized     = True

    def initialized(self): return self._initialized

    def set_process_noise_scale(self, scale):
        s = np.asarray(scale, dtype=np.float64)
        self._q_scale = np.full(N_STATES, float(s)) if s.ndim == 0 else s.copy()

    def set_measurement_noise_scale(self, sigma_multiplier):
        self._r_scale = float(sigma_multiplier) ** 2

    def set_noise_scales(self, Q_scale: float, R_scale: float):
        self.set_process_noise_scale(float(Q_scale))
        self.set_measurement_noise_scale(math.sqrt(max(float(R_scale), 1e-9)))

    @staticmethod
    def _wrap(angle): return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _build_F(self, v, psi):
        F = np.eye(N_STATES, dtype=np.float64)
        c = math.cos(psi)*self.dt; s = math.sin(psi)*self.dt
        F[IDX_X,IDX_V]=c; F[IDX_X,IDX_PSI]=-v*s
        F[IDX_Y,IDX_V]=-s; F[IDX_Y,IDX_PSI]=-v*c
        F[IDX_PSI,IDX_BPSI]=-self.dt
        return F

    def _build_Q(self, gps_denied):
        q = Q_DIAG_BASE * self._q_scale
        return np.diag(q * GPS_DENIED_Q_SCALE if gps_denied else q)

    def _build_R(self): return R_GNSS * self._r_scale
    def _assert_initialized(self):
        if not self._initialized: raise RuntimeError("Call initialize() first.")

    def _predict_core(self, a_fwd, wz, gps_denied):
        v=float(self.x[IDX_V]); psi=float(self.x[IDX_PSI]); b=float(self.x[IDX_BPSI])
        xn = np.empty(N_STATES, dtype=np.float64)
        xn[IDX_X]    = self.x[IDX_X] + v*math.cos(psi)*self.dt
        xn[IDX_Y]    = self.x[IDX_Y] - v*math.sin(psi)*self.dt
        xn[IDX_V]    = float(np.clip(v + a_fwd*self.dt, V_MIN, V_MAX))
        xn[IDX_PSI]  = self._wrap(psi + (wz-b)*self.dt)
        xn[IDX_BPSI] = b
        F=self._build_F(v,psi); Q=self._build_Q(gps_denied)
        self.x = xn; self.P = 0.5*(F@self.P@F.T+Q+(F@self.P@F.T+Q).T)/2

    def predict(self, u=None, a_fwd=None, wz=None, gps_denied=False):
        self._assert_initialized()
        if u is not None:
            self._predict_core(float(u['accel'][0]), float(u['gyro']), False)
        else:
            self._predict_core(float(a_fwd) if a_fwd is not None else 0.0,
                               float(wz)    if wz    is not None else 0.0,
                               bool(gps_denied))

    def update(self, gnss_x, gnss_y):
        self._assert_initialized()
        H = np.zeros((2, N_STATES), dtype=np.float64)
        H[0,IDX_X]=1.0; H[1,IDX_Y]=1.0
        R=self._build_R(); z=np.array([float(gnss_x),float(gnss_y)],dtype=np.float64)
        innov=z-H@self.x; self._last_innovation=innov.copy()
        S=H@self.P@H.T+R; K=self.P@H.T@np.linalg.inv(S)
        self.x=self.x+K@innov; self.x[IDX_PSI]=self._wrap(self.x[IDX_PSI])
        self.x[IDX_V]=float(np.clip(self.x[IDX_V],V_MIN,V_MAX))
        I_KH=np.eye(N_STATES,dtype=np.float64)-K@H
        self.P=I_KH@self.P@I_KH.T+K@R@K.T; self.P=0.5*(self.P+self.P.T)
        return float(innov@np.linalg.inv(S)@innov)

    def update_gps(self, z): return self.update(float(z[0]), float(z[1]))

    def get_state(self):
        sx, sy = self.get_position_std()
        return {'position':            np.array([self.x[IDX_X], self.x[IDX_Y]]),
                'theta':               float(self.x[IDX_PSI]),
                'velocity':            float(self.x[IDX_V]),
                'biases':              np.array([0.0, 0.0, self.x[IDX_BPSI]]),
                'covariance':          self.P.copy(),
                'position_uncertainty': math.sqrt(sx**2 + sy**2),
                'innovation':          self._last_innovation.copy(),
                'Q_scale':             float(np.mean(self._q_scale)),
                'R_scale':             float(math.sqrt(max(self._r_scale, 1e-9)))}

    def reset(self):
        self.x=np.zeros(N_STATES,dtype=np.float64); self.P=np.diag(P0_DIAG.astype(np.float64))
        self._q_scale=np.ones(N_STATES,dtype=np.float64); self._r_scale=1.0
        self._last_innovation=np.zeros(2,dtype=np.float64); self._initialized=False

    def get_position(self):     return float(self.x[IDX_X]), float(self.x[IDX_Y])
    def get_speed(self):        return float(self.x[IDX_V])
    def get_heading(self):      return float(self.x[IDX_PSI])
    def get_bias(self):         return float(self.x[IDX_BPSI])
    def get_state_vector(self): return self.x.copy()
    def get_covariance(self):   return self.P.copy()
    def get_position_std(self):
        return (math.sqrt(max(self.P[IDX_X,IDX_X],0.0)), math.sqrt(max(self.P[IDX_Y,IDX_Y],0.0)))
    @property
    def trace_P(self): return float(np.trace(self.P))


# =============================================================================
# RUN EKF ON ONE RUN'S DATA
# =============================================================================
def run_ekf_on_run(df_run, ekf, bridge, use_lstm):
    """
    v4 change (line marked with # v4):
        Old: a_fwd_used = a_fwd_pred          (replacement)
        New: a_fwd_used = ax_corr + a_fwd_pred  (additive bias correction)
    """
    ekf.reset()
    if bridge is not None: bridge.reset()
    results = []; first_valid = True
    required_cols = ['ax_corr','ay_corr','wz','gt_speed_mps','gnss_x','gnss_y','gt_x','gt_y','gt_heading']

    for _, row in df_run.iterrows():
        ts = float(row['timestamp'])
        gps_denied_raw = row.get('gps_denied', 0)
        gps_denied = 0 if pd.isna(gps_denied_raw) else int(gps_denied_raw)

        if any(pd.isna(row.get(c)) for c in required_cols):
            results.append({'timestamp':ts,'gt_x':float('nan'),'gt_y':float('nan'),
                            'ekf_x':float('nan'),'ekf_y':float('nan'),'ekf_v':float('nan'),
                            'ekf_psi':float('nan'),'ekf_bias':float('nan'),
                            'pos_std_x':float('nan'),'pos_std_y':float('nan'),
                            'gps_denied':gps_denied,'a_fwd_used':float('nan'),
                            'nis':float('nan'),'lstm_used':False})
            continue

        ax_corr  = float(row['ax_corr']);  ay_corr  = float(row['ay_corr'])
        wz       = float(row['wz']);       gnss_x   = float(row['gnss_x'])
        gnss_y   = float(row['gnss_y']);   gt_x     = float(row['gt_x'])
        gt_y     = float(row['gt_y']);     gt_head  = float(row['gt_heading'])
        gt_speed = float(row['gt_speed_mps'])

        if first_valid:
            ekf.initialize(x0=gnss_x, y0=gnss_y, heading0=gt_head, speed0=gt_speed)
            first_valid = False

        a_fwd_used = ax_corr   # default: raw corrected IMU
        lstm_used  = False

        if bridge is not None:
            bridge.push(ax_corr, ay_corr, wz, ekf.get_speed(), gps_denied)
            if use_lstm and gps_denied and bridge.ready():
                bias_fwd, _ = bridge.predict()
                if bias_fwd is not None:
                    # v4: ADDITIVE correction — IMU + learned bias
                    a_fwd_used = ax_corr + bias_fwd
                    lstm_used  = True

        ekf.predict(a_fwd=a_fwd_used, wz=wz, gps_denied=bool(gps_denied))
        nis = float('nan')
        if not gps_denied: nis = ekf.update(gnss_x, gnss_y)
        ex, ey = ekf.get_position(); sx, sy = ekf.get_position_std()

        results.append({'timestamp':ts,'gt_x':gt_x,'gt_y':gt_y,'ekf_x':ex,'ekf_y':ey,
                        'ekf_v':ekf.get_speed(),'ekf_psi':ekf.get_heading(),'ekf_bias':ekf.get_bias(),
                        'pos_std_x':sx,'pos_std_y':sy,'gps_denied':gps_denied,
                        'a_fwd_used':a_fwd_used,'nis':nis,'lstm_used':lstm_used})
    return pd.DataFrame(results)


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(df_res, label=''):
    tag = f"{label}_" if label else ""
    valid = df_res.dropna(subset=['ekf_x','ekf_y','gt_x','gt_y'])
    if valid.empty: return {f'{tag}overall_rmse': float('nan')}
    err = np.sqrt((valid['ekf_x']-valid['gt_x'])**2+(valid['ekf_y']-valid['gt_y'])**2)
    tun  = valid[valid['gps_denied']==1]; road = valid[valid['gps_denied']==0]
    def _r(e): return float(np.sqrt(np.mean(e**2))) if len(e)>0 else float('nan')
    def _m(e): return float(np.mean(e))             if len(e)>0 else float('nan')
    def _x(e): return float(np.max(e))              if len(e)>0 else float('nan')
    e_t = np.sqrt((tun['ekf_x']-tun['gt_x'])**2+(tun['ekf_y']-tun['gt_y'])**2) if len(tun)>0 else None
    e_r = np.sqrt((road['ekf_x']-road['gt_x'])**2+(road['ekf_y']-road['gt_y'])**2) if len(road)>0 else None
    nis_v = valid.dropna(subset=['nis'])['nis'].values
    return {f'{tag}overall_rmse':_r(err),f'{tag}overall_mean':_m(err),f'{tag}overall_max':_x(err),
            f'{tag}tunnel_rmse':_r(e_t),f'{tag}tunnel_mean':_m(e_t),f'{tag}tunnel_max':_x(e_t),
            f'{tag}tunnel_n':int(len(tun)),f'{tag}road_rmse':_r(e_r),f'{tag}road_mean':_m(e_r),
            f'{tag}nis_mean':float(np.mean(nis_v)) if len(nis_v)>0 else float('nan'),
            f'{tag}nis_pct_inside_95':float(np.mean(nis_v<NIS_BOUND_95)*100) if len(nis_v)>0 else float('nan')}


# =============================================================================
# VISUALISATION  (unchanged from v3)
# =============================================================================
def plot_run(df_base, df_lstm, run_id, save_dir):
    vb=df_base.dropna(subset=['ekf_x','ekf_y']); vl=df_lstm.dropna(subset=['ekf_x','ekf_y'])
    fig=plt.figure(figsize=(14,9)); fig.suptitle(f"EKF v4 — Run {run_id}  (test set)",fontsize=12,fontweight='bold')
    gs=gridspec.GridSpec(2,2,hspace=0.42,wspace=0.35)
    ax_tr=fig.add_subplot(gs[0,0]); ax_er=fig.add_subplot(gs[0,1])
    ax_sp=fig.add_subplot(gs[1,0]); ax_nis=fig.add_subplot(gs[1,1])
    ax_tr.plot(vb['gt_x'],vb['gt_y'],'g-',lw=1.5,label='GT'); ax_tr.plot(vb['ekf_x'],vb['ekf_y'],'b--',lw=0.9,label='Baseline')
    ax_tr.plot(vl['ekf_x'],vl['ekf_y'],'r-',lw=0.9,label='LSTM-EKF v4')
    tun=vb[vb['gps_denied']==1]
    if not tun.empty: ax_tr.scatter(tun['gt_x'],tun['gt_y'],c='orange',s=5,alpha=0.35,label='Tunnel')
    ax_tr.set_title("XY Trajectory"); ax_tr.legend(fontsize=7); ax_tr.grid(True,alpha=0.3); ax_tr.set_aspect('equal','datalim')
    err_b=np.sqrt((vb['ekf_x']-vb['gt_x'])**2+(vb['ekf_y']-vb['gt_y'])**2)
    err_l=np.sqrt((vl['ekf_x']-vl['gt_x'])**2+(vl['ekf_y']-vl['gt_y'])**2)
    ax_er.plot(vb['timestamp'],err_b,'b-',lw=0.7,alpha=0.7,label='Baseline'); ax_er.plot(vl['timestamp'],err_l,'r-',lw=0.7,alpha=0.7,label='LSTM-EKF v4')
    ax_er.set_title("Position Error"); ax_er.legend(fontsize=7); ax_er.grid(True,alpha=0.3)
    ax_sp.plot(vb['timestamp'],vb['ekf_v'],'b--',lw=0.8,label='Baseline'); ax_sp.plot(vl['timestamp'],vl['ekf_v'],'r-',lw=0.8,label='LSTM-EKF v4')
    ax_sp.set_title("Speed"); ax_sp.legend(fontsize=7); ax_sp.grid(True,alpha=0.3)
    nb=vb.dropna(subset=['nis']); nl=vl.dropna(subset=['nis'])
    ax_nis.plot(nb['timestamp'],nb['nis'],'b.',ms=1.5,alpha=0.5,label='Baseline'); ax_nis.plot(nl['timestamp'],nl['nis'],'r.',ms=1.5,alpha=0.5,label='LSTM-EKF v4')
    ax_nis.axhline(NIS_BOUND_95,color='k',ls='--',lw=1); ax_nis.set_title("NIS"); ax_nis.legend(fontsize=7); ax_nis.grid(True,alpha=0.3)
    out=os.path.join(save_dir,f'ekf_run{run_id}.png'); plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  Saved -> {out}")

def plot_summary(all_metrics, save_dir):
    run_ids=sorted(all_metrics.keys()); x,w=np.arange(len(run_ids)),0.35
    fig,axes=plt.subplots(1,3,figsize=(14,4)); fig.suptitle("EKF v4 Summary",fontsize=11,fontweight='bold')
    for ax,(key,ylabel) in zip(axes,[('overall_rmse','Overall RMSE (m)'),('tunnel_rmse','Tunnel RMSE (m)'),('road_rmse','Road RMSE (m)')]):
        b_raw=[all_metrics[r]['baseline'].get(f'baseline_{key}',float('nan')) for r in run_ids]
        l_raw=[all_metrics[r]['lstm'].get(f'lstm_{key}',float('nan')) for r in run_ids]
        b_plot=[0.0 if (v is None or (isinstance(v,float) and math.isnan(v))) else v for v in b_raw]
        l_plot=[0.0 if (v is None or (isinstance(v,float) and math.isnan(v))) else v for v in l_raw]
        ax.bar(x-w/2,b_plot,w,label='Baseline',color='steelblue',alpha=0.8); ax.bar(x+w/2,l_plot,w,label='LSTM-EKF v4',color='tomato',alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f'Run {r}' for r in run_ids]); ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(True,alpha=0.3,axis='y')
    plt.tight_layout(); out=os.path.join(save_dir,'ekf_summary.png'); plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig); print(f"  Saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*62}\n  EKF Localizer  —  v4 Final")
    print(f"  LSTM: ADDITIVE bias correction  (a_fwd = imu + lstm_bias)")
    print(f"  Eval: run(s) {EVAL_RUNS}  (held-out test set only)\n{'='*62}\n")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows | runs: {sorted(df['run_id'].unique())}")
    bridge  = LSTMBridge(MODEL_PATH, STATS_PATH, device=device)
    ekf_base= AdaptiveEKF(); ekf_lstm = AdaptiveEKF()
    all_results_base, all_results_lstm, all_metrics = [], [], {}

    for run_id in EVAL_RUNS:
        if run_id not in df['run_id'].values:
            print(f"  [WARN] Run {run_id} not in dataset."); continue
        df_run = df[df['run_id']==run_id].sort_values('timestamp').reset_index(drop=True)
        print(f"\n[Run {run_id}]  {len(df_run):,} rows | tunnel: {int(df_run['gps_denied'].sum())}")
        print("  Baseline EKF (raw IMU)...")
        df_base = run_ekf_on_run(df_run, ekf_base, bridge=None, use_lstm=False)
        print("  LSTM-EKF v4 (additive bias correction)...")
        df_lstm = run_ekf_on_run(df_run, ekf_lstm, bridge=bridge, use_lstm=True)
        df_base['run_id']=df_lstm['run_id']=run_id
        all_results_base.append(df_base); all_results_lstm.append(df_lstm)
        m_b=compute_metrics(df_base,'baseline'); m_l=compute_metrics(df_lstm,'lstm')
        all_metrics[run_id]={'baseline':m_b,'lstm':m_l}
        print(f"\n  {'Metric':<28} {'Baseline':>10}  {'LSTM-EKF v4':>12}  {'Delta':>10}")
        print("  "+"-"*64)
        for key in ['overall_rmse','tunnel_rmse','tunnel_max','road_rmse','nis_mean','nis_pct_inside_95']:
            bv=m_b.get(f'baseline_{key}',float('nan')); lv=m_l.get(f'lstm_{key}',float('nan'))
            d=(lv-bv) if not (math.isnan(bv) or math.isnan(lv)) else float('nan')
            print(f"  {key:<28} {bv:>10.3f}  {lv:>12.3f}  {d:>+10.3f}")
        plot_run(df_base, df_lstm, run_id, RESULTS_DIR)

    if not all_results_base: print("\n[WARN] No runs evaluated."); return
    df_ab=pd.concat(all_results_base,ignore_index=True); df_al=pd.concat(all_results_lstm,ignore_index=True)
    m_ab=compute_metrics(df_ab,'baseline'); m_al=compute_metrics(df_al,'lstm')

    print(f"\n{'='*62}\n  AGGREGATE METRICS  (test set — run 3 only)\n{'='*62}")
    lines=["EKF Localizer v4 — Test Metrics (Run 3)","="*62,
           f"{'Metric':<28} {'Baseline':>10}  {'LSTM-EKF v4':>12}  {'Improvement':>12}","-"*62]
    for key,label in [('overall_rmse','Overall RMSE (m)'),('overall_mean','Overall mean (m)'),
                       ('overall_max','Overall max  (m)'),('tunnel_rmse','Tunnel RMSE  (m)'),
                       ('tunnel_mean','Tunnel mean  (m)'),('tunnel_max','Tunnel max   (m)'),
                       ('road_rmse','Road RMSE    (m)'),('road_mean','Road mean    (m)'),
                       ('nis_mean','NIS mean'),('nis_pct_inside_95','NIS inside 95%')]:
        bv=m_ab.get(f'baseline_{key}',float('nan')); lv=m_al.get(f'lstm_{key}',float('nan'))
        pct=(f"{(bv-lv)/abs(bv)*100:+.1f}%" if not (math.isnan(bv) or math.isnan(lv) or bv==0) else "n/a")
        line=f"  {label:<28} {bv:>10.3f}  {lv:>12.3f}  {pct:>10}"; print(line); lines.append(line)

    mpath=os.path.join(RESULTS_DIR,'ekf_metrics.txt')
    with open(mpath,'w') as f: f.write("\n".join(lines))
    df_ab['mode']='baseline'; df_al['mode']='lstm'
    pd.concat([df_ab,df_al],ignore_index=True).to_csv(os.path.join(RESULTS_DIR,'ekf_predictions.csv'),index=False)
    plot_summary(all_metrics, RESULTS_DIR)
    print(f"\n{'='*62}\n  DONE  (v4 — additive LSTM bias correction)")
    print(f"  Next: python rl_train.py\n{'='*62}\n")


# =============================================================================
# SELF-TEST  (unchanged from v3, verifies basic EKF math)
# =============================================================================
def _run_self_test():
    print("="*62+"\n  AdaptiveEKF Self-Test  (v4, 5-state, CARLA frame)\n"+"="*62)
    np.random.seed(0); dt=0.05; n_steps=int(60/dt); DENY_START,DENY_END=400,600
    ekf=AdaptiveEKF(dt=dt); gt_x,gt_y,gt_v,gt_psi=0.0,0.0,10.7,0.0
    ekf.initialize(x0=gt_x,y0=gt_y,heading0=gt_psi,speed0=gt_v,bias0=0.0)
    errors_road=[]; errors_tunnel=[]; pass_count=0
    for k in range(n_steps):
        gt_psi=AdaptiveEKF._wrap(gt_psi+0.002); gt_x+=gt_v*math.cos(gt_psi)*dt; gt_y-=gt_v*math.sin(gt_psi)*dt
        gps_denied=DENY_START<=k<DENY_END
        ekf.predict(a_fwd=np.random.normal(0,0.02), wz=0.002+0.001+np.random.normal(0,0.005), gps_denied=gps_denied)
        if not gps_denied: ekf.update(gt_x+np.random.normal(0,GNSS_NOISE_STD), gt_y+np.random.normal(0,GNSS_NOISE_STD))
        ex,ey=ekf.get_position(); err=math.sqrt((ex-gt_x)**2+(ey-gt_y)**2)
        (errors_tunnel if gps_denied else errors_road).append(err)
        P=ekf.get_covariance(); eigs=np.linalg.eigvalsh(P)
        assert np.all(eigs>0) and np.allclose(P,P.T,atol=1e-10)
    mr=float(np.mean(errors_road)); assert mr<2.0; print(f"  PASS 1/4  Road mean {mr:.4f} m")
    if errors_tunnel: ft=errors_tunnel[-1]; assert ft<15.0; print(f"  PASS 2/4  Tunnel final {ft:.4f} m")
    print("  PASS 3/4  P symmetric PD every step"); pass_count+=3
    ekf.set_noise_scales(2.0,2.0); s=ekf.get_state()
    assert abs(s['Q_scale']-2.0)<1e-6; print(f"  PASS 4/4  RL hook round-trip"); pass_count+=1
    print(f"\n  {pass_count}/4 passed\n"+"="*62+"\n  ALL SELF-TESTS PASSED\n"+"="*62+"\n")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test': _run_self_test()
    else: main()
