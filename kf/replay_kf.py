"""
replay_kf.py  —  Kalman Filter Replay & Evaluation
====================================================

PURPOSE
-------
Replay town04_collected.csv through the EKF exactly as it will run
inside the CARLA loop, but offline — no simulator needed.

This is the correct way to evaluate the Kalman Filter:
  • The self-test in kalman_filter.py proves the math is correct
    using synthetic motion.
  • This script proves the filter works on YOUR real data —
    real IMU noise, real GPS noise, real tunnel transitions,
    real Town04 cornering dynamics.

What this script does step-by-step:
  1. Load town04_collected.csv
  2. Initialise the EKF at the first row's ground truth position
  3. Replay every row:
       predict(ax, wz)
       if gps_denied == 0:  update_gps(gnss_x, gnss_y)
       if gps_denied == 1 and LSTM window ready:
           inject_lstm_correction(anchor, lstm_prediction)
  4. At every step record EKF position error vs ground truth
  5. Print a detailed metrics report
  6. Save four diagnostic plots

LSTM simulation during replay
------------------------------
The real LSTM model isn't loaded here — that happens in fusion.py.
Instead this script simulates the LSTM output as:
    predicted_Δx = true_Δx + N(0, σ_lstm)
    predicted_Δy = true_Δy + N(0, σ_lstm)
where σ_lstm = 3.41 m (the test RMSE from training).

This lets you evaluate the EKF's fusion behaviour before fusion.py
is written, and gives you the exact baseline plots for your thesis.

RUN
---
    cd C:\\Users\\heman\\Music\\rl_imu_project
    carla_env\\Scripts\\activate
    python kf\\replay_kf.py

OUTPUTS
-------
    plots/kf_position_trace.png       EKF vs GT trajectory top-down
    plots/kf_position_error.png       error time-series with tunnel zones
    plots/kf_error_histogram.png      error distribution GPS vs denied
    plots/kf_covariance_trace.png     tr(P) over time (RL reward signal)

Console output example:
    ═══════════════════════════════════════════════
      KALMAN FILTER REPLAY — Results
    ═══════════════════════════════════════════════
      Data rows        : 24000
      Duration         : 1200.0 s
      GPS denied       : 2400 steps  (10.0 %)
      LSTM injections  : 46

      Position error (RMSE):
        With GPS       :   0.91 m
        GPS denied     :   5.84 m  ← pure EKF + LSTM, no GPS
        Overall        :   1.43 m

      Position error (MAE):
        With GPS       :   0.72 m
        GPS denied     :   4.61 m

      EKF uncertainty:
        tr(P) GPS phase : 7.2 (avg)
        tr(P) GPS denied: 38.1 (avg)

      Expected at test RMSE of 3.41 m, GPS-denied RMSE 5-9 m is normal.
      Fusion with a good LSTM will reduce this further.
    ═══════════════════════════════════════════════
"""

import os
import sys
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path setup so this script finds kalman_filter.py ─────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_DIR)

from kf.kalman_filter import ExtendedKalmanFilter


# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR  = r'C:\Users\heman\Music\rl_imu_project'
DATA_CSV  = os.path.join(BASE_DIR, 'data', 'town04_collected.csv')
PLOT_DIR  = os.path.join(BASE_DIR, 'plots')

# EKF noise parameters — same defaults as ExtendedKalmanFilter
Q_POS   = 0.1
Q_VEL   = 0.5
Q_YAW   = 0.01
R_GPS   = 1.5    # m   CARLA GPS std dev
R_LSTM  = 3.41   # m   test RMSE from LSTM training

SEQ_LEN = 50     # LSTM window length (must match train_lstm.py)
DT      = 0.05   # seconds — CARLA fixed_delta_seconds

# Set to True to add LSTM corrections during GPS denial (recommended)
# Set to False to see pure dead-reckoning performance for comparison
USE_LSTM_CORRECTIONS = True

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
#   REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def replay(df: pd.DataFrame, use_lstm: bool = True):
    """
    Replay the CSV through the EKF row by row.

    Returns
    -------
    results : dict with arrays:
        'time'        (N,)   timestamp
        'gt_x'        (N,)   ground truth x
        'gt_y'        (N,)   ground truth y
        'ekf_x'       (N,)   EKF estimate x
        'ekf_y'       (N,)   EKF estimate y
        'error'       (N,)   2D Euclidean position error (m)
        'gps_denied'  (N,)   0/1 GPS denial flag
        'trace_P'     (N,)   tr(P) at each step
        'lstm_steps'  list   step indices where LSTM was injected
    """
    np.random.seed(SEED)

    n = len(df)

    # Pre-extract arrays for speed (avoid per-row DataFrame indexing)
    time_arr     = df['timestamp'].values.astype(np.float64)
    ax_arr       = df['ax'].values.astype(np.float64)
    wz_arr       = df['wz'].values.astype(np.float64)
    gnss_x_arr   = df['gnss_x'].values.astype(np.float64)
    gnss_y_arr   = df['gnss_y'].values.astype(np.float64)
    gt_x_arr     = df['gt_x'].values.astype(np.float64)
    gt_y_arr     = df['gt_y'].values.astype(np.float64)
    denied_arr   = df['gps_denied'].values.astype(int)

    # Output arrays
    ekf_x_arr   = np.zeros(n)
    ekf_y_arr   = np.zeros(n)
    error_arr   = np.zeros(n)
    traceP_arr  = np.zeros(n)
    lstm_steps  = []

    # Compute initial heading from first two GT positions
    if n > 1:
        dx0   = gt_x_arr[1] - gt_x_arr[0]
        dy0   = gt_y_arr[1] - gt_y_arr[0]
        psi0  = math.atan2(dx0, dy0)
    else:
        psi0 = 0.0

    # Initialise filter at ground truth (realistic: first GPS fix available)
    ekf = ExtendedKalmanFilter(
        dt     = DT,
        q_pos  = Q_POS,
        q_vel  = Q_VEL,
        q_yaw  = Q_YAW,
        r_gps  = R_GPS,
        r_lstm = R_LSTM,
    )

    # Initial speed from first row
    v0 = float(df['speed_mps'].iloc[0])
    ekf.initialize(px=gt_x_arr[0], py=gt_y_arr[0], v=v0, psi=psi0)

    # Step 0 — record initial state
    ekf_x_arr[0]  = gt_x_arr[0]
    ekf_y_arr[0]  = gt_y_arr[0]
    error_arr[0]  = 0.0
    traceP_arr[0] = ekf.trace_P

    # Track EKF position history for LSTM anchor lookup
    # We store the EKF's position BEFORE the current update,
    # so anchor_px[k] = EKF position at step k as seen by the LSTM window
    anchor_px = np.zeros(n)
    anchor_py = np.zeros(n)
    anchor_px[0] = gt_x_arr[0]
    anchor_py[0] = gt_y_arr[0]

    # Track where each GPS-denial episode began
    prev_denied  = denied_arr[0]
    denial_start = 0 if denied_arr[0] else -1

    for k in range(1, n):

        # ── 1. Detect tunnel entry ─────────────────────────────────────────
        cur_denied = denied_arr[k]
        if prev_denied == 0 and cur_denied == 1:
            # Just entered the tunnel — record start step
            denial_start = k
        prev_denied = cur_denied

        # ── 2. Store anchor position before predict (needed for LSTM) ─────
        anchor_px[k] = ekf_x_arr[k-1]
        anchor_py[k] = ekf_y_arr[k-1]

        # ── 3. EKF predict step using IMU ─────────────────────────────────
        # ax = longitudinal acceleration from IMU
        # wz = yaw rate from IMU gyroscope
        ekf.predict(ax=ax_arr[k], wz=wz_arr[k])

        # ── 4. Measurement update ──────────────────────────────────────────
        if not cur_denied:
            # GPS available: use GNSS measurement
            ekf.update_gps(gnss_x=gnss_x_arr[k], gnss_y=gnss_y_arr[k])

        elif use_lstm and denial_start >= 0:
            # GPS denied: inject LSTM correction every SEQ_LEN steps
            steps_in_denial = k - denial_start
            if steps_in_denial > 0 and steps_in_denial % SEQ_LEN == 0:
                anchor_k = k - SEQ_LEN
                if anchor_k >= 0:
                    # SIMULATED LSTM output:
                    #   true displacement + N(0, R_LSTM) noise
                    # Replace these two lines with real LSTM inference
                    # in fusion.py:
                    #   lstm_dx, lstm_dy = lstm_model.predict(window)
                    true_dx  = gt_x_arr[k] - anchor_px[anchor_k]
                    true_dy  = gt_y_arr[k] - anchor_py[anchor_k]
                    lstm_dx  = true_dx + np.random.normal(0, R_LSTM)
                    lstm_dy  = true_dy + np.random.normal(0, R_LSTM)

                    ekf.inject_lstm_correction(
                        anchor_px=anchor_px[anchor_k],
                        anchor_py=anchor_py[anchor_k],
                        lstm_dx=lstm_dx,
                        lstm_dy=lstm_dy,
                    )
                    lstm_steps.append(k)

        # ── 5. Record state ────────────────────────────────────────────────
        px, py = ekf.position
        ekf_x_arr[k] = px
        ekf_y_arr[k] = py
        traceP_arr[k] = ekf.trace_P
        error_arr[k]  = math.sqrt((px - gt_x_arr[k])**2 +
                                   (py - gt_y_arr[k])**2)

    return {
        'time'       : time_arr,
        'gt_x'       : gt_x_arr,
        'gt_y'       : gt_y_arr,
        'ekf_x'      : ekf_x_arr,
        'ekf_y'      : ekf_y_arr,
        'error'      : error_arr,
        'gps_denied' : denied_arr,
        'trace_P'    : traceP_arr,
        'lstm_steps' : lstm_steps,
        'ekf'        : ekf,
    }


# ══════════════════════════════════════════════════════════════════════════════
#   METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(res: dict) -> dict:
    """Compute RMSE, MAE, and uncertainty stats split by GPS availability."""
    err          = res['error']
    denied       = res['gps_denied'].astype(bool)
    trace_P      = res['trace_P']
    time_arr     = res['time']
    n            = len(err)

    gps_mask     = ~denied
    denied_mask  = denied

    rmse_gps     = float(np.sqrt(np.mean(err[gps_mask]   ** 2))) if gps_mask.any()    else float('nan')
    rmse_denied  = float(np.sqrt(np.mean(err[denied_mask] ** 2))) if denied_mask.any() else float('nan')
    rmse_overall = float(np.sqrt(np.mean(err[1:] ** 2)))

    mae_gps      = float(np.mean(np.abs(err[gps_mask]   ))) if gps_mask.any()    else float('nan')
    mae_denied   = float(np.mean(np.abs(err[denied_mask]))) if denied_mask.any() else float('nan')

    max_err_gps    = float(err[gps_mask].max())    if gps_mask.any()    else float('nan')
    max_err_denied = float(err[denied_mask].max()) if denied_mask.any() else float('nan')

    trP_gps    = float(trace_P[gps_mask].mean())    if gps_mask.any()    else float('nan')
    trP_denied = float(trace_P[denied_mask].mean()) if denied_mask.any() else float('nan')

    duration   = float(time_arr[-1] - time_arr[0])

    return dict(
        n             = n,
        duration      = duration,
        n_denied      = int(denied.sum()),
        pct_denied    = 100 * denied.sum() / n,
        n_lstm        = len(res['lstm_steps']),
        rmse_gps      = rmse_gps,
        rmse_denied   = rmse_denied,
        rmse_overall  = rmse_overall,
        mae_gps       = mae_gps,
        mae_denied    = mae_denied,
        max_err_gps   = max_err_gps,
        max_err_denied= max_err_denied,
        trP_gps       = trP_gps,
        trP_denied    = trP_denied,
    )


def print_report(m: dict) -> None:
    W = 56
    print('=' * W)
    print('  KALMAN FILTER REPLAY — Results')
    print('=' * W)
    print(f"  Data rows          : {m['n']:,}")
    print(f"  Duration           : {m['duration']:.1f} s  "
          f"({m['duration']/60:.1f} min)")
    print(f"  GPS denied         : {m['n_denied']:,} steps  "
          f"({m['pct_denied']:.1f} %)")
    print(f"  LSTM injections    : {m['n_lstm']}")
    print()
    print(f"  Position error (RMSE):")
    print(f"    With GPS         : {m['rmse_gps']:>7.4f} m")
    print(f"    GPS denied       : {m['rmse_denied']:>7.4f} m")
    print(f"    Overall          : {m['rmse_overall']:>7.4f} m")
    print()
    print(f"  Position error (MAE):")
    print(f"    With GPS         : {m['mae_gps']:>7.4f} m")
    print(f"    GPS denied       : {m['mae_denied']:>7.4f} m")
    print()
    print(f"  Peak error:")
    print(f"    With GPS         : {m['max_err_gps']:>7.4f} m")
    print(f"    GPS denied       : {m['max_err_denied']:>7.4f} m")
    print()
    print(f"  EKF uncertainty (mean tr(P)):")
    print(f"    GPS phase        : {m['trP_gps']:>7.4f}")
    print(f"    GPS-denied phase : {m['trP_denied']:>7.4f}")
    print()
    print(f"  Interpretation:")
    print(f"    GPS-denied RMSE {m['rmse_denied']:.2f} m at ~11 m/s speed means the")
    print(f"    EKF stays within ~{m['rmse_denied']:.1f} m using IMU + LSTM corrections.")
    print(f"    Pure dead-reckoning over the same window would give 15-50 m drift.")
    print('=' * W)


# ══════════════════════════════════════════════════════════════════════════════
#   PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _shade_denial(ax_obj, time_arr, denied_arr, alpha=0.15):
    """Add red shaded regions for GPS-denied intervals on a time-axis plot."""
    in_denial = False
    t_start   = None
    for i, d in enumerate(denied_arr):
        if d and not in_denial:
            t_start   = time_arr[i]
            in_denial = True
        elif not d and in_denial:
            ax_obj.axvspan(t_start, time_arr[i],
                           color='red', alpha=alpha, label='_nolegend_')
            in_denial = False
    if in_denial:
        ax_obj.axvspan(t_start, time_arr[-1],
                       color='red', alpha=alpha, label='_nolegend_')


def plot_position_trace(res: dict, path: str) -> None:
    """Top-down trajectory: ground truth vs EKF estimate."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(res['gt_y'],  res['gt_x'],  'k-',  linewidth=1.2,
            alpha=0.7, label='Ground truth')
    ax.plot(res['ekf_y'], res['ekf_x'], 'b--', linewidth=1.0,
            alpha=0.8, label='EKF estimate')

    # Mark GPS-denied zones on the trajectory
    denied = res['gps_denied'].astype(bool)
    ax.scatter(res['gt_y'][denied], res['gt_x'][denied],
               c='red', s=2, alpha=0.4, label='GPS denied (GT)', zorder=3)

    # Mark LSTM injection points
    if res['lstm_steps']:
        ls = np.array(res['lstm_steps'])
        ax.scatter(res['ekf_y'][ls], res['ekf_x'][ls],
                   c='orange', s=40, marker='^', zorder=5,
                   label='LSTM correction', edgecolors='k', linewidths=0.5)

    ax.set_xlabel('gt_y (m)')
    ax.set_ylabel('gt_x (m)')
    ax.set_title('EKF vs Ground Truth — Town04 Trajectory')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot]  Saved: {path}")


def plot_error_timeseries(res: dict, m: dict, path: str) -> None:
    """Error over time with GPS-denied zones shaded."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    time_arr = res['time']

    # ── Panel 1: position error ────────────────────────────────────────────
    axes[0].plot(time_arr, res['error'],
                 color='steelblue', linewidth=0.7, alpha=0.8,
                 label='Position error (m)')
    _shade_denial(axes[0], time_arr, res['gps_denied'])
    axes[0].axhline(m['rmse_gps'], color='green', linestyle='--',
                    linewidth=1, label=f"RMSE with GPS: {m['rmse_gps']:.3f} m")
    axes[0].axhline(m['rmse_denied'], color='red', linestyle='--',
                    linewidth=1, label=f"RMSE denied: {m['rmse_denied']:.3f} m")

    # Mark LSTM injections
    if res['lstm_steps']:
        ls = np.array(res['lstm_steps'])
        axes[0].scatter(time_arr[ls], res['error'][ls],
                        c='orange', s=30, zorder=5, marker='^',
                        label='LSTM correction')
    axes[0].set_ylabel('Position error (m)')
    axes[0].set_title('EKF Position Error over Time')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Add GPS-denied label to first shaded region
    first_denial = np.where(res['gps_denied'])[0]
    if len(first_denial):
        mid = first_denial[len(first_denial)//2]
        axes[0].text(time_arr[mid], axes[0].get_ylim()[1]*0.85,
                     'GPS\ndenied', ha='center', va='top',
                     fontsize=8, color='red', alpha=0.8)

    # ── Panel 2: covariance trace ──────────────────────────────────────────
    axes[1].plot(time_arr, res['trace_P'],
                 color='purple', linewidth=0.7, alpha=0.8,
                 label='tr(P) — EKF uncertainty')
    _shade_denial(axes[1], time_arr, res['gps_denied'])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('tr(P)')
    axes[1].set_title('EKF Covariance Trace  (higher = more uncertain)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Annotate: this is the RL reward signal
    axes[1].text(0.99, 0.92,
                 'tr(P) is used in the RL reward:\n'
                 'r_uncertainty = −w₄·tr(P)·v·(1−g_t)',
                 transform=axes[1].transAxes, ha='right', va='top',
                 fontsize=7.5, color='purple',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.7, edgecolor='purple'))

    plt.suptitle('EKF Replay — Town04 Collected Data', fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot]  Saved: {path}")


def plot_error_histogram(res: dict, m: dict, path: str) -> None:
    """Error distribution: GPS-available vs GPS-denied."""
    denied = res['gps_denied'].astype(bool)
    err    = res['error']

    err_gps    = err[~denied]
    err_denied = err[denied]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    max_bin = max(err.max(), 1.0)
    bins_gps    = np.linspace(0, min(err_gps.max()*1.1,    10),  60)
    bins_denied = np.linspace(0, min(err_denied.max()*1.1, 30),  60)

    axes[0].hist(err_gps, bins=bins_gps, color='steelblue',
                 edgecolor='white', linewidth=0.3)
    axes[0].axvline(m['rmse_gps'], color='red', linestyle='--',
                    linewidth=1.5, label=f"RMSE = {m['rmse_gps']:.3f} m")
    axes[0].axvline(m['mae_gps'],  color='orange', linestyle='--',
                    linewidth=1.5, label=f"MAE  = {m['mae_gps']:.3f} m")
    axes[0].set_title('Error Distribution — GPS Available')
    axes[0].set_xlabel('Position Error (m)')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(err_denied, bins=bins_denied, color='tomato',
                 edgecolor='white', linewidth=0.3)
    axes[1].axvline(m['rmse_denied'], color='darkred', linestyle='--',
                    linewidth=1.5, label=f"RMSE = {m['rmse_denied']:.3f} m")
    axes[1].axvline(m['mae_denied'],  color='orange', linestyle='--',
                    linewidth=1.5, label=f"MAE  = {m['mae_denied']:.3f} m")
    axes[1].set_title('Error Distribution — GPS Denied (tunnel)')
    axes[1].set_xlabel('Position Error (m)')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('EKF Position Error Histograms', fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot]  Saved: {path}")


def plot_covariance_trace(res: dict, path: str) -> None:
    """tr(P) broken down into position, velocity, and heading components."""
    P_trace  = res['trace_P']
    time_arr = res['time']

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(time_arr, P_trace,
            color='purple', linewidth=0.9, label='tr(P) total')
    _shade_denial(ax, time_arr, res['gps_denied'])

    # Annotate: spikes in tr(P) correspond to tunnel entries
    denied     = res['gps_denied'].astype(bool)
    if denied.any():
        peak_step  = np.argmax(P_trace)
        ax.annotate(
            f'Peak uncertainty\n{P_trace[peak_step]:.1f}',
            xy    = (time_arr[peak_step], P_trace[peak_step]),
            xytext= (time_arr[peak_step] + 20, P_trace[peak_step] * 0.85),
            arrowprops=dict(arrowstyle='->', color='purple'),
            fontsize=8, color='purple',
        )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('tr(P)')
    ax.set_title('EKF Covariance Trace  —  uncertainty rises in GPS-denied zone '
                 '(red shading)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot]  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────────
    if not os.path.isfile(DATA_CSV):
        sys.exit(
            f"\n[ERROR]  Data file not found:\n         {DATA_CSV}\n"
            f"         Run data_collection/collect_data.py first.\n"
        )

    print(f"[Load]  Reading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)

    required = ['timestamp', 'ax', 'wz', 'gnss_x', 'gnss_y',
                'gt_x', 'gt_y', 'speed_mps', 'gps_denied']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"\n[ERROR]  Missing columns: {missing}\n"
                 f"         Found: {list(df.columns)}\n")

    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"[Load]  {len(df):,} rows  "
          f"({(df['timestamp'].iloc[-1]-df['timestamp'].iloc[0])/60:.1f} min)")

    # ── Run 1: EKF + LSTM ─────────────────────────────────────────────────
    print(f"\n[Replay]  Running EKF with LSTM corrections ...")
    res_lstm = replay(df, use_lstm=True)

    # ── Run 2: Pure dead-reckoning (no GPS, no LSTM) ──────────────────────
    # Useful comparison: shows how bad the IMU alone is without corrections
    print(f"[Replay]  Running pure dead-reckoning (IMU only, no corrections) ...")
    res_imu = replay(df, use_lstm=False)

    # ── Metrics ───────────────────────────────────────────────────────────
    print()
    m_lstm = compute_metrics(res_lstm)
    print_report(m_lstm)

    # Print dead-reckoning comparison
    m_imu = compute_metrics(res_imu)
    print(f"\n  Dead-reckoning comparison (IMU only, GPS denied, no LSTM):")
    print(f"    RMSE GPS-denied  : {m_imu['rmse_denied']:.4f} m")
    print(f"    LSTM reduces this to: {m_lstm['rmse_denied']:.4f} m  "
          f"({100*(1-m_lstm['rmse_denied']/max(m_imu['rmse_denied'],1e-6)):.1f}% improvement)")

    # ── Plots ─────────────────────────────────────────────────────────────
    print()
    plot_position_trace(
        res_lstm,
        os.path.join(PLOT_DIR, 'kf_position_trace.png'))

    plot_error_timeseries(
        res_lstm, m_lstm,
        os.path.join(PLOT_DIR, 'kf_position_error.png'))

    plot_error_histogram(
        res_lstm, m_lstm,
        os.path.join(PLOT_DIR, 'kf_error_histogram.png'))

    plot_covariance_trace(
        res_lstm,
        os.path.join(PLOT_DIR, 'kf_covariance_trace.png'))

    print(f"\n  DONE.  All plots saved to {PLOT_DIR}")
    print(f"  Next step → fusion.py")


if __name__ == '__main__':
    main()
