r"""
demo.py  —  Live CARLA Demonstration
=====================================
For: LSTM + RL-Adaptive EKF Localization Project
Purpose: Professor demo — shows accurate position estimation in real time,
         including inside the GPS-denied Town04 tunnel.

What this script does
---------------------
1. Connects to CARLA, spawns a Tesla Model 3 on Town04's highway.
2. Loads your trained LSTM and RL agent (graceful fallback if missing).
3. Runs a live matplotlib window with 4 panels updating every tick:
     Panel 1 — XY Trajectory: GT (green) vs LSTM-EKF (blue) vs Baseline (red)
     Panel 2 — Position Error: error in metres over time, tunnel shaded red
     Panel 3 — Q / R Scale: how the RL agent adapts noise parameters live
     Panel 4 — Live Metrics: 6 numbers updated every step
4. Prints a colour-coded console line every step.
5. On Ctrl-C or episode end: saves demo_final.png + demo_results.csv.

Key demo moment
---------------
When the vehicle enters the tunnel, GPS cuts out.
Watch Panel 2: the red baseline line spikes upward (3-5m error).
The blue LSTM-EKF line stays near 0.5m.
Panel 3 shows Q jumping up as the RL agent detects GPS denial.
That is your proof-of-concept in one visual.

Usage
-----
    # Terminal 1 — start CARLA first
    CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

    # Terminal 2
    cd C:\\Users\\heman\\Music\\rl_imu_project
    carla_env37\\Scripts\\activate
    python demo.py                         # full demo (LSTM + RL)
    python demo.py --no-rl                 # LSTM-EKF only, no RL adaptation
    python demo.py --steps 800             # longer run (default 500)
    python demo.py --no-render             # hide CARLA window (faster)

Output files
------------
    results/demo_final.png       screenshot of final panel state
    results/demo_results.csv     per-step data (gt_x, gt_y, ekf_x, ekf_y, error, ...)
"""

import sys
import os
import argparse
import time
import math
import csv
import logging
import signal

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# =============================================================================
# PATH SETUP
# =============================================================================
ROOT_DIR  = r'C:\Users\heman\Music\rl_imu_project'
CARLA_DIR = os.path.join(ROOT_DIR, 'carla_implementation')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CARLA_DIR)

from ekf          import AdaptiveEKF, LSTMBridge
from rl_agent     import PPOAgent
from carla_sensor_bridge   import CARLASensorBridge
from carla_rl_environment  import CARLALocalizationEnv, _correct_imu_for_gravity
from carla_config import (
    FIXED_DELTA_SECONDS, MAX_STEPS,
    LSTM_MODEL_PATH, LSTM_STATS_PATH,
    BEST_MODEL_PATH, RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('Demo')

# ANSI colours for console output
_RED    = '\033[91m'
_GREEN  = '\033[92m'
_BLUE   = '\033[94m'
_YELLOW = '\033[93m'
_RESET  = '\033[0m'
_BOLD   = '\033[1m'


# =============================================================================
# ARGUMENT PARSER
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Live CARLA demo — LSTM-EKF localization')
    p.add_argument('--steps',     type=int,  default=500,
                   help='Number of simulation steps (default 500 = 25s)')
    p.add_argument('--no-rl',     action='store_true',
                   help='Disable RL adaptive tuning (LSTM-EKF only)')
    p.add_argument('--no-render', action='store_true',
                   help='Headless CARLA (faster, no CARLA window)')
    p.add_argument('--spawn',     type=int,  default=16,
                   help='Spawn point index (default 16 = highway near tunnel)')
    return p.parse_args()


# =============================================================================
# LIVE DASHBOARD
# =============================================================================
class LiveDashboard:
    """
    4-panel matplotlib window that updates every simulation step.

    Panels:
      [0,0] XY Trajectory  — GT (green), LSTM-EKF (blue), Baseline (red dot)
      [0,1] Position Error  — error over time; tunnel shaded red
      [1,0] Q/R Scale       — RL agent adaptation (or flat 1.0 if --no-rl)
      [1,1] Live Metrics    — 6 stat cards updating every step
    """

    WINDOW_LEN = 300   # number of steps shown on rolling plots

    def __init__(self, total_steps: int, use_rl: bool):
        self.total_steps = total_steps
        self.use_rl      = use_rl

        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#0f0f0f')
        self.fig.suptitle(
            'RL-Adaptive EKF Localization  —  Live Demo  |  Town04',
            fontsize=12, fontweight='bold', color='white')

        gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                               left=0.07, right=0.97, top=0.92, bottom=0.06)
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        self.ax_err  = self.fig.add_subplot(gs[0, 1])
        self.ax_qr   = self.fig.add_subplot(gs[1, 0])
        self.ax_met  = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_traj, self.ax_err, self.ax_qr):
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#aaa', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')

        self.ax_met.set_facecolor('#1a1a1a')
        self.ax_met.axis('off')

        plt.show(block=False)

    # ── single update called every N steps ────────────────────────────────────

    def update(self, data: dict, step: int):
        """
        data keys:
          gt_x, gt_y           list[float]   ground truth path
          ekf_x, ekf_y         list[float]   LSTM-EKF estimate
          base_x, base_y       list[float]   baseline EKF estimate
          errors_ekf           list[float]   LSTM-EKF error per step
          errors_base          list[float]   baseline error per step
          q_scales, r_scales   list[float]   RL noise scales
          gps_denied           list[int]     0/1 per step
          tunnel_err_ekf       float         mean tunnel error (LSTM-EKF)
          tunnel_err_base      float         mean tunnel error (baseline)
          road_err_ekf         float         mean road error (LSTM-EKF)
          cur_q, cur_r         float         current Q/R scales
          improvement_pct      float         % improvement vs baseline in tunnel
        """
        # Trim to rolling window
        w = self.WINDOW_LEN
        gt_x  = data['gt_x'][-w:]
        gt_y  = data['gt_y'][-w:]
        ek_x  = data['ekf_x'][-w:]
        ek_y  = data['ekf_y'][-w:]
        bx_x  = data['base_x'][-w:]
        bx_y  = data['base_y'][-w:]
        e_ekf = data['errors_ekf'][-w:]
        e_bas = data['errors_base'][-w:]
        qsc   = data['q_scales'][-w:]
        rsc   = data['r_scales'][-w:]
        gps   = np.array(data['gps_denied'][-w:], dtype=bool)
        t_ax  = np.arange(len(e_ekf)) * FIXED_DELTA_SECONDS

        # ── Panel 1: Trajectory ───────────────────────────────────────────────
        ax = self.ax_traj
        ax.cla(); ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaa', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#333')

        # Shade tunnel region on trajectory (GPS denied points)
        if gps.any() and len(gt_x) > 1:
            ax.scatter(np.array(gt_x)[gps], np.array(gt_y)[gps],
                       c='#ff3333', s=6, alpha=0.3, zorder=1)

        if len(gt_x) > 1:
            ax.plot(gt_x, gt_y, '-',  color='#4caf50', lw=2.0,
                    label='Ground truth', zorder=3)
            ax.plot(ek_x, ek_y, '--', color='#42a5f5', lw=1.5,
                    label='LSTM-EKF', zorder=4)
            ax.plot(bx_x, bx_y, ':',  color='#ef5350', lw=1.5,
                    label='Baseline EKF', zorder=2)

            # Current position marker
            ax.scatter([gt_x[-1]], [gt_y[-1]], c='#4caf50', s=60,
                       zorder=5, marker='o')

        ax.set_title('XY Trajectory', color='white', fontsize=9, fontweight='bold')
        ax.set_xlabel('x (m)', color='#aaa', fontsize=8)
        ax.set_ylabel('y (m)', color='#aaa', fontsize=8)
        ax.legend(fontsize=7, loc='upper left',
                  facecolor='#222', labelcolor='white',
                  edgecolor='#444')
        ax.grid(True, alpha=0.15, color='#555')

        # ── Panel 2: Error over time ──────────────────────────────────────────
        ax = self.ax_err
        ax.cla(); ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaa', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#333')

        if len(e_ekf) > 1:
            # Shade tunnel intervals
            if gps.any():
                in_tunnel = False
                t_start   = 0.0
                for i, denied in enumerate(gps):
                    t_i = i * FIXED_DELTA_SECONDS
                    if denied and not in_tunnel:
                        t_start   = t_i
                        in_tunnel = True
                    elif not denied and in_tunnel:
                        ax.axvspan(t_start, t_i,
                                   alpha=0.18, color='#ff5555',
                                   label='_nolegend_')
                        in_tunnel = False
                if in_tunnel:
                    ax.axvspan(t_start, t_ax[-1],
                               alpha=0.18, color='#ff5555')

            ax.plot(t_ax, e_bas, '-', color='#ef5350', lw=1.5,
                    alpha=0.85, label='Baseline EKF')
            ax.plot(t_ax, e_ekf, '-', color='#42a5f5', lw=2.0,
                    label='LSTM-EKF')

        ax.set_title('Position Error (m)', color='white',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Time (s)', color='#aaa', fontsize=8)
        ax.set_ylabel('Error (m)', color='#aaa', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, facecolor='#222',
                  labelcolor='white', edgecolor='#444')
        ax.grid(True, alpha=0.15, color='#555')

        # Annotate tunnel
        if gps.any():
            tunnel_t = t_ax[gps]
            if len(tunnel_t) > 0:
                mid = (tunnel_t[0] + tunnel_t[-1]) / 2
                ax.text(mid, ax.get_ylim()[1] * 0.92, 'TUNNEL',
                        ha='center', va='top', fontsize=7,
                        color='#ff5555', fontweight='bold')

        # ── Panel 3: Q / R scales ─────────────────────────────────────────────
        ax = self.ax_qr
        ax.cla(); ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaa', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#333')

        if len(qsc) > 1:
            ax.axhline(1.0, color='#555', lw=0.8, ls='--')
            ax.plot(t_ax[:len(qsc)], qsc, '-',
                    color='#66bb6a', lw=1.8, label='Q scale')
            ax.plot(t_ax[:len(rsc)], rsc, '-',
                    color='#ab47bc', lw=1.8, label='R scale')
            if gps.any():
                for i, denied in enumerate(gps):
                    if denied:
                        ax.axvspan(i*FIXED_DELTA_SECONDS,
                                   (i+1)*FIXED_DELTA_SECONDS,
                                   alpha=0.08, color='#ff5555',
                                   label='_nolegend_')

        title = 'Q / R Scale (RL Agent)' if self.use_rl \
                else 'Q / R Scale (fixed — no RL)'
        ax.set_title(title, color='white', fontsize=9, fontweight='bold')
        ax.set_xlabel('Time (s)', color='#aaa', fontsize=8)
        ax.set_ylabel('Scale', color='#aaa', fontsize=8)
        ax.legend(fontsize=7, facecolor='#222',
                  labelcolor='white', edgecolor='#444')
        ax.grid(True, alpha=0.15, color='#555')

        # ── Panel 4: Live metric cards ────────────────────────────────────────
        ax = self.ax_met
        ax.cla(); ax.set_facecolor('#1a1a1a'); ax.axis('off')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')

        t_ekf  = data.get('tunnel_err_ekf',  float('nan'))
        t_base = data.get('tunnel_err_base', float('nan'))
        r_ekf  = data.get('road_err_ekf',    float('nan'))
        impr   = data.get('improvement_pct', float('nan'))
        cur_q  = data.get('cur_q', 1.0)
        cur_r  = data.get('cur_r', 1.0)
        gps_st = 'DENIED' if (len(data['gps_denied']) > 0
                               and data['gps_denied'][-1]) else 'OK'
        gps_col = '#ff3333' if gps_st == 'DENIED' else '#4caf50'

        # Step / GPS status header
        ax.text(0.5, 0.97,
                f'Step {step} / {self.total_steps}   |   GPS: {gps_st}',
                ha='center', va='top', transform=ax.transAxes,
                fontsize=10, fontweight='bold', color=gps_col)

        # Metric card helper
        def _card(x, y, label, value, val_color='white'):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), 0.42, 0.18,
                boxstyle='round,pad=0.02',
                facecolor='#2a2a2a', edgecolor='#444',
                transform=ax.transAxes, linewidth=0.8))
            ax.text(x + 0.21, y + 0.13, label,
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=7.5, color='#888')
            ax.text(x + 0.21, y + 0.05, value,
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=13, fontweight='bold', color=val_color)

            nan_str = lambda v, fmt=':.2f': (f'{v:{fmt[1:]}}' if not (v != v) else 'n/a')

        t_ekf_s  = f'{t_ekf:.2f} m'  if not math.isnan(t_ekf)  else 'n/a'
        t_base_s = f'{t_base:.2f} m' if not math.isnan(t_base) else 'n/a'
        r_ekf_s  = f'{r_ekf:.2f} m'  if not math.isnan(r_ekf)  else 'n/a'
        impr_s   = f'{impr:.0f}%'    if not math.isnan(impr)   else 'n/a'

        _card(0.04, 0.70, 'Tunnel error — LSTM-EKF', t_ekf_s,  '#42a5f5')
        _card(0.54, 0.70, 'Tunnel error — Baseline', t_base_s, '#ef5350')
        _card(0.04, 0.44, 'Road error (both ~equal)', r_ekf_s, '#aaa')
        _card(0.54, 0.44, 'Tunnel improvement',       impr_s,  '#4caf50')
        _card(0.04, 0.18, 'Current Q scale',  f'{cur_q:.2f}',  '#66bb6a')
        _card(0.54, 0.18, 'Current R scale',  f'{cur_r:.2f}',  '#ab47bc')

        # ── Flush ─────────────────────────────────────────────────────────────
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fig.savefig(path, dpi=150, bbox_inches='tight',
                         facecolor=self.fig.get_facecolor())
        log.info(f'Dashboard saved -> {path}')

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# =============================================================================
# CONSOLE LINE PRINTER
# =============================================================================
def print_step(step, total, gps_denied, ekf_err, base_err, q, r, lstm_on):
    gps_s   = f'{_RED}GPS DENIED{_RESET}' if gps_denied \
              else f'{_GREEN}GPS OK    {_RESET}'
    lstm_s  = f'{_GREEN}LSTM ON {_RESET}' if lstm_on \
              else f'{_YELLOW}LSTM WAIT{_RESET}'
    ekf_col = _BLUE if ekf_err < base_err else _YELLOW
    print(
        f'\r[{step:>4}/{total}]  {gps_s}  {lstm_s}  '
        f'EKF: {ekf_col}{ekf_err:>5.2f}m{_RESET}  '
        f'Base: {_RED}{base_err:>5.2f}m{_RESET}  '
        f'Q={q:.2f} R={r:.2f}',
        end='', flush=True,
    )


# =============================================================================
# MAIN DEMO FUNCTION
# =============================================================================
def run_demo(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('\n' + '='*65)
    print(f'{_BOLD}  RL-Adaptive EKF  —  Live Demo{_RESET}')
    print(f'  Steps   : {args.steps}  ({args.steps * FIXED_DELTA_SECONDS:.0f}s)')
    print(f'  RL agent: {"DISABLED (--no-rl)" if args.no_rl else "Enabled"}')
    print(f'  LSTM    : Enabled (falls back to raw IMU if weights missing)')
    print('='*65 + '\n')

    # ── Load LSTM bridge ──────────────────────────────────────────────────────
    log.info('Loading LSTM bridge...')
    lstm_bridge = LSTMBridge(
        model_path=LSTM_MODEL_PATH,
        stats_path=LSTM_STATS_PATH,
    )

    # ── Load RL agent ─────────────────────────────────────────────────────────
    rl_agent = None
    if not args.no_rl:
        if os.path.isfile(BEST_MODEL_PATH):
            rl_agent = PPOAgent(obs_dim=10, action_dim=2)
            rl_agent.load(BEST_MODEL_PATH)
            log.info(f'RL agent loaded from {BEST_MODEL_PATH}')
        else:
            log.warning(f'RL model not found at {BEST_MODEL_PATH} '
                        f'— running without RL adaptation.')

    # ── Two EKF instances: one LSTM-aided, one baseline ───────────────────────
    ekf_lstm = AdaptiveEKF()    # uses LSTM + RL (main system)
    ekf_base = AdaptiveEKF()    # raw IMU only  (baseline for comparison)

    # ── Connect to CARLA via RL environment ───────────────────────────────────
    log.info('Connecting to CARLA...')
    env = CARLALocalizationEnv(
        ekf_instance=ekf_lstm,
        render=not args.no_render,
    )
    log.info('CARLA environment ready.')

    # ── Initialise dashboard ──────────────────────────────────────────────────
    dashboard = LiveDashboard(
        total_steps=args.steps,
        use_rl=(rl_agent is not None),
    )

    # ── Data buffers ──────────────────────────────────────────────────────────
    buf = {
        'gt_x': [], 'gt_y': [],
        'ekf_x': [], 'ekf_y': [],
        'base_x': [], 'base_y': [],
        'errors_ekf': [], 'errors_base': [],
        'q_scales': [], 'r_scales': [],
        'gps_denied': [],
    }

    # Per-run accumulators for tunnel / road metric cards
    tun_errs_ekf,  tun_errs_base = [], []
    road_errs_ekf, road_errs_base = [], []

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'demo_results.csv')
    csv_f    = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_w    = csv.DictWriter(csv_f, fieldnames=[
        'step', 'timestamp',
        'gt_x', 'gt_y',
        'ekf_x', 'ekf_y', 'ekf_err',
        'base_x', 'base_y', 'base_err',
        'gps_denied', 'lstm_active',
        'q_scale', 'r_scale',
    ])
    csv_w.writeheader()

    # ── Graceful shutdown on Ctrl-C ───────────────────────────────────────────
    _stop = [False]
    def _sigint(sig, frame):
        print(f'\n{_YELLOW}  Ctrl-C received — saving and shutting down...{_RESET}')
        _stop[0] = True
    signal.signal(signal.SIGINT, _sigint)

    # ── Reset environment ─────────────────────────────────────────────────────
    obs = env.reset()
    ekf_base.reset()

    # Seed the baseline EKF at the same starting position as the main EKF
    # (we read it from env after reset() has called ekf_lstm.initialize())
    state0 = ekf_lstm.get_state()
    pos0   = state0['position']
    ekf_base.initialize(
        x0       = float(pos0[0]),
        y0       = float(pos0[1]),
        heading0 = float(state0['theta']),
        speed0   = float(state0['velocity']),
    )
    lstm_bridge.reset()

    log.info(f'Starting demo — {args.steps} steps. Watch the tunnel!\n')

    q_scale = r_scale = 1.0   # default; overridden by RL agent if active

    for step in range(1, args.steps + 1):
        if _stop[0]:
            break

        # ── RL agent selects noise scales ──────────────────────────────────────
        if rl_agent is not None:
            action, _, _ = rl_agent.select_action(obs, deterministic=True)
            # action = [delta_Q, delta_R] in [-0.5, 0.5]
            q_scale = float(np.clip(q_scale + action[0], 0.1, 3.0))
            r_scale = float(np.clip(r_scale + action[1], 0.1, 3.0))
        # else: q_scale / r_scale stay at 1.0

        # ── Environment step (ticks CARLA, runs main LSTM-EKF internally) ──────
        obs, reward, done, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
            if rl_agent is None
            else action
        )

        gt_x       = info['gt_x']
        gt_y       = info['gt_y']
        ekf_x      = info['ekf_x']
        ekf_y      = info['ekf_y']
        gps_denied = bool(info['gps_denied'])
        lstm_on    = info.get('lstm_active', False)
        ekf_err    = info['position_error']

        # ── Baseline EKF step (raw IMU, no LSTM, fixed Q/R=1) ─────────────────
        # Pull the same sensor data that was just used by the main EKF.
        # We replicate the predict/update pattern from carla_rl_environment.py
        # but with raw ax (no LSTM) and fixed noise scales.
        last_bundle = env._last_bundle
        if last_bundle is not None:
            ax_raw = last_bundle.imu.forward_accel
            ay_raw = last_bundle.imu.accel_y
            wz     = last_bundle.imu.yaw_rate
            pitch  = last_bundle.ground_truth.pitch_deg
            roll   = last_bundle.ground_truth.roll_deg
            ax_c, _ = _correct_imu_for_gravity(ax_raw, ay_raw, pitch, roll)

            ekf_base.predict(a_fwd=ax_c, wz=wz,
                             gps_denied=gps_denied)
            if not gps_denied and last_bundle.gnss is not None:
                ekf_base.update(last_bundle.gnss.local_x,
                                last_bundle.gnss.local_y)

        bx, by   = ekf_base.get_position()
        base_err = math.sqrt((bx - gt_x)**2 + (by - gt_y)**2)

        # ── Buffer append ──────────────────────────────────────────────────────
        buf['gt_x'].append(gt_x);   buf['gt_y'].append(gt_y)
        buf['ekf_x'].append(ekf_x); buf['ekf_y'].append(ekf_y)
        buf['base_x'].append(bx);   buf['base_y'].append(by)
        buf['errors_ekf'].append(ekf_err)
        buf['errors_base'].append(base_err)
        buf['q_scales'].append(q_scale)
        buf['r_scales'].append(r_scale)
        buf['gps_denied'].append(int(gps_denied))

        if gps_denied:
            tun_errs_ekf.append(ekf_err)
            tun_errs_base.append(base_err)
        else:
            road_errs_ekf.append(ekf_err)

        # ── Compute metric card values ─────────────────────────────────────────
        t_ekf  = float(np.mean(tun_errs_ekf))   if tun_errs_ekf  else float('nan')
        t_base = float(np.mean(tun_errs_base))  if tun_errs_base else float('nan')
        r_ekf  = float(np.mean(road_errs_ekf))  if road_errs_ekf else float('nan')
        if not (math.isnan(t_ekf) or math.isnan(t_base)) and t_base > 0:
            impr = (t_base - t_ekf) / t_base * 100.0
        else:
            impr = float('nan')

        # ── CSV write ──────────────────────────────────────────────────────────
        csv_w.writerow({
            'step':        step,
            'timestamp':   round(step * FIXED_DELTA_SECONDS, 3),
            'gt_x':        round(gt_x, 4),  'gt_y':  round(gt_y, 4),
            'ekf_x':       round(ekf_x, 4), 'ekf_y': round(ekf_y, 4),
            'ekf_err':     round(ekf_err, 4),
            'base_x':      round(bx, 4),    'base_y': round(by, 4),
            'base_err':    round(base_err, 4),
            'gps_denied':  int(gps_denied),
            'lstm_active': int(lstm_on),
            'q_scale':     round(q_scale, 4),
            'r_scale':     round(r_scale, 4),
        })
        csv_f.flush()

        # ── Console ───────────────────────────────────────────────────────────
        print_step(step, args.steps, gps_denied,
                   ekf_err, base_err, q_scale, r_scale, lstm_on)

        # ── Dashboard update every 5 steps ────────────────────────────────────
        if step % 5 == 0 or step == 1:
            dashboard.update({
                **buf,
                'tunnel_err_ekf':  t_ekf,
                'tunnel_err_base': t_base,
                'road_err_ekf':    r_ekf,
                'improvement_pct': impr,
                'cur_q':           q_scale,
                'cur_r':           r_scale,
            }, step=step)

        if done:
            log.info('\nEpisode ended by environment.')
            break

    # ── Final summary ──────────────────────────────────────────────────────────
    print('\n')
    print('='*65)
    print(f'{_BOLD}  DEMO COMPLETE{_RESET}')
    print('='*65)
    print(f'\n  Steps run          : {step}')

    if tun_errs_ekf and tun_errs_base:
        t_ekf_f  = np.mean(tun_errs_ekf)
        t_base_f = np.mean(tun_errs_base)
        impr_f   = (t_base_f - t_ekf_f) / t_base_f * 100
        print(f'\n  {_BOLD}Tunnel (GPS denied):{_RESET}')
        print(f'    LSTM-EKF   : {_BLUE}{t_ekf_f:.2f} m{_RESET}')
        print(f'    Baseline   : {_RED}{t_base_f:.2f} m{_RESET}')
        print(f'    Improvement: {_GREEN}{impr_f:.1f}%{_RESET}')
    else:
        print(f'\n  {_YELLOW}No tunnel traversal recorded in this run.{_RESET}')
        print(f'  Try increasing --steps or changing --spawn index.')

    if road_errs_ekf:
        print(f'\n  Open road error : {np.mean(road_errs_ekf):.2f} m '
              f'(GNSS dominates — both EKFs similar)')

    # ── Save outputs ───────────────────────────────────────────────────────────
    png_path = os.path.join(RESULTS_DIR, 'demo_final.png')
    dashboard.update({
        **buf,
        'tunnel_err_ekf':  float(np.mean(tun_errs_ekf))  if tun_errs_ekf  else float('nan'),
        'tunnel_err_base': float(np.mean(tun_errs_base)) if tun_errs_base else float('nan'),
        'road_err_ekf':    float(np.mean(road_errs_ekf)) if road_errs_ekf else float('nan'),
        'improvement_pct': impr_f if tun_errs_ekf and tun_errs_base else float('nan'),
        'cur_q': q_scale, 'cur_r': r_scale,
    }, step=step)
    dashboard.save(png_path)
    csv_f.close()

    print(f'\n  {_GREEN}Saved:{_RESET}')
    print(f'    {png_path}')
    print(f'    {csv_path}')
    print(f'\n  Window stays open. Close it manually when done.')
    print('='*65 + '\n')

    # Keep window open until user closes it
    plt.ioff()
    try:
        plt.show(block=True)
    except Exception:
        pass

    env.close()
    dashboard.close()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    args = parse_args()
    try:
        run_demo(args)
    except RuntimeError as e:
        log.error(f'\n{_RED}Error: {e}{_RESET}')
        log.error('Make sure CarlaUE4.exe is running before starting demo.py')
        sys.exit(1)
    except KeyboardInterrupt:
        log.info('\nDemo stopped.')
