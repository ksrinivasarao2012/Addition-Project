r"""
rl_train.py  —  RL Adaptive Filter Tuning  (Main Training Script)
==================================================================
For: LSTM + RL-Adaptive EKF Localization in CARLA Town04

Architecture (follows project diagram exactly):
    CARLA Simulation
        ↓
    Sensor Data Collection  (carla_sensor_bridge.py)
        ↓              ↓
    IMU Data        GPS Ground Truth
        ↓              ↓
    Kalman Filter   LSTM Drift Compensation
    Localization    (ekf.py → LSTMBridge)
        ↓              ↓
        KF-LSTM Fusion Module  (AdaptiveEKF in ekf.py)
              ↓
    RL Adaptive Filter Tuning  ← THIS FILE drives this block
    (PPOAgent learns Q/R scales)
              ↓
    Vehicle Control Commands  (CARLA autopilot, feedback loop)

What this script does:
    1. Connects to running CARLA server
    2. Creates AdaptiveEKF (with LSTM bridge loaded if available)
    3. Creates PPOAgent (obs=8, action=2)
    4. Runs NUM_EPISODES episodes:
          - Vehicle drives autonomously in Town04
          - Every 0.05s: IMU → EKF.predict(), GPS → EKF.update()
          - RL agent observes EKF state, outputs delta_Q / delta_R
          - Reward = f(position_error, tunnel_penalty, smoothness)
          - PPO updates policy after each episode
    5. Saves best model, training logs, live dashboard

Usage:
    # Terminal 1 — start CARLA first
    cd C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor
    CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

    # Terminal 2 — run training
    cd C:\Users\heman\Music\rl_imu_project
    carla_env37\Scripts\activate
    python rl_train.py
    python rl_train.py --episodes 200          # more episodes
    python rl_train.py --resume models/best_carla_model.pth  # resume
    python rl_train.py --no-render             # headless (faster)

Outputs:
    models/best_carla_model.pth    best model by mean position error
    models/latest_carla_model.pth  model at end of training
    models/checkpoint_epN.pth      periodic checkpoints
    logs/rl_training_log.csv       per-episode metrics
    results/training_final.png     6-panel training dashboard
"""

import sys
import os
import argparse
import time
import csv
import logging
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# PATH SETUP
# All project files live in rl_imu_project root.
# carla_implementation files are in the subdirectory.
# =============================================================================
import os

# The 'r' prefix tells Python to treat backslashes as literal characters
ROOT_DIR  = r'C:\Users\heman\Music\rl_imu_project'
CARLA_DIR = os.path.join(ROOT_DIR, 'carla_implementation')

sys.path.insert(0, ROOT_DIR)    # finds ekf.py, rl_agent.py
sys.path.insert(0, CARLA_DIR)   # finds carla_sensor_bridge.py, carla_config.py

from ekf import AdaptiveEKF
from rl_agent import PPOAgent
from carla_rl_environment import CARLALocalizationEnv
from carla_config import (
    NUM_EPISODES, MAX_STEPS, WARMUP_EPISODES,
    FIXED_DELTA_SECONDS, VEHICLE_BLUEPRINT, TARGET_SPEED,
    CARLA_TOWN, MODEL_DIR, RESULTS_DIR, LOG_DIR,
    BEST_MODEL_PATH, LATEST_MODEL_PATH, TRAINING_LOG_PATH,
    PLOT_UPDATE_INTERVAL, SAVE_INTERVAL,
)

# Optional LSTM bridge — loaded if model exists, skipped gracefully if not
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'lstm_drift_predictor.pth')
LSTM_STATS_PATH = os.path.join(ROOT_DIR, 'models', 'lstm_normalisation.npz')

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RLTrain")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="RL Adaptive EKF Training in CARLA Town04")
    p.add_argument("--episodes",  type=int, default=NUM_EPISODES,
                   help="Total training episodes (default: from carla_config)")
    p.add_argument("--no-render", action="store_true",
                   help="Headless mode — no CARLA window (faster training)")
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to checkpoint .pth to resume training from")
    p.add_argument("--no-lstm",   action="store_true",
                   help="Disable LSTM drift compensation (baseline mode)")
    return p.parse_args()


# =============================================================================
# LIVE TRAINING DASHBOARD  (6 panels)
# =============================================================================

class Dashboard:
    """
    Live 6-panel training dashboard.
    Updates every PLOT_UPDATE_INTERVAL episodes.

    Panels:
      [0,0] Episode return      — should trend upward
      [0,1] Mean position error — should trend downward
      [1,0] Tunnel error        — key metric (GPS denied)
      [1,1] Q scale over time   — should spike in tunnels
      [2,0] R scale over time   — GPS trust adaptation
      [2,1] Latest trajectory   — EKF vs Ground Truth (2D)
    """

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(
            "RL Adaptive EKF — Live Training Dashboard",
            fontsize=13, fontweight='bold')
        gs = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.35)

        self.ax = {
            'ret':  self.fig.add_subplot(gs[0, 0]),
            'err':  self.fig.add_subplot(gs[0, 1]),
            'tun':  self.fig.add_subplot(gs[1, 0]),
            'q':    self.fig.add_subplot(gs[1, 1]),
            'r':    self.fig.add_subplot(gs[2, 0]),
            'traj': self.fig.add_subplot(gs[2, 1]),
        }
        for ax in self.ax.values():
            ax.grid(True, alpha=0.3)
        plt.show(block=False)

    def update(self, history: dict, traj: dict, episode: int, total: int):
        eps = history['episodes']
        if len(eps) < 2:
            return

        def smooth(arr, w=7):
            if len(arr) < w:
                return np.array(arr)
            return np.convolve(arr, np.ones(w) / w, mode='valid')

        configs = [
            ('ret',  history['returns'],       'Returns',             'blue',   None),
            ('err',  history['mean_errors'],   'Mean Error (m)',      'red',    None),
            ('tun',  history['tunnel_errors'], 'Tunnel Error (m)',    'orange', None),
            ('q',    history['q_scales'],      'Q Scale',             'green',  1.0),
            ('r',    history['r_scales'],      'R Scale',             'purple', 1.0),
        ]

        for key, data, title, color, hline in configs:
            ax = self.ax[key]
            ax.cla(); ax.grid(True, alpha=0.3)
            arr = np.array(data)
            ax.plot(eps, arr, alpha=0.25, color=color, lw=1)
            if len(arr) >= 7:
                ax.plot(eps[6:], smooth(arr), color=color, lw=2,
                        label=f'{arr[-1]:.2f}')
                ax.legend(fontsize=8, loc='upper right')
            if hline:
                ax.axhline(hline, color='gray', ls='--', alpha=0.5, lw=1)
            ax.set_title(f"{title}  [Ep {episode}/{total}]",
                         fontsize=9, fontweight='bold')
            ax.set_xlabel("Episode", fontsize=8)

        # Trajectory panel
        ax = self.ax['traj']
        ax.cla(); ax.grid(True, alpha=0.3)
        ax.set_title("Latest Trajectory (EKF vs GT)", fontsize=9,
                     fontweight='bold')
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

        if traj and len(traj.get('gt_x', [])) > 1:
            gt_x  = np.array(traj['gt_x'])
            gt_y  = np.array(traj['gt_y'])
            ek_x  = np.array(traj['ekf_x'])
            ek_y  = np.array(traj['ekf_y'])
            gps_d = np.array(traj['gps_denied'], dtype=bool)

            ax.plot(gt_x, gt_y,  'g-',  lw=2,   label='Ground Truth', zorder=3)
            ax.plot(ek_x, ek_y,  'b--', lw=1.5, label='EKF Estimate',  zorder=2)
            if gps_d.any():
                ax.scatter(gt_x[gps_d], gt_y[gps_d],
                           c='red', s=4, alpha=0.5, zorder=4,
                           label='GPS Denied')
            ax.scatter([gt_x[0]],  [gt_y[0]],  c='lime',   s=80,
                       marker='o', zorder=5, label='Start')
            ax.scatter([gt_x[-1]], [gt_y[-1]], c='yellow', s=80,
                       marker='*', zorder=5, label='End')
            ax.legend(fontsize=7, loc='upper left')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        log.info(f"Dashboard saved → {path}")


# =============================================================================
# CONSOLE EPISODE BANNER
# =============================================================================

def print_banner(ep, total, summary, ep_return, duration, lstm_active):
    mean_err   = summary.get('mean_error', 0.0)
    tun_err    = summary.get('tunnel_mean_error', 0.0)
    steps      = summary.get('steps', 0)
    q          = summary.get('mean_q_scale', 1.0)
    r          = summary.get('mean_r_scale', 1.0)
    tun_steps  = summary.get('tunnel_steps', 0)
    tun_str    = f"🚇 {tun_err:.2f}m ({tun_steps}steps)" if tun_steps > 0 else "open road only"
    lstm_str   = "LSTM+EKF" if lstm_active else "EKF only"

    print(
        f"\n{'='*68}\n"
        f"  Ep {ep:>4}/{total}  |  {duration:.1f}s  |  {steps} steps  |  {lstm_str}\n"
        f"  Return : {ep_return:>8.2f}  (>0=good, 0=5m avg, <0=needs work)\n"
        f"  Error  : {mean_err:>6.2f}m overall  |  {tun_str}\n"
        f"  Q={q:.3f}  R={r:.3f}\n"
        f"{'='*68}"
    )


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(args):
    print("\n" + "="*68)
    print("  RL Adaptive EKF  —  Training")
    print("="*68)
    print(f"  Episodes    : {args.episodes}")
    print(f"  Steps/ep    : {MAX_STEPS}  ({MAX_STEPS * FIXED_DELTA_SECONDS:.0f}s)")
    print(f"  Vehicle     : {VEHICLE_BLUEPRINT} @ {TARGET_SPEED} km/h")
    print(f"  Map         : {CARLA_TOWN}")
    print(f"  LSTM        : {'DISABLED (--no-lstm)' if args.no_lstm else 'Enabled if model exists'}")
    print("="*68 + "\n")

    # ── Create output directories ─────────────────────────────────────────────
    for d in [MODEL_DIR, RESULTS_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Initialize AdaptiveEKF with optional LSTM bridge ──────────────────────
    # The AdaptiveEKF class (ekf.py) handles LSTM internally via LSTMBridge.
    # If LSTM model files exist and --no-lstm is not set, it loads automatically.
    ekf = AdaptiveEKF()
    log.info("✓ AdaptiveEKF initialised")

    # ── Initialize PPO Agent ──────────────────────────────────────────────────
    # obs_dim=8 must match CARLALocalizationEnv.OBS_DIM
    # action_dim=2: [delta_Q_scale, delta_R_scale]
    agent = PPOAgent(obs_dim=8, action_dim=2)

    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        log.info(f"✓ Resumed from checkpoint: {args.resume}")
    else:
        log.info("✓ PPOAgent initialised (fresh)")

    # ── Connect to CARLA and create RL environment ────────────────────────────
    log.info("Connecting to CARLA... (CarlaUE4.exe must be running)")
    env = CARLALocalizationEnv(
        ekf_instance=ekf,
        render=not args.no_render,
    )
    log.info("✓ CARLA environment ready")

    # ── Training state ────────────────────────────────────────────────────────
    dashboard = Dashboard()
    history   = {
        'episodes':      [],
        'returns':       [],
        'mean_errors':   [],
        'tunnel_errors': [],
        'q_scales':      [],
        'r_scales':      [],
    }
    latest_traj = {}
    best_error  = float('inf')

    # CSV log
    csv_f = open(TRAINING_LOG_PATH, 'w', newline='')
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        'episode', 'return', 'mean_error', 'tunnel_error',
        'q_scale', 'r_scale', 'steps', 'tunnel_steps',
        'duration_s', 'lstm_used',
    ])
    csv_w.writeheader()

    log.info(f"\nStarting {args.episodes} episodes...\n")

    # ==========================================================================
    # EPISODE LOOP
    # ==========================================================================
    for episode in range(1, args.episodes + 1):
        ep_start  = time.time()
        obs       = env.reset()
        ep_return = 0.0

        # Warmup: first WARMUP_EPISODES use zero action (EKF only, no RL)
        # This lets the EKF stabilise before the agent starts exploring
        use_rl = (episode > WARMUP_EPISODES)

        traj = {'gt_x': [], 'gt_y': [], 'ekf_x': [], 'ekf_y': [],
                'gps_denied': []}

        # ── STEP LOOP ─────────────────────────────────────────────────────────
        for step in range(MAX_STEPS):

            # Agent selects action
            if use_rl:
                # Returns (action, value, log_prob)
                # action = [delta_Q, delta_R] in [-0.5, 0.5]
                action, value, log_prob = agent.select_action(obs)
            else:
                # Warmup: no action → Q/R stay at 1.0
                action   = np.zeros(2, dtype=np.float32)
                value    = 0.0
                log_prob = 0.0

            # Environment step
            # carla_rl_environment.py:
            #   - applies delta_Q/delta_R to EKF noise scales
            #   - ticks CARLA world
            #   - runs EKF predict() + update_gps() as appropriate
            #   - computes position error vs ground truth
            #   - returns (next_obs, reward, done, info)
            next_obs, reward, done, info = env.step(action)

            # Store in PPO buffer
            agent.store_transition(obs, action, reward, value, log_prob, done)

            # Record trajectory for dashboard
            traj['gt_x'].append(info['gt_x'])
            traj['gt_y'].append(info['gt_y'])
            traj['ekf_x'].append(info['ekf_x'])
            traj['ekf_y'].append(info['ekf_y'])
            traj['gps_denied'].append(int(info['gps_denied']))

            ep_return += reward
            obs        = next_obs

            # Step-level console output every 100 steps
            if step % 100 == 0 and step > 0:
                log.info(
                    f"  Step {step:>4} | "
                    f"Error: {info['position_error']:>6.2f}m | "
                    f"GPS: {'DENIED' if info['gps_denied'] else 'OK    '} | "
                    f"Q:{info['q_scale']:.2f}  R:{info['r_scale']:.2f}"
                )

            if done:
                break

        # ── PPO UPDATE ────────────────────────────────────────────────────────
        # After each episode, update policy using stored transitions.
        # agent.update() computes GAE advantages, runs 4 PPO epochs,
        # clears buffer, returns training stats.
        if use_rl and len(env.episode_errors) > 5:
            ppo_stats = agent.update(next_obs=obs)
            if episode % 10 == 0:
                log.info(
                    f"  PPO | policy_loss={ppo_stats.get('policy_loss',0):.4f} "
                    f"value_loss={ppo_stats.get('value_loss',0):.4f} "
                    f"entropy={ppo_stats.get('entropy',0):.4f}"
                )

        # ── EPISODE SUMMARY ───────────────────────────────────────────────────
        summary    = env.get_episode_summary()
        duration   = time.time() - ep_start
        mean_error = summary.get('mean_error',        0.0)
        tun_error  = summary.get('tunnel_mean_error', 0.0)
        mean_q     = summary.get('mean_q_scale',      1.0)
        mean_r     = summary.get('mean_r_scale',      1.0)

        print_banner(episode, args.episodes, summary,
                     ep_return, duration, use_rl)

        history['episodes'].append(episode)
        history['returns'].append(ep_return)
        history['mean_errors'].append(mean_error)
        history['tunnel_errors'].append(tun_error)
        history['q_scales'].append(mean_q)
        history['r_scales'].append(mean_r)
        latest_traj = traj

        csv_w.writerow({
            'episode':      episode,
            'return':       round(ep_return, 3),
            'mean_error':   round(mean_error, 4),
            'tunnel_error': round(tun_error, 4),
            'q_scale':      round(mean_q, 4),
            'r_scale':      round(mean_r, 4),
            'steps':        summary.get('steps', 0),
            'tunnel_steps': summary.get('tunnel_steps', 0),
            'duration_s':   round(duration, 2),
            'lstm_used':    not args.no_lstm,
        })
        csv_f.flush()

        # ── UPDATE DASHBOARD ──────────────────────────────────────────────────
        if episode % PLOT_UPDATE_INTERVAL == 0:
            dashboard.update(history, latest_traj, episode, args.episodes)

        # ── SAVE BEST MODEL ───────────────────────────────────────────────────
        if use_rl and 0 < mean_error < best_error:
            best_error = mean_error
            agent.save(BEST_MODEL_PATH)
            log.info(f"  ★  Best model saved  (error: {best_error:.2f}m)")

        # ── PERIODIC CHECKPOINT ───────────────────────────────────────────────
        if episode % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(MODEL_DIR, f'checkpoint_ep{episode}.pth')
            agent.save(ckpt_path)
            dashboard.save(os.path.join(RESULTS_DIR,
                                        f'dashboard_ep{episode}.png'))

    # ==========================================================================
    # TRAINING COMPLETE
    # ==========================================================================
    csv_f.close()
    agent.save(LATEST_MODEL_PATH)

    # Final dashboard save
    dashboard.update(history, latest_traj, args.episodes, args.episodes)
    dashboard.save(os.path.join(RESULTS_DIR, 'training_final.png'))

    print("\n" + "="*68)
    print("  ✓  TRAINING COMPLETE")
    print("="*68)
    print(f"  Best mean error : {best_error:.2f}m")
    print(f"  Best model      : {BEST_MODEL_PATH}")
    print(f"  Latest model    : {LATEST_MODEL_PATH}")
    print(f"  Training log    : {TRAINING_LOG_PATH}")
    print(f"  Dashboard       : {RESULTS_DIR}training_final.png")
    print("="*68)
    print("\n  Next: python evaluate_carla.py\n")

    env.close()
    plt.ioff()
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    args = parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        log.info("\nTraining stopped by user. Best model already saved.")
    except RuntimeError as e:
        log.error(f"\n✗ {e}")
        log.error("  Make sure CarlaUE4.exe is running before starting training.")
        sys.exit(1)
