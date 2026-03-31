# =============================================================================
# train_carla.py  —  CARLA + RL-Adaptive EKF Training
#
# Improvements in this version:
#   - Normalized reward (returns in [-500, +500] range, readable)
#   - Slower vehicle speed (30 km/h, realistic and visible)
#   - Better live dashboard with step-level EKF error panel
#   - Console shows clear per-episode summary
#   - Spectator camera follows vehicle smoothly
# =============================================================================

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
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from ekf import AdaptiveEKF
from rl_agent import PPOAgent
from carla_rl_environment import CARLALocalizationEnv
from carla_config import *

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("TrainCARLA")


# =============================================================================
# ARGS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int, default=NUM_EPISODES)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--resume",    type=str, default=None)
    return parser.parse_args()


# =============================================================================
# LIVE DASHBOARD  — 6 panels, updates every PLOT_UPDATE_INTERVAL episodes
# =============================================================================

class LiveDashboard:
    """
    6-panel training dashboard:
      [0,0] Episode return over time       — are we improving overall?
      [0,1] Mean position error over time  — core metric, should drop
      [1,0] Tunnel (GPS-denied) error      — key challenge metric
      [1,1] Q scale adaptation             — shows agent learning tunnel pattern
      [2,0] R scale adaptation             — GPS trust tuning
      [2,1] Latest trajectory              — EKF vs ground truth in 2D
    """

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.patch.set_facecolor('#1a1a2e')
        self.fig.suptitle(
            "CARLA + RL-Adaptive EKF  |  Live Training Dashboard",
            fontsize=14, fontweight="bold", color="white"
        )
        gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35,
                               top=0.93, bottom=0.06)

        axes = [self.fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
        self.ax_return, self.ax_error, self.ax_tunnel, \
        self.ax_q, self.ax_r, self.ax_traj = axes

        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white', labelsize=8)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            ax.grid(True, alpha=0.2, color='white')

        plt.show(block=False)
        self._episode_text = None

    def update(self, history: dict, latest_traj: dict, current_ep: int):
        eps = history["episodes"]
        if len(eps) < 2:
            return

        def smooth(arr, w=7):
            if len(arr) < w:
                return np.array(arr)
            return np.convolve(arr, np.ones(w) / w, mode="valid")

        def splot(ax, eps, raw, color, title, ylabel, hline=None):
            ax.cla()
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, alpha=0.2, color='white')
            ax.set_title(title, color='white', fontsize=9, fontweight='bold')
            ax.set_xlabel("Episode", color='white', fontsize=8)
            ax.set_ylabel(ylabel, color='white', fontsize=8)
            raw = np.array(raw)
            ax.plot(eps, raw, alpha=0.25, color=color, lw=1)
            if len(raw) >= 7:
                s = smooth(raw)
                ax.plot(eps[6:], s, color=color, lw=2.5,
                        label=f"Smoothed: {s[-1]:.2f}")
                ax.legend(fontsize=7, facecolor='#1a1a2e',
                          labelcolor='white', framealpha=0.8)
            if hline is not None:
                ax.axhline(y=hline, color='gray', linestyle='--',
                           alpha=0.6, lw=1)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

        returns       = history["returns"]
        mean_errors   = history["mean_errors"]
        tunnel_errors = history["tunnel_errors"]
        q_scales      = history["q_scales"]
        r_scales      = history["r_scales"]

        splot(self.ax_return, eps, returns,
              '#00d4ff', f"Episode Return  (Ep {current_ep}/{NUM_EPISODES})",
              "Return", hline=0)
        splot(self.ax_error, eps, mean_errors,
              '#ff6b6b', "Mean Position Error (m)", "Error (m)")
        splot(self.ax_tunnel, eps, tunnel_errors,
              '#ffa500', "Tunnel (GPS-Denied) Error (m)", "Error (m)")
        splot(self.ax_q, eps, q_scales,
              '#51cf66', "Q Scale  (↑ in tunnel = smart!)", "Q Scale", hline=1.0)
        splot(self.ax_r, eps, r_scales,
              '#cc5de8', "R Scale  (GPS trust)", "R Scale", hline=1.0)

        # Trajectory panel
        self.ax_traj.cla()
        self.ax_traj.set_facecolor('#16213e')
        self.ax_traj.tick_params(colors='white', labelsize=8)
        self.ax_traj.grid(True, alpha=0.2, color='white')
        self.ax_traj.set_title("Latest Episode Trajectory",
                               color='white', fontsize=9, fontweight='bold')
        self.ax_traj.set_xlabel("X (m)", color='white', fontsize=8)
        self.ax_traj.set_ylabel("Y (m)", color='white', fontsize=8)
        for spine in self.ax_traj.spines.values():
            spine.set_edgecolor('#444')

        if latest_traj:
            gt_x  = np.array(latest_traj.get("gt_x",  []))
            gt_y  = np.array(latest_traj.get("gt_y",  []))
            ekf_x = np.array(latest_traj.get("ekf_x", []))
            ekf_y = np.array(latest_traj.get("ekf_y", []))
            gps_d = np.array(latest_traj.get("gps_denied", []), dtype=bool)

            if len(gt_x) > 1:
                self.ax_traj.plot(gt_x, gt_y, color='#51cf66',
                                  lw=2, label="Ground Truth", zorder=3)
                self.ax_traj.plot(ekf_x, ekf_y, color='#00d4ff',
                                  lw=1.5, linestyle='--',
                                  label="EKF Estimate", zorder=2)
                if gps_d.any():
                    self.ax_traj.scatter(gt_x[gps_d], gt_y[gps_d],
                                         color='#ff6b6b', s=6, zorder=4,
                                         label="GPS Denied")
                # Mark start and end
                self.ax_traj.scatter([gt_x[0]], [gt_y[0]],
                                     color='lime', s=80, marker='o',
                                     zorder=5, label="Start")
                self.ax_traj.scatter([gt_x[-1]], [gt_y[-1]],
                                     color='yellow', s=80, marker='*',
                                     zorder=5, label="End")
                self.ax_traj.legend(fontsize=7, facecolor='#1a1a2e',
                                    labelcolor='white', framealpha=0.8,
                                    loc='upper left')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def save(self, path: str):
        self.fig.savefig(path, dpi=150, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        log.info(f"Dashboard saved: {path}")


# =============================================================================
# CONSOLE PROGRESS PRINTER
# =============================================================================

def print_episode_banner(ep, total, summary, ep_return, duration):
    """Print a clear per-episode summary to console."""
    mean_err   = summary.get("mean_error", 0.0)
    tunnel_err = summary.get("tunnel_mean_error", 0.0)
    steps      = summary.get("steps", 0)
    q          = summary.get("mean_q_scale", 1.0)
    r          = summary.get("mean_r_scale", 1.0)
    t_steps    = summary.get("tunnel_steps", 0)

    tunnel_str = f"🚇 {tunnel_err:.2f}m ({t_steps}steps)" if t_steps > 0 else "no tunnel"

    print(
        f"\n{'='*65}\n"
        f"  Episode {ep:>4} / {total}  |  {duration:.1f}s  |  {steps} steps\n"
        f"  Return:  {ep_return:>8.2f}   "
        f"(>0 = good, 0 = 5m avg error, <0 = needs work)\n"
        f"  Error:   {mean_err:>6.2f}m overall  |  {tunnel_str}\n"
        f"  Q scale: {q:.3f}  |  R scale: {r:.3f}\n"
        f"{'='*65}"
    )


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(args):
    print("\n" + "="*65)
    print("  CARLA + RL-Adaptive EKF  |  Training Starting")
    print("="*65)
    print(f"  Episodes:    {args.episodes}")
    print(f"  Steps/ep:    {MAX_STEPS}  ({MAX_STEPS * FIXED_DELTA_SECONDS:.0f}s per episode)")
    print(f"  Vehicle:     {VEHICLE_BLUEPRINT} @ {TARGET_SPEED} km/h")
    print(f"  Map:         {CARLA_TOWN}")
    print(f"  Reward:      normalized  (+1=great, 0=5m error, -1=10m error)")
    print("="*65 + "\n")

    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,     exist_ok=True)

    ekf   = AdaptiveEKF()
    agent = PPOAgent(obs_dim=8, action_dim=2)

    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        log.info(f"Resumed from: {args.resume}")

    log.info("Connecting to CARLA... (make sure CarlaUE4.exe is running)")
    env = CARLALocalizationEnv(ekf_instance=ekf, render=not args.no_render)

    dashboard   = LiveDashboard()
    history     = {"episodes": [], "returns": [], "mean_errors": [],
                   "tunnel_errors": [], "q_scales": [], "r_scales": []}
    latest_traj = {}
    best_error  = float("inf")

    csv_file   = open(TRAINING_LOG_PATH, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode", "return", "mean_error", "tunnel_error",
        "q_scale", "r_scale", "steps", "tunnel_steps", "duration_s"
    ])
    csv_writer.writeheader()

    log.info(f"\nStarting {args.episodes} episodes. Watch the dashboard!\n")

    for episode in range(1, args.episodes + 1):
        ep_start  = time.time()
        obs       = env.reset()
        ep_return = 0.0
        use_rl    = (episode > WARMUP_EPISODES)

        traj = {"gt_x": [], "gt_y": [], "ekf_x": [], "ekf_y": [],
                "gps_denied": []}

        # ── Episode rollout ──────────────────────────────────────────
        for step in range(MAX_STEPS):

            if use_rl:
                action, value, log_prob = agent.select_action(obs)
            else:
                action   = np.zeros(2, dtype=np.float32)
                value    = 0.0
                log_prob = 0.0

            next_obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, value, log_prob, done)

            traj["gt_x"].append(info["gt_x"])
            traj["gt_y"].append(info["gt_y"])
            traj["ekf_x"].append(info["ekf_x"])
            traj["ekf_y"].append(info["ekf_y"])
            traj["gps_denied"].append(int(info["gps_denied"]))

            ep_return += reward
            obs = next_obs

            # Print step-level info every 100 steps so you can see it's alive
            if step % 100 == 0 and step > 0:
                log.info(
                    f"  Step {step:>4} | "
                    f"Error: {info['position_error']:>6.2f}m | "
                    f"GPS: {'DENIED' if info['gps_denied'] else 'OK    '} | "
                    f"Q:{info['q_scale']:.2f} R:{info['r_scale']:.2f}"
                )

            if done:
                break

        # ── PPO Update ───────────────────────────────────────────────
        if use_rl and len(env.episode_errors) > 5:
            agent.update(next_obs=obs)

        # ── Summary ──────────────────────────────────────────────────
        summary     = env.get_episode_summary()
        duration    = time.time() - ep_start
        mean_error  = summary.get("mean_error", 0.0)
        tun_error   = summary.get("tunnel_mean_error", 0.0)
        mean_q      = summary.get("mean_q_scale", 1.0)
        mean_r      = summary.get("mean_r_scale", 1.0)

        print_episode_banner(episode, args.episodes, summary, ep_return, duration)

        history["episodes"].append(episode)
        history["returns"].append(ep_return)
        history["mean_errors"].append(mean_error)
        history["tunnel_errors"].append(tun_error)
        history["q_scales"].append(mean_q)
        history["r_scales"].append(mean_r)
        latest_traj = traj

        csv_writer.writerow({
            "episode":      episode,
            "return":       round(ep_return, 3),
            "mean_error":   round(mean_error, 4),
            "tunnel_error": round(tun_error, 4),
            "q_scale":      round(mean_q, 4),
            "r_scale":      round(mean_r, 4),
            "steps":        summary.get("steps", 0),
            "tunnel_steps": summary.get("tunnel_steps", 0),
            "duration_s":   round(duration, 2),
        })
        csv_file.flush()

        # Update dashboard
        if episode % PLOT_UPDATE_INTERVAL == 0:
            dashboard.update(history, latest_traj, episode)

        # Save best model
        if use_rl and 0 < mean_error < best_error:
            best_error = mean_error
            agent.save(BEST_MODEL_PATH)
            log.info(f"  ★ Best model saved  (error: {best_error:.2f}m)")

        # Periodic checkpoint
        if episode % SAVE_INTERVAL == 0:
            agent.save(f"{MODEL_DIR}checkpoint_ep{episode}.pth")
            dashboard.save(f"{RESULTS_DIR}dashboard_ep{episode}.png")

    # ── Done ─────────────────────────────────────────────────────────
    csv_file.close()
    agent.save(LATEST_MODEL_PATH)
    dashboard.update(history, latest_traj, args.episodes)
    dashboard.save(f"{RESULTS_DIR}training_final.png")

    print("\n" + "="*65)
    print("  ✓  TRAINING COMPLETE!")
    print(f"     Best mean error : {best_error:.2f}m")
    print(f"     Best model      : {BEST_MODEL_PATH}")
    print(f"     Training plot   : {RESULTS_DIR}training_final.png")
    print(f"     CSV log         : {TRAINING_LOG_PATH}")
    print("="*65)
    print("\n  Next step:  python evaluate_carla.py\n")

    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        log.info("\nTraining stopped by user. Progress saved.")
    except RuntimeError as e:
        log.error(f"\n✗ {e}")
        sys.exit(1)
