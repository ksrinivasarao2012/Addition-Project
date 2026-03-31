# =============================================================================
# evaluate_carla.py
# Evaluation script — RL-Adaptive EKF vs Static EKF in CARLA
#
# Runs both configurations through the same CARLA scenario and
# produces publication-quality comparison plots.
#
# Usage:
#   python evaluate_carla.py
#   python evaluate_carla.py --model models/best_carla_model.pth --episodes 10
# =============================================================================

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from ekf import AdaptiveEKF
from rl_agent import PPOAgent
from carla_rl_environment import CARLALocalizationEnv
from carla_config import *

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("Evaluate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default=BEST_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=5,
                        help="Evaluation episodes per configuration")
    parser.add_argument("--static-q", type=float, default=1.0,
                        help="Fixed Q scale for static EKF baseline")
    parser.add_argument("--static-r", type=float, default=1.0,
                        help="Fixed R scale for static EKF baseline")
    return parser.parse_args()


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def run_evaluation_episode(env: CARLALocalizationEnv,
                            agent: PPOAgent,
                            use_rl: bool,
                            static_q: float = 1.0,
                            static_r: float = 1.0) -> Dict:
    """
    Run one evaluation episode.

    Args:
        env:      CARLA environment
        agent:    Trained PPO agent
        use_rl:   If True, use agent. If False, use fixed Q/R (static baseline).
        static_q: Q scale for static baseline
        static_r: R scale for static baseline

    Returns:
        Dictionary with trajectory data and performance metrics.
    """
    obs = env.reset()

    # Storage
    gt_x, gt_y         = [], []
    ekf_x, ekf_y       = [], []
    errors              = []
    tunnel_errors       = []
    q_scales, r_scales  = [], []
    gps_denied_flags    = []

    total_reward = 0.0

    for step in range(MAX_STEPS):
        if use_rl:
            action, _, _ = agent.select_action(obs)
        else:
            # Static: force Q/R to fixed values by passing near-zero action
            # (environment clips and adds to current scale — we set scale each step)
            current_q = env.q_scale
            current_r = env.r_scale
            action = np.array([
                static_q - current_q,  # Drive Q toward static_q
                static_r - current_r,
            ], dtype=np.float32)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        gt_x.append(info["gt_x"])
        gt_y.append(info["gt_y"])
        ekf_x.append(info["ekf_x"])
        ekf_y.append(info["ekf_y"])
        errors.append(info["position_error"])
        q_scales.append(info["q_scale"])
        r_scales.append(info["r_scale"])
        gps_denied_flags.append(int(info["gps_denied"]))

        if info["gps_denied"]:
            tunnel_errors.append(info["position_error"])

        if done:
            break

    errors_arr = np.array(errors)
    tunnel_arr = np.array(tunnel_errors) if tunnel_errors else np.array([0.0])

    return {
        "gt_x":           np.array(gt_x),
        "gt_y":           np.array(gt_y),
        "ekf_x":          np.array(ekf_x),
        "ekf_y":          np.array(ekf_y),
        "errors":         errors_arr,
        "tunnel_errors":  tunnel_arr,
        "q_scales":       np.array(q_scales),
        "r_scales":       np.array(r_scales),
        "gps_denied":     np.array(gps_denied_flags, dtype=bool),
        "total_reward":   total_reward,
        "mean_error":     float(np.mean(errors_arr)),
        "max_error":      float(np.max(errors_arr)),
        "tunnel_error":   float(np.mean(tunnel_arr)),
        "steps":          len(errors),
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(rl_results: List[Dict],
                    static_results: List[Dict],
                    save_path: str):
    """
    Generate publication-quality 6-panel comparison figure.

    Panels:
      [0,0] 2D trajectory comparison (best episode)
      [0,1] Position error over time (average ± std)
      [1,0] Error CDF comparison
      [1,1] Q scale adaptation (RL only)
      [2,0] R scale adaptation (RL only)
      [2,1] Bar chart: key metrics comparison
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("CARLA Evaluation: RL-Adaptive EKF vs Static EKF",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    ax_traj  = fig.add_subplot(gs[0, 0])
    ax_err   = fig.add_subplot(gs[0, 1])
    ax_cdf   = fig.add_subplot(gs[1, 0])
    ax_q     = fig.add_subplot(gs[1, 1])
    ax_r     = fig.add_subplot(gs[2, 0])
    ax_bar   = fig.add_subplot(gs[2, 1])

    # ---- 1. Trajectory (best RL episode = lowest mean error) ----
    best_rl  = min(rl_results,     key=lambda r: r["mean_error"])
    best_st  = min(static_results, key=lambda r: r["mean_error"])

    ax_traj.plot(best_rl["gt_x"],  best_rl["gt_y"],  "g-",  lw=2.0, label="Ground Truth",    zorder=3)
    ax_traj.plot(best_rl["ekf_x"], best_rl["ekf_y"], "b--", lw=1.5, label="RL-Adaptive EKF", zorder=2)
    ax_traj.plot(best_st["ekf_x"], best_st["ekf_y"], "r:",  lw=1.5, label="Static EKF",       zorder=1)

    # Shade GPS-denied regions
    denied = best_rl["gps_denied"]
    if denied.any():
        gt_x_d = best_rl["gt_x"][denied]
        gt_y_d = best_rl["gt_y"][denied]
        ax_traj.scatter(gt_x_d, gt_y_d, color="red", s=4, alpha=0.5,
                        label="GPS Denied Zone", zorder=4)

    ax_traj.set_title("2D Trajectory Comparison")
    ax_traj.set_xlabel("X (m)"); ax_traj.set_ylabel("Y (m)")
    ax_traj.legend(fontsize=7, loc="upper left")
    ax_traj.grid(True, alpha=0.3)

    # ---- 2. Error over time ----
    max_len = max(max(len(r["errors"]) for r in rl_results),
                  max(len(r["errors"]) for r in static_results))

    def pad_array(arr, length):
        padded = np.full(length, arr[-1] if len(arr) > 0 else 0.0)
        padded[:len(arr)] = arr
        return padded

    rl_errs = np.array([pad_array(r["errors"], max_len) for r in rl_results])
    st_errs = np.array([pad_array(r["errors"], max_len) for r in static_results])

    steps = np.arange(max_len) * FIXED_DELTA_SECONDS

    ax_err.plot(steps, np.mean(rl_errs, axis=0), "b-",  lw=2, label="RL-Adaptive")
    ax_err.plot(steps, np.mean(st_errs, axis=0), "r--", lw=2, label="Static EKF")
    ax_err.fill_between(steps,
                         np.mean(rl_errs, axis=0) - np.std(rl_errs, axis=0),
                         np.mean(rl_errs, axis=0) + np.std(rl_errs, axis=0),
                         alpha=0.2, color="blue")

    # Mark GPS denial periods
    denied_time = steps[best_rl["gps_denied"][:max_len]]
    if len(denied_time) > 0:
        ax_err.axvspan(denied_time[0], denied_time[-1],
                       alpha=0.1, color="red", label="GPS Denied")

    ax_err.set_title("Position Error Over Time")
    ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("Error (m)")
    ax_err.legend(fontsize=8); ax_err.grid(True, alpha=0.3)

    # ---- 3. Error CDF ----
    all_rl = np.concatenate([r["errors"] for r in rl_results])
    all_st = np.concatenate([r["errors"] for r in static_results])

    for arr, color, label in [
        (all_rl, "blue",  "RL-Adaptive"),
        (all_st, "red",   "Static EKF"),
    ]:
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax_cdf.plot(sorted_arr, cdf, color=color, linewidth=2, label=label)

    ax_cdf.set_title("Error CDF (lower-left = better)")
    ax_cdf.set_xlabel("Position Error (m)"); ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.legend(fontsize=8); ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_xlim(left=0)

    # ---- 4. Q Scale Adaptation ----
    rl_q = np.array([pad_array(r["q_scales"], max_len) for r in rl_results])
    ax_q.plot(steps, np.mean(rl_q, axis=0), "g-", lw=2, label="Q Scale")
    ax_q.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (1.0)")

    # Shade GPS denial
    if len(denied_time) > 0:
        ax_q.axvspan(denied_time[0], denied_time[-1],
                     alpha=0.1, color="red", label="GPS Denied")

    ax_q.set_title("Q Scale Adaptation (RL Agent)")
    ax_q.set_xlabel("Time (s)"); ax_q.set_ylabel("Q Scale")
    ax_q.legend(fontsize=8); ax_q.grid(True, alpha=0.3)
    ax_q.annotate("Q increases\nin tunnel →", xy=(denied_time[0] if len(denied_time) > 0 else 0, 1.5),
                  fontsize=8, color="darkred",
                  arrowprops=dict(arrowstyle="->", color="darkred"))

    # ---- 5. R Scale Adaptation ----
    rl_r = np.array([pad_array(r["r_scales"], max_len) for r in rl_results])
    ax_r.plot(steps, np.mean(rl_r, axis=0), "purple", lw=2, label="R Scale")
    ax_r.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (1.0)")
    if len(denied_time) > 0:
        ax_r.axvspan(denied_time[0], denied_time[-1],
                     alpha=0.1, color="red", label="GPS Denied")

    ax_r.set_title("R Scale Adaptation (RL Agent)")
    ax_r.set_xlabel("Time (s)"); ax_r.set_ylabel("R Scale")
    ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)

    # ---- 6. Bar chart summary ----
    metrics = {
        "Mean\nError (m)":   (np.mean([r["mean_error"]   for r in rl_results]),
                               np.mean([r["mean_error"]   for r in static_results])),
        "Tunnel\nError (m)": (np.mean([r["tunnel_error"] for r in rl_results]),
                               np.mean([r["tunnel_error"] for r in static_results])),
        "Max\nError (m)":    (np.mean([r["max_error"]    for r in rl_results]),
                               np.mean([r["max_error"]    for r in static_results])),
    }

    x = np.arange(len(metrics))
    width = 0.35
    rl_vals = [v[0] for v in metrics.values()]
    st_vals = [v[1] for v in metrics.values()]

    bars1 = ax_bar.bar(x - width/2, rl_vals, width, label="RL-Adaptive",
                       color="steelblue", alpha=0.85)
    bars2 = ax_bar.bar(x + width/2, st_vals, width, label="Static EKF",
                       color="tomato", alpha=0.85)

    # Improvement percentage labels
    for i, (rl_v, st_v) in enumerate(zip(rl_vals, st_vals)):
        improvement = (st_v - rl_v) / st_v * 100
        ax_bar.text(x[i], max(rl_v, st_v) + 0.3,
                    f"-{improvement:.0f}%", ha="center",
                    fontsize=9, fontweight="bold", color="darkgreen")

    ax_bar.set_title("Performance Summary")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metrics.keys())
    ax_bar.legend(fontsize=8); ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.set_ylabel("Error (m)")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info(f"✓ Comparison plot saved: {save_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Load model
    if not os.path.exists(args.model):
        log.error(f"Model not found: {args.model}")
        log.error("Run train_carla.py first to train the model.")
        sys.exit(1)

    # Initialize
    ekf_rl     = AdaptiveEKF()
    ekf_static = AdaptiveEKF()

    agent = PPOAgent(
        obs_dim    = CARLALocalizationEnv.OBS_DIM,
        action_dim = CARLALocalizationEnv.ACTION_DIM,
    )
    agent.load(args.model)
    # Note: rl_agent.py has no set_eval_mode() method.
    # The policy network uses torch.no_grad() inside select_action when deterministic=True
    log.info(f"✓ Loaded model: {args.model}")

    # Connect to CARLA (single environment, run both configs sequentially)
    log.info("Connecting to CARLA...")
    env = CARLALocalizationEnv(ekf_instance=ekf_rl, render=True)

    # ------------------------------------------------------------------
    # Run RL evaluation
    # ------------------------------------------------------------------
    log.info(f"\n--- Evaluating RL-Adaptive EKF ({args.episodes} episodes) ---")
    rl_results = []
    for ep in range(args.episodes):
        result = run_evaluation_episode(env, agent, use_rl=True)
        rl_results.append(result)
        log.info(f"  RL Episode {ep+1}: error={result['mean_error']:.2f}m | "
                 f"tunnel={result['tunnel_error']:.2f}m | "
                 f"steps={result['steps']}")

    # ------------------------------------------------------------------
    # Run Static EKF evaluation
    # ------------------------------------------------------------------
    log.info(f"\n--- Evaluating Static EKF ({args.episodes} episodes) ---")
    env.ekf = ekf_static  # Swap EKF instance
    static_results = []
    for ep in range(args.episodes):
        result = run_evaluation_episode(env, agent, use_rl=False,
                                        static_q=args.static_q,
                                        static_r=args.static_r)
        static_results.append(result)
        log.info(f"  Static Episode {ep+1}: error={result['mean_error']:.2f}m | "
                 f"tunnel={result['tunnel_error']:.2f}m | "
                 f"steps={result['steps']}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<25} {'Static EKF':>15} {'RL-Adaptive':>15} {'Improvement':>12}")
    print("-" * 65)

    def fmt_improvement(rl_v, st_v):
        imp = (st_v - rl_v) / st_v * 100
        sign = "-" if imp > 0 else "+"
        return f"{sign}{abs(imp):.1f}%"

    metrics = [
        ("Mean Error (m)",
         np.mean([r["mean_error"]   for r in static_results]),
         np.mean([r["mean_error"]   for r in rl_results])),
        ("Tunnel Error (m)",
         np.mean([r["tunnel_error"] for r in static_results]),
         np.mean([r["tunnel_error"] for r in rl_results])),
        ("Max Error (m)",
         np.mean([r["max_error"]    for r in static_results]),
         np.mean([r["max_error"]    for r in rl_results])),
    ]

    for name, st_v, rl_v in metrics:
        print(f"{name:<25} {st_v:>15.3f} {rl_v:>15.3f} {fmt_improvement(rl_v, st_v):>12}")

    print("=" * 65)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = f"{RESULTS_DIR}carla_comparison.png"
    plot_comparison(rl_results, static_results, save_path)

    env.close()
    log.info("\n✓ Evaluation complete!")
    log.info(f"  Plot saved to: {save_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        log.error(f"Error: {e}")
        sys.exit(1)
