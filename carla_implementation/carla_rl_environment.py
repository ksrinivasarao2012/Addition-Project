# =============================================================================
# carla_rl_environment.py  —  v4  (two targeted changes from v3)
#
# CHANGE 1 — LSTM applied as ADDITIVE BIAS CORRECTION (not replacement)
#   v3: a_fwd = lstm_output
#   v4: a_fwd = ax_corr + lstm_output   (bias_fwd predicted by train_lstm v4)
#
# CHANGE 2 — OBS_DIM 8 → 10  (two new observation dimensions)
#   dim 9 (index 8): lstm_disagreement
#       = |lstm_bias_fwd| / 5.0  (normalised)
#       Tells PPO how large a correction the LSTM is currently making.
#       Large disagreement = IMU has significant bias = agent should adjust Q.
#   dim 10 (index 9): lstm_ready
#       = 1.0 if LSTMBridge buffer is full, 0.0 otherwise
#       Tells PPO whether LSTM is actively correcting or still warming up.
#       Without this, agent cannot distinguish "GPS denied + LSTM active"
#       from "GPS denied + LSTM still building context window".
#
# ekf.py signatures (v4):
#   reset()
#   predict(u)                       → u = {'accel':[ax,ay], 'gyro':omega}
#   update_gps(z)                    → z = np.array([x, y])
#   set_noise_scales(Q_scale, R_scale)
#   get_state() → dict with keys:
#       'position', 'theta', 'velocity', 'biases',
#       'covariance', 'position_uncertainty', 'innovation',
#       'Q_scale', 'R_scale'
#
# rl_agent.py signatures (v4):
#   select_action(obs) → (action, value, log_prob)
#   store_transition(obs, action, reward, value, log_prob, done)
#   update(next_obs)   → stats dict
#   obs_dim = 10  ← updated from 8 (requires retraining PPO)
# =============================================================================

import numpy as np
import math
import logging
from typing import Tuple, Optional

from carla_sensor_bridge import CARLASensorBridge, SensorBundle
from carla_config import (
    FIXED_DELTA_SECONDS, RANDOMIZE_SPAWN, MAX_STEPS,
    G, LSTM_MODEL_PATH, LSTM_STATS_PATH,
)
from carla_config import *

log = logging.getLogger("CARLAEnv")


def _correct_imu_for_gravity(ax_raw: float, ay_raw: float,
                              pitch_deg: float, roll_deg: float):
    """
    Remove gravitational component from raw IMU.
    Matches collect_data.py correct_imu_for_gravity() exactly.
    """
    pitch_rad = math.radians(pitch_deg)
    roll_rad  = math.radians(roll_deg)
    ax_corr   = ax_raw + G * math.sin(pitch_rad)
    ay_corr   = ay_raw - G * math.sin(roll_rad) * math.cos(pitch_rad)
    return ax_corr, ay_corr


class CARLALocalizationEnv:
    """
    RL environment wrapping CARLA + AdaptiveEKF.

    v4 changes:
      - OBS_DIM = 10  (was 8)
      - LSTM applied as additive bias: a_fwd = ax_corr + lstm_bias
      - Two new observation dims: lstm_disagreement, lstm_ready
    """

    # CHANGE 2: OBS_DIM 8 → 10
    OBS_DIM    = 10   # UPDATED — must match PPOAgent obs_dim=10
    ACTION_DIM = 2

    Q_SCALE_MIN = 0.1
    Q_SCALE_MAX = 3.0
    R_SCALE_MIN = 0.1
    R_SCALE_MAX = 3.0

    def __init__(self, ekf_instance, render: bool = True):
        self.ekf    = ekf_instance
        self.bridge = CARLASensorBridge()
        self.render = render

        self.q_scale:        float = 1.0
        self.r_scale:        float = 1.0
        self.step_count:     int   = 0
        self.time_since_gps: float = 0.0
        self.episode_count:  int   = 0

        self._prev_position_error: float = 0.0
        self._gps_denied_steps:    int   = 0
        self._total_steps:         int   = 0

        self.episode_errors:        list = []
        self.episode_tunnel_errors: list = []
        self.episode_q_scales:      list = []
        self.episode_r_scales:      list = []

        self._last_bundle: Optional[SensorBundle] = None

        # CHANGE 2 state: track last LSTM output for observation building
        self._last_lstm_bias_fwd: float = 0.0   # last bias prediction (m/s²)
        self._lstm_ready:         bool  = False  # is buffer full?

        # Load LSTM bridge (v4 bridge asserts output_is_bias=True in checkpoint)
        from ekf import LSTMBridge
        self._lstm_bridge = LSTMBridge(
            model_path=LSTM_MODEL_PATH,
            stats_path=LSTM_STATS_PATH,
        )
        if self._lstm_bridge.loaded():
            log.info("LSTM bridge v4 loaded — additive bias correction enabled")
        else:
            log.warning("LSTM bridge not loaded — EKF uses raw IMU only")

        log.info("Initializing CARLA environment...")
        if not self.bridge.connect():
            raise RuntimeError(
                "Could not connect to CARLA!\n"
                "Launch CarlaUE4.exe first, then run training.")
        log.info("CARLA environment ready")

    # ── RESET ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.episode_count += 1
        log.info(f"Episode {self.episode_count}: resetting...")

        success = self.bridge.reset_episode(randomize=RANDOMIZE_SPAWN)
        if not success:
            raise RuntimeError("Failed to reset CARLA episode.")

        bundle = self.bridge.get_sensor_bundle()
        if bundle is None:
            raise RuntimeError("No sensor data after reset.")

        self.ekf.initialize(
            x0       = 0.0,
            y0       = 0.0,
            heading0 = bundle.ground_truth.heading,
            speed0   = bundle.ground_truth.velocity,
            bias0    = 0.0,
        )

        self.q_scale              = 1.0
        self.r_scale              = 1.0
        self.step_count           = 0
        self.time_since_gps       = 0.0
        self._prev_position_error = 0.0
        self._gps_denied_steps    = 0
        self._last_bundle         = bundle

        # CHANGE 2: reset LSTM tracking state
        self._last_lstm_bias_fwd = 0.0
        self._lstm_ready         = False

        self.episode_errors.clear()
        self.episode_tunnel_errors.clear()
        self.episode_q_scales.clear()
        self.episode_r_scales.clear()

        self._lstm_bridge.reset()
        self.ekf.set_noise_scales(self.q_scale, self.r_scale)

        obs = self._build_observation(bundle)
        log.info(f"  Episode {self.episode_count} started | "
                 f"Spawn: ({bundle.ground_truth.x:.1f}, {bundle.ground_truth.y:.1f})")
        return obs

    # ── STEP ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.step_count   += 1
        self._total_steps += 1

        delta_q = float(np.clip(action[0], -0.5, 0.5))
        delta_r = float(np.clip(action[1], -0.5, 0.5))
        self.q_scale = float(np.clip(self.q_scale + delta_q, self.Q_SCALE_MIN, self.Q_SCALE_MAX))
        self.r_scale = float(np.clip(self.r_scale + delta_r, self.R_SCALE_MIN, self.R_SCALE_MAX))
        self.ekf.set_noise_scales(self.q_scale, self.r_scale)

        bundle = self.bridge.get_sensor_bundle()
        if bundle is None:
            obs = self._build_observation(self._last_bundle)
            return obs, -10.0, False, {"error": "no_sensor_data"}
        self._last_bundle = bundle

        # Gravity correction
        ax_corr, ay_corr = _correct_imu_for_gravity(
            bundle.imu.forward_accel, bundle.imu.accel_y,
            bundle.ground_truth.pitch_deg, bundle.ground_truth.roll_deg,
        )
        wz = bundle.imu.yaw_rate

        # Push to LSTM buffer every tick (open road too — keeps buffer warm)
        self._lstm_bridge.push(ax_corr, ay_corr, wz, self.ekf.get_speed(), int(bundle.gps_denied))
        self._lstm_ready = self._lstm_bridge.ready()

        # CHANGE 1: ADDITIVE BIAS CORRECTION
        # a_fwd = ax_corr (IMU baseline, always)
        # If GPS denied and LSTM ready, add the predicted bias correction
        a_fwd      = ax_corr    # default: raw corrected IMU
        lstm_active = False
        if bundle.gps_denied and self._lstm_bridge.ready():
            bias_fwd, _ = self._lstm_bridge.predict()
            if bias_fwd is not None:
                a_fwd = ax_corr + bias_fwd   # ADDITIVE — not replacement
                self._last_lstm_bias_fwd = bias_fwd
                lstm_active = True

        # EKF predict using corrected acceleration
        self.ekf.predict(u={
            'accel': np.array([a_fwd, ay_corr]),
            'gyro':  wz,
        })

        # EKF update when GPS available
        if bundle.gnss is not None and not bundle.gps_denied:
            self.ekf.update_gps(z=np.array([bundle.gnss.local_x, bundle.gnss.local_y]))
            self.time_since_gps = 0.0
        else:
            self.time_since_gps += FIXED_DELTA_SECONDS
            self._gps_denied_steps += 1

        state   = self.ekf.get_state()
        ekf_pos = state['position']
        ekf_x   = float(ekf_pos[0]); ekf_y = float(ekf_pos[1])
        gt      = bundle.ground_truth
        position_error = math.sqrt((ekf_x - gt.x)**2 + (ekf_y - gt.y)**2)

        reward = self._compute_reward(position_error, bundle.gps_denied, delta_q, delta_r)

        self.episode_errors.append(position_error)
        self.episode_q_scales.append(self.q_scale)
        self.episode_r_scales.append(self.r_scale)
        if bundle.gps_denied:
            self.episode_tunnel_errors.append(position_error)
        self._prev_position_error = position_error

        done = self._check_done(position_error, bundle)
        obs  = self._build_observation(bundle, ekf_state=state)

        info = {
            "position_error": position_error,
            "gps_denied":     bundle.gps_denied,
            "in_tunnel":      gt.in_tunnel,
            "q_scale":        self.q_scale,
            "r_scale":        self.r_scale,
            "time_since_gps": self.time_since_gps,
            "step":           self.step_count,
            "gt_x":           gt.x,
            "gt_y":           gt.y,
            "ekf_x":          ekf_x,
            "ekf_y":          ekf_y,
            "lstm_active":    lstm_active,
        }
        return obs, reward, done, info

    # ── OBSERVATION  (10-dim) ─────────────────────────────────────────────────

    def _build_observation(self, bundle: SensorBundle,
                           ekf_state: dict = None) -> np.ndarray:
        """
        10-dim observation vector:

          [0]  innovation_x          normalised to [-1, 1]
          [1]  innovation_y          normalised to [-1, 1]
          [2]  position_uncertainty  normalised to [0, 1]
          [3]  time_since_gps        normalised to [0, 1]
          [4]  q_scale               normalised to [0, 1]
          [5]  r_scale               normalised to [0, 1]
          [6]  gps_denied flag       0 or 1
          [7]  vehicle speed         normalised to [0, 1]
          [8]  lstm_disagreement     |lstm_bias_fwd| / 5.0, clipped [0, 1]
               Tells agent how large a correction LSTM is making.
               Large = IMU has significant bias → agent may want to adjust Q.
          [9]  lstm_ready            1.0 if buffer full (LSTM active), else 0.0
               Tells agent whether LSTM is actually correcting or still warming up.
        """
        if ekf_state is None:
            ekf_state = self.ekf.get_state()

        innovation  = ekf_state.get('innovation', np.zeros(2))
        innov_x     = float(innovation[0]) if len(innovation) > 0 else 0.0
        innov_y     = float(innovation[1]) if len(innovation) > 1 else 0.0
        uncertainty = float(ekf_state.get('position_uncertainty', 0.0))
        gt_velocity = bundle.ground_truth.velocity if bundle else 0.0
        gps_denied  = float(bundle.gps_denied) if bundle else 0.0

        # CHANGE 2: new dims 8 and 9
        # Disagreement = how large is the LSTM correction currently?
        # 5.0 m/s² is a conservative normalisation scale for IMU bias.
        lstm_disagreement = abs(self._last_lstm_bias_fwd) / 5.0
        lstm_ready        = 1.0 if self._lstm_ready else 0.0

        obs = np.array([
            np.clip(innov_x / 10.0,     -1.0, 1.0),   # [0]
            np.clip(innov_y / 10.0,     -1.0, 1.0),   # [1]
            np.clip(uncertainty / 20.0,  0.0, 1.0),   # [2]
            np.clip(self.time_since_gps / 30.0, 0.0, 1.0),  # [3]
            (self.q_scale - self.Q_SCALE_MIN) / (self.Q_SCALE_MAX - self.Q_SCALE_MIN),  # [4]
            (self.r_scale - self.R_SCALE_MIN) / (self.R_SCALE_MAX - self.R_SCALE_MIN),  # [5]
            gps_denied,                                # [6]
            np.clip(gt_velocity / 30.0,  0.0, 1.0),  # [7]
            np.clip(lstm_disagreement,   0.0, 1.0),  # [8] NEW
            lstm_ready,                                # [9] NEW
        ], dtype=np.float32)

        return obs

    # ── REWARD  (unchanged from v3) ───────────────────────────────────────────

    def _compute_reward(self, position_error: float, in_tunnel: bool,
                        delta_q: float, delta_r: float) -> float:
        normalized  = (5.0 - position_error) / 5.0
        normalized  = float(np.clip(normalized, -2.0, 1.0))
        if in_tunnel:
            normalized *= 1.5
        improvement = (self._prev_position_error - position_error) / 5.0
        normalized += 0.3 * float(np.clip(improvement, -1.0, 1.0))
        normalized -= 0.05 * (abs(delta_q) + abs(delta_r))
        return float(np.clip(normalized, -3.0, 2.0))

    # ── TERMINATION  (unchanged) ──────────────────────────────────────────────

    def _check_done(self, position_error: float, bundle: SensorBundle) -> bool:
        if self.step_count >= MAX_STEPS:        return True
        if position_error > 100.0:              return True
        if self._vehicle_is_stuck(bundle):      return True
        return False

    def _vehicle_is_stuck(self, bundle: SensorBundle) -> bool:
        return bundle.ground_truth.velocity < 0.5 and self.step_count > 50

    # ── SUMMARY  (unchanged) ─────────────────────────────────────────────────

    def get_episode_summary(self) -> dict:
        if not self.episode_errors: return {}
        errors = np.array(self.episode_errors)
        t_errs = np.array(self.episode_tunnel_errors) if self.episode_tunnel_errors else np.array([0.0])
        return {
            "episode":           self.episode_count,
            "steps":             self.step_count,
            "mean_error":        float(np.mean(errors)),
            "max_error":         float(np.max(errors)),
            "final_error":       float(errors[-1]),
            "tunnel_mean_error": float(np.mean(t_errs)),
            "tunnel_steps":      self._gps_denied_steps,
            "mean_q_scale":      float(np.mean(self.episode_q_scales)),
            "mean_r_scale":      float(np.mean(self.episode_r_scales)),
        }

    def close(self):
        self.bridge.destroy()
        log.info("CARLA environment closed.")

    def __enter__(self):  return self
    def __exit__(self, *args): self.close()
