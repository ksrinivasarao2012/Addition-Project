# =============================================================================
# carla_rl_environment.py  —  fully matched to your ekf.py + rl_agent.py
#
# ekf.py signatures:
#   reset()                          → no args
#   predict(u)                       → u = {'accel':[ax,ay], 'gyro':omega}
#   update_gps(z)                    → z = np.array([x, y])
#   set_noise_scales(Q_scale, R_scale)
#   get_state() → dict with keys:
#       'position', 'theta', 'velocity', 'biases',
#       'covariance', 'position_uncertainty', 'innovation',
#       'Q_scale', 'R_scale'
#
# rl_agent.py signatures:
#   select_action(obs) → (action, value, log_prob)
#   store_transition(obs, action, reward, value, log_prob, done)
#   update(next_obs)   → stats dict
#   obs_dim = 8  ← agent expects 8-dim obs
# =============================================================================

import numpy as np
import math
import logging
from typing import Tuple, Optional

from carla_sensor_bridge import CARLASensorBridge, SensorBundle
from carla_config import *

log = logging.getLogger("CARLAEnv")


class CARLALocalizationEnv:
    """
    RL environment wrapping CARLA + EKF.
    Observation is 8-dim to match PPOAgent(obs_dim=8).
    """

    OBS_DIM    = 8   # Must match PPOAgent obs_dim=8
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

        log.info("Initializing CARLA environment...")
        if not self.bridge.connect():
            raise RuntimeError(
                "Could not connect to CARLA!\n"
                "Launch CarlaUE4.exe first, then run training."
            )
        log.info("✓ CARLA environment ready")

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset for new episode. Returns 8-dim observation."""
        self.episode_count += 1
        log.info(f"Episode {self.episode_count}: resetting...")

        success = self.bridge.reset_episode(randomize=RANDOMIZE_SPAWN)
        if not success:
            raise RuntimeError("Failed to reset CARLA episode.")

        bundle = self.bridge.get_sensor_bundle()
        if bundle is None:
            raise RuntimeError("No sensor data after reset.")

        # Initialise EKF at local (0,0) — GNSS also outputs (0,0) at spawn.
        # Use initialize() not direct index access — state layout changed in v2:
        #   OLD: x[2]=theta, x[3]=v
        #   NEW: x[2]=v,     x[3]=psi
        # Direct index access would silently swap heading and velocity.
        self.ekf.initialize(
            x0       = 0.0,
            y0       = 0.0,
            heading0 = bundle.ground_truth.heading,
            speed0   = bundle.ground_truth.velocity,
            bias0    = 0.0,
        )

        # Reset tracking
        self.q_scale              = 1.0
        self.r_scale              = 1.0
        self.step_count           = 0
        self.time_since_gps       = 0.0
        self._prev_position_error = 0.0
        self._gps_denied_steps    = 0
        self._last_bundle         = bundle

        self.episode_errors.clear()
        self.episode_tunnel_errors.clear()
        self.episode_q_scales.clear()
        self.episode_r_scales.clear()

        self.ekf.set_noise_scales(self.q_scale, self.r_scale)

        obs = self._build_observation(bundle)
        log.info(f"  Episode {self.episode_count} started | "
                 f"Spawn: ({bundle.ground_truth.x:.1f}, {bundle.ground_truth.y:.1f})")
        return obs

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step. Returns (obs, reward, done, info)."""
        self.step_count   += 1
        self._total_steps += 1

        # 1. Clip action to valid range
        delta_q = float(np.clip(action[0], -0.5, 0.5))
        delta_r = float(np.clip(action[1], -0.5, 0.5))

        self.q_scale = float(np.clip(self.q_scale + delta_q,
                                     self.Q_SCALE_MIN, self.Q_SCALE_MAX))
        self.r_scale = float(np.clip(self.r_scale + delta_r,
                                     self.R_SCALE_MIN, self.R_SCALE_MAX))

        self.ekf.set_noise_scales(self.q_scale, self.r_scale)

        # 2. Get sensor data from CARLA
        bundle = self.bridge.get_sensor_bundle()
        if bundle is None:
            obs = self._build_observation(self._last_bundle)
            return obs, -10.0, False, {"error": "no_sensor_data"}
        self._last_bundle = bundle

        # 3. EKF predict — u = {'accel': [ax, ay], 'gyro': omega}
        dt = FIXED_DELTA_SECONDS
        self.ekf.predict(u={
            'accel': np.array([bundle.imu.forward_accel,
                               bundle.imu.accel_y]),
            'gyro':  bundle.imu.yaw_rate,
        })

        # 4. EKF update — only when GPS available
        if bundle.gnss is not None and not bundle.gps_denied:
            self.ekf.update_gps(z=np.array([bundle.gnss.local_x,
                                            bundle.gnss.local_y]))
            self.time_since_gps = 0.0
        else:
            self.time_since_gps += dt
            self._gps_denied_steps += 1

        # 5. Compute position error
        # get_state() returns dict — position is at key 'position'
        state   = self.ekf.get_state()
        ekf_pos = state['position']      # np.array([x, y])
        ekf_x   = float(ekf_pos[0])
        ekf_y   = float(ekf_pos[1])
        gt      = bundle.ground_truth

        position_error = math.sqrt(
            (ekf_x - gt.x) ** 2 +
            (ekf_y - gt.y) ** 2
        )

        # 6. Reward
        reward = self._compute_reward(position_error, bundle.gps_denied,
                                      delta_q, delta_r)

        # 7. Log
        self.episode_errors.append(position_error)
        self.episode_q_scales.append(self.q_scale)
        self.episode_r_scales.append(self.r_scale)
        if bundle.gps_denied:
            self.episode_tunnel_errors.append(position_error)
        self._prev_position_error = position_error

        # 8. Done check
        done = self._check_done(position_error, bundle)

        # 9. Next observation
        obs = self._build_observation(bundle, ekf_state=state)

        # 10. Info
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
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # OBSERVATION  (8-dim to match PPOAgent obs_dim=8)
    # ------------------------------------------------------------------

    def _build_observation(self, bundle: SensorBundle,
                           ekf_state: dict = None) -> np.ndarray:
        """
        8-dim observation vector matching PPOAgent(obs_dim=8):
          [0] innovation_x          (from ekf state['innovation'][0])
          [1] innovation_y          (from ekf state['innovation'][1])
          [2] position_uncertainty  (from ekf state['position_uncertainty'])
          [3] time_since_gps        (normalized 0-1)
          [4] q_scale               (normalized 0-1)
          [5] r_scale               (normalized 0-1)
          [6] gps_denied flag       (0 or 1)
          [7] vehicle speed         (normalized 0-1, max 30 m/s)
        """
        if ekf_state is None:
            ekf_state = self.ekf.get_state()

        innovation  = ekf_state.get('innovation', np.zeros(2))
        innov_x     = float(innovation[0]) if len(innovation) > 0 else 0.0
        innov_y     = float(innovation[1]) if len(innovation) > 1 else 0.0
        uncertainty = float(ekf_state.get('position_uncertainty', 0.0))

        gt_velocity = bundle.ground_truth.velocity if bundle else 0.0
        gps_denied  = float(bundle.gps_denied) if bundle else 0.0

        obs = np.array([
            np.clip(innov_x     / 10.0,  -1.0, 1.0),
            np.clip(innov_y     / 10.0,  -1.0, 1.0),
            np.clip(uncertainty / 20.0,   0.0, 1.0),
            np.clip(self.time_since_gps / 30.0, 0.0, 1.0),
            (self.q_scale - self.Q_SCALE_MIN) / (self.Q_SCALE_MAX - self.Q_SCALE_MIN),
            (self.r_scale - self.R_SCALE_MIN) / (self.R_SCALE_MAX - self.R_SCALE_MIN),
            gps_denied,
            np.clip(gt_velocity / 30.0,   0.0, 1.0),
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------------

    def _compute_reward(self, position_error: float, in_tunnel: bool,
                        delta_q: float, delta_r: float) -> float:
        """
        Normalized reward so episode returns are human-readable.

        Design:
          +1.0  if position error = 0m  (perfect)
           0.0  if position error = 5m  (acceptable)
          -1.0  if position error = 10m (bad)
          -2.0  if position error = 15m+ (very bad)

        This keeps episode returns in roughly [-500, +500] range
        instead of the raw [-5000, 0] range from -position_error.
        """
        # Normalize: map error to [-2, +1] range
        # 5m error = 0 reward (breakeven point)
        normalized = (5.0 - position_error) / 5.0
        normalized = float(np.clip(normalized, -2.0, 1.0))

        # Tunnel bonus: correct behavior during GPS denial is worth more
        if in_tunnel:
            normalized *= 1.5

        # Improvement bonus: reward getting better vs last step
        improvement = (self._prev_position_error - position_error) / 5.0
        normalized += 0.3 * float(np.clip(improvement, -1.0, 1.0))

        # Smoothness: small penalty for thrashing Q/R
        normalized -= 0.05 * (abs(delta_q) + abs(delta_r))

        return float(np.clip(normalized, -3.0, 2.0))

    # ------------------------------------------------------------------
    # TERMINATION
    # ------------------------------------------------------------------

    def _check_done(self, position_error: float,
                    bundle: SensorBundle) -> bool:
        if self.step_count >= MAX_STEPS:
            log.info(f"Episode ended: max steps ({MAX_STEPS})")
            return True
        if position_error > 100.0:
            log.warning(f"Episode ended: EKF diverged ({position_error:.1f}m)")
            return True
        if self._vehicle_is_stuck(bundle):
            log.info("Episode ended: vehicle stuck")
            return True
        return False

    def _vehicle_is_stuck(self, bundle: SensorBundle) -> bool:
        return bundle.ground_truth.velocity < 0.5 and self.step_count > 50

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    def get_episode_summary(self) -> dict:
        if not self.episode_errors:
            return {}
        errors = np.array(self.episode_errors)
        t_errs = np.array(self.episode_tunnel_errors) \
                 if self.episode_tunnel_errors else np.array([0.0])
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
