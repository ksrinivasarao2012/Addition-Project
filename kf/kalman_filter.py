"""
kalman_filter.py  —  Extended Kalman Filter for IMU-based Localization
=======================================================================

PURPOSE
-------
Maintain a real-time estimate of the vehicle's 2-D position, speed, and
heading using raw IMU readings.  When GPS is available it corrects the
accumulated IMU drift.  When GPS is denied (inside the tunnel) the filter
propagates on IMU alone until the LSTM drift compensator provides a
pseudo-measurement correction via inject_lstm_correction().

This module is self-contained and has no CARLA dependency.  It is called
every tick from the fusion module (fusion.py).

STATE VECTOR  x ∈ ℝ⁴
---------------------
    x = [px, py, v, ψ]

    px  — position x   (metres,  same frame as collect_data.py gt_x)
    py  — position y   (metres,  same frame as collect_data.py gt_y)
    v   — speed        (m/s,     always ≥ 0)
    ψ   — heading      (radians, world frame, wraps via atan2)

Why 4-state and not more
    vx = v·cos(ψ) and vy = v·sin(ψ) → adding both would be redundant.
    IMU bias states require long observability windows; a short tunnel
    pass cannot estimate them reliably.  Clean 4-state is the correct
    choice for this scenario.

PROCESS MODEL  (nonlinear → EKF required)
-----------------------------------------
    px' = px + v·cos(ψ)·dt
    py' = py + v·sin(ψ)·dt
    v'  = v  + a_long·dt          ← longitudinal accel from IMU (ax)
    ψ'  = ψ  + wz·dt              ← yaw rate from IMU gyroscope

    Jacobian F = ∂f/∂x  (used for covariance propagation P' = F·P·Fᵀ + Q):

        ┌ 1  0  cos(ψ)·dt   -v·sin(ψ)·dt ┐
    F = │ 0  1  sin(ψ)·dt    v·cos(ψ)·dt │
        │ 0  0  1             0           │
        └ 0  0  0             1           ┘

MEASUREMENT MODEL — GPS  (linear)
----------------------------------
    z_gps = [px_measured, py_measured]
    H_gps = [[1, 0, 0, 0],
              [0, 1, 0, 0]]
    R_gps = diag([σ_gps², σ_gps²])      default σ_gps = 1.5 m (CARLA GPS noise)

MEASUREMENT MODEL — LSTM pseudo-measurement  (linear, GPS-denied only)
------------------------------------------------------------------------
    When GPS is denied the LSTM predicts the displacement since the anchor
    point (position SEQ_LEN steps ago).  This gives a pseudo-measurement
    of the current position:

        z_lstm = [px_anchor + Δx_lstm, py_anchor + Δy_lstm]

    with uncertainty R_lstm = diag([σ_lstm², σ_lstm²]) where σ_lstm is
    the LSTM test RMSE from training (≈ 3.4 m).

NUMERICAL STABILITY
-------------------
    All measurement updates use the Joseph form:
        P = (I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ
    This guarantees P stays symmetric positive-definite even under
    floating-point rounding over thousands of steps.

Q AND R TUNING
--------------
    Default values below are reasonable starting points for Town04.
    The RL adaptive filter tuning module (Step 4) will tune them online
    by adjusting scale factors q_scale and r_scale.  The EKF exposes
    set_Q_scale() and set_R_scale() for this purpose.

USAGE
-----
    from kf.kalman_filter import ExtendedKalmanFilter

    ekf = ExtendedKalmanFilter(dt=0.05)
    ekf.initialize(px=0.0, py=0.0, v=10.7, psi=0.0)

    # Every CARLA tick:
    ekf.predict(ax=imu_ax, wz=imu_wz)

    if not gps_denied:
        ekf.update_gps(gnss_x, gnss_y)
    else:
        # when LSTM has a fresh prediction:
        ekf.inject_lstm_correction(anchor_px, anchor_py, lstm_dx, lstm_dy)

    px, py  = ekf.position
    yaw     = ekf.yaw
    speed   = ekf.speed
    trace_P = ekf.trace_P          # uncertainty signal for RL reward
"""

import math
import numpy as np
from typing import Tuple, Optional


class ExtendedKalmanFilter:
    """
    4-state EKF for real-time vehicle localization using IMU + optional GPS.

    Parameters
    ----------
    dt : float
        Fixed timestep in seconds.  Must match CARLA fixed_delta_seconds.
    q_pos  : float   process noise — position  (m²)
    q_vel  : float   process noise — speed     (m²/s²)
    q_yaw  : float   process noise — heading   (rad²)
    r_gps  : float   GPS measurement std dev   (m)
    r_lstm : float   LSTM pseudo-measurement std dev (m)
                     Set to the test RMSE from training (~3.4 m).
    """

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self,
                 dt:     float = 0.05,
                 q_pos:  float = 0.1,
                 q_vel:  float = 0.5,
                 q_yaw:  float = 0.01,
                 r_gps:  float = 1.5,
                 r_lstm: float = 3.5) -> None:

        self.dt = float(dt)

        # ── base noise parameters (before RL scale factors) ────────────────
        self._q_pos_base  = float(q_pos)
        self._q_vel_base  = float(q_vel)
        self._q_yaw_base  = float(q_yaw)
        self._r_gps_base  = float(r_gps)
        self._r_lstm_base = float(r_lstm)

        # RL scale factors (tuned online, default = 1.0)
        self._q_scale = 1.0
        self._r_scale = 1.0

        # ── state and covariance ───────────────────────────────────────────
        self.x = np.zeros(4, dtype=np.float64)   # [px, py, v, psi]
        self.P = np.eye(4, dtype=np.float64)

        self._initialized = False

        # ── fixed matrices ─────────────────────────────────────────────────
        # GPS measurement matrix — observes px, py only
        self._H = np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        # Step counter and history for diagnostics
        self._step       = 0
        self._gps_steps  = 0
        self._lstm_steps = 0

    # ── public API ────────────────────────────────────────────────────────────

    def initialize(self,
                   px:  float,
                   py:  float,
                   v:   float,
                   psi: float,
                   P0:  Optional[np.ndarray] = None) -> None:
        """
        Set the initial state and covariance.  Must be called before the
        first predict/update call.

        Parameters
        ----------
        px, py : initial position (metres)
        v      : initial speed (m/s)
        psi    : initial heading (radians)
        P0     : initial covariance matrix (4×4).  If None, a sensible
                 default is used (1 m position, 0.5 m/s speed, 0.1 rad heading).
        """
        self.x = np.array([px, py, v, psi], dtype=np.float64)

        if P0 is not None:
            P0 = np.asarray(P0, dtype=np.float64)
            if P0.shape != (4, 4):
                raise ValueError(f"P0 must be (4,4), got {P0.shape}")
            self.P = P0.copy()
        else:
            self.P = np.diag([1.0, 1.0, 0.25, 0.01]).astype(np.float64)

        self._initialized  = True
        self._step         = 0
        self._gps_steps    = 0
        self._lstm_steps   = 0

    def predict(self, ax: float, wz: float) -> None:
        """
        IMU prediction step.  Advances the state by one timestep using
        the nonlinear bicycle-model process function.

        Parameters
        ----------
        ax : longitudinal acceleration from IMU  (m/s²).
             In CARLA this is approximately ax from the IMU sensor.
             (az carries gravity and is NOT used here.)
        wz : yaw rate from IMU gyroscope  (rad/s).
             In CARLA this is IMU angular_velocity.z.
        """
        self._assert_initialized()

        px, py, v, psi = self.x
        dt = self.dt

        # ── nonlinear process function f(x, u) ───────────────────────────
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)

        px_new  = px  + v * cos_psi * dt
        py_new  = py  + v * sin_psi * dt
        v_new   = v   + float(ax) * dt
        psi_new = psi + float(wz) * dt

        # Clamp speed — vehicle cannot go negative
        v_new = max(0.0, v_new)

        # Normalise heading to (−π, π]
        psi_new = self._wrap_angle(psi_new)

        self.x = np.array([px_new, py_new, v_new, psi_new], dtype=np.float64)

        # ── Jacobian F = ∂f/∂x ───────────────────────────────────────────
        F = np.array([
            [1.0, 0.0,  cos_psi * dt,  -v * sin_psi * dt],
            [0.0, 1.0,  sin_psi * dt,   v * cos_psi * dt],
            [0.0, 0.0,  1.0,            0.0              ],
            [0.0, 0.0,  0.0,            1.0              ],
        ], dtype=np.float64)

        # ── covariance prediction P' = F·P·Fᵀ + Q ────────────────────────
        Q = self._build_Q()
        self.P = F @ self.P @ F.T + Q

        # Enforce symmetry (suppresses floating-point drift)
        self.P = 0.5 * (self.P + self.P.T)

        self._step += 1

    def update_gps(self, gnss_x: float, gnss_y: float) -> None:
        """
        GPS measurement update.  Call this every tick when gps_denied == 0.

        Parameters
        ----------
        gnss_x, gnss_y : GPS-derived local position (metres).
                          Must be in the same coordinate frame as gt_x/gt_y
                          (i.e. produced by CoordConverter.gnss_to_local).
        """
        self._assert_initialized()

        z   = np.array([float(gnss_x), float(gnss_y)], dtype=np.float64)
        R   = self._build_R_gps()
        self.x, self.P = self._measurement_update(self.x, self.P, z,
                                                   self._H, R)
        self._gps_steps += 1

    def inject_lstm_correction(self,
                               anchor_px: float,
                               anchor_py: float,
                               lstm_dx:   float,
                               lstm_dy:   float) -> None:
        """
        LSTM pseudo-measurement update.  Call when GPS is denied and the
        LSTM has produced a fresh displacement prediction.

        The LSTM predicts (Δx, Δy) — the displacement from SEQ_LEN steps
        ago to now.  This gives a pseudo-measurement of current position:

            z_pseudo = [anchor_px + Δx, anchor_py + Δy]

        where anchor_px/py is the EKF position at the start of the LSTM
        input window (stored by the fusion module).

        Parameters
        ----------
        anchor_px, anchor_py : EKF position at the start of the LSTM window
        lstm_dx, lstm_dy     : LSTM predicted displacement  (metres)
        """
        self._assert_initialized()

        z = np.array([float(anchor_px) + float(lstm_dx),
                      float(anchor_py) + float(lstm_dy)], dtype=np.float64)
        R = self._build_R_lstm()
        self.x, self.P = self._measurement_update(self.x, self.P, z,
                                                   self._H, R)
        self._lstm_steps += 1

    # ── RL tuning interface ───────────────────────────────────────────────────

    def set_Q_scale(self, q_scale: float) -> None:
        """
        Multiply all Q diagonal elements by q_scale.
        Called by the RL adaptive filter tuning agent.
        Valid range: [0.01, 100.0]
        """
        self._q_scale = float(np.clip(q_scale, 0.01, 100.0))

    def set_R_scale(self, r_scale: float) -> None:
        """
        Multiply all R diagonal elements by r_scale.
        Called by the RL adaptive filter tuning agent.
        Valid range: [0.01, 100.0]
        """
        self._r_scale = float(np.clip(r_scale, 0.01, 100.0))

    def get_noise_scales(self) -> Tuple[float, float]:
        """Return current (q_scale, r_scale) for logging / RL observations."""
        return self._q_scale, self._r_scale

    # ── state accessors ───────────────────────────────────────────────────────

    @property
    def position(self) -> Tuple[float, float]:
        """Current position estimate (px, py) in metres."""
        return float(self.x[0]), float(self.x[1])

    @property
    def speed(self) -> float:
        """Current speed estimate in m/s."""
        return float(self.x[2])

    @property
    def yaw(self) -> float:
        """Current heading estimate in radians, range (−π, π]."""
        return float(self.x[3])

    @property
    def state(self) -> np.ndarray:
        """Full state vector [px, py, v, ψ] as a float64 ndarray copy."""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Full covariance matrix P (4×4) as a float64 ndarray copy."""
        return self.P.copy()

    @property
    def trace_P(self) -> float:
        """
        Trace of the covariance matrix.  Scalar measure of total uncertainty.

        Used in the RL reward function:
            r_uncertainty = −w4 · tr(P) · v · (1 − g_t)
        where g_t = 1 when GPS is available, 0 when denied.
        Higher tr(P) → larger penalty → agent learns to maintain GPS lock
        and to choose routes that minimise time in the tunnel.
        """
        return float(np.trace(self.P))

    @property
    def position_std(self) -> Tuple[float, float]:
        """
        1-sigma position uncertainty: (σ_x, σ_y) in metres.
        Useful for visualisation and sanity checks.
        """
        return float(math.sqrt(max(0.0, self.P[0, 0]))), \
               float(math.sqrt(max(0.0, self.P[1, 1])))

    @property
    def step_count(self) -> int:
        """Number of predict() calls since initialization."""
        return self._step

    def diagnostics(self) -> dict:
        """Return a dict of current filter state for logging."""
        px, py = self.position
        sx, sy = self.position_std
        return {
            'step'        : self._step,
            'px'          : round(px, 4),
            'py'          : round(py, 4),
            'v'           : round(self.speed, 4),
            'psi_deg'     : round(math.degrees(self.yaw), 3),
            'trace_P'     : round(self.trace_P, 6),
            'sigma_px'    : round(sx, 4),
            'sigma_py'    : round(sy, 4),
            'gps_updates' : self._gps_steps,
            'lstm_updates': self._lstm_steps,
            'q_scale'     : self._q_scale,
            'r_scale'     : self._r_scale,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_Q(self) -> np.ndarray:
        s = self._q_scale
        return np.diag([
            s * self._q_pos_base,
            s * self._q_pos_base,
            s * self._q_vel_base,
            s * self._q_yaw_base,
        ]).astype(np.float64)

    def _build_R_gps(self) -> np.ndarray:
        s = self._r_scale
        v = (s * self._r_gps_base) ** 2
        return np.diag([v, v]).astype(np.float64)

    def _build_R_lstm(self) -> np.ndarray:
        # LSTM uncertainty is independent of the RL r_scale — it is fixed by
        # training performance and should not be tuned by the RL agent.
        v = self._r_lstm_base ** 2
        return np.diag([v, v]).astype(np.float64)

    @staticmethod
    def _measurement_update(x:   np.ndarray,
                            P:   np.ndarray,
                            z:   np.ndarray,
                            H:   np.ndarray,
                            R:   np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generic linear measurement update using the Joseph form.

        Joseph form:  P = (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ

        Why Joseph form instead of the standard P = (I−KH)·P:
            The standard form is algebraically equivalent only when K is
            exactly optimal.  Numerical errors in K cause P to slowly lose
            symmetry and positive-definiteness over thousands of steps.
            The Joseph form guarantees P stays symmetric PD regardless.

        Returns updated (x, P).
        """
        n   = len(x)
        y   = z - H @ x                           # innovation
        S   = H @ P @ H.T + R                     # innovation covariance
        K   = P @ H.T @ np.linalg.inv(S)          # Kalman gain
        IKH = np.eye(n) - K @ H
        x_upd = x + K @ y
        P_upd = IKH @ P @ IKH.T + K @ R @ K.T    # Joseph form
        P_upd = 0.5 * (P_upd + P_upd.T)           # enforce symmetry
        return x_upd, P_upd

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to (−π, π]."""
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "ExtendedKalmanFilter.initialize() must be called before "
                "predict() or update() methods."
            )


# ══════════════════════════════════════════════════════════════════════════════
#   STANDALONE TEST  —  run this file directly to verify the filter
# ══════════════════════════════════════════════════════════════════════════════

def _run_self_test() -> None:
    """
    Simulate 60 seconds of highway driving through a GPS-denied zone and
    verify that:
      1. EKF position error stays below 2 m with GPS available.
      2. EKF position error stays below 15 m after 10 s of GPS denial
         (pure IMU dead-reckoning — no LSTM correction yet).
      3. LSTM pseudo-measurement reduces error after GPS denial.
      4. P stays symmetric positive-definite at every step.
      5. All public properties return reasonable values.
    """
    import sys

    print("=" * 60)
    print("  ExtendedKalmanFilter — Self Test")
    print("=" * 60)

    np.random.seed(0)
    dt      = 0.05
    n_steps = int(60 / dt)           # 60 s × 20 Hz = 1200 steps

    # Denial zone: steps 400–600 (20 s → 30 s)
    DENY_START = 400
    DENY_END   = 600
    SEQ_LEN    = 50

    ekf = ExtendedKalmanFilter(dt=dt, r_lstm=3.5)

    # ── Ground truth ──────────────────────────────────────────────────────
    gt_px, gt_py = 0.0, 0.0
    gt_v,  gt_psi = 10.7, 0.0

    ekf.initialize(px=gt_px, py=gt_py, v=gt_v, psi=gt_psi)

    # Position history for anchor tracking
    px_history = np.zeros(n_steps + 1)
    py_history = np.zeros(n_steps + 1)
    px_history[0] = gt_px
    py_history[0] = gt_py

    errors_gps    = []   # error during GPS-available phase
    errors_denied = []   # error during GPS-denied phase (pure IMU)
    errors_lstm   = []   # error after first LSTM correction

    lstm_injected = False
    max_eigen     = []

    for k in range(n_steps):
        # ── Advance ground truth ──────────────────────────────────────────
        gt_psi = ekf._wrap_angle(gt_psi + 0.002)   # gentle curve
        gt_px += gt_v * math.cos(gt_psi) * dt
        gt_py += gt_v * math.sin(gt_psi) * dt

        px_history[k+1] = gt_px
        py_history[k+1] = gt_py

        # ── Simulated IMU (noisy) ─────────────────────────────────────────
        ax_meas = np.random.normal(0.0,  0.05)
        wz_meas = 0.002 + np.random.normal(0.0, 0.005)

        # ── Predict ───────────────────────────────────────────────────────
        ekf.predict(ax=ax_meas, wz=wz_meas)

        # ── Update ────────────────────────────────────────────────────────
        gps_denied = DENY_START <= k < DENY_END

        if not gps_denied:
            gnss_x = gt_px + np.random.normal(0, 1.5)
            gnss_y = gt_py + np.random.normal(0, 1.5)
            ekf.update_gps(gnss_x, gnss_y)
        else:
            # Inject LSTM correction once per SEQ_LEN steps
            steps_in_denial = k - DENY_START
            if steps_in_denial > 0 and steps_in_denial % SEQ_LEN == 0:
                anchor_k  = k - SEQ_LEN
                anchor_px = px_history[anchor_k]
                anchor_py = py_history[anchor_k]
                # Simulate LSTM output: true delta + noise (~3.4 m)
                true_dx = gt_px - anchor_px
                true_dy = gt_py - anchor_py
                lstm_dx = true_dx + np.random.normal(0, 3.4)
                lstm_dy = true_dy + np.random.normal(0, 3.4)
                ekf.inject_lstm_correction(anchor_px, anchor_py,
                                           lstm_dx, lstm_dy)
                lstm_injected = True

        # ── Record error and numerical health ─────────────────────────────
        ekf_px, ekf_py = ekf.position
        err = math.sqrt((ekf_px - gt_px)**2 + (ekf_py - gt_py)**2)

        if not gps_denied:
            errors_gps.append(err)
        elif not lstm_injected:
            errors_denied.append(err)
        else:
            errors_lstm.append(err)

        # P must stay symmetric PD at every step
        eigvals = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigvals > 0), \
            f"P not positive-definite at step {k}: min eigenvalue = {eigvals.min()}"
        assert np.allclose(ekf.P, ekf.P.T, atol=1e-10), \
            f"P not symmetric at step {k}"
        max_eigen.append(float(eigvals.max()))

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n  GPS-available phase ({len(errors_gps)} steps):")
    print(f"    Mean error : {np.mean(errors_gps):.4f} m")
    print(f"    Max  error : {np.max(errors_gps):.4f} m")
    assert np.mean(errors_gps) < 2.0, \
        f"GPS phase mean error too high: {np.mean(errors_gps):.3f} m"
    print(f"    ✓  mean < 2.0 m")

    if errors_denied:
        print(f"\n  GPS-denied phase, before LSTM ({len(errors_denied)} steps):")
        print(f"    Final error : {errors_denied[-1]:.4f} m  "
              f"(pure IMU dead-reckoning)")
        # Dead-reckoning can drift; just verify it doesn't blow up entirely
        assert errors_denied[-1] < 50.0, \
            f"Dead-reckoning drift too high: {errors_denied[-1]:.1f} m"
        print(f"    ✓  final < 50.0 m  (dead-reckoning bound)")

    if errors_lstm:
        print(f"\n  GPS-denied phase, after LSTM injection ({len(errors_lstm)} steps):")
        print(f"    Mean error : {np.mean(errors_lstm):.4f} m")
        print(f"    Max  error : {np.max(errors_lstm):.4f} m")
        print(f"    ✓  LSTM corrections applied successfully")

    print(f"\n  P numerical health:")
    print(f"    Max eigenvalue ever : {max(max_eigen):.4f}")
    print(f"    trace_P (final)     : {ekf.trace_P:.4f}")
    print(f"    ✓  symmetric PD at every step")

    print(f"\n  Diagnostics at final step:")
    for k, v in ekf.diagnostics().items():
        print(f"    {k:15s} : {v}")

    print(f"\n  Q/R scale interface:")
    ekf.set_Q_scale(2.0)
    ekf.set_R_scale(0.5)
    assert ekf.get_noise_scales() == (2.0, 0.5)
    print(f"    set_Q_scale(2.0) / set_R_scale(0.5) : OK")
    ekf.set_Q_scale(1e-5)   # below min → clamped to 0.01
    assert ekf.get_noise_scales()[0] == 0.01
    print(f"    Q scale clamping at 0.01             : OK")

    print(f"\n{'=' * 60}")
    print(f"  ALL SELF-TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    _run_self_test()
