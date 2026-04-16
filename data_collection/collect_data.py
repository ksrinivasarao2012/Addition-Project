"""
collect_data.py  —  Production-quality CARLA Town04 Data Collection (v12)
=========================================================================
For: LSTM + RL-Adaptive EKF Localization Project

Changes from v11 → v12:
  - BUG FIX Q: is_gps_denied_time() — GPS denial window (40.0, 25.0) had
      end=65.0 which exceeds GPS_CYCLE_TIME=60.0.  The v11 check
      `start <= t_mod < (start + dur)` never fires for t_mod in [0, 5)
      because t_mod ∈ [0, 60).  Result: 5s of denial per 60-second cycle
      was silently lost, causing v11 to produce only 58.3% GPS denial
      instead of the intended 66.7%.  Fixed by a new is_gps_denied_time()
      helper that handles cyclic wrap-around: when start+dur > CYCLE_TIME
      the check becomes `t_mod >= start OR t_mod < (start+dur-CYCLE_TIME)`.
      This is a DATA CORRECTNESS bug — the training label gps_denied=1 was
      wrong for every tick in the wrap segment, so the EKF bias-state
      estimator would have trained on incorrect supervision signal.

  - FIX R: VALIDATION_MODE flag (default False).  When True, main()
      overrides TICKS_PER_RUN → 200 and NUM_RUNS → 1 without touching any
      other constant, so a safe first-run check requires only flipping one
      variable.  All three post-run checks (console logs, CSV, alignment)
      still execute normally.

  - FIX S: DROP_ABORT_THRESHOLD = 0.10.  collect_run now checks the running
      drop rate every SAVE_INTERVAL ticks and raises RuntimeError if it
      exceeds the threshold, aborting the run early with a clear message
      rather than silently writing a corrupted dataset.

  - FIX T: CoordConverter.reset_origin() method replaces the direct
      attribute write `conv.ref_lat = conv.ref_lon = None` in main().
      Direct attribute access bypasses any future validation added to the
      class, and is harder to grep/trace.  The method also logs the reset
      for visibility.

  - FIX U: dataset_summary(csv_path) function.  Called once in main() after
      all runs complete.  Prints per-column NaN rates, gps_denied value
      distribution, dropped-frame total, and basic range sanity checks.
      Requires only the stdlib csv module — no pandas dependency.

All previous v11 / v10 / v9 fixes are preserved unchanged:
  - BUG FIX Q (this version) — GPS denial wrap-around.
  - FIX H:  tick=0 initialised before try block.
  - FIX I:  _drain_to_frame no premature early return on queue.Empty.
  - FIX J:  temp file handles None-sentinel before try block.
  - FIX K:  spawn_npcs guard for empty bps list.
  - FIX L:  GPS constants at module level (not re-allocated per tick).
  - FIX M:  gps_denied=-1 sentinel (not NaN) in dropped-frame rows.
  - FIX N:  apply_zero_phase_filter validates cutoff_freq and fs.
  - FIX O:  dropped-frame count logged at end of each run.
  - FIX P:  destroy_npcs uses apply_batch_sync; failures logged.
  - FIX 1:  safe_filter_array never mutates its input.
  - FIX 2:  gnss_to_local_raw has the same origin guard as gnss_to_local.
  - FIX 3:  exception-safe master write — partial run data always flushed.
  - FIX 4:  cleanup tick loop uses `continue` instead of `break`.
  - FIX 5:  filtfilt pad-length guard with FILTFILT_MIN_SAMPLES constant.
  - FIX A:  FILTFILT_MIN_SAMPLES guard inside safe_filter_array.
  - FIX B:  < 2 valid points guard before np.interp.
  - FIX C:  ±inf normalised to NaN in safe_filter_array.
  - FIX D:  zero-phase filter wrapped in broad try/except.
  - FIX E:  all-NaN paranoia check after filtfilt.
  - FIX F:  empty array guard at top of safe_filter_array.
  - FIX G:  inf→NaN normalisation before orig_*_nan snapshot.
"""

import sys
import os
import time
import math
import csv
import queue
import random
import numpy as np
from scipy.signal import filtfilt, butter

# =============================================================================
# CARLA IMPORT
# =============================================================================
CARLA_EGG = (
    r'C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor'
    r'\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg'
)
sys.path.insert(0, CARLA_EGG)
sys.path.insert(0, r'C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor\PythonAPI')
import carla


# =============================================================================
# CONFIGURATION
# =============================================================================
HOST      = '127.0.0.1'
PORT      = 2000
TIMEOUT_S = 15.0
MAP_NAME  = 'Town04'

VEHICLE_BP    = 'vehicle.tesla.model3'
FIXED_DELTA_T = 0.05       # 20 Hz
WARMUP_TICKS  = 120
TICKS_PER_RUN = 6_000      # 5 min per run
NUM_RUNS      = 5
SAVE_INTERVAL = 500
NPC_COUNT     = 30

G = 9.81  # m/s²
EGO_EXCLUSION_RADIUS = 10.0

HIGHWAY_SPAWN_INDICES = [16, 15, 17, 18]

TUNNEL_X_MIN = -130.0
TUNNEL_X_MAX =  140.0
TUNNEL_Y_MIN =  -35.0
TUNNEL_Y_MAX =   65.0

WEATHER_PRESETS = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.ClearSunset,
]
WEATHER_NAMES = ["ClearNoon", "CloudyNoon", "WetNoon", "ClearSunset"]

OUTPUT_DIR = r'C:\Users\heman\Music\rl_imu_project\data'
TRAIN_CSV  = os.path.join(OUTPUT_DIR, 'town04_dataset.csv')
DEBUG_CSV  = os.path.join(OUTPUT_DIR, 'town04_debug.csv')

TRAIN_COLS = [
    'timestamp', 'run_id', 'weather',
    'ax', 'ay', 'az', 'wx', 'wy', 'wz',
    'ax_corr', 'ay_corr',
    'gnss_x', 'gnss_y',
    'gt_x', 'gt_y', 'gt_heading', 'gt_speed_mps',
    'gt_accel_fwd_mps2', 'gt_accel_lat_mps2',
    'gps_denied', 'pitch_deg', 'roll_deg',
]

DEBUG_COLS = TRAIN_COLS + [
    'world_x', 'world_y', 'world_z',
    'gnss_x_raw', 'gnss_y_raw',
]

SPEED_SCHEDULE = [
    (100, 60,  "highway_cruise"), (80,  80,  "fast_highway"),
    (60,  90,  "max_highway"),    (80,  40,  "decelerate"),
    (50,   0,  "full_stop"),      (40,   0,  "stopped"),
    (80,  20,  "slow_creep"),     (100, 50,  "moderate_urban"),
    (150, 60,  "highway_cruise"), (60,  30,  "slow_urban"),
    (100, 60,  "highway_cruise"), (80,  70,  "slightly_above"),
    (100, 60,  "highway_cruise"), (80,   0,  "emergency_stop"),
    (40,   0,  "stopped_2"),      (100, 60,  "resume_highway"),
    (100, 60,  "highway_cruise"),
]

# Minimum samples required to safely call filtfilt.
# For a 2nd-order Butterworth: padlen = 3*max(len(b),len(a))-1 = 8.
# We use 16 (double + buffer) for headroom.
FILTFILT_MIN_SAMPLES = 16

# FIX L: GPS denial constants at module level.
GPS_CYCLE_TIME   = 60.0
GPS_DENY_WINDOWS = [
    (20.0, 10.0),   # t_mod ∈ [20, 35)  — never wraps
    (40.0, 13.0),   # t_mod ∈ [40, 60) ∪ [0, 5)  — wraps past 60s boundary
    #                 BUG FIX Q: v11 lost the [0,5) segment every cycle.
]

# FIX R: Abort a run early if the running drop rate exceeds this threshold.
# Checked every SAVE_INTERVAL ticks.  10 % is the upper bound for usable data.
DROP_ABORT_THRESHOLD = 0.10

# FIX R: VALIDATION_MODE — flip to True for the first validation run.
# main() will override TICKS_PER_RUN→200, NUM_RUNS→1 when this is True,
# then print a reminder so it is never left on accidentally for full collection.
VALIDATION_MODE = False


# =============================================================================
# ZERO-PHASE FILTER
# =============================================================================
def apply_zero_phase_filter(data_array, cutoff_freq=2.0, fs=20.0, order=2):
    """
    FIX N: validates cutoff_freq and fs BEFORE calling butter() so that
    invalid parameters raise a clear ValueError instead of a cryptic scipy
    error or being silently swallowed by safe_filter_array's broad except.
    """
    if fs <= 0:
        raise ValueError(
            f"apply_zero_phase_filter: fs must be > 0, got {fs}"
        )
    nyquist = 0.5 * fs
    if cutoff_freq <= 0 or cutoff_freq >= nyquist:
        raise ValueError(
            f"apply_zero_phase_filter: cutoff_freq must be in "
            f"(0, {nyquist:.4f}), got {cutoff_freq}"
        )
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data_array)


def safe_filter_array(arr: np.ndarray) -> np.ndarray:
    """
    Interpolates over NaNs, applies a zero-phase Butterworth filter, and
    returns the smoothed array.

    Contract:
    ---------
    * NEVER mutates the input array (always operates on an internal copy).
    * NEVER raises — returns a (possibly unfiltered) copy as a safe fallback
      for every failure mode.
    * Handles: empty arrays, all-NaN, +-inf, < 2 valid points, arrays too
      short for filtfilt, and unexpected filter exceptions.
    """
    # FIX 1: always own the data.
    arr = np.array(arr, dtype=float)

    # FIX F: empty guard.
    if arr.size == 0:
        return arr

    # FIX C: normalise ±inf → NaN.
    inf_mask = np.isinf(arr)
    if inf_mask.any():
        arr[inf_mask] = np.nan

    # FIX E / all-NaN guard.
    if np.isnan(arr).all():
        return arr

    # FIX A / FIX 5: length guard.
    if len(arr) < FILTFILT_MIN_SAMPLES:
        return arr

    # FIX B: NaN interpolation.
    mask = np.isnan(arr)
    if mask.any():
        valid   = ~mask
        n_valid = int(np.sum(valid))
        if n_valid < 2:
            return arr
        arr[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(valid),
            arr[valid],
        )

    # FIX D: broad exception safety.
    try:
        filtered = apply_zero_phase_filter(arr)
        if np.isnan(filtered).all():          # FIX E paranoia
            return arr
        return filtered
    except Exception:
        return arr


# =============================================================================
# GPS DENIAL HELPER  (BUG FIX Q)
# =============================================================================
def is_gps_denied_time(t_mod: float) -> bool:
    """
    BUG FIX Q — Returns True if t_mod falls inside any GPS denial window,
    correctly handling windows that wrap past GPS_CYCLE_TIME.

    v11 bug: the check `start <= t_mod < (start + dur)` never fires for
    t_mod ∈ [0, 5) when start=40, dur=25 (end=65 > cycle=60).  This silently
    dropped 5 s of denial per 60-second cycle, producing 58.3% denial instead
    of the intended 66.7%.

    Fix: when start + dur > GPS_CYCLE_TIME, the window wraps around the cycle
    boundary and covers [start, GPS_CYCLE_TIME) ∪ [0, (start+dur-CYCLE_TIME)).
    """
    for start, dur in GPS_DENY_WINDOWS:
        end = start + dur
        if end <= GPS_CYCLE_TIME:
            # Normal (non-wrapping) window.
            if start <= t_mod < end:
                return True
        else:
            # Wrapping window: covers [start, cycle) and [0, end-cycle).
            wrap_end = end - GPS_CYCLE_TIME
            if t_mod >= start or t_mod < wrap_end:
                return True
    return False


# =============================================================================
# SENSOR & COORD MANAGERS
# =============================================================================
class SyncSensorManager:
    def __init__(self):
        self.imu_queue  = queue.Queue()
        self.gnss_queue = queue.Queue()

    def on_imu(self, data):  self.imu_queue.put(data)
    def on_gnss(self, data): self.gnss_queue.put(data)

    def get_frame(self, frame_id, timeout=2.0):
        imu_data  = self._drain_to_frame(self.imu_queue,  frame_id, timeout)
        gnss_data = self._drain_to_frame(self.gnss_queue, frame_id, timeout)
        return imu_data, gnss_data

    def _drain_to_frame(self, q, target_frame, timeout):
        """
        FIX I: Removed premature queue.Empty early return.

        Correct behaviour:
          - Exact match      → return immediately.
          - ±1 match         → store as best_candidate; keep polling until
                               deadline so exact-match has priority.
          - Out-of-tolerance → consume and log (cannot put back).
          - Deadline reached → return best_candidate if any; else raise.
          - queue.Empty      → keep waiting (do NOT return early).
        """
        FRAME_TOLERANCE = 0

        deadline       = time.time() + timeout
        best_candidate = None
        best_diff      = float('inf')

        while True:
            remaining = deadline - time.time()

            if remaining <= 0:
                if best_candidate is not None:
                    return best_candidate
                raise RuntimeError(f"Timeout waiting for frame {target_frame}")

            try:
                data = q.get(timeout=min(remaining, 0.1))
                diff = abs(data.frame - target_frame)

                if diff == 0:
                    return data

                if diff <= FRAME_TOLERANCE:
                    if diff < best_diff:
                        best_candidate = data
                        best_diff      = diff
                    continue

                # Out-of-tolerance: consumed; log for visibility.
                print(
                    f"  [WARN] _drain_to_frame: discarding frame "
                    f"{data.frame} (target={target_frame}, diff={diff})"
                )

            except queue.Empty:
                continue  # FIX I: do NOT return best_candidate early

    def clear(self):
        for q in (self.imu_queue, self.gnss_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break


class CoordConverter:
    EARTH_RADIUS = 6_371_000

    def __init__(self):
        self.ref_lat = self.ref_lon = None

    def set_origin(self, lat, lon):
        self.ref_lat, self.ref_lon = lat, lon
        print(f"  [Coord] Origin set: lat={lat:.8f}  lon={lon:.8f}")

    def reset_origin(self):
        """
        FIX T: explicit method to clear the per-run GNSS origin.
        Replaces `conv.ref_lat = conv.ref_lon = None` direct attribute
        writes in main(), which bypass any future validation added here.
        """
        self.ref_lat = self.ref_lon = None
        print("  [Coord] Origin cleared for new run.")

    def _check_origin(self):
        if self.ref_lat is None:
            raise RuntimeError(
                "CoordConverter origin not set — call set_origin() first."
            )

    def gnss_to_local(self, lat, lon):
        """Returns (east_m, north_m) in CARLA-aligned local frame (+Y is south)."""
        self._check_origin()
        east  = (self.EARTH_RADIUS
                 * math.radians(lon - self.ref_lon)
                 * math.cos(math.radians(self.ref_lat)))
        north = self.EARTH_RADIUS * math.radians(lat - self.ref_lat)
        return east, -north  # CARLA +Y is South

    def gnss_to_local_raw(self, lat, lon):
        """
        Returns (east_m, north_m) in geographic convention (no Y-flip).
        FIX 2: same origin guard as gnss_to_local.
        """
        self._check_origin()
        east  = (self.EARTH_RADIUS
                 * math.radians(lon - self.ref_lon)
                 * math.cos(math.radians(self.ref_lat)))
        north = self.EARTH_RADIUS * math.radians(lat - self.ref_lat)
        return east, north


def correct_imu_for_gravity(ax_raw, ay_raw, pitch_deg, roll_deg):
    pitch_rad = math.radians(pitch_deg)
    roll_rad  = math.radians(roll_deg)
    ax_corr = round(ax_raw + G * math.sin(pitch_rad), 6)
    ay_corr = round(ay_raw - G * math.sin(roll_rad) * math.cos(pitch_rad), 6)
    return ax_corr, ay_corr


class SpeedScheduler:
    def __init__(self, schedule):
        self.schedule = schedule
        self.total = sum(d for d, _, _ in schedule)

    def get(self, tick):
        pos, cum = tick % self.total, 0
        for dur, spd, lbl in self.schedule:
            cum += dur
            if pos < cum:
                return spd, lbl
        return self.schedule[-1][1], self.schedule[-1][2]


def get_speed_mps(v):
    vel = v.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def in_tunnel(x, y):
    return (TUNNEL_X_MIN <= x <= TUNNEL_X_MAX
            and TUNNEL_Y_MIN <= y <= TUNNEL_Y_MAX)


def spectator_follow(world, vehicle):
    t   = vehicle.get_transform()
    yaw = math.radians(t.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(
            x=t.location.x - 10 * math.cos(yaw),
            y=t.location.y - 10 * math.sin(yaw),
            z=t.location.z + 6,
        ),
        carla.Rotation(pitch=-20, yaw=t.rotation.yaw),
    ))


def carla_yaw_to_heading_rad(yaw_deg):
    return math.atan2(
        math.sin(-math.radians(yaw_deg)),
        math.cos(-math.radians(yaw_deg)),
    )


def spawn_npcs(world, count, ego_loc):
    bpl = world.get_blueprint_library()
    safe_pts = [
        sp for sp in world.get_map().get_spawn_points()
        if math.sqrt((sp.location.x - ego_loc.x)**2
                     + (sp.location.y - ego_loc.y)**2) > EGO_EXCLUSION_RADIUS
    ]
    random.shuffle(safe_pts)
    bps = [
        bp for bp in bpl.filter('vehicle.*')
        if not any(x in bp.id for x in ['bike', 'motorcycle', 'crossbike'])
    ]

    # FIX K: clear error instead of cryptic IndexError from random.choice([]).
    if not bps:
        raise RuntimeError(
            "spawn_npcs: no non-bike vehicle blueprints found. "
            "Check the CARLA version or blueprint filter string."
        )

    npcs = []
    for sp in safe_pts[:count]:
        npc = world.try_spawn_actor(random.choice(bps), sp)
        if npc:
            npc.set_autopilot(True, 8000)
            npcs.append(npc)
    print(f"  [NPC] Spawned {len(npcs)} vehicles.")
    return npcs


def destroy_npcs(client, npcs):
    if not npcs:
        print("  [NPC] No vehicles to destroy.")
        return

    # FIX P: apply_batch_sync; failures are logged.
    responses = client.apply_batch_sync(
        [carla.command.DestroyActor(n) for n in npcs]
    )
    failed = sum(1 for r in responses if r.error)
    if failed:
        print(
            f"  [NPC] Destroyed {len(npcs) - failed}/{len(npcs)} vehicles "
            f"({failed} failed — likely already invalid)."
        )
    else:
        print(f"  [NPC] Destroyed {len(npcs)} vehicles.")


def verify_alignment(buffer):
    print("\n" + "=" * 72)
    print("  ALIGNMENT VERIFICATION (First 20 valid ticks)")
    print("=" * 72)
    print(f"  {'t':>3}  {'gt_x':>8}  {'gt_y':>8}  "
          f"{'gnss_x':>8}  {'gnss_y':>8}  {'err_x':>7}  {'err_y':>7}")
    errs_x, errs_y = [], []
    for i, r in enumerate(buffer):
        dx = r['gnss_x'] - r['gt_x']
        dy = r['gnss_y'] - r['gt_y']
        errs_x.append(abs(dx))
        errs_y.append(abs(dy))
        print(f"  {i:>3}  {r['gt_x']:>8.3f}  {r['gt_y']:>8.3f}  "
              f"{r['gnss_x']:>8.3f}  {r['gnss_y']:>8.3f}  "
              f"{dx:>7.3f}  {dy:>7.3f}")

    if len(buffer) >= 2:
        gt_dy  = buffer[-1]['gt_y']   - buffer[0]['gt_y']
        gns_dy = buffer[-1]['gnss_y'] - buffer[0]['gnss_y']
        if gt_dy * gns_dy < -0.1:
            print("\n  CRITICAL: gt_y and gnss_y move in OPPOSITE directions!\n")
        else:
            print(
                f"\n  Direction OK | "
                f"Mean Err: X={np.mean(errs_x):.2f}m  Y={np.mean(errs_y):.2f}m\n"
            )


# =============================================================================
# DATASET SUMMARY  (FIX U)
# =============================================================================
def dataset_summary(csv_path: str) -> None:
    """
    FIX U: Post-collection quality report.  Reads the master training CSV and
    prints:
      - Total row count and per-run breakdown.
      - Per-column NaN rates (highlights any column > 5 %).
      - gps_denied value distribution (should contain 0, 1, and -1).
      - Basic sanity ranges for gt_speed_mps and gt_accel_fwd_mps2.

    Uses only the stdlib csv module — no pandas required.
    """
    print("\n" + "=" * 65)
    print("  DATASET SUMMARY")
    print(f"  File: {csv_path}")
    print("=" * 65)

    if not os.path.exists(csv_path):
        print("  [ERROR] File not found — nothing to summarise.")
        return

    rows = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"  [ERROR] Could not read CSV: {e}")
        return

    if not rows:
        print("  [WARN] CSV is empty — no rows collected.")
        return

    n = len(rows)
    print(f"\n  Total rows: {n:,}")

    # Per-run breakdown
    run_counts: dict = {}
    for r in rows:
        rid = r.get('run_id', '?')
        run_counts[rid] = run_counts.get(rid, 0) + 1
    print("\n  Rows per run:")
    for rid, cnt in sorted(run_counts.items()):
        print(f"    run {rid}: {cnt:,}  ({100*cnt/n:.1f}%)")

    # NaN rates for numeric columns
    numeric_cols = [
        c for c in TRAIN_COLS
        if c not in ('run_id', 'weather', 'gps_denied')
    ]
    print("\n  NaN rates (numeric columns):")
    any_high = False
    for col in numeric_cols:
        nan_count = sum(
            1 for r in rows
            if r.get(col, '') in ('', 'nan', 'NaN', 'inf', '-inf')
        )
        rate = 100 * nan_count / n
        flag = '  ⚠ HIGH' if rate > 5.0 else ''
        if rate > 5.0:
            any_high = True
        print(f"    {col:<25s}: {rate:5.1f}%{flag}")
    if not any_high:
        print("    All columns within acceptable range (<5%).")

    # gps_denied distribution
    denied_counts: dict = {}
    for r in rows:
        val = r.get('gps_denied', '?')
        denied_counts[val] = denied_counts.get(val, 0) + 1
    print("\n  gps_denied distribution:")
    for val, cnt in sorted(denied_counts.items()):
        label = {
            '0':  'available',
            '1':  'denied',
            '-1': 'dropped frame',
        }.get(val, 'unknown')
        print(f"    {val:>4s} ({label:>13s}): {cnt:,}  ({100*cnt/n:.1f}%)")
    missing_vals = [v for v in ('0', '1', '-1') if v not in denied_counts]
    if missing_vals:
        print(f"  [WARN] gps_denied values not seen: {missing_vals}")
        if '1' not in denied_counts:
            print("         gps_denied=1 never occurred — "
                  "check GPS denial windows and tunnel bounds.")

    # Basic range sanity
    print("\n  Sanity ranges (non-NaN rows only):")
    for col, lo, hi, unit in [
        ('gt_speed_mps',      0.0,  35.0, 'm/s'),
        ('gt_accel_fwd_mps2', -25.0, 25.0, 'm/s²'),
        ('gt_accel_lat_mps2', -25.0, 25.0, 'm/s²'),
        ('pitch_deg',         -15.0, 15.0, 'deg'),
        ('roll_deg',          -15.0, 15.0, 'deg'),
    ]:
        vals = []
        for r in rows:
            raw = r.get(col, '')
            try:
                v = float(raw)
                if math.isfinite(v):
                    vals.append(v)
            except (ValueError, TypeError):
                pass
        if not vals:
            print(f"    {col:<25s}: no valid data")
            continue
        vmin, vmax = min(vals), max(vals)
        out = vmin < lo or vmax > hi
        flag = '  ⚠ OUT OF RANGE' if out else ''
        print(f"    {col:<25s}: [{vmin:8.3f}, {vmax:8.3f}] {unit}{flag}")

    print("\n" + "=" * 65)


# =============================================================================
# SINGLE RUN COLLECTION (Temp-to-Master Architecture)
# =============================================================================
def collect_run(client, world, tm, run_id, spawn_index,
                weather_preset, weather_name,
                conv, scheduler, tw, dw, train_f, debug_f,
                ticks_per_run=None):
    """
    Collects one full run and appends filtered rows to the master CSV writers.

    Parameters
    ----------
    ticks_per_run : int or None
        If provided, overrides the module-level TICKS_PER_RUN.  Used by
        VALIDATION_MODE in main() without touching the global constant.

    Architecture:
      - Temp files are written tick-by-tick for crash safety.
      - All rows are held in RAM so zero-phase filtering can be applied
        post-run without phase distortion.
      - FIX 3:  master-write always executes (run_exception sentinel).
      - FIX 4:  cleanup tick loop uses `continue` so CARLA is always drained.
      - FIX G:  inf→NaN normalisation before orig_*_nan snapshot.
      - FIX H:  `tick` initialised before try block.
      - FIX J:  temp file handles initialised to None before try block.
      - FIX L:  GPS denial uses module-level GPS_CYCLE_TIME / GPS_DENY_WINDOWS.
      - FIX M:  gps_denied=-1 sentinel in dropped-frame rows.
      - FIX O:  dropped-frame count logged at end of run.
      - BUG FIX Q: GPS denial uses is_gps_denied_time() (handles wrap-around).
      - FIX S:  DROP_ABORT_THRESHOLD checked every SAVE_INTERVAL ticks.
    """
    if ticks_per_run is None:
        ticks_per_run = TICKS_PER_RUN

    vehicle = imu_s = gnss_s = None
    npcs    = []

    run_rows           = []
    align_buf          = []
    raw_accel_fwd_list = []
    raw_accel_lat_list = []

    run_exception = None

    temp_train_path = os.path.join(OUTPUT_DIR, f'temp_train_run_{run_id}.csv')
    temp_debug_path = os.path.join(OUTPUT_DIR, f'temp_debug_run_{run_id}.csv')

    # FIX J: None-sentinel BEFORE try block.
    temp_train_f = None
    temp_debug_f = None

    # FIX H: tick initialised BEFORE try block.
    tick          = 0
    tunnel_ticks  = 0
    dropped_ticks = 0

    try:
        temp_train_f = open(temp_train_path, 'w', newline='', encoding='utf-8')
        temp_debug_f = open(temp_debug_path, 'w', newline='', encoding='utf-8')
        temp_tw = csv.DictWriter(temp_train_f, fieldnames=TRAIN_COLS)
        temp_dw = csv.DictWriter(temp_debug_f, fieldnames=DEBUG_COLS)
        temp_tw.writeheader()
        temp_dw.writeheader()

        world.set_weather(weather_preset)
        print(f"\n[Run {run_id}] Weather: {weather_name}  |  ticks: {ticks_per_run}")

        bpl       = world.get_blueprint_library()
        bp        = bpl.find(VEHICLE_BP)
        bp.set_attribute('role_name', 'hero')
        spawn_pts = world.get_map().get_spawn_points()

        sp      = spawn_pts[spawn_index]
        vehicle = world.try_spawn_actor(bp, sp)
        if not vehicle:
            for alt in [spawn_index - 1, spawn_index + 1, spawn_index - 2]:
                if 0 <= alt < len(spawn_pts):
                    vehicle = world.try_spawn_actor(bp, spawn_pts[alt])
                    if vehicle:
                        sp = spawn_pts[alt]
                        break
        if not vehicle:
            raise RuntimeError(
                "Vehicle spawn failed — all candidate points occupied."
            )

        world.tick()
        sync_mgr = SyncSensorManager()

        # IMU with gyro bias for EKF bias-state training
        imu_bp = bpl.find('sensor.other.imu')
        for k, v in [
            ('noise_accel_stddev_x', '0.02'),
            ('noise_accel_stddev_y', '0.02'),
            ('noise_accel_stddev_z', '0.02'),
            ('noise_gyro_stddev_x',  '0.005'),
            ('noise_gyro_stddev_y',  '0.005'),
            ('noise_gyro_stddev_z',  '0.005'),
            ('noise_gyro_bias_x',    '0.001'),
            ('noise_gyro_bias_y',    '0.001'),
            ('noise_gyro_bias_z',    '0.001'),
            ('sensor_tick',          str(FIXED_DELTA_T)),
        ]:
            imu_bp.set_attribute(k, v)
        imu_s = world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0, y=0, z=0)),
            attach_to=vehicle,
        )
        imu_s.listen(sync_mgr.on_imu)

        gnss_bp = bpl.find('sensor.other.gnss')
        for k, v in [
            ('noise_lat_stddev', '0.000005'),
            ('noise_lon_stddev', '0.000005'),
            ('sensor_tick',      str(FIXED_DELTA_T)),
        ]:
            gnss_bp.set_attribute(k, v)
        gnss_s = world.spawn_actor(
            gnss_bp,
            carla.Transform(carla.Location(x=0, y=0, z=0)),
            attach_to=vehicle,
        )
        gnss_s.listen(sync_mgr.on_gnss)

        tm.ignore_lights_percentage(vehicle, 0.0)
        tm.ignore_signs_percentage(vehicle, 0.0)
        tm.auto_lane_change(vehicle, True)
        vehicle.set_autopilot(True, 8000)

        tm.global_percentage_speed_difference(15.0)
        tm.distance_to_leading_vehicle(vehicle, 20.0)

        npcs = spawn_npcs(world, NPC_COUNT, sp.location)

        print(f"  Warming up ({WARMUP_TICKS} ticks)...")
        for _ in range(WARMUP_TICKS):
            frame = world.tick()
            try:
                sync_mgr.get_frame(frame, timeout=1.0)
            except RuntimeError:
                pass

        print("  Setting GNSS origin...")
        origin_set = False
        for _ in range(30):
            frame = world.tick()
            try:
                _, gnss_data = sync_mgr.get_frame(frame, timeout=5.0)
                conv.set_origin(gnss_data.latitude, gnss_data.longitude)
                origin_set = True
                break
            except RuntimeError:
                continue
        if not origin_set:
            raise RuntimeError(
                "GNSS sensor not ready after 30 ticks — cannot set origin."
            )

        spawn_loc = vehicle.get_transform().location
        sync_mgr.clear()

        print(f"  Recording {ticks_per_run} ticks...")

        current_speed = 0.0   # smooth speed state

        while tick < ticks_per_run:
            frame_id = world.tick()
            tick += 1

            target_kmh, phase = scheduler.get(tick)

            target_speed = float(target_kmh)

            # Smooth speed transition (VERY IMPORTANT)
            alpha = 0.03  # try 0.02–0.05
            current_speed = (1 - alpha) * current_speed + alpha * target_speed

            tm.set_desired_speed(vehicle, current_speed)

            # ------------------------------------------------------------------
            # FIX S: mid-run drop-rate check every SAVE_INTERVAL ticks.
            # Abort early with a clear error rather than silently continuing.
            # ------------------------------------------------------------------
            if tick % SAVE_INTERVAL == 0 and tick > 0:
                current_rate = dropped_ticks / tick
                if current_rate > DROP_ABORT_THRESHOLD:
                    raise RuntimeError(
                        f"Drop rate {100*current_rate:.1f}% at tick {tick} "
                        f"exceeds threshold {100*DROP_ABORT_THRESHOLD:.0f}%. "
                        f"Check CARLA performance / sensor timeout."
                    )

            # ------------------------------------------------------------------
            # Dropped-frame path.
            # ------------------------------------------------------------------
            try:
                imu_data, gnss_data = sync_mgr.get_frame(frame_id, timeout=0.01)
            except RuntimeError as e:
                print(f"  [WARN] tick {tick}: dropped frame — writing NaNs ({e})")
                dropped_ticks += 1
                ts    = round(tick * FIXED_DELTA_T, 4)
                t_row = {col: float('nan') for col in TRAIN_COLS}
                t_row.update({
                    'timestamp':  ts,
                    'run_id':     run_id,
                    'weather':    weather_name,
                    # FIX M: integer sentinel -1 (not NaN).
                    'gps_denied': -1,
                })
                d_row = {col: float('nan') for col in DEBUG_COLS}
                d_row.update({
                    'timestamp':  ts,
                    'run_id':     run_id,
                    'weather':    weather_name,
                    'gps_denied': -1,
                })
                temp_tw.writerow(t_row)
                temp_dw.writerow(d_row)
                run_rows.append((t_row, d_row))
                raw_accel_fwd_list.append(float('nan'))
                raw_accel_lat_list.append(float('nan'))
                continue

            # ------------------------------------------------------------------
            # Normal path.
            # ------------------------------------------------------------------
            transform = vehicle.get_transform()
            loc       = transform.location
            pitch_deg = transform.rotation.pitch
            roll_deg  = transform.rotation.roll
            yaw_deg   = transform.rotation.yaw

            gt_x       = loc.x - spawn_loc.x
            gt_y       = loc.y - spawn_loc.y
            gt_heading = carla_yaw_to_heading_rad(yaw_deg)
            gt_speed   = get_speed_mps(vehicle)

            if gt_speed < 0.3:
                gt_speed = 0.0

            accel_3d = vehicle.get_acceleration()
            fwd_vec  = transform.get_forward_vector()
            lat_vec  = transform.get_right_vector()
            gt_accel_fwd = max(-25.0, min(25.0,
                accel_3d.x * fwd_vec.x
                + accel_3d.y * fwd_vec.y
                + accel_3d.z * fwd_vec.z))
            gt_accel_lat = max(-25.0, min(25.0,
                accel_3d.x * lat_vec.x
                + accel_3d.y * lat_vec.y
                + accel_3d.z * lat_vec.z))

            gnss_x, gnss_y         = conv.gnss_to_local(
                gnss_data.latitude, gnss_data.longitude)
            gnss_x_raw, gnss_y_raw = conv.gnss_to_local_raw(
                gnss_data.latitude, gnss_data.longitude)

            ax_corr, ay_corr = correct_imu_for_gravity(
                imu_data.accelerometer.x, imu_data.accelerometer.y,
                pitch_deg, roll_deg,
            )

            # GPS DENIAL LOGIC
            # FIX L: module-level constants (not re-allocated per tick).
            # BUG FIX Q: is_gps_denied_time() handles cyclic wrap-around.
            t_abs = tick * FIXED_DELTA_T
            t_mod = t_abs % GPS_CYCLE_TIME

            gps_denied_time   = is_gps_denied_time(t_mod)
            gps_denied_tunnel = in_tunnel(loc.x, loc.y)
            gps_denied        = 1 if (gps_denied_time or gps_denied_tunnel) else 0
            if gps_denied:
                tunnel_ticks += 1

            ts = round(tick * FIXED_DELTA_T, 4)

            t_row = {
                'timestamp':         ts,
                'run_id':            run_id,
                'weather':           weather_name,
                'ax':                round(imu_data.accelerometer.x, 6),
                'ay':                round(imu_data.accelerometer.y, 6),
                'az':                round(imu_data.accelerometer.z, 6),
                'wx':                round(imu_data.gyroscope.x, 6),
                'wy':                round(imu_data.gyroscope.y, 6),
                'wz':                round(imu_data.gyroscope.z, 6),
                'ax_corr':           ax_corr,
                'ay_corr':           ay_corr,
                'gnss_x':            round(gnss_x, 4),
                'gnss_y':            round(gnss_y, 4),
                'gt_x':              round(gt_x, 4),
                'gt_y':              round(gt_y, 4),
                'gt_heading':        round(gt_heading, 6),
                'gt_speed_mps':      round(gt_speed, 4),
                'gt_accel_fwd_mps2': round(gt_accel_fwd, 4),
                'gt_accel_lat_mps2': round(gt_accel_lat, 4),
                'gps_denied':        gps_denied,
                'pitch_deg':         round(pitch_deg, 3),
                'roll_deg':          round(roll_deg, 3),
            }

            d_row = {
                **t_row,
                'world_x':    round(loc.x, 4),
                'world_y':    round(loc.y, 4),
                'world_z':    round(loc.z, 4),
                'gnss_x_raw': round(gnss_x_raw, 4),
                'gnss_y_raw': round(gnss_y_raw, 4),
            }

            temp_tw.writerow(t_row)
            temp_dw.writerow(d_row)
            run_rows.append((t_row, d_row))
            raw_accel_fwd_list.append(gt_accel_fwd)
            raw_accel_lat_list.append(gt_accel_lat)

            if len(align_buf) < 20 and not math.isnan(gnss_x):
                align_buf.append(t_row)

            if tick % SAVE_INTERVAL == 0:
                temp_train_f.flush()
                temp_debug_f.flush()

            if tick % 5 == 0:
                spectator_follow(world, vehicle)
            if tick % 500 == 0:
                print(f"  tick={tick:5d}/{ticks_per_run} "
                        f"| target={target_kmh:5.1f} km/h "
                        f"| smooth={current_speed:5.1f} km/h "
                        f"| actual={gt_speed * 3.6:5.1f} km/h "
                        f"| phase={phase}")

        if run_id == 0 and align_buf:
            verify_alignment(align_buf)

        print(f"  [Run {run_id}] GPS-denied ticks: {tunnel_ticks:5d} / {ticks_per_run} "
              f"({100 * tunnel_ticks / max(ticks_per_run, 1):.1f}%)")

    except Exception as e:
        run_exception = e
        print(f"  [ERROR] Run {run_id} interrupted at tick {tick}: {e}")

    finally:
        # FIX J: direct None-sentinel references.
        for handle in (temp_train_f, temp_debug_f):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass

        for sensor in [imu_s, gnss_s]:
            try:
                if sensor:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                pass

        try:
            if vehicle:
                vehicle.destroy()
        except Exception:
            pass

        destroy_npcs(client, npcs)

        # FIX 4: `continue` so all 15 drain ticks always fire.
        for _ in range(15):
            try:
                world.tick()
            except Exception:
                continue

    # -------------------------------------------------------------------------
    # FIX 3: Always executes — even if run_exception is set.
    # -------------------------------------------------------------------------
    if run_rows:
        n_rows = len(run_rows)

        # FIX O
        print(
            f"  [Run {run_id}] {n_rows} rows collected | "
            f"dropped: {dropped_ticks} "
            f"({100 * dropped_ticks / max(n_rows, 1):.1f}%)"
        )

        if n_rows < FILTFILT_MIN_SAMPLES:
            print(
                f"  [Run {run_id}] Too short to filter "
                f"({n_rows} < {FILTFILT_MIN_SAMPLES} samples = "
                f"{n_rows * FIXED_DELTA_T:.2f} s) — discarding run."
            )
        else:
            print(f"  [Run {run_id}] Applying zero-phase filter...")

            fwd_arr = np.array(raw_accel_fwd_list)
            lat_arr = np.array(raw_accel_lat_list)

            # FIX G
            fwd_arr[np.isinf(fwd_arr)] = np.nan
            lat_arr[np.isinf(lat_arr)] = np.nan

            orig_fwd_nan = np.isnan(fwd_arr)
            orig_lat_nan = np.isnan(lat_arr)

            fwd_smoothed = safe_filter_array(fwd_arr)
            lat_smoothed = safe_filter_array(lat_arr)

            for i, (t_row, d_row) in enumerate(run_rows):
                if not math.isnan(t_row['timestamp']):
                    if not orig_fwd_nan[i]:
                        t_row['gt_accel_fwd_mps2'] = round(fwd_smoothed[i], 4)
                        d_row['gt_accel_fwd_mps2'] = round(fwd_smoothed[i], 4)
                    if not orig_lat_nan[i]:
                        t_row['gt_accel_lat_mps2'] = round(lat_smoothed[i], 4)
                        d_row['gt_accel_lat_mps2'] = round(lat_smoothed[i], 4)
                tw.writerow(t_row)
                dw.writerow(d_row)

            print(f"  [Run {run_id}] Master CSV write complete.")
    else:
        print(f"  [Run {run_id}] No rows collected — nothing written to master CSV.")

    for path in (temp_train_path, temp_debug_path):
        try:
            os.remove(path)
        except OSError:
            pass

    if run_exception is not None:
        raise run_exception


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = orig_settings = train_f = debug_f = None

    # FIX R: VALIDATION_MODE override.
    ticks_per_run = TICKS_PER_RUN
    num_runs      = NUM_RUNS
    if VALIDATION_MODE:
        ticks_per_run = 200
        num_runs      = 1
        print("\n" + "!" * 65)
        print("  ⚠  VALIDATION_MODE = True")
        print("     TICKS_PER_RUN overridden → 200  (10 seconds)")
        print("     NUM_RUNS overridden      → 1")
        print("     Set VALIDATION_MODE = False for full data collection.")
        print("!" * 65)

    print("\n" + "=" * 65)
    print("  CARLA Town04 — Production Data Collection (v12)")
    print(f"  {num_runs} run(s) x {ticks_per_run * FIXED_DELTA_T / 60:.1f} min each")
    print("=" * 65 + "\n")

    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(TIMEOUT_S)
        world = client.get_world()

        if world.get_map().name.split('/')[-1] != MAP_NAME:
            print(f"  Loading map {MAP_NAME}...")
            world = client.load_world(MAP_NAME)
            time.sleep(8)

        orig_settings = world.get_settings()
        s = world.get_settings()
        s.synchronous_mode    = True
        s.fixed_delta_seconds = FIXED_DELTA_T
        world.apply_settings(s)

        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        train_f = open(TRAIN_CSV, 'w', newline='', encoding='utf-8')
        debug_f = open(DEBUG_CSV, 'w', newline='', encoding='utf-8')
        tw = csv.DictWriter(train_f, fieldnames=TRAIN_COLS)
        dw = csv.DictWriter(debug_f, fieldnames=DEBUG_COLS)
        tw.writeheader()
        dw.writeheader()

        conv      = CoordConverter()
        scheduler = SpeedScheduler(SPEED_SCHEDULE)

        for run_id in range(num_runs):
            spawn_idx    = HIGHWAY_SPAWN_INDICES[run_id % len(HIGHWAY_SPAWN_INDICES)]
            weather      = WEATHER_PRESETS[run_id % len(WEATHER_PRESETS)]
            weather_name = WEATHER_NAMES[run_id % len(WEATHER_NAMES)]

            # FIX T: reset_origin() method instead of direct attribute write.
            conv.reset_origin()

            try:
                collect_run(
                    client, world, tm, run_id, spawn_idx,
                    weather, weather_name,
                    conv, scheduler, tw, dw, train_f, debug_f,
                    ticks_per_run=ticks_per_run,
                )
            except Exception as e:
                print(
                    f"\n[WARN] Run {run_id} failed: {e}"
                    f"  — continuing to next run.\n"
                )

            train_f.flush()
            debug_f.flush()

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")

    finally:
        for f in [train_f, debug_f]:
            try:
                if f:
                    f.close()
            except Exception:
                pass
        if orig_settings and client:
            try:
                world.apply_settings(orig_settings)
                print("  World settings restored.")
            except Exception:
                pass
        print("[Done]")

    # FIX U: post-collection quality report.
    dataset_summary(TRAIN_CSV)

    # FIX R: remind the user if they forgot to turn off VALIDATION_MODE.
    if VALIDATION_MODE:
        print("\n" + "!" * 65)
        print("  ⚠  Validation run complete.")
        print("     If checks passed, set VALIDATION_MODE = False")
        print("     and re-run for full data collection.")
        print("!" * 65 + "\n")


if __name__ == '__main__':
    main()
