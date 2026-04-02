# =============================================================================
# carla_config.py
# Central configuration for CARLA + RL-EKF Integration
# All tunable parameters live here — edit this file to customize behavior
#
# PATHS are absolute using PROJECT_ROOT derived from this file's location.
# Previously MODEL_DIR = "models/" was relative to CWD, which resolved
# correctly from rl_imu_project\ but WRONG from carla_implementation\.
# =============================================================================

import os

# This file lives in:  rl_imu_project\carla_implementation\carla_config.py
# PROJECT_ROOT is:     rl_imu_project\
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

# -----------------------------------------------------------------------------
# CARLA SERVER CONNECTION
# -----------------------------------------------------------------------------
CARLA_HOST = "localhost"          # CARLA server IP (localhost = same machine)
CARLA_PORT = 2000                 # Default CARLA port
CARLA_TIMEOUT = 10.0              # Seconds to wait for server response
CARLA_TOWN = "Town04"             # Town04 has a highway tunnel — perfect for us!
                                  # Alternatives: Town01 (simple), Town05 (urban)

# -----------------------------------------------------------------------------
# SIMULATION SETTINGS
# -----------------------------------------------------------------------------
SYNC_MODE = True                  # Synchronous mode = deterministic, better for RL
FIXED_DELTA_SECONDS = 0.05        # Simulation timestep (20 Hz) — matches IMU rate
RENDER_MODE = True                # Set False for headless/faster training

# -----------------------------------------------------------------------------
# VEHICLE SETTINGS
# -----------------------------------------------------------------------------
VEHICLE_BLUEPRINT = "vehicle.tesla.model3"   # Vehicle to spawn
SPAWN_POINT_INDEX = 0                         # Which spawn point to use (0 = default)

# Autopilot route settings
USE_AUTOPILOT = True              # Let CARLA drive the vehicle (we control EKF, not driving)
TARGET_SPEED = 30                 # km/h target speed for autopilot (slow = realistic + visible)

# -----------------------------------------------------------------------------
# IMU SENSOR CONFIGURATION
# These match realistic automotive-grade IMU specs
# -----------------------------------------------------------------------------
IMU_CONFIG = {
    "noise_accel_stddev_x": 0.01,   # Accelerometer noise (m/s²) — automotive grade
    "noise_accel_stddev_y": 0.01,
    "noise_accel_stddev_z": 0.01,
    "noise_gyro_bias_x":    0.001,  # Gyroscope bias (rad/s) — realistic drift
    "noise_gyro_bias_y":    0.001,
    "noise_gyro_bias_z":    0.001,
    "noise_gyro_stddev_x":  0.005,  # Gyroscope noise (rad/s)
    "noise_gyro_stddev_y":  0.005,
    "noise_gyro_stddev_z":  0.005,
    "sensor_tick":          "0.05", # 20 Hz IMU update rate
}

# -----------------------------------------------------------------------------
# GNSS (GPS) SENSOR CONFIGURATION
# -----------------------------------------------------------------------------
GNSS_CONFIG = {
    "noise_lat_bias":   0.0,        # GPS bias in latitude degrees
    "noise_lon_bias":   0.0,        # GPS bias in longitude degrees
    "noise_lat_stddev": 0.00001,    # ~1m position noise (realistic civilian GPS)
    "noise_lon_stddev": 0.00001,
    "sensor_tick":      "0.1",      # 10 Hz GPS update rate
}

# -----------------------------------------------------------------------------
# GPS DENIAL ZONES
# FIX: Coordinates here are LOCAL EKF frame coords (relative to spawn point),
# matching collect_data.py which uses TUNNEL_X_MIN/MAX, TUNNEL_Y_MIN/MAX.
# The old values (-400,-200,10,60) were raw CARLA world coordinates —
# but GPSDenialManager._check_zones() receives LOCAL coords from carla_to_local().
# These corrected bounds match collect_data.py's tunnel detection exactly so
# gps_denied=1 fires at the same positions in both training and RL evaluation.
# Format: (x_min, x_max, y_min, y_max) in LOCAL EKF frame (metres from spawn)
# -----------------------------------------------------------------------------
GPS_DENIAL_ZONES = [
    # Town04 main tunnel — matches TUNNEL_X/Y in collect_data.py
    (-130.0, 140.0, -35.0, 65.0),
]

# How to determine GPS denial:
# "zone"    = purely coordinate-based (above)
# "tunnel"  = use CARLA's OpenDRIVE tunnel detection (z < -1.0)
# "both"    = use both methods (recommended — belt and braces)
GPS_DENIAL_METHOD = "both"

# -----------------------------------------------------------------------------
# RL TRAINING PARAMETERS
# -----------------------------------------------------------------------------
NUM_EPISODES     = 150            # Total training episodes
MAX_STEPS        = 500            # Max steps per episode (500 × 0.05s = 25 seconds per episode)
EVAL_INTERVAL    = 10             # Evaluate every N episodes
SAVE_INTERVAL    = 25             # Save model checkpoint every N episodes
WARMUP_EPISODES  = 5              # Episodes before RL starts (pure EKF warmup)

# Episode reset behavior
RANDOMIZE_SPAWN  = True           # Random spawn point each episode (better generalization)
RANDOMIZE_WEATHER = True          # Random weather conditions

# Weather presets to cycle through during training
WEATHER_PRESETS = [
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "HardRainNoon",
    "ClearSunset",
]

# -----------------------------------------------------------------------------
# COORDINATE SYSTEM
# CARLA uses (x, y, z) in meters. We project to local 2D (x, y) for the EKF.
# Reference point = vehicle spawn location (set at runtime)
# -----------------------------------------------------------------------------
USE_LOCAL_COORDS = True           # Convert CARLA world coords to local EKF coords

# -----------------------------------------------------------------------------
# FILE PATHS — absolute, correct regardless of which directory you run from
# -----------------------------------------------------------------------------
MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_DIR     = os.path.join(PROJECT_ROOT, "logs")

BEST_MODEL_PATH     = os.path.join(MODEL_DIR, "best_carla_model.pth")
LATEST_MODEL_PATH   = os.path.join(MODEL_DIR, "latest_carla_model.pth")
TRAINING_LOG_PATH   = os.path.join(LOG_DIR,   "carla_training_log.csv")

# LSTM model files — produced by train_lstm.py, consumed by ekf.py LSTMBridge
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_drift_predictor.pth")
LSTM_STATS_PATH = os.path.join(MODEL_DIR, "lstm_normalisation.npz")

# Gravity constant for IMU correction (matches collect_data.py G = 9.81)
G = 9.81  # m/s²

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
PLOT_UPDATE_INTERVAL = 5          # Update live plots every N episodes
SAVE_TRAJECTORY_VIDEO = False     # Save episode trajectories as video (slow)
SHOW_CARLA_WINDOW    = True       # Show CARLA rendering window during training
