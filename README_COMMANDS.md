# RL-Adaptive EKF Localization — Run Guide

## File Structure

```
C:\Users\heman\Music\rl_imu_project\
├── ekf.py
├── rl_agent.py
├── rl_train.py
├── requirements.txt
├── data_collection\
│   └── collect_data.py
├── lstm\
│   └── train_lstm.py
├── carla_implementation\
│   ├── carla_config.py
│   ├── carla_sensor_bridge.py
│   ├── carla_rl_environment.py
│   ├── train_carla.py
│   └── evaluate_carla.py
├── data\
├── models\
├── results\
└── logs\
```

---

## One-time Setup

```cmd
cd C:\Users\heman\Music\rl_imu_project
carla_env37\Scripts\activate
pip install -r requirements.txt
pip install C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-cp37-cp37m-win_amd64.whl
```

---

## Step 1 — Collect Data

**Terminal 1 — start CARLA:**
```cmd
cd C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600
```
Wait ~30 seconds for CARLA to fully load.

**Terminal 2 — run collection:**
```cmd
cd C:\Users\heman\Music\rl_imu_project
carla_env37\Scripts\activate
python data_collection\collect_data.py
```
Runs 4 × 5 min = ~20 min total. Outputs:
- `data\town04_dataset.csv`
- `data\town04_debug.csv`

---

## Step 2 — Train LSTM

CARLA does **not** need to be running for this step.

```cmd
cd C:\Users\heman\Music\rl_imu_project
carla_env37\Scripts\activate
python lstm\train_lstm.py
```
Runs ~150 epochs. Outputs:
- `models\lstm_drift_predictor.pth`
- `models\lstm_normalisation.npz`
- `results\lstm_training.png`
- `results\lstm_metrics.txt`

---

## Step 3 — Run EKF Offline Evaluation

CARLA does **not** need to be running for this step.

```cmd
cd C:\Users\heman\Music\rl_imu_project
carla_env37\Scripts\activate

# Normal evaluation (Baseline vs LSTM-EKF on run 3)
python ekf.py

# Self-test (no data needed — runs in 2 seconds)
python ekf.py --test
```
Outputs:
- `results\ekf_run3.png`
- `results\ekf_summary.png`
- `results\ekf_metrics.txt`
- `results\ekf_predictions.csv`

---

## Step 4 — RL Adaptive Filter Training

**Terminal 1 — start CARLA:**
```cmd
cd C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600
```

**Terminal 2 — run RL training:**
```cmd
cd C:\Users\heman\Music\rl_imu_project
carla_env37\Scripts\activate

# Standard run (150 episodes)
python rl_train.py

# More episodes
python rl_train.py --episodes 200

# Headless (faster, no CARLA window)
python rl_train.py --no-render

# Resume from checkpoint
python rl_train.py --resume models\best_carla_model.pth
```
Outputs:
- `models\best_carla_model.pth`
- `models\latest_carla_model.pth`
- `logs\carla_training_log.csv`
- `results\training_final.png`

---

## Step 5 — Evaluate RL vs Static EKF

**Terminal 1 — CARLA must be running** (same as Step 4).

**Terminal 2:**
```cmd
cd C:\Users\heman\Music\rl_imu_project\carla_implementation
carla_env37\Scripts\activate

# Default evaluation (5 episodes each)
python evaluate_carla.py

# More episodes, specific model
python evaluate_carla.py --model ..\models\best_carla_model.pth --episodes 10
```
Outputs:
- `results\carla_comparison.png`

---

## Quick Reference

| Step | Script | CARLA needed? | Runtime |
|------|--------|--------------|---------|
| 1 — Collect data | `data_collection\collect_data.py` | Yes | ~20 min |
| 2 — Train LSTM | `lstm\train_lstm.py` | No | ~10 min |
| 3 — EKF eval | `ekf.py` | No | ~1 min |
| 3 — EKF self-test | `ekf.py --test` | No | 2 sec |
| 4 — RL training | `rl_train.py` | Yes | ~2 hr |
| 5 — Final eval | `carla_implementation\evaluate_carla.py` | Yes | ~15 min |

---

## Troubleshooting

**"Cannot connect to CARLA"** — Wait longer after launching CarlaUE4.exe, or check port 2000 is free:
```cmd
netstat -an | findstr 2000
```

**"No module named carla"** — Re-run the pip install from the setup step above.

**"No module named ekf"** — Make sure you are running from `rl_imu_project\`, not a subdirectory.

**CARLA crashes on start** — Add `-RenderOffScreen` flag:
```cmd
CarlaUE4.exe -RenderOffScreen -quality-level=Low -opengl
```

**Low speed / vehicle stuck** — Reduce `TARGET_SPEED` in `carla_implementation\carla_config.py`.


### demo file running
# Terminal 1
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

# Terminal 2
cd C:\Users\heman\Music\rl_imu_project
carla_env37\Scripts\activate
python demo.py

optional flags:
python demo.py --steps 800      # longer run, more tunnel passes
python demo.py --no-rl          # show LSTM-EKF without RL (simpler demo)
python demo.py --no-render      # hide CARLA window, only show dashboard
