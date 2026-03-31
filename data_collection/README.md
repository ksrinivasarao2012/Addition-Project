# Step 1 — CARLA Data Collection

## What this does
Drives a Tesla Model 3 autonomously around Town04 for ~20 minutes.
Logs synchronized IMU + GNSS + Ground Truth every 0.05 seconds (20 Hz).
Automatically detects the underground tunnel and flags those rows as GPS-denied.

## Output
| File | Purpose |
|---|---|
| `data/town04_collected.csv` | Clean training data for LSTM |
| `data/town04_debug.csv` | Full log with CARLA world coordinates |

## Columns
| Column | Description |
|---|---|
| timestamp | Seconds from recording start |
| ax, ay, az | Linear acceleration m/s² (IMU, noisy) |
| wx, wy, wz | Angular velocity rad/s (IMU, noisy) |
| gnss_x, gnss_y | GNSS position in local frame (metres) |
| gt_x, gt_y, gt_z | Ground truth position in local frame (metres) |
| speed_mps | Vehicle speed m/s |
| gps_denied | 1 = inside tunnel, 0 = open road |

## How to run

### Step 1 — Launch CARLA
```
cd C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600
```
Wait ~30 seconds until the CARLA window fully loads.

### Step 2 — Install requirements
```
cd C:\Users\heman\Music\rl_imu_project
carla_env37\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Run collection
```
python data_collection\collect_data.py
```

### Step 4 — Stop when ready
Press Ctrl-C at any time. Data collected so far is saved automatically.
Minimum recommended: 10 minutes (12,000 ticks) so the vehicle passes
through the tunnel multiple times.

## What to watch in console
```
🛣  tick=  400 | t=  20.0s | speed=40.1km/h | pos=(  45.2,  -12.3) | z= 0.12 | tunnel= 0.0% | elapsed=22s
🚇  tick=  800 | t=  40.0s | speed=38.7km/h | pos=( -12.1,  -89.4) | z=-3.21 | tunnel=18.2% | elapsed=44s
```
- `🚇` = vehicle inside tunnel (gps_denied=1)
- `🛣` = vehicle on open road (gps_denied=0)
- z values below -1.5 confirm tunnel detection is working

## Troubleshooting

### "Connection refused" error
CARLA is not running or still loading. Wait and retry.

### "No spawn points found"
Town04 did not load correctly. Restart CARLA and try again.

### gps_denied never shows 1 (no tunnel detected)
The tunnel Z threshold needs adjustment for your route.
Open `data/town04_debug.csv` and look at the `world_z` column.
Find rows where the vehicle is visibly in the tunnel and note the z value.
Update `TUNNEL_Z_THRESHOLD` in `collect_data.py` to match.

### Vehicle not moving
Traffic Manager connection issue.
Ensure CARLA launched without -carla-rpc-port flag conflicts.

## Collected data looks good when
- At least 5-10% of rows have gps_denied=1
- speed_mps is consistently 8-12 m/s (30-40 km/h)
- gnss_x/y and gt_x/y are close (within 1-2m) on open road
- gnss_x/y stops updating (stays constant) inside tunnel rows
  → this is expected; LSTM will learn to compensate for this gap
