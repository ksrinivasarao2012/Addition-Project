"""
collect_data.py  —  CARLA Town04 data collection for LSTM training
v4: Correct spawn on highway + geographic tunnel/underpass detection
"""

import sys, os, time, math, csv
from threading import Lock

CARLA_EGG = (
    r'C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor'
    r'\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg'
)
sys.path.insert(0, CARLA_EGG)
sys.path.insert(0, r'C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor\PythonAPI')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carla
from coord_converter import CoordConverter

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOST             = '127.0.0.1'
PORT             = 2000
TIMEOUT_S        = 15.0
MAP_NAME         = 'Town04'
VEHICLE_BP       = 'vehicle.tesla.model3'
TARGET_SPEED_KMH = 60
FIXED_DELTA_T    = 0.05          # 20 Hz
WARMUP_TICKS     = 80
MAX_TICKS        = 24_000        # 20 minutes
SAVE_INTERVAL    = 200

# ── SPAWN: Index 16 confirmed on main highway (road_id=35, speed_limit=90) ───
FORCED_SPAWN_INDEX = 16
# world = (389.16, -185.33, 0.28)

# ── TUNNEL DETECTION (geographic bounds) ─────────────────────────────────────
# Town04 underpass: highway passes under elevated structure at z~11m
# Elevated spawn points 53-60 are at x: -105 to +113, y: -4 to +47
# Highway driving under them is our GPS-denied zone
# Using slightly wider bounds to catch full underpass passage
TUNNEL_X_MIN = -130.0
TUNNEL_X_MAX =  140.0
TUNNEL_Y_MIN =  -35.0
TUNNEL_Y_MAX =   65.0

OUTPUT_DIR = r'C:\Users\heman\Music\rl_imu_project\data'
TRAIN_CSV  = os.path.join(OUTPUT_DIR, 'town04_collected.csv')
DEBUG_CSV  = os.path.join(OUTPUT_DIR, 'town04_debug.csv')

TRAIN_COLS = ['timestamp','ax','ay','az','wx','wy','wz',
              'gnss_x','gnss_y','gt_x','gt_y','gt_z','speed_mps','gps_denied']
DEBUG_COLS = TRAIN_COLS + ['world_x','world_y','world_z']


# ── Sensor holder ─────────────────────────────────────────────────────────────
class SensorData:
    def __init__(self):
        self.lock = Lock()
        self.ax=self.ay=self.az=0.0
        self.wx=self.wy=self.wz=0.0
        self.lat=self.lon=0.0
        self.imu_ready=self.gnss_ready=False

    def on_imu(self, d):
        with self.lock:
            self.ax,self.ay,self.az = (d.accelerometer.x,
                                       d.accelerometer.y,
                                       d.accelerometer.z)
            self.wx,self.wy,self.wz = (d.gyroscope.x,
                                       d.gyroscope.y,
                                       d.gyroscope.z)
            self.imu_ready = True

    def on_gnss(self, d):
        with self.lock:
            self.lat,self.lon = d.latitude, d.longitude
            self.gnss_ready = True

    def snap(self):
        with self.lock:
            return dict(ax=self.ax, ay=self.ay, az=self.az,
                        wx=self.wx, wy=self.wy, wz=self.wz,
                        lat=self.lat, lon=self.lon,
                        imu_ready=self.imu_ready,
                        gnss_ready=self.gnss_ready)


def get_speed(v):
    vel = v.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def in_tunnel(world_x, world_y):
    """
    Returns True when vehicle is in the Town04 underpass zone.
    The highway passes under an elevated structure in this x/y region.
    This is where we simulate GPS denial.
    """
    return (TUNNEL_X_MIN <= world_x <= TUNNEL_X_MAX and
            TUNNEL_Y_MIN <= world_y <= TUNNEL_Y_MAX)


def spectator_follow(world, vehicle):
    t  = vehicle.get_transform()
    yr = math.radians(t.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(
            x=t.location.x - 8*math.cos(yr),
            y=t.location.y - 8*math.sin(yr),
            z=t.location.z + 4),
        carla.Rotation(pitch=-15, yaw=t.rotation.yaw)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = vehicle = imu_s = gnss_s = None
    orig_settings = train_f = debug_f = None

    try:
        # ── Connect ───────────────────────────────────────────────────────────
        print("[CARLA] Connecting ...")
        client = carla.Client(HOST, PORT)
        client.set_timeout(TIMEOUT_S)
        print(f"[CARLA] Connected — server {client.get_server_version()}")

        # ── Load map ──────────────────────────────────────────────────────────
        world = client.get_world()
        if world.get_map().name.split('/')[-1] != MAP_NAME:
            print(f"[CARLA] Loading {MAP_NAME} ...")
            world = client.load_world(MAP_NAME)
            time.sleep(6)
        print(f"[CARLA] Map: {world.get_map().name}")

        # ── Sync mode ─────────────────────────────────────────────────────────
        orig_settings = world.get_settings()
        s = world.get_settings()
        s.synchronous_mode    = True
        s.fixed_delta_seconds = FIXED_DELTA_T
        world.apply_settings(s)

        # ── Spawn on highway (index 16) ───────────────────────────────────────
        bpl       = world.get_blueprint_library()
        bp        = bpl.find(VEHICLE_BP)
        bp.set_attribute('role_name', 'hero')
        spawn_pts = world.get_map().get_spawn_points()
        sp        = spawn_pts[FORCED_SPAWN_INDEX]

        vehicle = world.try_spawn_actor(bp, sp)
        if vehicle is None:
            # Try neighbours if 16 is blocked
            for fallback in [15, 17, 18, 19, 20, 21]:
                vehicle = world.try_spawn_actor(bp, spawn_pts[fallback])
                if vehicle:
                    sp = spawn_pts[fallback]
                    print(f"[Spawn] Index 16 blocked, using {fallback}")
                    break
        if vehicle is None:
            raise RuntimeError("Cannot spawn vehicle")

        print(f"[Spawn] Index {FORCED_SPAWN_INDEX}: "
              f"world=({sp.location.x:.2f}, {sp.location.y:.2f}, "
              f"{sp.location.z:.2f})  road_id=35  speed_limit=90")

        # ── Attach sensors ────────────────────────────────────────────────────
        sd = SensorData()

        imu_bp = bpl.find('sensor.other.imu')
        for k, v in [('noise_accel_stddev_x','0.02'),
                     ('noise_accel_stddev_y','0.02'),
                     ('noise_accel_stddev_z','0.02'),
                     ('noise_gyro_stddev_x', '0.005'),
                     ('noise_gyro_stddev_y', '0.005'),
                     ('noise_gyro_stddev_z', '0.005'),
                     ('sensor_tick', str(FIXED_DELTA_T))]:
            imu_bp.set_attribute(k, v)
        imu_s = world.spawn_actor(
            imu_bp, carla.Transform(carla.Location(z=0.5)),
            attach_to=vehicle)
        imu_s.listen(sd.on_imu)

        gnss_bp = bpl.find('sensor.other.gnss')
        for k, v in [('noise_lat_stddev', '0.000005'),
                     ('noise_lon_stddev', '0.000005'),
                     ('sensor_tick', str(FIXED_DELTA_T))]:
            gnss_bp.set_attribute(k, v)
        gnss_s = world.spawn_actor(
            gnss_bp, carla.Transform(carla.Location(z=0.5)),
            attach_to=vehicle)
        gnss_s.listen(sd.on_gnss)

        # ── Manual throttle kick ──────────────────────────────────────────────
        print("\n[Drive] Manual throttle kick (2 sec) ...")
        vehicle.set_autopilot(False)
        for i in range(40):
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.7, steer=0.0, brake=0.0, hand_brake=False))
            world.tick()
            if i % 10 == 0:
                print(f"  kick {i:2d}/40 | "
                      f"speed={get_speed(vehicle)*3.6:.1f} km/h")
        print(f"[Drive] Speed after kick: {get_speed(vehicle)*3.6:.1f} km/h")

        # ── Traffic Manager ───────────────────────────────────────────────────
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        vehicle.set_autopilot(True, 8000)
        tm.ignore_lights_percentage(vehicle, 100.0)
        tm.ignore_signs_percentage(vehicle, 100.0)
        tm.global_percentage_speed_difference(-30.0)
        tm.set_desired_speed(vehicle, TARGET_SPEED_KMH)
        tm.auto_lane_change(vehicle, True)
        tm.keep_right_rule_percentage(vehicle, 100.0)
        print(f"[TM] Autopilot active — target {TARGET_SPEED_KMH} km/h")

        # ── Warm-up ───────────────────────────────────────────────────────────
        print(f"[Warmup] {WARMUP_TICKS} ticks ...")
        for i in range(WARMUP_TICKS):
            world.tick()
            if i % 20 == 0:
                loc_w = vehicle.get_transform().location
                tun   = "IN TUNNEL" if in_tunnel(loc_w.x, loc_w.y) else "open road"
                print(f"  warmup {i:3d}/{WARMUP_TICKS} | "
                      f"speed={get_speed(vehicle)*3.6:.1f} km/h | "
                      f"world=({loc_w.x:.1f},{loc_w.y:.1f}) | {tun}")

        # ── Coordinate origin ─────────────────────────────────────────────────
        snap = sd.snap()
        if not snap['gnss_ready']:
            raise RuntimeError("GNSS not ready after warmup")
        conv = CoordConverter()
        conv.set_origin(snap['lat'], snap['lon'])
        spawn_loc = vehicle.get_transform().location
        print(f"\n[Origin] Recording origin = "
              f"world({spawn_loc.x:.2f}, {spawn_loc.y:.2f}, {spawn_loc.z:.2f})")
        print(f"[Check]  Speed = {get_speed(vehicle)*3.6:.1f} km/h")
        print(f"[Tunnel] GPS denied when world_x in [{TUNNEL_X_MIN}, {TUNNEL_X_MAX}]"
              f" AND world_y in [{TUNNEL_Y_MIN}, {TUNNEL_Y_MAX}]")

        # ── Open CSV files ────────────────────────────────────────────────────
        train_f = open(TRAIN_CSV, 'w', newline='')
        debug_f = open(DEBUG_CSV, 'w', newline='')
        tw = csv.DictWriter(train_f, fieldnames=TRAIN_COLS)
        dw = csv.DictWriter(debug_f, fieldnames=DEBUG_COLS)
        tw.writeheader()
        dw.writeheader()

        # ── Recording loop ────────────────────────────────────────────────────
        print("\n" + "="*62)
        print(f"  Recording {MAX_TICKS} ticks "
              f"({MAX_TICKS*FIXED_DELTA_T/60:.0f} min) — Ctrl-C to stop early")
        print(f"  Tunnel zone: x[{TUNNEL_X_MIN},{TUNNEL_X_MAX}] "
              f"y[{TUNNEL_Y_MIN},{TUNNEL_Y_MAX}]")
        print("="*62 + "\n")

        tick = tunnel_ticks = 0
        t0   = time.time()
        first_tunnel_tick = None

        while tick < MAX_TICKS:
            world.tick()
            tick += 1

            snap = sd.snap()
            loc  = vehicle.get_transform().location

            # Ground truth in local frame
            gt_x = loc.x - spawn_loc.x
            gt_y = loc.y - spawn_loc.y
            gt_z = loc.z - spawn_loc.z

            # GNSS in local frame
            gnss_x, gnss_y = conv.gnss_to_local(snap['lat'], snap['lon'])

            speed      = get_speed(vehicle)
            gps_denied = 1 if in_tunnel(loc.x, loc.y) else 0

            if gps_denied:
                tunnel_ticks += 1
                if first_tunnel_tick is None:
                    first_tunnel_tick = tick
                    print(f"\n  *** TUNNEL ENTERED at tick={tick} "
                          f"t={tick*FIXED_DELTA_T:.1f}s "
                          f"world=({loc.x:.1f},{loc.y:.1f}) ***\n")

            ts  = round(tick * FIXED_DELTA_T, 4)
            row = dict(
                timestamp  = ts,
                ax=round(snap['ax'],6), ay=round(snap['ay'],6),
                az=round(snap['az'],6),
                wx=round(snap['wx'],6), wy=round(snap['wy'],6),
                wz=round(snap['wz'],6),
                gnss_x=round(gnss_x,4), gnss_y=round(gnss_y,4),
                gt_x  =round(gt_x,4),   gt_y  =round(gt_y,4),
                gt_z  =round(gt_z,4),
                speed_mps =round(speed,4),
                gps_denied=gps_denied,
            )
            tw.writerow(row)
            dw.writerow({**row,
                         'world_x': round(loc.x, 4),
                         'world_y': round(loc.y, 4),
                         'world_z': round(loc.z, 4)})

            if tick % SAVE_INTERVAL == 0:
                train_f.flush()
                debug_f.flush()

            if tick % 5 == 0:
                spectator_follow(world, vehicle)

            if tick % 400 == 0:
                t_pct = 100 * tunnel_ticks / tick
                icon  = "🚇" if gps_denied else "🛣 "
                print(f"  {icon} tick={tick:5d} | t={ts:6.1f}s | "
                      f"spd={speed*3.6:4.1f}km/h | "
                      f"world=({loc.x:7.1f},{loc.y:7.1f}) | "
                      f"tunnel={t_pct:4.1f}% | "
                      f"elapsed={time.time()-t0:.0f}s")

        print(f"\n[Done] {tick} ticks = {tick*FIXED_DELTA_T:.1f}s")
        _print_summary(tunnel_ticks, tick)

    except KeyboardInterrupt:
        print("\n[Stopped] Saving ...")
        try:
            _print_summary(tunnel_ticks, tick)
        except Exception:
            pass

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        for f in [train_f, debug_f]:
            try:
                if f: f.flush(); f.close()
            except Exception:
                pass
        print(f"[Saved] {TRAIN_CSV}")
        print(f"[Saved] {DEBUG_CSV}")
        for actor in [imu_s, gnss_s, vehicle]:
            try:
                if actor: actor.destroy()
            except Exception:
                pass
        if orig_settings:
            try:
                world.apply_settings(orig_settings)
            except Exception:
                pass
        print("[Cleanup] Done.")


def _print_summary(tunnel_ticks, total_ticks):
    if total_ticks == 0:
        return
    t_pct = 100 * tunnel_ticks / total_ticks
    print(f"\n{'='*50}")
    print(f"  Total ticks   : {total_ticks}")
    print(f"  Tunnel ticks  : {tunnel_ticks} ({t_pct:.1f}%)")
    print(f"  Data duration : {total_ticks * FIXED_DELTA_T:.1f}s")
    if tunnel_ticks == 0:
        print("\n  [WARN] No tunnel data collected!")
        print("  [WARN] The vehicle did not pass through the tunnel zone.")
        print("  [WARN] Check world coordinates in town04_debug.csv")
        print("  [WARN] Look for rows where world_x is near 0 and world_y near 0-50")
        print("  [WARN] Update TUNNEL_X_MIN/MAX and TUNNEL_Y_MIN/MAX accordingly")
    else:
        print(f"\n  [OK] Tunnel zone was detected correctly.")
        print(f"  [OK] Data is ready for LSTM training.")
    print('='*50)


if __name__ == '__main__':
    main()
