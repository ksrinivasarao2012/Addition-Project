# =============================================================================
# carla_sensor_bridge.py
# Replaces simulator.py — connects to CARLA and streams real sensor data
#
# This module handles ALL CARLA communication:
#   - Server connection + world setup
#   - Vehicle spawning and autopilot
#   - IMU sensor (accelerometer + gyroscope)
#   - GNSS sensor (GPS with realistic noise)
#   - GPS denial zone detection (tunnels + custom zones)
#   - Coordinate conversion (CARLA world → local EKF frame)
#   - Ground truth extraction for training reward
# =============================================================================

import carla
import numpy as np
import math
import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from carla_config import *

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("CARLABridge")


# =============================================================================
# DATA CLASSES  — clean containers for sensor readings
# =============================================================================

@dataclass
class IMUReading:
    """Raw IMU data from CARLA IMU sensor."""
    timestamp:    float           # Simulation time (seconds)
    accel_x:      float           # Accelerometer X (m/s²) — forward
    accel_y:      float           # Accelerometer Y (m/s²) — lateral
    accel_z:      float           # Accelerometer Z (m/s²) — vertical
    gyro_x:       float           # Gyroscope X (rad/s) — roll rate
    gyro_y:       float           # Gyroscope Y (rad/s) — pitch rate
    gyro_z:       float           # Gyroscope Z (rad/s) — yaw rate

    @property
    def forward_accel(self) -> float:
        """Acceleration in forward direction (relevant for 2D EKF)."""
        return self.accel_x

    @property
    def yaw_rate(self) -> float:
        """Yaw rate (rotation about Z axis) — key for heading estimation."""
        return self.gyro_z


@dataclass
class GNSSReading:
    """Raw GNSS (GPS) data from CARLA GNSS sensor."""
    timestamp:  float             # Simulation time (seconds)
    latitude:   float             # Degrees
    longitude:  float             # Degrees
    altitude:   float             # Meters
    valid:      bool = True       # False when inside GPS denial zone


@dataclass
class GroundTruth:
    """Vehicle ground truth from CARLA — used for reward computation only."""
    timestamp:  float
    x:          float             # World X position (meters, local frame)
    y:          float             # World Y position (meters, local frame)
    heading:    float             # Yaw angle (radians)
    velocity:   float             # Forward speed (m/s)
    in_tunnel:  bool = False      # Whether vehicle is in a tunnel


@dataclass
class SensorBundle:
    """All sensor data for one timestep — what the EKF receives."""
    imu:          IMUReading
    gnss:         Optional[GNSSReading]   # None when GPS denied
    ground_truth: GroundTruth
    step:         int
    gps_denied:   bool


# =============================================================================
# COORDINATE CONVERTER  — CARLA world coords → local EKF frame
# =============================================================================

class CoordinateConverter:
    """
    Converts CARLA's world coordinate system to a local flat 2D frame
    centered at the vehicle's spawn point. This makes EKF state variables
    start near zero and remain numerically well-conditioned.

    CARLA uses:  X = East, Y = South (left-handed), Z = Up
    EKF uses:    X = East, Y = North (standard NED-like), theta = heading
    """

    def __init__(self):
        self.origin_x: Optional[float] = None
        self.origin_y: Optional[float] = None
        self.origin_yaw: float = 0.0
        # GNSS reference point (set at spawn — lat/lon at origin)
        self.gnss_ref_lat: Optional[float] = None
        self.gnss_ref_lon: Optional[float] = None

    def set_origin(self, carla_transform: carla.Transform):
        """Call once after vehicle spawns to set local coordinate origin."""
        self.origin_x   = carla_transform.location.x
        self.origin_y   = carla_transform.location.y
        self.origin_yaw = math.radians(carla_transform.rotation.yaw)
        log.info(f"Coordinate origin set: ({self.origin_x:.2f}, {self.origin_y:.2f})")

    def set_gnss_origin(self, lat: float, lon: float):
        """Call once after first GNSS reading at spawn to set GPS reference."""
        self.gnss_ref_lat = lat
        self.gnss_ref_lon = lon

    def carla_to_local(self, x: float, y: float) -> Tuple[float, float]:
        """Convert CARLA world (x,y) to local EKF frame (x,y)."""
        if self.origin_x is None:
            return x, y
        # Translate
        dx = x - self.origin_x
        dy = y - self.origin_y
        # CARLA Y is inverted (left-handed system) → flip Y
        dy = -dy
        return dx, dy

    def carla_yaw_to_heading(self, carla_yaw_deg: float) -> float:
        """
        Convert CARLA yaw (degrees, clockwise from North) to
        standard heading (radians, counterclockwise from East).
        """
        # CARLA yaw: 0=East, 90=South, -90=North (clockwise)
        # Standard: 0=East, pi/2=North (counterclockwise)
        heading = -math.radians(carla_yaw_deg)
        # Normalize to [-pi, pi]
        return math.atan2(math.sin(heading), math.cos(heading))

    def gnss_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert GPS lat/lon to local EKF (x,y).
        Uses the GNSS reading at spawn as reference so output starts at (0,0)
        and matches the EKF state which is also seeded at (0,0) relative coords.
        """
        if self.gnss_ref_lat is None:
            # Reference not set yet — return zero
            return 0.0, 0.0

        EARTH_RADIUS = 6_371_000  # meters
        ref_lat = self.gnss_ref_lat
        ref_lon = self.gnss_ref_lon

        # Equirectangular projection relative to spawn GNSS reading
        dx =  EARTH_RADIUS * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
        dy =  EARTH_RADIUS * math.radians(lat - ref_lat)

        # CARLA GNSS: increasing lat = North = positive Y in EKF
        # CARLA GNSS: increasing lon = East  = positive X in EKF
        return float(dx), float(dy)


# =============================================================================
# GPS DENIAL MANAGER
# =============================================================================

class GPSDenialManager:
    """
    Determines whether the vehicle currently has GPS reception.
    Uses two methods:
      1. Zone-based: predefined rectangular zones from config
      2. Tunnel detection: reads CARLA's OpenDRIVE road topology
    """

    def __init__(self, world: carla.World):
        self.world  = world
        self.zones  = GPS_DENIAL_ZONES
        self.method = GPS_DENIAL_METHOD
        self._tunnel_cache: dict = {}   # Cache tunnel checks (expensive)

    def is_gps_denied(self, x: float, y: float,
                      carla_location: carla.Location) -> bool:
        """
        Returns True if the vehicle should not receive GPS at this position.
        x, y are local EKF coordinates.
        carla_location is the raw CARLA world location.
        """
        if self.method in ("zone", "both"):
            if self._check_zones(x, y):
                return True

        if self.method in ("tunnel", "both"):
            if self._check_tunnel(carla_location):
                return True

        return False

    def _check_zones(self, x: float, y: float) -> bool:
        """Check if (x,y) falls inside any configured denial zone."""
        for (xmin, xmax, ymin, ymax) in self.zones:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

    def _check_tunnel(self, location: carla.Location) -> bool:
        """
        Check if the vehicle is inside a CARLA tunnel using waypoint topology.
        Waypoints inside tunnels have is_junction=False and are underground.
        We cache results by grid cell for performance.
        """
        # Round to 5m grid for cache key
        key = (round(location.x / 5) * 5, round(location.y / 5) * 5)

        if key not in self._tunnel_cache:
            try:
                map_ = self.world.get_map()
                wp   = map_.get_waypoint(location,
                                         project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
                # Town04's tunnel waypoints have z < -1.0 (underground)
                in_tunnel = (wp is not None and location.z < -1.0)
                self._tunnel_cache[key] = in_tunnel
            except Exception:
                self._tunnel_cache[key] = False

        return self._tunnel_cache[key]


# =============================================================================
# CARLA SENSOR BRIDGE  — main class
# =============================================================================

class CARLASensorBridge:
    """
    Main interface between CARLA and the RL-EKF system.

    Usage:
        bridge = CARLASensorBridge()
        bridge.connect()
        bridge.spawn_vehicle()

        for step in range(MAX_STEPS):
            bundle = bridge.get_sensor_bundle()
            # feed bundle.imu and bundle.gnss to your EKF
            # use bundle.ground_truth for reward

        bridge.destroy()

    Thread safety:
        Sensor callbacks run on CARLA's thread.
        Data is passed via thread-safe queues.
    """

    def __init__(self):
        # CARLA objects
        self.client:  Optional[carla.Client] = None
        self.world:   Optional[carla.World]  = None
        self.vehicle: Optional[carla.Actor]  = None
        self.sensors: List[carla.Actor]      = []

        # Sensor data queues (thread-safe)
        self._imu_queue:  queue.Queue = queue.Queue(maxsize=10)
        self._gnss_queue: queue.Queue = queue.Queue(maxsize=10)

        # Helper classes
        self.coord_conv  = CoordinateConverter()
        self.gps_manager: Optional[GPSDenialManager] = None

        # State tracking
        self.step_count: int   = 0
        self.origin_set: bool  = False
        self._latest_imu:  Optional[IMUReading]  = None
        self._latest_gnss: Optional[GNSSReading] = None
        self._last_gt:     Optional[GroundTruth]  = None

        # Performance tracking
        self._imu_count:  int = 0
        self._gnss_count: int = 0

    # ------------------------------------------------------------------
    # CONNECTION & SETUP
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Connect to the CARLA server.
        Make sure CarlaUE4.exe is running before calling this.

        Returns True on success, False on failure.
        """
        log.info(f"Connecting to CARLA at {CARLA_HOST}:{CARLA_PORT} ...")
        try:
            self.client = carla.Client(CARLA_HOST, CARLA_PORT)
            self.client.set_timeout(CARLA_TIMEOUT)

            # Load the correct map
            current_map = self.client.get_world().get_map().name
            if CARLA_TOWN not in current_map:
                log.info(f"Loading {CARLA_TOWN} (this takes ~30 seconds)...")
                self.world = self.client.load_world(CARLA_TOWN)
                time.sleep(5)  # Wait for map to fully load
            else:
                self.world = self.client.get_world()

            # Configure simulation settings
            self._apply_world_settings()

            # Initialize GPS denial manager
            self.gps_manager = GPSDenialManager(self.world)

            log.info(f"✓ Connected to CARLA | Map: {self.world.get_map().name}")
            return True

        except Exception as e:
            log.error(f"✗ CARLA connection failed: {e}")
            log.error("  Make sure CARLA (CarlaUE4.exe) is running!")
            return False

    def _apply_world_settings(self):
        """Apply synchronous mode and fixed timestep settings."""
        settings = self.world.get_settings()
        settings.synchronous_mode    = SYNC_MODE
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        settings.no_rendering_mode   = not RENDER_MODE
        self.world.apply_settings(settings)
        log.info(f"World settings: sync={SYNC_MODE}, dt={FIXED_DELTA_SECONDS}s, "
                 f"render={RENDER_MODE}")

    def spawn_vehicle(self, randomize: bool = False) -> bool:
        """
        Spawn the ego vehicle in the CARLA world.

        Args:
            randomize: If True, pick a random spawn point (better for training)

        Returns True on success.
        """
        bp_library = self.world.get_blueprint_library()

        # Get vehicle blueprint
        vehicle_bp = bp_library.find(VEHICLE_BLUEPRINT)
        if vehicle_bp is None:
            # Fallback to any car
            vehicle_bp = bp_library.filter("vehicle.*")[0]
            log.warning(f"Blueprint {VEHICLE_BLUEPRINT} not found, using {vehicle_bp.id}")

        # Choose spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            log.error("No spawn points found in map!")
            return False

        if randomize:
            spawn_transform = np.random.choice(spawn_points)
        else:
            idx = min(SPAWN_POINT_INDEX, len(spawn_points) - 1)
            spawn_transform = spawn_points[idx]

        # Spawn vehicle
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if self.vehicle is None:
            # Try random spawn if preferred failed
            for sp in np.random.permutation(spawn_points)[:5]:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle is not None:
                    spawn_transform = sp
                    break

        if self.vehicle is None:
            log.error("Could not spawn vehicle!")
            return False

        # Tick world once to register the vehicle
        if SYNC_MODE:
            self.world.tick()

        # Set coordinate origin at spawn point
        self.coord_conv.set_origin(self.vehicle.get_transform())
        self.origin_set = True

        # Enable autopilot — CARLA drives, we control the EKF
        if USE_AUTOPILOT:
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            # Positive % = slower than speed limit, negative = faster
            # Town04 speed limit ~90km/h, we want ~30km/h = 67% reduction
            speed_reduction = max(0, int((1.0 - TARGET_SPEED / 90.0) * 100))
            traffic_manager.vehicle_percentage_speed_difference(self.vehicle, speed_reduction)
            self.vehicle.set_autopilot(True, 8000)

        # Attach sensors
        self._attach_imu_sensor()
        self._attach_gnss_sensor()

        # Move spectator camera to follow the vehicle (so you see it moving in CARLA)
        self._attach_spectator()

        # Tick once more so sensors initialize
        if SYNC_MODE:
            self.world.tick()

        log.info(f"✓ Vehicle spawned: {self.vehicle.id} "
                 f"at ({spawn_transform.location.x:.1f}, "
                 f"{spawn_transform.location.y:.1f})")
        return True

    # ------------------------------------------------------------------
    # SENSOR ATTACHMENT
    # ------------------------------------------------------------------

    def _attach_imu_sensor(self):
        """Attach IMU sensor to vehicle with realistic noise parameters."""
        bp_library = self.world.get_blueprint_library()
        imu_bp = bp_library.find("sensor.other.imu")

        # Apply noise configuration
        for key, value in IMU_CONFIG.items():
            if imu_bp.has_attribute(key):
                imu_bp.set_attribute(key, str(value))

        # Mount at vehicle center (slight upward offset for realism)
        transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.5))
        imu_sensor = self.world.spawn_actor(imu_bp, transform,
                                            attach_to=self.vehicle)
        imu_sensor.listen(self._imu_callback)
        self.sensors.append(imu_sensor)
        log.info("✓ IMU sensor attached (20 Hz, automotive-grade noise)")

    def _attach_gnss_sensor(self):
        """Attach GNSS/GPS sensor to vehicle."""
        bp_library = self.world.get_blueprint_library()
        gnss_bp = bp_library.find("sensor.other.gnss")

        for key, value in GNSS_CONFIG.items():
            if gnss_bp.has_attribute(key):
                gnss_bp.set_attribute(key, str(value))

        transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.5))
        gnss_sensor = self.world.spawn_actor(gnss_bp, transform,
                                             attach_to=self.vehicle)
        gnss_sensor.listen(self._gnss_callback)
        self.sensors.append(gnss_sensor)
        log.info("✓ GNSS sensor attached (10 Hz, ~1m noise)")

    # ------------------------------------------------------------------
    # SPECTATOR CAMERA
    # ------------------------------------------------------------------

    def _attach_spectator(self):
        """Move CARLA spectator to bird-eye view above the vehicle."""
        try:
            spectator = self.world.get_spectator()
            transform = self.vehicle.get_transform()
            # Position 30m above vehicle, looking down at an angle
            spectator_loc = carla.Location(
                x=transform.location.x - 10.0 * transform.get_forward_vector().x,
                y=transform.location.y - 10.0 * transform.get_forward_vector().y,
                z=transform.location.z + 25.0
            )
            spectator_rot = carla.Rotation(pitch=-45.0, yaw=transform.rotation.yaw)
            spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))
        except Exception:
            pass  # Non-critical — training continues even if spectator fails

    def update_spectator(self):
        """Call each step to keep camera following vehicle."""
        if self.vehicle and self.vehicle.is_alive:
            self._attach_spectator()

    # ------------------------------------------------------------------
    # SENSOR CALLBACKS  (run on CARLA's thread)
    # ------------------------------------------------------------------

    def _imu_callback(self, imu_data: carla.IMUMeasurement):
        """Called by CARLA every 0.05s with new IMU data."""
        reading = IMUReading(
            timestamp = imu_data.timestamp,
            accel_x   = imu_data.accelerometer.x,
            accel_y   = imu_data.accelerometer.y,
            accel_z   = imu_data.accelerometer.z,
            gyro_x    = imu_data.gyroscope.x,
            gyro_y    = imu_data.gyroscope.y,
            gyro_z    = imu_data.gyroscope.z,
        )
        self._latest_imu = reading
        self._imu_count += 1

        # Put in queue; discard oldest if full (don't block)
        try:
            self._imu_queue.put_nowait(reading)
        except queue.Full:
            try:
                self._imu_queue.get_nowait()  # discard oldest
                self._imu_queue.put_nowait(reading)
            except queue.Empty:
                pass

    def _gnss_callback(self, gnss_data: carla.GnssMeasurement):
        """Called by CARLA every 0.1s with new GNSS data."""
        reading = GNSSReading(
            timestamp = gnss_data.timestamp,
            latitude  = gnss_data.latitude,
            longitude = gnss_data.longitude,
            altitude  = gnss_data.altitude,
        )
        self._latest_gnss = reading
        self._gnss_count += 1

        try:
            self._gnss_queue.put_nowait(reading)
        except queue.Full:
            try:
                self._gnss_queue.get_nowait()
                self._gnss_queue.put_nowait(reading)
            except queue.Empty:
                pass

    # ------------------------------------------------------------------
    # GROUND TRUTH
    # ------------------------------------------------------------------

    def get_ground_truth(self) -> GroundTruth:
        """
        Extract precise vehicle state from CARLA physics engine.
        This is used ONLY for computing training rewards — the RL agent
        does NOT see this during inference.
        """
        transform  = self.vehicle.get_transform()
        velocity   = self.vehicle.get_velocity()
        location   = transform.location

        # Convert to local EKF frame
        lx, ly = self.coord_conv.carla_to_local(location.x, location.y)
        heading = self.coord_conv.carla_yaw_to_heading(transform.rotation.yaw)

        # Speed magnitude (m/s)
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # Check tunnel status
        in_tunnel = self.gps_manager.is_gps_denied(lx, ly, location)

        gt = GroundTruth(
            timestamp = self.world.get_snapshot().timestamp.elapsed_seconds,
            x         = lx,
            y         = ly,
            heading   = heading,
            velocity  = speed,
            in_tunnel = in_tunnel,
        )
        self._last_gt = gt
        return gt

    # ------------------------------------------------------------------
    # MAIN DATA ACCESS
    # ------------------------------------------------------------------

    def get_sensor_bundle(self, timeout: float = 1.0) -> Optional[SensorBundle]:
        """
        Get one complete sensor bundle for the current timestep.

        In synchronous mode, you must call world.tick() BEFORE calling this
        to advance the simulation. This method reads the MOST RECENT data.

        Returns None if sensor data is not yet available.
        """
        # Tick the world forward (synchronous mode)
        if SYNC_MODE:
            self.world.tick()

        self.step_count += 1

        # Keep spectator camera following the vehicle
        self.update_spectator()

        # Get IMU reading (required)
        imu = self._get_latest_imu(timeout)
        if imu is None:
            log.warning(f"Step {self.step_count}: No IMU data available")
            return None

        # Get ground truth
        gt = self.get_ground_truth()

        # Determine GPS denial
        location = self.vehicle.get_transform().location
        gps_denied = self.gps_manager.is_gps_denied(gt.x, gt.y, location)

        # Get GNSS reading (optional — None if GPS denied or unavailable)
        gnss = None
        if not gps_denied:
            gnss = self._get_latest_gnss(timeout=0.0)  # Non-blocking
            if gnss is not None:
                # Convert lat/lon to local EKF coords for the EKF update step
                lx, ly = self.coord_conv.gnss_to_local(gnss.latitude,
                                                        gnss.longitude)
                # Store converted coords as attributes (EKF uses these)
                gnss.local_x = lx
                gnss.local_y = ly
                gnss.valid   = True
        else:
            gnss = None  # GPS denied — EKF must rely on IMU only

        return SensorBundle(
            imu          = imu,
            gnss         = gnss,
            ground_truth = gt,
            step         = self.step_count,
            gps_denied   = gps_denied,
        )

    def _get_latest_imu(self, timeout: float = 1.0) -> Optional[IMUReading]:
        """Get the most recent IMU reading, waiting up to timeout seconds."""
        try:
            return self._imu_queue.get(timeout=timeout)
        except queue.Empty:
            return self._latest_imu  # Return last known if queue empty

    def _get_latest_gnss(self, timeout: float = 0.5) -> Optional[GNSSReading]:
        """Get the most recent GNSS reading."""
        try:
            return self._gnss_queue.get(timeout=timeout)
        except queue.Empty:
            return self._latest_gnss

    # ------------------------------------------------------------------
    # WEATHER CONTROL
    # ------------------------------------------------------------------

    def set_random_weather(self):
        """Randomize weather for episode diversity during training."""
        preset_name = np.random.choice(WEATHER_PRESETS)
        preset = getattr(carla.WeatherParameters, preset_name,
                         carla.WeatherParameters.ClearNoon)
        self.world.set_weather(preset)
        log.info(f"Weather: {preset_name}")

    # ------------------------------------------------------------------
    # EPISODE RESET
    # ------------------------------------------------------------------

    def reset_episode(self, randomize: bool = None) -> bool:
        """
        Reset for a new training episode:
        1. Destroy old vehicle + sensors
        2. Spawn fresh vehicle (optionally at new location)
        3. Optionally randomize weather
        4. Reset step counter and data queues

        Returns True on success.
        """
        randomize = randomize if randomize is not None else RANDOMIZE_SPAWN

        # Clear queues
        for q in (self._imu_queue, self._gnss_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Reset state
        self._latest_imu  = None
        self._latest_gnss = None
        self.step_count   = 0

        # Destroy existing actors
        self._destroy_actors()

        # Tick to allow CARLA to process destruction
        if SYNC_MODE and self.world:
            for _ in range(5):
                self.world.tick()

        # Randomize weather
        if RANDOMIZE_WEATHER:
            self.set_random_weather()

        # Spawn fresh vehicle
        success = self.spawn_vehicle(randomize=randomize)

        # Warmup: tick a few times so sensors initialize
        if SYNC_MODE and success:
            for _ in range(10):
                self.world.tick()
            # Drain initial sensor data (often contains zeros)
            time.sleep(0.1)
            while not self._imu_queue.empty():
                self._imu_queue.get_nowait()
            while not self._gnss_queue.empty():
                self._gnss_queue.get_nowait()

            # Collect a fresh GNSS reading and use it as the coordinate origin
            # This ensures gnss_to_local() returns (0,0) at spawn — matching EKF seed
            for _ in range(20):
                self.world.tick()
            gnss_seed = self._get_latest_gnss(timeout=2.0)
            if gnss_seed is not None:
                self.coord_conv.set_gnss_origin(gnss_seed.latitude, gnss_seed.longitude)
                log.info(f"GNSS origin set: lat={gnss_seed.latitude:.6f} lon={gnss_seed.longitude:.6f}")
            else:
                log.warning("Could not get GNSS seed — coordinate conversion may be off")

        return success

    # ------------------------------------------------------------------
    # STATISTICS & DIAGNOSTICS
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return current bridge statistics for logging."""
        return {
            "step":         self.step_count,
            "imu_received": self._imu_count,
            "gps_received": self._gnss_count,
            "in_tunnel":    self._last_gt.in_tunnel if self._last_gt else False,
            "position":     (self._last_gt.x, self._last_gt.y)
                            if self._last_gt else (0.0, 0.0),
        }

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def _destroy_actors(self):
        """Safely destroy all spawned actors."""
        actors_to_destroy = []

        # Sensors first (they must be destroyed before vehicle)
        for sensor in self.sensors:
            try:
                sensor.stop()
                actors_to_destroy.append(sensor)
            except Exception:
                pass
        self.sensors.clear()

        # Then vehicle
        if self.vehicle and self.vehicle.is_alive:
            actors_to_destroy.append(self.vehicle)
        self.vehicle = None

        # Batch destroy for efficiency
        if actors_to_destroy:
            self.client.apply_batch([carla.command.DestroyActor(a)
                                     for a in actors_to_destroy])

    def destroy(self):
        """Full cleanup — call this when done."""
        log.info("Shutting down CARLA bridge...")
        self._destroy_actors()

        # Restore world settings to async mode
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode    = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        log.info("✓ CARLA bridge destroyed cleanly")

    # ------------------------------------------------------------------
    # CONTEXT MANAGER SUPPORT  (use with 'with' statement)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.destroy()


# =============================================================================
# QUICK TEST  — run directly to verify CARLA connection
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CARLA Sensor Bridge — Connection Test")
    print("=" * 60)
    print("Make sure CarlaUE4.exe is running before proceeding!\n")

    bridge = CARLASensorBridge()

    if not bridge.connect():
        print("\n✗ Could not connect to CARLA.")
        print("  1. Launch CARLA: CarlaUE4.exe -quality-level=Low")
        print("  2. Wait for the CARLA window to fully open")
        print("  3. Run this script again")
        exit(1)

    print("\nSpawning vehicle...")
    if not bridge.spawn_vehicle():
        print("✗ Could not spawn vehicle. Check map has spawn points.")
        bridge.destroy()
        exit(1)

    print("\nCollecting sensor data (10 steps)...")
    print(f"{'Step':>4}  {'IMU Accel':>10}  {'IMU Gyro':>10}  {'GPS X':>8}  {'GPS Y':>8}  {'GPS Denied':>10}  {'In Tunnel':>9}")
    print("-" * 75)

    for step in range(10):
        bundle = bridge.get_sensor_bundle()
        if bundle is None:
            print(f"  {step:>3}: No data received")
            continue

        gnss_x = bundle.gnss.local_x if bundle.gnss else float("nan")
        gnss_y = bundle.gnss.local_y if bundle.gnss else float("nan")

        print(f"  {bundle.step:>3}  "
              f"{bundle.imu.forward_accel:>10.4f}  "
              f"{bundle.imu.yaw_rate:>10.4f}  "
              f"{gnss_x:>8.2f}  "
              f"{gnss_y:>8.2f}  "
              f"{'YES' if bundle.gps_denied else 'no':>10}  "
              f"{'YES' if bundle.ground_truth.in_tunnel else 'no':>9}")

    print("\n✓ Sensor bridge working correctly!")
    print(f"  Stats: {bridge.get_stats()}")

    bridge.destroy()
    print("\nBridge destroyed cleanly. Ready to train!")
