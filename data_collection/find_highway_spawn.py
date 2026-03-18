"""
find_highway_spawn.py
---------------------
Prints ALL Town04 spawn points with road info so you can identify
which ones are on the main highway (not surface streets).

RUN THIS FIRST before collect_data.py.
Copy the index numbers that show highway road IDs.

Usage:
    python data_collection\find_highway_spawn.py
"""
import sys

CARLA_EGG = (
    r'C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor'
    r'\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg'
)
sys.path.insert(0, CARLA_EGG)
sys.path.insert(0, r'C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor\PythonAPI')

import carla

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
world  = client.get_world()
cmap   = world.get_map()

spawn_pts = cmap.get_spawn_points()
print(f"\nTotal spawn points: {len(spawn_pts)}\n")
print(f"{'Idx':>4}  {'world_x':>9}  {'world_y':>9}  {'world_z':>7}  "
      f"{'road_id':>8}  {'lane_id':>7}  {'speed_limit':>11}  {'lane_type'}")
print("-" * 80)

highway_candidates = []

for i, sp in enumerate(spawn_pts):
    wp = cmap.get_waypoint(sp.location,
                           project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    if wp is None:
        continue

    # Get speed limit from waypoint
    # Higher speed limits = highway
    speed_limit = 0
    try:
        # Try to get speed limit from nearby landmarks
        landmarks = wp.get_landmarks_of_type(30.0, '274')  # speed limit sign type
        if landmarks:
            speed_limit = int(landmarks[0].value)
    except Exception:
        pass

    lane_type_str = str(wp.lane_type).split('.')[-1]

    print(f"  {i:3d}  {sp.location.x:9.2f}  {sp.location.y:9.2f}  "
          f"{sp.location.z:7.2f}  {wp.road_id:8d}  {wp.lane_id:7d}  "
          f"{speed_limit:11d}  {lane_type_str}")

    # Highway heuristic: large road_id or far from town center
    # Town04's main highway has road_ids typically > 30
    # Surface roads cluster around smaller road_ids
    dist_from_center = (sp.location.x**2 + sp.location.y**2)**0.5
    if wp.road_id > 20 or dist_from_center > 200:
        highway_candidates.append((i, sp.location.x, sp.location.y, sp.location.z, wp.road_id))

print("\n" + "="*60)
print("HIGHWAY CANDIDATES (road_id > 20 or far from center):")
print("="*60)
for idx, x, y, z, rid in highway_candidates:
    print(f"  spawn_pts[{idx:3d}]  world=({x:8.2f}, {y:8.2f}, {z:.2f})  road_id={rid}")

print("\nLook for spawn points with:")
print("  - Large road_id (highway has higher IDs in Town04)")
print("  - world_y around -150 to -400 (south part of map = highway)")
print("  - world_x between 100 and 500")
print("\nThen update HIGHWAY_SPAWN_INDICES in collect_data.py with those indices.")
