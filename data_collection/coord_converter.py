"""
coord_converter.py
------------------
Converts CARLA GNSS readings (latitude, longitude) to a local
Cartesian frame (x, y) in meters, using the vehicle spawn point
as the origin (0, 0).

Usage:
    conv = CoordConverter()
    conv.set_origin(lat0, lon0)        # call once after warmup
    x, y = conv.gnss_to_local(lat, lon)
"""

import math


class CoordConverter:
    """
    Equirectangular projection from GNSS to local (x, y) in meters.

    - Origin is set from the first stable GNSS reading at spawn.
    - x increases East, y increases North.
    - This matches CARLA's world frame convention closely enough
      for Town04 distances (< 2km error).
    """

    EARTH_RADIUS_M = 6_371_000.0   # mean Earth radius in metres

    def __init__(self):
        self._origin_lat = None
        self._origin_lon = None
        self._origin_set = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_origin(self, lat: float, lon: float):
        """
        Call this ONCE after the warm-up ticks, passing the first
        stable GNSS reading from the spawn location.
        After this, gnss_to_local() will return (0, 0) for the spawn.
        """
        self._origin_lat = math.radians(lat)
        self._origin_lon = math.radians(lon)
        self._origin_set = True
        print(f"[CoordConverter] Origin set → lat={lat:.6f}, lon={lon:.6f}")

    @property
    def is_ready(self) -> bool:
        return self._origin_set

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def gnss_to_local(self, lat: float, lon: float):
        """
        Convert GNSS (lat, lon) degrees → local (x, y) metres.

        Returns:
            (x, y) tuple in metres relative to spawn origin.
            x = East,  y = North.

        Raises:
            RuntimeError if set_origin() has not been called yet.
        """
        if not self._origin_set:
            raise RuntimeError(
                "CoordConverter: call set_origin() before gnss_to_local()"
            )

        lat_r = math.radians(lat)
        lon_r = math.radians(lon)

        # Equirectangular approximation
        x = self.EARTH_RADIUS_M * (lon_r - self._origin_lon) * math.cos(self._origin_lat)
        y = self.EARTH_RADIUS_M * (lat_r  - self._origin_lat)

        return x, y

    # ------------------------------------------------------------------
    # Inverse (optional – useful for debugging)
    # ------------------------------------------------------------------

    def local_to_gnss(self, x: float, y: float):
        """
        Convert local (x, y) metres → (lat, lon) degrees.
        Inverse of gnss_to_local().
        """
        if not self._origin_set:
            raise RuntimeError(
                "CoordConverter: call set_origin() before local_to_gnss()"
            )

        lat_r = (y / self.EARTH_RADIUS_M) + self._origin_lat
        lon_r = (x / (self.EARTH_RADIUS_M * math.cos(self._origin_lat))) + self._origin_lon

        return math.degrees(lat_r), math.degrees(lon_r)
