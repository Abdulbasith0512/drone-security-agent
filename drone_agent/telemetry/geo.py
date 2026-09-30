"""Small-area geometry for the synthetic property (equirectangular approximation, metres).

The property spans about 100 m, so a flat-earth approximation is accurate to well under a metre.
Coordinates in this project are synthetic and near (0, 0); nothing here refers to a real place.
"""

import math

METERS_PER_DEGREE_LAT = 111_320.0

# The circle the drone must stay inside, covering every patrol zone with margin.
GEOFENCE_CENTER: tuple[float, float] = (0.0004, 0.00045)
GEOFENCE_RADIUS_M = 100.0


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Ground distance in metres between two nearby points."""
    mean_lat = math.radians((lat1 + lat2) / 2)
    dy = (lat2 - lat1) * METERS_PER_DEGREE_LAT
    dx = (lon2 - lon1) * METERS_PER_DEGREE_LAT * math.cos(mean_lat)
    return math.hypot(dx, dy)


def east_offset_degrees(meters: float, latitude: float) -> float:
    """Longitude change (degrees) that moves a point `meters` to the east at `latitude`."""
    return meters / (METERS_PER_DEGREE_LAT * math.cos(math.radians(latitude)))
