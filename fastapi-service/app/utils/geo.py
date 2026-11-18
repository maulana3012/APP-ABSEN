import math

DEFAULT_CENTER_LAT = -6.387683
DEFAULT_CENTER_LON = 106.939161
DEFAULT_MAX_RADIUS_METERS = 100.0

def calculate_distance(lat1: float, lon1: float, lat2: float = DEFAULT_CENTER_LAT, lon2: float = DEFAULT_CENTER_LON) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c

def is_within_geofence(lat: float, lon: float, max_radius: float = DEFAULT_MAX_RADIUS_METERS) -> tuple[bool, float]:
    distance = calculate_distance(lat, lon)
    return (distance <= max_radius), round(distance, 2)