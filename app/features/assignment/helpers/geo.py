"""
Pure geometry helpers for the assignment feature.

Extracted verbatim from the legacy algorithm/base/base.py; `coords_to_km`
was previously fed from module config globals whose values derive only from
the office latitude (LAT_TO_KM = 111.0, LON_TO_KM = 111.0*cos(office_lat)),
computed inline here — behavior-identical.
"""

import math


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(
        dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371
    return c * r


def bearing_difference(b1, b2):
    """Compute minimum difference between two bearings"""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def coords_to_km(lat, lon, office_lat, office_lon):
    """Convert lat/lon coordinates to km from office using local approximation"""
    lat_km = (lat - office_lat) * 111.0
    lon_km = (lon - office_lon) * (111.0 * math.cos(math.radians(office_lat)))
    return lat_km, lon_km
