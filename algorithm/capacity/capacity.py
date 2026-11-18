import os
import math
import time
import json
import copy
import warnings
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# optional imports from your codebase
try:
    from ordering import apply_route_ordering
    ORDERING_AVAILABLE = True
except Exception:
    ORDERING_AVAILABLE = False

try:
    from algorithm.algorithm_cache import get_algorithm_cache, is_cache_disabled
    # Check configuration to enable/disable cache
    if is_cache_disabled():
        ALGORITHM_CACHE_AVAILABLE = False
        print("🚫 Algorithm caching is disabled via configuration - will run without caching")
    else:
        ALGORITHM_CACHE_AVAILABLE = True
except Exception:
    ALGORITHM_CACHE_AVAILABLE = False

# Import pricebook calculator
from algorithm.pricebook.pricebook_calculator import calculate_pricebook_id

# import base helpers used by the original file
from algorithm.base.base import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, bearing_difference,
    calculate_bearing_vectorized, coords_to_km,
    optimize_route_sequence_improved, update_route_metrics_improved,
    calculate_average_bearing_improved, _get_all_drivers_as_unassigned,
    _convert_users_to_unassigned_format)

# import response builder functions
from algorithm.response.response_builder import (
    build_standard_response,
    save_standardized_response,
    log_response_metrics
)


# ----------------------------
# Configuration loader
# ----------------------------
def load_capacity_config():
    script_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(script_dir, "config.json")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}

    mode_cfg = cfg.get("mode_configs", {}).get("capacity_optimization", {})

    C = {}
    C['GEOGRAPHIC_CLUSTER_RADIUS_KM'] = float(
        mode_cfg.get("geographic_cluster_radius_km",
                     cfg.get("geographic_cluster_radius_km", 1.2)))
    C['MAX_CLUSTER_SIZE'] = int(
        mode_cfg.get("max_cluster_size", cfg.get("max_cluster_size", 6)))
    C['MIN_SAMPLES_DBSCAN'] = int(
        mode_cfg.get("min_samples_dbscan", cfg.get("min_samples_dbscan", 1)))
    C['DISTANCE_WEIGHT'] = float(
        mode_cfg.get("distance_weight", cfg.get("distance_weight", 0.7)))
    C['CAPACITY_WASTE_WEIGHT'] = float(
        mode_cfg.get("capacity_waste_weight",
                     cfg.get("capacity_waste_weight", 0.3)))
    C['MIN_CAPACITY_UTILIZATION'] = float(
        mode_cfg.get("min_capacity_utilization",
                     cfg.get("min_capacity_utilization", 0.4)))
    C['MAX_ON_ROUTE_DETOUR_KM'] = float(
        mode_cfg.get("max_on_route_detour_km",
                     cfg.get("max_on_route_detour_km", 2.0)))
    C['BEARING_TOLERANCE_DEGREES'] = float(
        mode_cfg.get("bearing_tolerance_degrees",
                     cfg.get("bearing_tolerance_degrees", 45)))
    C['DIRECTION_TOLERANCE_DEGREES'] = float(
        mode_cfg.get("direction_tolerance_degrees",
                     cfg.get("direction_tolerance_degrees", 60)))
    C['MAX_MERGE_DISTANCE_KM'] = float(
        mode_cfg.get("max_merge_distance_km",
                     cfg.get("max_merge_distance_km", 4.0)))
    C['MERGE_SCORE_THRESHOLD'] = float(
        mode_cfg.get("merge_score_threshold",
                     cfg.get("merge_score_threshold", 1.5)))
    C['SMALL_ROUTE_THRESHOLD'] = int(
        mode_cfg.get("small_route_threshold",
                     cfg.get("small_route_threshold", 2)))
    C['CAPACITY_PRIORITY_WEIGHT'] = float(
        mode_cfg.get("capacity_priority_weight",
                     cfg.get("capacity_priority_weight", 2.0)))
    C['OVERFLOW_PENALTY_PER_SEAT'] = float(
        mode_cfg.get("overflow_penalty_per_seat",
                     cfg.get("overflow_penalty_per_seat", 3.0)))
    C['UTILIZATION_BONUS_THRESHOLD'] = float(
        mode_cfg.get("utilization_bonus_threshold",
                     cfg.get("utilization_bonus_threshold", 0.8)))

    C['OFFICE_LAT'] = float(
        os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    C['OFFICE_LON'] = float(
        os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    return C


CONFIG = load_capacity_config()


# ----------------------------
# Safety & normalization helpers
# ----------------------------
def normalize_user_id(u):
    if u is None:
        return ""
    if isinstance(u, (str, int)):
        return str(u)
    try:
        return str(
            u.get('user_id') or u.get('id') or u.get('uid') or u.get('userId')
            or '')
    except Exception:
        return ""


def build_user_dict_from_row(user_row):
    if hasattr(user_row, "to_dict"):
        d = user_row.to_dict()
    else:
        d = dict(user_row)
    return {
        'user_id': str(d.get('user_id', d.get('id', ''))),
        'lat': float(d.get('latitude', d.get('lat', 0)) or 0),
        'lng': float(d.get('longitude', d.get('lng', 0)) or 0),
        'latitude': float(d.get('latitude', d.get('lat', 0)) or 0),
        'longitude': float(d.get('longitude', d.get('lng', 0)) or 0),
        'first_name': d.get('first_name') or d.get('firstName') or '',
        'last_name': d.get('last_name') or d.get('lastName') or '',
        'email': d.get('email') or '',
        'phone': d.get('phone') or '',
        'address': d.get('address') or '',
        'employee_shift': d.get('employee_shift') or d.get('employeeShift') or '',
        'shift_type': str(d.get('shift_type', '')) if d.get('shift_type') else '',
        'office_distance': float(d.get('office_distance', 0) or 0),
        '_original': d
    }


def route_capacity(route):
    """Return canonical capacity for route (prefers capacity, vehicle_capacity, vehicle_type)"""
    try:
        return int(
            route.get(
                'capacity',
                route.get('vehicle_capacity', route.get('vehicle_type', 0))
                or 0))
    except Exception:
        return 0


def route_has_capacity(route):
    try:
        cap = route_capacity(route)
        return len(route.get('assigned_users', [])) < cap
    except Exception:
        return False


def collect_assigned_user_ids(routes):
    s = set()
    for r in routes:
        for u in r.get('assigned_users', []):
            uid = normalize_user_id(u)
            if uid:
                s.add(uid)
    return s


def is_user_on_way_to_office_capacity(user_pos, driver_pos, office_pos,
                                      assigned_users):
    """
    Determines if a user is roughly on the way from driver → office.
    Returns (is_on_way, extra_detour_km)
    """
    try:
        ux, uy = user_pos
        dx, dy = driver_pos
        ox, oy = office_pos

        # direct driver->office distance
        direct = haversine_distance(dx, dy, ox, oy)

        # driver->user->office distance
        via_user = (haversine_distance(dx, dy, ux, uy) +
                    haversine_distance(ux, uy, ox, oy))

        detour = via_user - direct

        # simple cut — user is on way if detour < some threshold
        return (detour <= CONFIG['MAX_ON_ROUTE_DETOUR_KM'], detour)

    except Exception:
        return (False, float('inf'))


def assign_cab_to_cluster_capacity(cluster_users, driver_df, office_lat,
                                   office_lon):
    """
    Chooses a driver for a cluster if capacity mismatch happens.
    Tries splitting cluster if needed.
    """

    cluster_size = len(cluster_users)

    # drivers with enough capacity
    capable = driver_df[driver_df.apply(lambda d: int(
        d.get('capacity',
              d.get('vehicle_capacity', d.get('vehicle_type', 0)) or 0)) >=
                                        cluster_size,
                                        axis=1)]

    if not capable.empty:
        # pick nearest capable driver
        center_lat = cluster_users['latitude'].mean()
        center_lon = cluster_users['longitude'].mean()
        capable['dist_to_cluster'] = capable.apply(
            lambda d: haversine_distance(center_lat, center_lon,
                                         float(d['latitude']),
                                         float(d['longitude'])),
            axis=1)
        return capable.sort_values('dist_to_cluster').iloc[0]

    # otherwise cluster is too large → split it
    chunk = CONFIG['MAX_CLUSTER_SIZE']
    if cluster_size > chunk:
        return None  # splitting handled by upstream cluster splitter

    # no one can take cluster — return None
    return None


def can_add_user_to_route(route, user_id, assigned_user_ids):
    if not user_id:
        return False
    if assigned_user_ids is not None and str(user_id) in assigned_user_ids:
        return False
    cap = route_capacity(route)
    if len(route.get('assigned_users', [])) >= cap:
        return False
    for u in route.get('assigned_users', []):
        if normalize_user_id(u) == str(user_id):
            return False
    return True


def safe_add_user_to_route(route, user_obj, assigned_user_ids=None):
    """
    Adds user to route safely:
      - normalizes user id
      - checks capacity (capacity/vehicle_capacity/vehicle_type)
      - checks global duplicate set (assigned_user_ids)
    assigned_user_ids can be None (will not check global duplicates) but will be updated if provided.
    Returns True if added; False otherwise.
    """
    uid = normalize_user_id(user_obj)
    if not uid:
        return False
    if not can_add_user_to_route(route, uid, assigned_user_ids):
        return False

    if isinstance(user_obj, dict) and 'user_id' in user_obj:
        u = copy.deepcopy(user_obj)
    else:
        u = build_user_dict_from_row(user_obj)
    u['user_id'] = str(uid)
    if 'lat' not in u:
        u['lat'] = float(u.get('latitude', 0) or 0)
    if 'lng' not in u:
        u['lng'] = float(u.get('longitude', 0) or 0)

    route.setdefault('assigned_users', []).append(u)
    if assigned_user_ids is not None:
        assigned_user_ids.add(uid)
    return True


def calculate_route_dispersion(users, route_center=None):
    """
    Calculate the geometric dispersion of a route.
    
    Dispersion = max distance from route centroid
    This measures how "spread out" the route is.
    
    Lower dispersion = better geometric cohesion
    Higher dispersion = user is far from center, hurts geometry
    """
    if not users:
        return 0.0
    
    if route_center is None:
        route_center = calculate_route_center(users)
    
    if not route_center:
        return 0.0
    
    max_dist = 0.0
    for user in users:
        user_pos = (float(user.get('latitude', 0)), float(user.get('longitude', 0)))
        dist = haversine_distance(user_pos[0], user_pos[1], route_center[0], route_center[1])
        max_dist = max(max_dist, dist)
    
    return max_dist


def find_best_eviction_candidate(route, new_user):
    """
    When route is full, find which user to evict (if any) to add new_user.
    
    Ruin-and-recreate logic:
    1. Create candidate set = current users + new user
    2. For each user: calculate route dispersion if that user is removed
    3. Pick the eviction that minimizes overall dispersion
    4. If removing the new user minimizes dispersion → reject new user
    5. Otherwise → evict the user whose removal helps geometry most
    
    Returns: (should_evict, evict_uid) tuple
    - should_evict: True if we should evict someone (False means new user is worst fit)
    - evict_uid: uid to remove, None if new user should be rejected
    """
    current_users = route.get('assigned_users', [])
    if not current_users:
        return False, None
    
    # Calculate current route center (baseline)
    current_center = calculate_route_center(current_users)
    current_dispersion = calculate_route_dispersion(current_users, current_center)
    
    # Build candidate set: current users + new user
    new_uid = normalize_user_id(new_user)
    candidate_users = current_users + [new_user]
    candidate_center = calculate_route_center(candidate_users)
    
    # For each user in candidates, calculate what happens if we remove them
    eviction_scores = {}
    
    for user in candidate_users:
        uid = normalize_user_id(user)
        # Set without this user
        without_user = [u for u in candidate_users if normalize_user_id(u) != uid]
        
        if not without_user:
            # Can't remove everyone
            eviction_scores[uid] = float('inf')
            continue
        
        # Calculate dispersion of route without this user
        without_center = calculate_route_center(without_user)
        dispersion_without = calculate_route_dispersion(without_user, without_center)
        eviction_scores[uid] = dispersion_without
    
    # Find the user whose removal minimizes dispersion
    best_evict_uid = min(eviction_scores.keys(), key=lambda x: eviction_scores[x])
    best_dispersion_after_evict = eviction_scores[best_evict_uid]
    
    # Decision logic:
    # - If removing the NEW user gives best geometry → reject new user
    # - If removing someone ELSE gives better geometry → evict them, accept new user
    
    if best_evict_uid == new_uid:
        # New user is worst fit → reject them
        return False, None
    else:
        # Someone else is worse fit → evict them
        return True, best_evict_uid


def safe_add_user_to_route_with_eviction(route, new_user, assigned_user_ids=None):
    """
    Add user to route with intelligent eviction (ruin-and-recreate).

    FIXED VERSION: Eviction is triggered EARLY (at capacity-1), not just when full.
    This prevents suboptimal users from "locking in" due to timing.

    Logic:
    ------
    1. If route is approaching capacity (capacity - 1):
       - ALWAYS evaluate eviction, even if there's technically space
       - This checks if new user is a good fit BEFORE adding

    2. If eviction logic says to evict:
       - Remove worst-fit user, add new user
       - Net: +1 better user, -1 worse user

    3. If route still has space (and not approaching):
       - Add directly (geometry is fine with room to spare)

    4. If route is full AND eviction says reject:
       - Reject new user (new user is worst fit)

    Returns: (was_added, evicted_uid) tuple
    """
    uid = normalize_user_id(new_user)
    if not uid:
        return False, None

    cap = route_capacity(route)
    current_size = len(route.get('assigned_users', []))

    # =========================================================================
    # KEY FIX: Check if we're APPROACHING capacity, not just AT capacity
    # =========================================================================
    approaching_capacity = current_size >= cap - 1

    if approaching_capacity:
        # =====================================================================
        # Route is nearly full - ALWAYS evaluate if eviction makes sense
        # This prevents bad geometry from forming due to greedy early additions
        # =====================================================================
        should_evict, evict_uid = find_best_eviction_candidate(route, new_user)

        if should_evict and evict_uid:
            # Evict the worst-fit user, add the new user
            route['assigned_users'] = [u for u in route['assigned_users']
                                       if normalize_user_id(u) != evict_uid]
            if assigned_user_ids is not None:
                assigned_user_ids.discard(evict_uid)

            # Add the new user
            if isinstance(new_user, dict) and 'user_id' in new_user:
                u = copy.deepcopy(new_user)
            else:
                u = build_user_dict_from_row(new_user)
            u['user_id'] = str(uid)
            if 'lat' not in u:
                u['lat'] = float(u.get('latitude', 0) or 0)
            if 'lng' not in u:
                u['lng'] = float(u.get('longitude', 0) or 0)

            route['assigned_users'].append(u)
            if assigned_user_ids is not None:
                assigned_user_ids.add(uid)

            return True, evict_uid
        else:
            # New user is worst fit, OR route is full and eviction didn't help
            if current_size < cap:
                # Route has space, new user just isn't great, but add it anyway
                # (still better than leaving seat empty)
                if isinstance(new_user, dict) and 'user_id' in new_user:
                    u = copy.deepcopy(new_user)
                else:
                    u = build_user_dict_from_row(new_user)
                u['user_id'] = str(uid)
                if 'lat' not in u:
                    u['lat'] = float(u.get('latitude', 0) or 0)
                if 'lng' not in u:
                    u['lng'] = float(u.get('longitude', 0) or 0)

                route['assigned_users'].append(u)
                if assigned_user_ids is not None:
                    assigned_user_ids.add(uid)
                return True, None
            else:
                # Route is full and new user is worst fit → reject
                return False, None

    else:
        # =====================================================================
        # Route has plenty of space (capacity - 2 or better)
        # Add directly without eviction check (geometry is fine)
        # =====================================================================
        if isinstance(new_user, dict) and 'user_id' in new_user:
            u = copy.deepcopy(new_user)
        else:
            u = build_user_dict_from_row(new_user)
        u['user_id'] = str(uid)
        if 'lat' not in u:
            u['lat'] = float(u.get('latitude', 0) or 0)
        if 'lng' not in u:
            u['lng'] = float(u.get('longitude', 0) or 0)

        route.setdefault('assigned_users', []).append(u)
        if assigned_user_ids is not None:
            assigned_user_ids.add(uid)
        return True, None


def safe_add_user_to_route_with_eviction_aggressive(route, new_user, assigned_user_ids=None):
    """
    AGGRESSIVE VERSION: Always evaluate if new user is a good fit.

    This version ALWAYS runs find_best_eviction_candidate, even when there's
    plenty of space. Use this if you want maximum geometry control.

    Note: Slightly higher compute cost, but better geometry quality.
    """
    uid = normalize_user_id(new_user)
    if not uid:
        return False, None

    cap = route_capacity(route)
    current_size = len(route.get('assigned_users', []))

    # Always evaluate if new user is a good fit for the route
    should_evict, evict_uid = find_best_eviction_candidate(route, new_user)

    if should_evict and evict_uid:
        # Evict worst-fit, add new user
        route['assigned_users'] = [u for u in route['assigned_users']
                                   if normalize_user_id(u) != evict_uid]
        if assigned_user_ids is not None:
            assigned_user_ids.discard(evict_uid)

        if isinstance(new_user, dict) and 'user_id' in new_user:
            u = copy.deepcopy(new_user)
        else:
            u = build_user_dict_from_row(new_user)
        u['user_id'] = str(uid)
        if 'lat' not in u:
            u['lat'] = float(u.get('latitude', 0) or 0)
        if 'lng' not in u:
            u['lng'] = float(u.get('longitude', 0) or 0)

        route['assigned_users'].append(u)
        if assigned_user_ids is not None:
            assigned_user_ids.add(uid)

        return True, evict_uid

    elif current_size < cap:
        # No eviction needed, but route has space → add directly
        if isinstance(new_user, dict) and 'user_id' in new_user:
            u = copy.deepcopy(new_user)
        else:
            u = build_user_dict_from_row(new_user)
        u['user_id'] = str(uid)
        if 'lat' not in u:
            u['lat'] = float(u.get('latitude', 0) or 0)
        if 'lng' not in u:
            u['lng'] = float(u.get('longitude', 0) or 0)

        route['assigned_users'].append(u)
        if assigned_user_ids is not None:
            assigned_user_ids.add(uid)
        return True, None

    else:
        # No space and new user is worst fit → reject
        return False, None


def remove_duplicate_users_from_routes(routes):
    # Keep best (most fields) user object for each uid, then rebuild route lists
    best = {}
    for ridx, r in enumerate(routes):
        for u in r.get('assigned_users', []):
            uid = normalize_user_id(u)
            if not uid:
                continue
            score = sum(1 for k in ('first_name', 'email', 'lat', 'lng')
                        if u.get(k))
            existing = best.get(uid)
            if existing is None or score > existing['score']:
                best[uid] = {
                    'score': score,
                    'user': copy.deepcopy(u),
                    'route_idx': ridx
                }

    route_map = {i: [] for i in range(len(routes))}
    for uid, info in best.items():
        route_map.setdefault(info['route_idx'], []).append(info['user'])

    new_routes = []
    for i, r in enumerate(routes):
        rc = r.copy()
        rc['assigned_users'] = route_map.get(i, [])
        new_routes.append(rc)
    return new_routes


# ----------------------------
# Core geospatial helpers
# ----------------------------
def calculate_route_center(users):
    if not users:
        return None
    lat_sum = 0.0
    lon_sum = 0.0
    valid = 0
    for u in users:
        try:
            lat = float(u.get('lat', u.get('latitude', 0)) or 0)
            lon = float(u.get('lng', u.get('longitude', 0)) or 0)
            if lat == 0 and lon == 0:
                continue
            lat_sum += lat
            lon_sum += lon
            valid += 1
        except Exception:
            continue
    if valid == 0:
        return None
    return (lat_sum / valid, lon_sum / valid)


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    lat2r = math.radians(lat2)
    lon2r = math.radians(lon2)
    dl = lon2r - lon1r
    y = math.sin(dl) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(
        lat2r) * math.cos(dl)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


# ================================
# Direction + Distance Merge Guard
# ================================
def bearing_between(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = map(math.radians, (lat1, lon1, lat2, lon2))
    dl = lon2r - lon1r
    y = math.sin(dl) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(
        lat2r) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angular_difference(b1, b2):
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


DIRECTION_THRESHOLD_DEGREES = 35
DISTANCE_THRESHOLD_KM = 2.0


def directional_merge_allowed(centerA, centerB, office_lat, office_lon):
    # Handle None centers
    if centerA is None or centerB is None:
        return False

    # compute geographic closeness
    dist = haversine_distance(centerA[0], centerA[1], centerB[0], centerB[1])
    close = dist <= DISTANCE_THRESHOLD_KM

    # compute directional alignment
    bA = calculate_bearing(centerA[0], centerA[1], office_lat, office_lon)
    bB = calculate_bearing(centerB[0], centerB[1], office_lat, office_lon)
    aligned = angular_difference(bA, bB) <= DIRECTION_THRESHOLD_DEGREES

    # allow merge if either close OR aligned
    return close or aligned


# ----------------------------
# Route creation helpers
# ----------------------------
def create_route_from_driver(driver, office_lat=None, office_lon=None):
    try:
        d = driver.to_dict() if hasattr(driver, "to_dict") else dict(driver)
        cap = int(
            d.get('capacity',
                  d.get('vehicle_capacity', d.get('vehicle_type', 0)) or 0))
        route = {
            'driver_id': str(d.get('driver_id', d.get('id', ''))),
            'vehicle_id': str(d.get('vehicle_id', '') or ''),
            'vehicle_type': cap,
            'capacity': cap,
            'latitude': float(d.get('latitude', d.get('lat', 0)) or 0),
            'longitude': float(d.get('longitude', d.get('lng', 0)) or 0),
            'first_name': d.get('first_name') or d.get('firstName') or '',
            'last_name': d.get('last_name') or d.get('lastName') or '',
            'email': d.get('email') or '',
            'vehicle_name': d.get('vehicle_name') or d.get('vehicleName') or '',
            'vehicle_no': d.get('vehicle_no') or d.get('vehicleNo') or '',
            'chasis_no': d.get('chasis_no') or d.get('chasisNo') or '',
            'color': d.get('color') or '',
            'registration_no': d.get('registration_no') or d.get('registrationNo') or '',
            'shift_type_id': d.get('shift_type_id') or d.get('shiftTypeId'),
            'assigned_users': []
        }
        return route
    except Exception as e:
        logger.error(f"Error creating route from driver: {e}")
        return None


def create_route_from_cluster_capacity(cluster_users, driver, office_lat,
                                       office_lon):
    try:
        cluster_size = len(cluster_users)
        driver_capacity = int(
            driver.get(
                'capacity',
                driver.get('vehicle_capacity', driver.get('vehicle_type', 0))
                or 0))
        if cluster_size > driver_capacity:
            logger.error(
                f"ROUTE ERROR: cluster {cluster_size} > driver capacity {driver_capacity}"
            )
            return None
        route = {
            'driver_id': str(driver.get('driver_id', driver.get('id', ''))),
            'vehicle_id': str(driver.get('vehicle_id', '') or ''),
            'vehicle_type': driver_capacity,
            'capacity': driver_capacity,
            'latitude': float(driver.get('latitude', driver.get('lat', 0)) or 0),
            'longitude': float(driver.get('longitude', driver.get('lng', 0)) or 0),
            'first_name': driver.get('first_name') or driver.get('firstName') or '',
            'last_name': driver.get('last_name') or driver.get('lastName') or '',
            'email': driver.get('email') or '',
            'vehicle_name': driver.get('vehicle_name') or driver.get('vehicleName') or '',
            'vehicle_no': driver.get('vehicle_no') or driver.get('vehicleNo') or '',
            'chasis_no': driver.get('chasis_no') or driver.get('chasisNo') or '',
            'color': driver.get('color') or '',
            'registration_no': driver.get('registration_no') or driver.get('registrationNo') or '',
            'shift_type_id': driver.get('shift_type_id') or driver.get('shiftTypeId'),
            'assigned_users': []
        }
        for _, user in cluster_users.iterrows():
            u = build_user_dict_from_row(user)
            route['assigned_users'].append(u)
        # Optimize & metrics
        try:
            route = optimize_route_sequence_improved(route, office_lat,
                                                     office_lon)
            update_route_metrics_improved(route, office_lat, office_lon)
        except Exception:
            pass
        if len(route['assigned_users']) > route_capacity(route):
            logger.error(
                f"CRITICAL: created route over capacity for driver {route['driver_id']}"
            )
            return None
        return route
    except Exception as e:
        logger.error(f"Error create_route_from_cluster_capacity: {e}")
        return None


# ----------------------------
# Clustering & splitting
# ----------------------------
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


def cluster_users_by_proximity(user_df, office_lat, office_lon):
    try:
        if len(user_df) < 2:
            user_df['geo_cluster'] = 0
            return user_df
        coords = user_df[['latitude', 'longitude']].values
        coords_km = np.array(
            [coords_to_km(c[0], c[1], office_lat, office_lon) for c in coords])
        distances = pdist(coords_km, metric='euclidean')
        Z = linkage(distances, method='single')
        clusters = fcluster(Z,
                            CONFIG['GEOGRAPHIC_CLUSTER_RADIUS_KM'],
                            criterion='distance')
        user_df['geo_cluster'] = clusters
        user_df = split_oversized_clusters_capacity(user_df)
        return user_df
    except Exception as e:
        logger.error(f"cluster_users_by_proximity error: {e}")
        user_df['geo_cluster'] = range(len(user_df))
        return user_df


def split_oversized_clusters_capacity(user_df):
    cluster_sizes = user_df['geo_cluster'].value_counts()
    oversized = cluster_sizes[cluster_sizes > CONFIG['MAX_CLUSTER_SIZE']].index
    if len(oversized) == 0:
        return user_df
    next_cluster = user_df['geo_cluster'].max() + 1
    for cid in oversized:
        cluster_users = user_df[user_df['geo_cluster'] == cid].copy()
        if len(cluster_users) <= CONFIG['MAX_CLUSTER_SIZE'] * 2:
            mid = len(cluster_users) // 2
            user_df.loc[cluster_users.index[mid:],
                        'geo_cluster'] = next_cluster
            next_cluster += 1
        else:
            center_lat = cluster_users['latitude'].mean()
            center_lon = cluster_users['longitude'].mean()
            cluster_users['dist_to_center'] = cluster_users.apply(
                lambda r: haversine_distance(r['latitude'], r['longitude'],
                                             center_lat, center_lon),
                axis=1)
            cluster_users = cluster_users.sort_values('dist_to_center')
            chunk = CONFIG['MAX_CLUSTER_SIZE']
            for i in range(0, len(cluster_users), chunk):
                if i == 0:
                    continue
                user_df.loc[cluster_users.index[i:i + chunk],
                            'geo_cluster'] = next_cluster
                next_cluster += 1
    return user_df


# ----------------------------
# Seat filling (safe)
# ----------------------------
def fill_remaining_seats_with_cluster_check_capacity(routes,
                                                     unassigned_users_df,
                                                     office_lat,
                                                     office_lon,
                                                     assigned_user_ids=None):
    if assigned_user_ids is None:
        assigned_user_ids = collect_assigned_user_ids(routes)
    filled_user_ids = set()
    seats_filled = 0

    # prepare route centers and bearings
    route_centers = []
    route_bearings = []
    for route in routes:
        if route.get('assigned_users'):
            c = calculate_route_center(route['assigned_users'])
            route_centers.append(c)
            route_bearings.append(
                calculate_average_bearing_improved(
                    route, office_lat, office_lon) if hasattr(
                        calculate_average_bearing_improved, '__call__') else 0)
        else:
            route_centers.append(
                (route.get('latitude'), route.get('longitude')))
            route_bearings.append(0)

    if unassigned_users_df is None or unassigned_users_df.empty:
        return routes, filled_user_ids

    pool = unassigned_users_df.copy()
    if 'geo_cluster' in pool.columns:
        pool['cluster_size'] = pool.groupby(
            'geo_cluster')['user_id'].transform('count')
        pool = pool.sort_values('cluster_size', ascending=True)

    for ridx, route in enumerate(routes):
        avail = route_capacity(route) - len(route.get('assigned_users', []))
        if avail <= 0:
            continue
        center = route_centers[ridx]
        route_bearing = route_bearings[ridx]
        candidates = []
        for _, user in pool.iterrows():
            uid = str(user['user_id'])
            if uid in filled_user_ids or uid in assigned_user_ids:
                continue
            user_pos = (user['latitude'], user['longitude'])
            dist_to_route = haversine_distance(
                user_pos[0], user_pos[1], center[0],
                center[1]) if center else float('inf')
            # check other clusters approx
            belongs_other = False
            for oi, oc in enumerate(route_centers):
                if oi == ridx:
                    continue
                d_other = haversine_distance(user_pos[0], user_pos[1], oc[0],
                                             oc[1]) if oc else float('inf')
                if d_other < dist_to_route - 0.3:
                    belongs_other = True
                    break
            if belongs_other:
                # permissive override
                if avail <= 2 and dist_to_route > 1.5:
                    continue
            # is on way?
            on_way, detour = is_user_on_way_to_office_capacity(
                user_pos, (route.get('latitude'), route.get('longitude')),
                (office_lat, office_lon), route.get('assigned_users', []))
            if not on_way or detour > CONFIG['MAX_ON_ROUTE_DETOUR_KM']:
                continue
            user_bearing = calculate_bearing(center[0], center[1], user_pos[0],
                                             user_pos[1]) if center else 0
            bearing_diff = abs(bearing_difference(
                user_bearing,
                route_bearing)) if route_bearing is not None else 0
            if bearing_diff > 180:
                bearing_diff = 360 - bearing_diff
            bearing_align = 1.0 - (bearing_diff / 180.0)
            new_util = (len(route.get('assigned_users', [])) +
                        1) / route_capacity(route)
            composite = dist_to_route * 0.4 + detour * 0.3 + (
                1 - bearing_align) * 0.2 + (1 - new_util) * 0.1
            candidates.append({
                'user': user,
                'score': composite,
                'dist': dist_to_route,
                'detour': detour,
                'util': new_util,
                'uid': uid
            })
        candidates.sort(key=lambda x: x['score'])
        filled_here = 0
        for c in candidates:
            if filled_here >= avail:
                # Route is full - try intelligent eviction (ruin-and-recreate)
                urow = c['user']
                user_obj = build_user_dict_from_row(urow)
                added, evicted_uid = safe_add_user_to_route_with_eviction(route, user_obj, assigned_user_ids)
                if added:
                    filled_user_ids.add(c['uid'])
                    if evicted_uid:
                        filled_user_ids.discard(evicted_uid)
                        print(f"  🔄 Evicted: user {evicted_uid} (bad geometry) → Added: user {c['uid']} (better fit)")
                        filled_here += 1  # Net +1 eviction/swap
                break
            urow = c['user']
            user_obj = build_user_dict_from_row(urow)
            added = safe_add_user_to_route(route, user_obj, assigned_user_ids)
            if added:
                filled_user_ids.add(c['uid'])
                filled_here += 1
                seats_filled += 1
        if filled_here > 0:
            try:
                optimize_route_sequence_improved(route, office_lat, office_lon)
                update_route_metrics_improved(route, office_lat, office_lon)
            except Exception:
                pass
        if filled_user_ids:
            pool = pool[~pool['user_id'].astype(str).isin(filled_user_ids)]
    return routes, filled_user_ids


# ----------------------------
# Fallback assignment (safe)
# ----------------------------
def apply_final_user_fallback_capacity(routes,
                                       unassigned_users_df,
                                       available_drivers,
                                       office_lat,
                                       office_lon,
                                       assigned_user_ids=None):
    if assigned_user_ids is None:
        assigned_user_ids = collect_assigned_user_ids(routes)
    if unassigned_users_df is None or unassigned_users_df.empty:
        return routes, unassigned_users_df
    if available_drivers is None or available_drivers.empty:
        return routes, unassigned_users_df

    remaining = unassigned_users_df.copy()
    drivers = available_drivers.copy()
    assigned_now = set()
    for _, user in remaining.iterrows():
        if drivers.empty:
            break
        uid = str(user['user_id'])
        if uid in assigned_user_ids or uid in assigned_now:
            continue
        # find first driver with capacity>=1
        cand = drivers[drivers.apply(lambda d: int(
            d.get('capacity',
                  d.get('vehicle_capacity', d.get('vehicle_type', 0)) or 0)) >=
                                     1,
                                     axis=1)]
        if cand.empty:
            break
        chosen = cand.iloc[0]
        route = create_route_from_cluster_capacity(pd.DataFrame([user]),
                                                   chosen, office_lat,
                                                   office_lon)
        if route:
            # add safely (route created with user already)
            routes.append(route)
            assigned_now.add(uid)
            drivers = drivers[drivers['driver_id'] != chosen['driver_id']]
    still_unassigned = remaining[~remaining['user_id'].astype(str).
                                 isin(assigned_now)]
    return routes, still_unassigned


# ----------------------------
# Merging (safe)
# ----------------------------
def perform_route_merge_capacity(route1, route2, office_lat, office_lon):
    try:
        # Direction + distance rule
        cA = calculate_route_center(route1.get('assigned_users', []))
        cB = calculate_route_center(route2.get('assigned_users', []))

        if not directional_merge_allowed(cA, cB, office_lat, office_lon):
            return None

        base = route1 if route1.get(
            'capacity', route1.get('vehicle_type', 0)) >= route2.get(
                'capacity', route2.get('vehicle_type', 0)) else route2
        other = route2 if base is route1 else route1
        merged_capacity = route_capacity(base)
        combined = base.get('assigned_users', []) + other.get(
            'assigned_users', [])
        if len(combined) > merged_capacity:
            return None
        merged = base.copy()
        merged['assigned_users'] = []
        assigned = set()
        for u in combined:
            uid = normalize_user_id(u)
            if not uid or uid in assigned:
                continue
            merged['assigned_users'].append(copy.deepcopy(u))
            assigned.add(uid)
        try:
            merged = optimize_route_sequence_improved(merged, office_lat,
                                                      office_lon)
            update_route_metrics_improved(merged, office_lat, office_lon)
        except Exception:
            pass
        if len(merged['assigned_users']) > merged_capacity:
            return None
        return merged
    except Exception as e:
        logger.error(f"perform_route_merge_capacity error: {e}")
        return None


def merge_small_routes_with_nearby_capacity(routes, office_lat, office_lon):
    if len(routes) < 2:
        return routes
    route_data = []
    for i, r in enumerate(routes):
        route_data.append({
            'index':
            i,
            'route':
            r,
            'user_count':
            len(r.get('assigned_users', [])),
            'capacity':
            route_capacity(r),
            'available_capacity':
            route_capacity(r) - len(r.get('assigned_users', [])),
            'utilization':
            len(r.get('assigned_users', [])) / (route_capacity(r) or 1)
        })
    route_data.sort(key=lambda x: x['user_count'])
    merged_routes = []
    used = set()
    merges = 0
    for i, rd in enumerate(route_data):
        if rd['index'] in used:
            continue
        if rd['user_count'] > CONFIG['SMALL_ROUTE_THRESHOLD'] and rd[
                'utilization'] >= CONFIG['MIN_CAPACITY_UTILIZATION']:
            merged_routes.append(rd['route'])
            used.add(rd['index'])
            continue
        best_candidate = None
        best_score = float('inf')
        for j, rd2 in enumerate(route_data):
            if i == j or rd2['index'] in used:
                continue
            combined = rd['user_count'] + rd2['user_count']
            max_cap = max(rd['capacity'], rd2['capacity'])
            if combined > max_cap:
                continue
            center1 = calculate_route_center(rd['route'].get(
                'assigned_users', [])) or (rd['route'].get('latitude'),
                                           rd['route'].get('longitude'))
            center2 = calculate_route_center(rd2['route'].get(
                'assigned_users', [])) or (rd2['route'].get('latitude'),
                                           rd2['route'].get('longitude'))

            # Direction + distance rule
            if not directional_merge_allowed(center1, center2, office_lat,
                                             office_lon):
                continue

            dist = haversine_distance(center1[0], center1[1], center2[0],
                                      center2[1])
            if dist > CONFIG['MAX_MERGE_DISTANCE_KM']:
                continue
            compat = calculate_direction_compatibility_capacity(
                rd['route'], rd2['route'], office_lat, office_lon)
            combined_util = combined / max_cap if max_cap else 0
            cap_bonus = -1.0 if combined_util >= CONFIG[
                'UTILIZATION_BONUS_THRESHOLD'] else 0
            score = dist * 0.3 + (1 - compat) * 2.0 + cap_bonus
            if rd['user_count'] <= 1 or rd2['user_count'] <= 1:
                score -= 1.0
            if score < best_score and score < CONFIG['MERGE_SCORE_THRESHOLD']:
                best_score = score
                best_candidate = rd2
        if best_candidate:
            merged = perform_route_merge_capacity(rd['route'],
                                                  best_candidate['route'],
                                                  office_lat, office_lon)
            if merged:
                merged_routes.append(merged)
                used.add(rd['index'])
                used.add(best_candidate['index'])
                merges += 1
                continue
        merged_routes.append(rd['route'])
        used.add(rd['index'])
    return merged_routes


def calculate_direction_compatibility_capacity(route1, route2, office_lat,
                                               office_lon):
    try:
        b1 = calculate_average_bearing_improved(route1, office_lat, office_lon)
        b2 = calculate_average_bearing_improved(route2, office_lat, office_lon)
        diff = bearing_difference(b1, b2)
        compat = 1.0 - (diff / CONFIG['DIRECTION_TOLERANCE_DEGREES'])
        return max(0.0, min(1.0, compat))
    except Exception:
        return 0.5


def assert_unique_drivers(routes):
    """
    🔒 Safety guard to catch duplicate driver assignments immediately.
    Turns a silent logic bug into an immediate failure.
    """
    seen = set()
    for r in routes:
        driver_id = r.get('driver_id')
        if not driver_id:
            continue
        if driver_id in seen:
            raise RuntimeError(f"🚨 DUPLICATE DRIVER ROUTE: {driver_id} appears in multiple routes!")
        seen.add(driver_id)
    logger.info(f"✅ Driver uniqueness check passed: {len(seen)} unique drivers in {len(routes)} routes")


# ----------------------------
# Swap helpers (safe)
# ----------------------------
def try_user_reallocation_capacity(route1, route2, office_lat, office_lon):
    # Attempt moving best user from route1 -> route2 if helps and capacity allows; safe-add approach
    best_improvement = 0
    best_user = None
    if len(route2.get('assigned_users', [])) >= route_capacity(route2):
        return 0
    c1 = calculate_route_center(route1.get('assigned_users', []))
    c2 = calculate_route_center(route2.get('assigned_users', []))
    if not c1 or not c2:
        return 0
    for u in route1.get('assigned_users', [])[:]:
        d1 = haversine_distance(u['lat'], u['lng'], c1[0], c1[1])
        d2 = haversine_distance(u['lat'], u['lng'], c2[0], c2[1])
        if d2 + 0.1 < d1:
            improvement = d1 - d2
            if improvement > best_improvement:
                best_improvement = improvement
                best_user = u
    if best_user and best_improvement > 0.3:
        # remove from route1 and safely add to route2
        try:
            route1['assigned_users'].remove(best_user)
            added = safe_add_user_to_route(route2, best_user, None)
            if not added:
                # rollback if cannot add
                route1['assigned_users'].append(best_user)
                return 0
            try:
                optimize_route_sequence_improved(route1, office_lat,
                                                 office_lon)
                optimize_route_sequence_improved(route2, office_lat,
                                                 office_lon)
            except Exception:
                pass
            return best_improvement
        except Exception:
            return 0
    return 0


def try_asymmetric_swap_capacity(route1, route2):
    # conservative swapping similar to old code but safe-add aware
    best_improvement = 0
    best_pair = None
    c1 = calculate_route_center(route1.get('assigned_users', []))
    c2 = calculate_route_center(route2.get('assigned_users', []))
    if not c1 or not c2:
        return 0
    # try moves
    for u1 in route1.get('assigned_users', [])[:]:
        if len(route2.get('assigned_users', [])) >= route_capacity(route2):
            continue
        d1 = haversine_distance(u1['lat'], u1['lng'], c1[0], c1[1])
        d2 = haversine_distance(u1['lat'], u1['lng'], c2[0], c2[1])
        if d2 < d1 - 0.3:
            imp = d1 - d2
            r1_util_after = (len(route1['assigned_users']) -
                             1) / (route_capacity(route1) or 1)
            r2_util_after = (len(route2['assigned_users']) +
                             1) / (route_capacity(route2) or 1)
            if r1_util_after >= 0.6 and r2_util_after >= 0.6:
                imp += 0.5
            if imp > best_improvement:
                best_improvement = imp
                best_pair = ('move', u1, None, route1, route2)
    if best_pair and best_improvement > 0.2:
        typ, u1, u2, from_r, to_r = best_pair
        try:
            from_r['assigned_users'].remove(u1)
            added = safe_add_user_to_route(to_r, u1, None)
            if not added:
                from_r['assigned_users'].append(u1)
                return 0
            try:
                optimize_route_sequence_improved(from_r, CONFIG['OFFICE_LAT'],
                                                 CONFIG['OFFICE_LON'])
                optimize_route_sequence_improved(to_r, CONFIG['OFFICE_LAT'],
                                                 CONFIG['OFFICE_LON'])
            except Exception:
                pass
            return best_improvement
        except Exception:
            return 0
    return 0


# ----------------------------
# Aggressive merging and fallback pipeline helpers
# ----------------------------
def merge_routes_by_direction_and_capacity(routes, office_lat, office_lon):
    if len(routes) < 2:
        return routes
    route_data = []
    for i, r in enumerate(routes):
        if not r.get('assigned_users'):
            continue
        center = calculate_route_center(r.get('assigned_users', []))
        if not center:
            continue
        bearing = calculate_bearing(center[0], center[1], office_lat,
                                    office_lon)
        route_data.append({
            'index': i,
            'route': r,
            'center': center,
            'bearing': bearing,
            'user_count': len(r['assigned_users']),
            'capacity': route_capacity(r)
        })
    merged = []
    used = set()
    for i, a in enumerate(route_data):
        if a['index'] in used:
            continue
        candidate = None
        for j, b in enumerate(route_data):
            if i >= j or b['index'] in used:
                continue
            total_users = a['user_count'] + b['user_count']
            max_cap = max(a['capacity'], b['capacity'])
            if total_users > max_cap:
                continue

            # Direction + distance rule
            if directional_merge_allowed(a['center'], b['center'], office_lat,
                                         office_lon):
                candidate = b
                break
        if candidate:
            merged_route = perform_route_merge_capacity(
                a['route'], candidate['route'], office_lat, office_lon)
            if merged_route:
                merged.append(merged_route)
                used.add(a['index'])
                used.add(candidate['index'])
                continue
        merged.append(a['route'])
        used.add(a['index'])
    return merged


def apply_final_capacity_optimization(routes, office_lat, office_lon):
    # final pass to combine any compatible routes
    changed = True
    max_iter = 3
    it = 0
    while changed and it < max_iter:
        changed = False
        new_routes = []
        used = set()
        for i, r1 in enumerate(routes):
            if i in used:
                continue
            merged_flag = False
            for j, r2 in enumerate(routes[i + 1:], start=i + 1):
                if j in used:
                    continue

                # Direction + distance rule
                cA = calculate_route_center(r1.get('assigned_users', []))
                cB = calculate_route_center(r2.get('assigned_users', []))

                if not directional_merge_allowed(cA, cB, office_lat,
                                                 office_lon):
                    continue

                merged = perform_route_merge_capacity(r1, r2, office_lat,
                                                      office_lon)
                if merged:
                    new_routes.append(merged)
                    used.add(i)
                    used.add(j)
                    merged_flag = True
                    changed = True
                    break
            if not merged_flag and i not in used:
                new_routes.append(r1)
                used.add(i)
        routes = new_routes
        it += 1
    return routes


# ----------------------------
# Main pipeline (restored & fixed)
# ----------------------------
def run_assignment_capacity(source_id: str,
                            parameter: int = 1,
                            string_param: str = "",
                            choice: str = ""):
    return run_capacity_assignment_simplified(source_id, parameter,
                                              string_param, choice)


def run_capacity_assignment_simplified(source_id: str,
                                       parameter: int = 1,
                                       string_param: str = "",
                                       choice: str = ""):
    start_time = time.time()
    print(
        f"🚀 Starting CAPACITY-OPTIMIZED assignment (Balance.py Architecture)")
    print(
        f"📋 Source: {source_id}, Parameter: {parameter}, String: {string_param}"
    )

    try:
        # load data exactly as original implementation expected
        data = load_env_and_fetch_data(source_id, parameter, string_param, choice)

        # caching (optional)
        db_name = source_id if source_id and source_id != "1" else data.get(
            "db", "default")
        cached_result = None
        if ALGORITHM_CACHE_AVAILABLE:
            try:
                cache = get_algorithm_cache(db_name, "capacity")
                current_signature = cache.generate_data_signature(
                    data, {
                        'parameter': parameter,
                        'string_param': string_param,
                        'choice': choice,
                        'algorithm': 'capacity'
                    })
                cached_result = cache.get_cached_result(current_signature)
                if cached_result is not None:
                    print("⚡ FAST RESPONSE: Using cached algorithm result")
                    cached_result['_execution_time'] = 0.001
                    cached_result['_cache_hit'] = True
                    return cached_result
            except Exception as e:
                print(f"Cache error: {e} - continuing")

        users = data.get('users', [])
        if not users:
            print("⚠️ No users found - returning empty assignment")
            return create_empty_response(data, start_time)

        # gather drivers
        all_drivers = []
        if "drivers" in data:
            drivers_data = data["drivers"]
            all_drivers.extend(drivers_data.get("driversUnassigned", []))
            all_drivers.extend(drivers_data.get("driversAssigned", []))
        else:
            all_drivers.extend(data.get("driversUnassigned", []))
            all_drivers.extend(data.get("driversAssigned", []))
        if not all_drivers:
            print("⚠️ No drivers available - all users unassigned")
            return create_no_drivers_response(data, start_time)

        print(
            f"📥 Data loaded - Users: {len(users)}, Drivers: {len(all_drivers)}"
        )

        # extract office coords and validate (validate_input_data expects single arg)
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        print("✅ Data validation passed")

        # prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        print(
            f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}"
        )

        # step 1: clustering
        print("📍 STEP 1: geographic clustering")
        user_df = cluster_users_by_proximity(user_df, office_lat, office_lon)
        
        # Log clustering results
        if 'geo_cluster' in user_df.columns:
            print(f"  📊 Created {user_df['geo_cluster'].nunique()} clusters")
            for cid in sorted(user_df['geo_cluster'].unique()):
                cluster_users = user_df[user_df['geo_cluster'] == cid]
                cluster_center = (cluster_users['latitude'].mean(), cluster_users['longitude'].mean())
                dists = []
                for _, user in cluster_users.iterrows():
                    dist = haversine_distance(user['latitude'], user['longitude'], cluster_center[0], cluster_center[1])
                    dists.append(dist)
                avg_spread = sum(dists) / len(dists) if dists else 0
                user_names = [u['first_name'] if pd.notna(u.get('first_name')) else 'Unknown' for _, u in cluster_users.iterrows()]
                print(f"    Cluster {cid}: {len(cluster_users)} users, center={cluster_center}, avg_spread={avg_spread:.2f}km")
                print(f"      Users: {', '.join(user_names)}")

        # step 1.5: CLUSTER REORGANIZATION - Merge nearby singleton clusters before assignment
        print("\n🔄 STEP 1.5: Optimizing cluster coherence (merging nearby singletons)")
        if 'geo_cluster' in user_df.columns:
            cluster_counts = user_df['geo_cluster'].value_counts()
            singleton_clusters = cluster_counts[cluster_counts == 1].index.tolist()
            
            for singleton_cid in singleton_clusters:
                singleton_user = user_df[user_df['geo_cluster'] == singleton_cid].iloc[0]
                singleton_pos = (float(singleton_user['latitude']), float(singleton_user['longitude']))
                singleton_name = singleton_user.get('first_name', 'Unknown')
                
                # Find nearest non-singleton cluster
                nearest_cluster = None
                nearest_distance = float('inf')
                
                for other_cid in user_df['geo_cluster'].unique():
                    if other_cid == singleton_cid:
                        continue
                    other_cluster = user_df[user_df['geo_cluster'] == other_cid]
                    if len(other_cluster) <= 1:  # Skip other singletons
                        continue
                    
                    other_center = (
                        other_cluster['latitude'].mean(),
                        other_cluster['longitude'].mean()
                    )
                    
                    dist = haversine_distance(singleton_pos[0], singleton_pos[1], 
                                            other_center[0], other_center[1])
                    
                    # Merge if within 2.5km
                    if dist < nearest_distance and dist < 2.5:
                        nearest_distance = dist
                        nearest_cluster = other_cid
                
                if nearest_cluster is not None:
                    user_df.loc[user_df['geo_cluster'] == singleton_cid, 'geo_cluster'] = nearest_cluster
                    other_users = user_df[user_df['geo_cluster'] == nearest_cluster]
                    other_names = [u.get('first_name', 'Unknown') for _, u in other_users.iterrows()]
                    print(f"  ✅ Merged {singleton_name} (singleton) with cluster {nearest_cluster}: {', '.join(other_names)} (dist: {nearest_distance:.2f}km)")

        # step 2: initial route creation from clusters
        routes = []
        assigned_user_ids = set()
        assigned_driver_ids = set()  # 🔐 Global driver lock to prevent duplicates
        if 'geo_cluster' in user_df.columns:
            clusters = user_df.groupby('geo_cluster')
            for cid, cluster_users in clusters:
                cluster_size = len(cluster_users)
                cand_drivers = driver_df[driver_df.apply(lambda d: int(
                    d.get(
                        'capacity',
                        d.get('vehicle_capacity', d.get('vehicle_type', 0)) or
                        0)) >= cluster_size,
                                                         axis=1)]
                if cand_drivers.empty:
                    # cluster too big; splitting handled in split_cluster_for_capacity_match within assign_cab_to_cluster_capacity
                    best_driver = assign_cab_to_cluster_capacity(
                        cluster_users, driver_df, office_lat, office_lon)
                    if best_driver is None:
                        continue
                    # best_driver may be a sub-cluster; handle fallback accordingly
                else:
                    # pick closest driver
                    center_lat = cluster_users['latitude'].mean()
                    center_lon = cluster_users['longitude'].mean()
                    cand_drivers['dist_to_cluster'] = cand_drivers.apply(
                        lambda d: haversine_distance(center_lat, center_lon,
                                                     float(d['latitude']),
                                                     float(d['longitude'])),
                        axis=1)
                    # 🔐 Global driver lock: only pick drivers not already assigned
                    available_cand_drivers = cand_drivers[~cand_drivers.apply(
                        lambda d: str(d.get('driver_id', d.get('id', ''))) in assigned_driver_ids, axis=1)]

                    if available_cand_drivers.empty:
                        continue  # No available drivers for this cluster

                    chosen = available_cand_drivers.sort_values(
                        'dist_to_cluster').iloc[0]
                    route = create_route_from_cluster_capacity(
                        cluster_users, chosen, office_lat, office_lon)
                    if route:
                        routes.append(route)
                        driver_name = chosen.get('first_name', 'Unknown')
                        driver_id = str(chosen.get('driver_id', chosen.get('id', '')))
                        dist_to_cluster = chosen['dist_to_cluster']
                        cap = int(chosen.get('capacity', chosen.get('vehicle_capacity', 0)))
                        cluster_user_names = [u['first_name'] if pd.notna(u.get('first_name')) else 'Unknown' for _, u in cluster_users.iterrows()]
                        print(f"  ✓ Cluster {cid} ({cluster_size} users) → {driver_name} ({driver_id}, cap={cap}) dist={dist_to_cluster:.2f}km")
                        print(f"      Assigned: {', '.join(cluster_user_names)}")

                        # 🔐 CRITICAL FIX: Global driver lock to prevent duplicates
                        assigned_driver_ids.add(driver_id)
                        driver_df = driver_df[driver_df['driver_id'] != driver_id]

        # step 3: create empty routes for remaining drivers
        for _, driver in driver_df.iterrows():
            r = create_route_from_driver(driver, office_lat, office_lon)
            if r:
                routes.append(r)

        # recalc assigned set
        assigned_user_ids = collect_assigned_user_ids(routes)
        
        # Log route status after clustering
        print(f"\n📊 ROUTE STATUS AFTER CLUSTERING:")
        print(f"{'Driver':<25} {'Cap':>3} {'Asg':>3} {'Util':>6} {'Center':>20} {'Spread':>8}")
        print("="*90)
        for route in routes:
            driver_name = route.get('first_name', 'Unknown')
            cap = route_capacity(route)
            asg = len(route.get('assigned_users', []))
            util = (asg / cap * 100) if cap > 0 else 0
            route_center = calculate_route_center(route.get('assigned_users', []))
            
            # Calculate spread
            spread = 0.0
            if len(route.get('assigned_users', [])) > 1 and route_center:
                dists = []
                for user in route.get('assigned_users', []):
                    user_pos = (float(user.get('latitude', 0)), float(user.get('longitude', 0)))
                    dist = haversine_distance(user_pos[0], user_pos[1], route_center[0], route_center[1])
                    dists.append(dist)
                spread = sum(dists) / len(dists)
            
            center_str = f"({route_center[0]:.2f},{route_center[1]:.2f})" if route_center else "None"
            print(f"{driver_name:<25} {cap:>3} {asg:>3} {util:>5.1f}% {center_str:>20} {spread:>7.2f}km")
        
        print(f"\nTotal assigned: {len(assigned_user_ids)}/{len(user_df)} users")

        # step 4: OUTLIER DETECTION & REMOVAL - Remove users too far from their route center
        print(f"\n🔍 OUTLIER detection pass (analyzing {len(routes)} routes)")
        outliers_to_reassign = set()
        outliers_found = 0
        for route_idx, route in enumerate(routes):
            if len(route.get('assigned_users', [])) < 2:
                continue
            
            driver_name = route.get('first_name', 'Unknown')
            route_center = calculate_route_center(route.get('assigned_users', []))
            if not route_center:
                continue
            
            # Calculate distances from each user to route center
            user_distances = []
            for user in route.get('assigned_users', []):
                uid = normalize_user_id(user)
                user_name = user.get('first_name', 'Unknown')
                user_pos = (float(user.get('latitude', 0)), float(user.get('longitude', 0)))
                dist = haversine_distance(user_pos[0], user_pos[1], route_center[0], route_center[1])
                user_distances.append((uid, user_name, user, dist))
            
            # If this route has outliers (users much farther than others), remove them
            if len(user_distances) > 1:
                distances_only = [d[3] for d in user_distances]
                avg_dist = sum(distances_only) / len(distances_only)
                max_dist = max(distances_only)
                
                # Log all users in this route with their distances
                print(f"  Route {driver_name}: center=({route_center[0]:.2f},{route_center[1]:.2f}), avg_dist={avg_dist:.2f}km, max_dist={max_dist:.2f}km")
                for uid, user_name, _, dist in sorted(user_distances, key=lambda x: x[3], reverse=True):
                    status = "✓" if dist <= avg_dist * 1.5 else "⚠️ " if dist <= avg_dist * 2.0 else "❌"
                    print(f"    {status} {user_name}: {dist:.2f}km")
                
                # If max distance is 1.8x+ average, that user is an outlier
                if max_dist > avg_dist * 1.8 and max_dist > 1.8:
                    for uid, user_name, user_obj, dist in user_distances:
                        if dist == max_dist:  # This is the outlier
                            route['assigned_users'].remove(user_obj)
                            outliers_to_reassign.add(uid)
                            assigned_user_ids.discard(uid)
                            outliers_found += 1
                            print(f"      ❌ OUTLIER REMOVED: {user_name} (dist: {dist:.2f} km vs avg: {avg_dist:.2f} km, ratio: {dist/avg_dist:.2f}x)")
        
        print(f"\n  Total outliers removed: {outliers_found}")
        
        # step 4b: greedy fill remaining seats (safe)
        unassigned_users_df = user_df[~user_df['user_id'].astype(str).
                                      isin(assigned_user_ids)].copy()
        
        print(f"\n📥 SEAT-FILLING PHASE ({len(unassigned_users_df)} users to assign)")
        routes, newly_filled = fill_remaining_seats_with_cluster_check_capacity(
            routes, unassigned_users_df, office_lat, office_lon,
            assigned_user_ids)
        assigned_user_ids.update(newly_filled)
        print(f"  ✓ Assigned {len(newly_filled)} users via seat-filling")
        
        # Log final route status after seat-filling
        print(f"\n📊 ROUTE STATUS AFTER SEAT-FILLING:")
        print(f"{'Driver':<25} {'Cap':>3} {'Asg':>3} {'Util':>6} {'Center':>20} {'Spread':>8}")
        print("="*90)
        for route in routes:
            driver_name = route.get('first_name', 'Unknown')
            cap = route_capacity(route)
            asg = len(route.get('assigned_users', []))
            util = (asg / cap * 100) if cap > 0 else 0
            route_center = calculate_route_center(route.get('assigned_users', []))
            
            # Calculate spread
            spread = 0.0
            if len(route.get('assigned_users', [])) > 1 and route_center:
                dists = []
                for user in route.get('assigned_users', []):
                    user_pos = (float(user.get('latitude', 0)), float(user.get('longitude', 0)))
                    dist = haversine_distance(user_pos[0], user_pos[1], route_center[0], route_center[1])
                    dists.append(dist)
                spread = sum(dists) / len(dists)
            
            center_str = f"({route_center[0]:.2f},{route_center[1]:.2f})" if route_center else "None"
            print(f"{driver_name:<25} {cap:>3} {asg:>3} {util:>5.1f}% {center_str:>20} {spread:>7.2f}km")

        # step 5: final fallback for remaining users
        remaining_unassigned = user_df[~user_df['user_id'].astype(str).
                                       isin(assigned_user_ids)].copy()
        if not remaining_unassigned.empty:
            used_driver_ids = {r['driver_id'] for r in routes}
            available_drivers = driver_df[~driver_df['driver_id'].
                                          isin(used_driver_ids)].copy()
            routes, still_unassigned = apply_final_user_fallback_capacity(
                routes, remaining_unassigned, available_drivers, office_lat,
                office_lon, assigned_user_ids)
            assigned_user_ids = collect_assigned_user_ids(routes)

        # step 6: merging small routes (safe)
        routes = merge_small_routes_with_nearby_capacity(
            routes, office_lat, office_lon)

        # step 7: final swapping passes (safe)
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                try_user_reallocation_capacity(routes[i], routes[j],
                                               office_lat, office_lon)
                try_asymmetric_swap_capacity(routes[i], routes[j])

        # step 8: aggressive merging and final optimization if needed
        routes = apply_final_capacity_optimization(routes, office_lat,
                                                   office_lon)

        # 🔒 Safety check: ensure no duplicate drivers (critical fix validation)
        assert_unique_drivers(routes)

        # dedupe & finalize
        routes = remove_duplicate_users_from_routes(routes)
        assigned_user_ids = collect_assigned_user_ids(routes)

        # final optimize & metrics
        final_routes = []
        for r in routes:
            if r.get('assigned_users'):
                try:
                    r = optimize_route_sequence_improved(
                        r, office_lat, office_lon)
                except Exception:
                    pass
                try:
                    update_route_metrics_improved(r, office_lat, office_lon)
                except Exception:
                    pass
                final_routes.append(r)
            else:
                # drop empty route
                pass

        # post-check
        for r in final_routes:
            cap = route_capacity(r)
            if len(r.get('assigned_users', [])) > cap:
                logger.error(
                    f"Post-check: route {r.get('driver_id')} over capacity {len(r.get('assigned_users', []))}/{cap}"
                )

        # Apply route ordering (pickup_order) using ordering integration
        if ORDERING_AVAILABLE:
            try:
                from ordering import apply_route_ordering
                db_name = source_id if source_id and source_id != "1" else data.get("db", "default")
                final_routes = apply_route_ordering(
                    final_routes, 
                    office_lat, 
                    office_lon,
                    db_name=db_name,
                    algorithm_name="capacity"
                )
                logger.info("✅ Route ordering applied successfully")
            except Exception as ordering_error:
                logger.warning(f"⚠️ Route ordering failed: {ordering_error} - continuing without ordering")

        print(
            f"Capacity assignment complete: {len(final_routes)} routes, {len(assigned_user_ids)} assigned users"
        )
        
        # Enhance routes with driver and user information from original data
        execution_time = time.time() - start_time
        
        # Get all drivers from original data
        if "drivers" in data:
            all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
        else:
            all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])
        
        # Enhance routes with driver information
        enhanced_routes = []
        for route in final_routes:
            enhanced_route = route.copy()
            driver_id = route['driver_id']
            driver_info = None
            
            # Find driver in original data
            for driver in all_drivers_data:
                if str(driver.get('id', driver.get('sub_user_id', ''))) == driver_id:
                    driver_info = driver
                    break
            
            if driver_info:
                enhanced_route.update({
                    'first_name': driver_info.get('first_name', ''),
                    'last_name': driver_info.get('last_name', ''),
                    'email': driver_info.get('email', ''),
                    'vehicle_name': driver_info.get('vehicle_name', ''),
                    'vehicle_no': driver_info.get('vehicle_no', ''),
                    'capacity': driver_info.get('capacity', ''),
                    'chasis_no': driver_info.get('chasis_no', ''),
                    'color': driver_info.get('color', ''),
                    'registration_no': driver_info.get('registration_no', ''),
                    'shift_type_id': driver_info.get('shift_type_id', '')
                })

            # Calculate pricebook_id for this route
            pricebook_id = calculate_pricebook_id(
                route=enhanced_route,
                data=data,
                company_lat=office_lat,
                company_lon=office_lon,
                logger=logger
            )
            enhanced_route['price_book_id'] = pricebook_id

            # Enhance user information
            enhanced_users = []
            original_users = data.get('users', [])
            
            for user in route['assigned_users']:
                enhanced_user = user.copy()
                
                for orig_user in original_users:
                    if str(orig_user.get('id', orig_user.get('sub_user_id', ''))) == user['user_id']:
                        enhanced_user.update({
                            'address': orig_user.get('address', ''),
                            'employee_shift': orig_user.get('employee_shift', ''),
                            'shift_type': orig_user.get('shift_type', ''),
                            'last_name': orig_user.get('last_name', ''),
                            'phone': orig_user.get('phone', '')
                        })
                        break
                
                enhanced_users.append(enhanced_user)
            
            enhanced_route['assigned_users'] = enhanced_users
            enhanced_routes.append(enhanced_route)
        
        # Enhance unassigned users
        enhanced_unassigned_users = []
        original_users = data.get('users', [])
        
        unassigned_user_records = [
            u for u in user_df.to_dict('records')
            if str(u['user_id']) not in assigned_user_ids
        ]
        
        for user in unassigned_user_records:
            enhanced_user = user.copy()
            
            for orig_user in original_users:
                if str(orig_user.get('id', orig_user.get('sub_user_id', ''))) == user['user_id']:
                    enhanced_user.update({
                        'address': orig_user.get('address', ''),
                        'employee_shift': orig_user.get('employee_shift', ''),
                        'shift_type': orig_user.get('shift_type', ''),
                        'last_name': orig_user.get('last_name', ''),
                        'phone': orig_user.get('phone', '')
                    })
                    break
            
            enhanced_unassigned_users.append(enhanced_user)
        
        # 🔧 FIX: Rebuild driver lists from routes (single source of truth)
        assigned_driver_ids = {str(route['driver_id']) for route in final_routes}

        # Get all original drivers data
        if "drivers" in data:
            all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
        else:
            all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])

        # Build proper unassigned drivers list (only those NOT in routes)
        enhanced_unassigned_drivers = []
        unassigned_drivers = []

        for driver in all_drivers_data:
            driver_id = str(driver.get('id', driver.get('sub_user_id', '')))
            if driver_id not in assigned_driver_ids:
                # Driver is truly unassigned
                unassigned_drivers.append({
                    'driver_id': driver_id,
                    'capacity': int(driver.get('capacity', 0)),
                    'vehicle_id': str(driver.get('vehicle_id', '')),
                    'latitude': float(driver.get('latitude', 0.0)),
                    'longitude': float(driver.get('longitude', 0.0)),
                    'first_name': driver.get('first_name', ''),
                    'last_name': driver.get('last_name', ''),
                    'email': driver.get('email', ''),
                    'vehicle_name': driver.get('vehicle_name', ''),
                    'vehicle_no': driver.get('vehicle_no', ''),
                    'chasis_no': driver.get('chasis_no', ''),
                    'color': driver.get('color', ''),
                    'registration_no': driver.get('registration_no', ''),
                    'shift_type_id': driver.get('shift_type_id', '')
                })

        # Data already enhanced above - just append
        enhanced_unassigned_drivers = unassigned_drivers
        
        result = build_standard_response(
            status="true",
            execution_time=execution_time,
            routes=enhanced_routes,
            unassigned_users=enhanced_unassigned_users,
            unassigned_drivers=enhanced_unassigned_drivers,
            optimization_mode="capacity",
            parameter=parameter,
            company=data.get('company', {}),
            shift=data.get('shift', {}),
            string_param=string_param,
            choice=choice
        )
        
        try:
            save_standardized_response(result, "drivers_and_routes.json")
        except Exception as save_error:
            logger.warning(f"Failed to save standardized response: {save_error}")
        
        try:
            log_response_metrics(result, "capacity_optimized")
        except Exception as log_error:
            logger.warning(f"Failed to log response metrics: {log_error}")
        
        return result
    except Exception as e:
        logger.error(f"Error in algorithm routing: {e}", exc_info=True)
        # build failure response
        try:
            return create_no_drivers_response({}, time.time())
        except Exception:
            return {'routes': [], 'assigned_user_count': 0, 'error': str(e)}


# ----------------------------
# Response builders (from old file)
# ----------------------------
def create_empty_response(data, start_time):
    try:
        from algorithm.response.response_builder import build_standard_response
        return build_standard_response(
            status="true",
            execution_time=time.time() - start_time,
            routes=[],
            unassigned_users=[],
            unassigned_drivers=_get_all_drivers_as_unassigned(data),
            optimization_mode="capacity",
            parameter=1)
    except Exception:
        return {'routes': [], 'unassigned_users': [], 'unassigned_drivers': []}


def create_no_drivers_response(data, start_time):
    try:
        from algorithm.response.response_builder import build_standard_response
        unassigned_users = _convert_users_to_unassigned_format(
            data.get('users', []))
        return build_standard_response(status="true",
                                       execution_time=time.time() - start_time,
                                       routes=[],
                                       unassigned_users=unassigned_users,
                                       unassigned_drivers=[],
                                       optimization_mode="capacity",
                                       parameter=1)
    except Exception:
        return {
            'routes': [],
            'unassigned_users':
            _convert_users_to_unassigned_format(data.get('users', [])) if
            hasattr(_convert_users_to_unassigned_format, '__call__') else [],
            'unassigned_drivers': []
        }