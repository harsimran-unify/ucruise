import os
import math
import requests
import numpy as np
import pandas as pd
import time
import json
from functools import lru_cache
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from logger import get_logger, start_session

# Start new session with cleared logs
logger = start_session()

warnings.filterwarnings('ignore')

# Import custom logging and progress tracking
from logger import get_logger
from progress import get_progress_tracker

# Import ordering system for optimal pickup sequences
try:
    from ordering import apply_route_ordering
    ORDERING_AVAILABLE = True
except ImportError:
    ORDERING_AVAILABLE = False
    logger.warning("Ordering system not available - routes will not have optimal pickup sequences")


# Load and validate configuration with route efficiency settings
def load_and_validate_config():
    """Load configuration with route efficiency settings"""
    # Find the config file relative to this script's location
    # __file__ is algorithm/base/base.py, so we need to go up 3 levels to reach project root
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go up from algorithm/base/ to project root
    config_path = os.path.join(script_dir, 'config.json')

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger = get_logger()
        logger.warning(
            f"Could not load config.json from {config_path}, using defaults. Error: {e}")
        cfg = {}

    # Always use route_efficiency mode
    current_mode = "route_efficiency"

    # Get route efficiency configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("route_efficiency", {})

    logger = get_logger()
    logger.info(f"🎯 Using optimization mode: ROUTE EFFICIENCY")

    # Validate and set configuration with mode-specific overrides
    config = {}

    # Distance configurations with mode overrides
    config['MAX_FILL_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("max_fill_distance_km",
                            cfg.get("max_fill_distance_km", 5.0))))
    config['MERGE_DISTANCE_KM'] = max(
        0.1,
        float(
            mode_config.get("merge_distance_km",
                            cfg.get("merge_distance_km", 3.0))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 1.5)))
    config['OVERFLOW_PENALTY_KM'] = max(
        0.0, float(cfg.get("overflow_penalty_km", 10.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(
        0.1, float(cfg.get("distance_issue_threshold_km", 8.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(
        0.0, float(cfg.get("swap_improvement_threshold_km", 0.5)))

    # Utilization thresholds (0-1 range)
    config['MIN_UTIL_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("min_util_threshold", 0.5))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.5))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan",
                                                      2)))
    config['MAX_SWAP_ITERATIONS'] = max(1,
                                        int(cfg.get("max_swap_iterations", 3)))
    config['MAX_USERS_FOR_FALLBACK'] = max(
        1, int(cfg.get("max_users_for_fallback", 3)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users",
                                                      2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users",
                                                      7)))

    # Angle configurations with mode overrides
    config['MAX_BEARING_DIFFERENCE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_bearing_difference",
                                cfg.get("max_bearing_difference", 20)))))
    config['MAX_TURNING_ANGLE'] = max(
        0,
        min(
            180,
            float(
                mode_config.get("max_allowed_turning_score",
                                cfg.get("max_allowed_turning_score", 35)))))

    # Cost penalties with mode overrides
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(
        0.0,
        float(
            mode_config.get("utilization_penalty_per_seat",
                            cfg.get("utilization_penalty_per_seat", 2.0))))

    # Office coordinates with environment variable fallbacks
    office_lat = float(
        os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(
        os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    # Validate coordinate bounds
    if not (-90 <= office_lat <= 90):
        logger.warning(f"Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        logger.warning(f"Invalid office longitude {office_lon}, using default")
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Route efficiency parameters (permanent)
    config['optimization_mode'] = "route_efficiency"
    config['aggressive_merging'] = mode_config.get('aggressive_merging', False)
    config['capacity_weight'] = mode_config.get('capacity_weight', 1.0)
    config['direction_weight'] = mode_config.get('direction_weight', 3.0)

    # Clustering and optimization parameters with mode overrides
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 3))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)
    config['angular_sectors'] = cfg.get('angular_sectors', 8)
    config['max_users_per_initial_cluster'] = cfg.get(
        'max_users_per_initial_cluster', 8)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 7)

    # Route efficiency parameters
    config['zigzag_penalty_weight'] = mode_config.get(
        'zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 3.0))
    config['route_split_turning_threshold'] = cfg.get(
        'route_split_turning_threshold', 35)
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 1.4)
    config['route_split_consistency_threshold'] = cfg.get(
        'route_split_consistency_threshold', 0.7)
    config['merge_tortuosity_improvement_required'] = cfg.get(
        'merge_tortuosity_improvement_required', True)

    # Latitude conversion factor for distance normalization
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    logger.info(
        f"   📊 Max bearing difference: {config['MAX_BEARING_DIFFERENCE']}°")
    logger.info(f"   📊 Max turning score: {config['MAX_TURNING_ANGLE']}°")
    logger.info(f"   📊 Max fill distance: {config['MAX_FILL_DISTANCE_KM']}km")
    logger.info(f"   📊 Capacity weight: {config['capacity_weight']}")
    logger.info(f"   📊 Direction weight: {config['direction_weight']}")

    return config


# Load validated configuration - always route efficiency
_config = load_and_validate_config()
MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM']
MIN_UTIL_THRESHOLD = _config['MIN_UTIL_THRESHOLD']
DBSCAN_EPS_KM = _config['DBSCAN_EPS_KM']
MIN_SAMPLES_DBSCAN = _config['MIN_SAMPLES_DBSCAN']
MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
SWAP_IMPROVEMENT_THRESHOLD = _config['SWAP_IMPROVEMENT_THRESHOLD']
MAX_SWAP_ITERATIONS = _config['MAX_SWAP_ITERATIONS']
UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']
OVERFLOW_PENALTY_KM = _config['OVERFLOW_PENALTY_KM']
DISTANCE_ISSUE_THRESHOLD = _config['DISTANCE_ISSUE_THRESHOLD']
LOW_UTILIZATION_THRESHOLD = _config['LOW_UTILIZATION_THRESHOLD']
MAX_USERS_FOR_FALLBACK = _config['MAX_USERS_FOR_FALLBACK']
FALLBACK_MIN_USERS = _config['FALLBACK_MIN_USERS']
FALLBACK_MAX_USERS = _config['FALLBACK_MAX_USERS']
OFFICE_LAT = _config['OFFICE_LAT']
OFFICE_LON = _config['OFFICE_LON']


def validate_input_data(data):
    """Comprehensive data validation with bounds checking"""
    logger = get_logger()
    if not isinstance(data, dict):
        raise ValueError("API response must be a dictionary")

    # Check for users
    users = data.get("users", [])
    if not users:
        raise ValueError("No users found in API response")

    if not isinstance(users, list):
        raise ValueError("Users must be a list")

    # Special handling for empty users
    if len(users) == 0:
        raise ValueError("Empty users list")

    # Validate each user comprehensively
    for i, user in enumerate(users):
        if not isinstance(user, dict):
            raise ValueError(f"User {i} must be a dictionary")

        required_fields = ["id", "latitude", "longitude"]
        for field in required_fields:
            if field not in user:
                raise ValueError(f"User {i} missing required field: {field}")
            if user[field] is None or user[field] == "":
                raise ValueError(f"User {i} has null/empty {field}")

        # Validate coordinate bounds
        try:
            lat = float(user["latitude"])
            lon = float(user["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(
                    f"User {i} invalid latitude: {lat} (must be -90 to 90)")
            if not (-180 <= lon <= 180):
                raise ValueError(
                    f"User {i} invalid longitude: {lon} (must be -180 to 180)")
        except (ValueError, TypeError) as e:
            raise ValueError(f"User {i} invalid coordinates: {e}")

    # Get all drivers from both sources
    all_drivers = []

    # Check nested format first
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))

    # Check flat format
    if not all_drivers:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    if not all_drivers:
        raise ValueError("No drivers found in API response")

    # Validate drivers comprehensively (allow duplicates for pick/drop scenarios)
    duplicate_driver_count = 0
    for i, driver in enumerate(all_drivers):
        if not isinstance(driver, dict):
            raise ValueError(f"Driver {i} must be a dictionary")

        required_fields = ["id", "capacity", "latitude", "longitude"]
        for field in required_fields:
            if field not in driver:
                raise ValueError(f"Driver {i} missing required field: {field}")
            if driver[field] is None or driver[field] == "":
                raise ValueError(f"Driver {i} has null/empty {field}")

        # Count duplicates but don't error (legitimate for pick/drop scenarios)
        driver_id = str(driver["id"])
        duplicate_count = sum(1 for d in all_drivers
                              if str(d.get("id", "")) == driver_id)
        if duplicate_count > 1:
            duplicate_driver_count += 1

        # Validate driver coordinates
        try:
            lat = float(driver["latitude"])
            lon = float(driver["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(f"Driver {i} invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Driver {i} invalid longitude: {lon}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid coordinates: {e}")

        # Validate capacity
        try:
            capacity = int(driver["capacity"])
            if capacity <= 0:
                raise ValueError(
                    f"Driver {i} invalid capacity: {capacity} (must be > 0)")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid capacity: {e}")

    if duplicate_driver_count > 0:
        logger.info(
            f"ℹ️ INFO: Found {duplicate_driver_count} duplicate driver entries (normal for pick/drop scenarios)"
        )

    logger.info(
        f"✅ Input data validation passed - {len(users)} users, {len(all_drivers)} drivers"
    )


def load_env_and_fetch_data(source_id: str,
                            parameter: int = 1,
                            string_param: str = "",
                            choice: str = ""):
    """Load environment variables and fetch data from API"""
    logger = get_logger()
    # Get absolute path to .env file - go up 3 levels from algorithm/base/ to project root
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go up from algorithm/base/ to project root
    env_path = os.path.join(script_dir, '.env')

    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    BASE_API_URL = os.getenv("API_URL")
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
    if not BASE_API_URL or not API_AUTH_TOKEN:
        raise ValueError("Both API_URL and API_AUTH_TOKEN must be set in .env")

    # Send all parameters along with source_id in the API URL
    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{string_param}/{choice}"
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    logger.info(f"📡 Making API call to: {API_URL}")
    resp = requests.get(API_URL, headers=headers)
    resp.raise_for_status()

    # Check if response body is empty
    if len(resp.text.strip()) == 0:
        raise ValueError(
            f"API returned empty response body. "
            f"Status: {resp.status_code}, "
            f"Content-Type: {resp.headers.get('content-type', 'unknown')}, "
            f"URL: {API_URL}")

    try:
        payload = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(
            f"API returned invalid JSON. "
            f"Status: {resp.status_code}, "
            f"Content-Type: {resp.headers.get('content-type', 'unknown')}, "
            f"Response body: '{resp.text[:200]}...', "
            f"JSON Error: {str(e)}")

    if not payload.get("status") or "data" not in payload:
        raise ValueError(
            "Unexpected response format: 'status' or 'data' missing")

    # Use the provided parameters
    data = payload["data"]
    data["_parameter"] = parameter
    data["_string_param"] = string_param
    data["_choice"] = choice

    # Handle nested drivers structure
    if "drivers" in data:
        drivers_data = data["drivers"]
        data["driversUnassigned"] = drivers_data.get("driversUnassigned", [])
        data["driversAssigned"] = drivers_data.get("driversAssigned", [])
    else:
        data["driversUnassigned"] = data.get("driversUnassigned", [])
        data["driversAssigned"] = data.get("driversAssigned", [])

    # Extract safety flag first - this overrides ride settings
    safety_flag = data.get("safety", 0)

    # Extract ride_settings to determine algorithm
    ride_settings = data.get("ride_settings", {})
    pic_priority = ride_settings.get("pic_priority")
    drop_priority = ride_settings.get("drop_priority")

    # Determine algorithm priority based on safety flag and ride_settings
    if safety_flag == 1:
        # Safety flag is set to 1 - force safety algorithm regardless of ride settings
        algorithm_priority = 2  # Safety algorithm priority
        logger.info(f"🔒 SAFETY FLAG DETECTED (safety=1) - Forcing safety algorithm regardless of ride settings")
    else:
        # Safety flag is 0 or missing - use ride settings as before
        algorithm_priority = pic_priority if pic_priority is not None else drop_priority
        logger.info(f"📊 Using ride_settings for algorithm selection (safety={safety_flag})")

    data["_algorithm_priority"] = algorithm_priority
    data["_safety_flag"] = safety_flag

    # Log the data structure for debugging
    logger.info(f"📊 API Response structure:")
    logger.info(f"   - users: {len(data.get('users', []))}")
    logger.info(
        f"   - driversUnassigned: {len(data.get('driversUnassigned', []))}")
    logger.info(
        f"   - driversAssigned: {len(data.get('driversAssigned', []))}")
    logger.info(f"   - ride_settings: {ride_settings}")
    logger.info(f"   - safety flag: {safety_flag}")
    logger.info(f"   - final algorithm priority: {algorithm_priority} ({'SAFETY ALGORITHM' if safety_flag == 1 else 'REGULAR ALGORITHM'})")

    return data


def extract_office_coordinates(data):
    """Extract dynamic office coordinates from API data"""
    company_data = data.get("company", {})
    office_lat = float(company_data.get("latitude", OFFICE_LAT))
    office_lon = float(company_data.get("longitude", OFFICE_LON))
    return office_lat, office_lon


def prepare_user_driver_dataframes(data):
    """Prepare user and driver dataframes from API data"""
    # Prepare user DataFrame
    users = data.get("users", [])
    user_data = []
    for user in users:
        user_data.append({
            'user_id':
            str(user.get('id', '')),
            'latitude':
            float(user.get('latitude', 0.0)),
            'longitude':
            float(user.get('longitude', 0.0)),
            'first_name':
            str(user.get('first_name', '')),
            'email':
            str(user.get('email', '')),
            'office_distance':
            float(user.get('office_distance', 0.0))
        })

    user_df = pd.DataFrame(user_data)

    # Prepare driver DataFrame
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    driver_data = []
    for i, driver in enumerate(all_drivers):
        driver_data.append({
            'driver_id': str(driver.get('id', '')),
            'latitude': float(driver.get('latitude', 0.0)),
            'longitude': float(driver.get('longitude', 0.0)),
            'capacity': int(driver.get('capacity', 1)),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'priority': i + 1  # Simple priority based on order
        })

    driver_df = pd.DataFrame(driver_data)

    return user_df, driver_df


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


def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized calculation of bearing from point A to B in degrees"""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def calculate_bearings_and_features(user_df, office_lat, office_lon):
    """Calculate bearings and add geographical features - OFFICE TO USER direction"""
    user_df = user_df.copy()

    # Calculate bearing FROM OFFICE TO USER
    user_df['bearing_from_office'] = calculate_bearing_vectorized(
        office_lat, office_lon, user_df['latitude'], user_df['longitude'])
    user_df['bearing_sin'] = np.sin(np.radians(user_df['bearing_from_office']))
    user_df['bearing_cos'] = np.cos(np.radians(user_df['bearing_from_office']))

    return user_df


def coords_to_km(lat, lon, office_lat, office_lon):
    """Convert lat/lon coordinates to km from office using local approximation"""
    lat_km = (lat - office_lat) * _config['LAT_TO_KM']
    lon_km = (lon - office_lon) * _config['LON_TO_KM']
    return lat_km, lon_km


def dbscan_clustering_metric(user_df, eps_km, min_samples, office_lat,
                             office_lon):
    """Perform DBSCAN clustering using proper metric coordinates"""
    # Convert coordinates to km from office
    coords_km = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'],
                                      office_lat, office_lon)
        coords_km.append([lat_km, lon_km])

    coords_km = np.array(coords_km)

    # Use DBSCAN with eps in km (no scaling needed now)
    dbscan = DBSCAN(eps=eps_km, min_samples=min_samples)
    labels = dbscan.fit_predict(coords_km)

    # Handle noise points: assign to nearest cluster if possible
    noise_mask = labels == -1
    if noise_mask.any():
        valid_labels = labels[~noise_mask]
        if len(valid_labels) > 0:
            # Find nearest cluster for each noise point
            for i in np.where(noise_mask)[0]:
                noise_point = coords_km[i]
                distances = cdist([noise_point], coords_km[~noise_mask])[0]
                nearest_cluster_idx = np.argmin(distances)
                labels[i] = valid_labels[nearest_cluster_idx]
        else:
            # If all points are noise, assign to a single cluster
            labels[:] = 0
    return labels


def kmeans_clustering_metric(user_df, n_clusters, office_lat, office_lon):
    """Perform KMeans clustering using metric coordinates"""
    # Convert coordinates to km from office
    coords_km = []
    user_ids = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'],
                                      office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
        user_ids.append(user['user_id'])

    # Sort by user_id for deterministic ordering
    sorted_data = sorted(zip(user_ids, coords_km), key=lambda x: x[0])
    coords_km = np.array([item[1] for item in sorted_data])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_km)

    # Map labels back to original order
    label_map = {
        user_id: label
        for (user_id, _), label in zip(sorted_data, labels)
    }
    return [label_map[user_id] for user_id in user_df['user_id']]


def estimate_clusters(user_df, config, office_lat, office_lon):
    """Estimate optimal number of clusters using silhouette score with metric coordinates"""
    # Convert coordinates to km from office
    coords_km = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'],
                                      office_lat, office_lon)
        coords_km.append([lat_km, lon_km])

    coords_km = np.array(coords_km)

    max_clusters = min(10, len(user_df) // 2)
    if max_clusters < 2:
        return 1

    scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_km)
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(coords_km, cluster_labels)
            scores.append((n_clusters, score))

    if not scores:
        return 1

    best_n_clusters = max(scores, key=lambda item: item[1])[0]
    return best_n_clusters


# STEP 1: DIRECTION-AWARE GEOGRAPHIC CLUSTERING
def create_geographic_clusters(user_df, office_lat, office_lon, config):
    """Create direction-aware geographic clusters using proper distance metrics"""
    if len(user_df) == 0:
        return user_df

    logger = get_logger()
    logger.info("  🗺️  Creating direction-aware geographic clusters...")

    # Calculate features including bearings
    user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)

    # Use sector-based clustering for direction awareness
    use_sweep = config.get('use_sweep_algorithm', True)

    if use_sweep and len(user_df) > 3:
        labels = sweep_clustering(user_df, config)
    else:
        labels = polar_sector_clustering(user_df, office_lat, office_lon,
                                         config)

    user_df['geo_cluster'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    logger.info(
        f"  ✅ Created {n_clusters} direction-aware geographic clusters")
    return user_df


def sweep_clustering(user_df, config):
    """Sweep algorithm: sort by bearing and group by capacity"""
    # Sort users by bearing from office
    sorted_df = user_df.sort_values('bearing_from_office')

    labels = []
    current_cluster = 0
    current_capacity = 0
    max_capacity = config.get('max_users_per_initial_cluster', 8)

    for idx, user in sorted_df.iterrows():
        if current_capacity >= max_capacity:
            current_cluster += 1
            current_capacity = 0

        labels.append(current_cluster)
        current_capacity += 1

    # Create label mapping back to original order
    result_labels = [-1] * len(user_df)
    for i, orig_idx in enumerate(sorted_df.index):
        result_labels[orig_idx] = labels[i]

    return result_labels


def polar_sector_clustering(user_df, office_lat, office_lon, config):
    """Partition into angular sectors then cluster within sectors using metric distances"""
    n_sectors = config.get('angular_sectors', 8)
    sector_angle = 360.0 / n_sectors

    # Assign users to sectors based on bearing
    user_df_copy = user_df.copy()
    user_df_copy['sector'] = (user_df_copy['bearing_from_office'] //
                              sector_angle).astype(int)

    labels = [-1] * len(user_df)
    current_cluster = 0

    # Cluster within each sector
    for sector in range(n_sectors):
        sector_users = user_df_copy[user_df_copy['sector'] == sector]
        if len(sector_users) == 0:
            continue

        if len(sector_users) <= 3:
            # Small sectors get single cluster
            for idx in sector_users.index:
                labels[idx] = current_cluster
            current_cluster += 1
        else:
            # Use spatial clustering within sector with proper metric
            eps_km = config.get('DBSCAN_EPS_KM', 1.5)
            sector_labels = dbscan_clustering_metric(sector_users, eps_km, 2,
                                                     office_lat, office_lon)

            for i, idx in enumerate(sector_users.index):
                if sector_labels[i] == -1:
                    labels[idx] = current_cluster
                    current_cluster += 1
                else:
                    labels[idx] = current_cluster + sector_labels[i]

            current_cluster += max(sector_labels) + 1 if len(
                sector_labels) > 0 else 1

    return labels


# STEP 2: DIRECTION-AWARE CAPACITY SUB-CLUSTERING
def create_capacity_subclusters(user_df, office_lat, office_lon, config):
    """Split geographic clusters by capacity and bearing constraints with direction awareness"""
    if len(user_df) == 0:
        return user_df

    logger = get_logger()
    logger.info("  🚗 Creating direction-aware capacity-based sub-clusters...")

    user_df['sub_cluster'] = -1
    sub_cluster_counter = 0
    max_bearing_diff = config.get('MAX_BEARING_DIFFERENCE', 20)

    for geo_cluster in user_df['geo_cluster'].unique():
        if geo_cluster == -1:
            continue

        geo_cluster_users = user_df[user_df['geo_cluster'] == geo_cluster]

        if len(geo_cluster_users) <= config.get('max_users_per_cluster', 7):
            user_df.loc[geo_cluster_users.index,
                        'sub_cluster'] = sub_cluster_counter
            sub_cluster_counter += 1
        else:
            # Use bearing-weighted clustering for sub-clusters
            sub_cluster_counter = create_bearing_aware_subclusters(
                geo_cluster_users, user_df, sub_cluster_counter, config,
                max_bearing_diff)

    logger.info(
        f"  ✅ Created {user_df['sub_cluster'].nunique()} direction-aware capacity-based sub-clusters"
    )
    return user_df


def create_bearing_aware_subclusters(geo_cluster_users, user_df,
                                     sub_cluster_counter, config,
                                     max_bearing_diff):
    """Create subclusters that maintain directional consistency"""
    # Sort by bearing to maintain direction
    sorted_users = geo_cluster_users.sort_values('bearing_from_office')
    max_users_per_cluster = config.get('max_users_per_cluster', 7)

    current_cluster_users = []

    for idx, user in sorted_users.iterrows():
        # Check if adding this user would violate bearing constraints
        if current_cluster_users:
            bearing_spread = calculate_bearing_spread(
                [u[1] for u in current_cluster_users] + [user])
            if len(
                    current_cluster_users
            ) >= max_users_per_cluster or bearing_spread > max_bearing_diff:
                # Assign current cluster
                for cluster_user_idx, _ in current_cluster_users:
                    user_df.loc[cluster_user_idx,
                                'sub_cluster'] = sub_cluster_counter
                sub_cluster_counter += 1
                current_cluster_users = []

        current_cluster_users.append((idx, user))

    # Assign remaining users
    if current_cluster_users:
        for cluster_user_idx, _ in current_cluster_users:
            user_df.loc[cluster_user_idx, 'sub_cluster'] = sub_cluster_counter
        sub_cluster_counter += 1

    return sub_cluster_counter


def calculate_bearing_spread(users):
    """Calculate the angular spread of users"""
    if len(users) <= 1:
        return 0

    bearings = [user['bearing_from_office'] for user in users]
    bearings.sort()

    # Handle circular nature of bearings
    max_gap = 0
    for i in range(len(bearings)):
        gap = bearings[(i + 1) % len(bearings)] - bearings[i]
        if gap < 0:
            gap += 360
        max_gap = max(max_gap, gap)

    # Return the complement of the largest gap (actual spread)
    return 360 - max_gap if max_gap > 180 else max_gap


# STEP 3: SEQUENCE-AWARE DRIVER ASSIGNMENT
def assign_drivers_by_priority(user_df, driver_df, office_lat, office_lon):
    """
    Step 3: Assign drivers based on priority and proximity using sequence-aware cost
    """
    logger = get_logger()
    logger.info("🚗 Step 3: Assigning drivers by priority...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort drivers by priority, then by driver_id for consistency
    available_drivers = driver_df.sort_values(
        ['priority', 'capacity', 'driver_id'], ascending=[True, False, True])

    # Process each sub-cluster in sorted order for consistency
    for sub_cluster_id in sorted(user_df['sub_cluster'].unique()):
        cluster_users = user_df[user_df['sub_cluster'] == sub_cluster_id]
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].
                                              isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        # Auto-split clusters that are larger than any available driver's capacity
        if not available_drivers.empty:
            max_driver_capacity = int(available_drivers['capacity'].max())
        else:
            max_driver_capacity = 0

        if max_driver_capacity > 0 and len(
                unassigned_in_cluster) > max_driver_capacity:
            parts = math.ceil(len(unassigned_in_cluster) / max_driver_capacity)
            logger.info(
                f"  📍 Auto-splitting large cluster {sub_cluster_id} ({len(unassigned_in_cluster)} users) into {parts} capacity-sized parts"
            )

            # Build feature matrix: km coords + bearing sin/cos
            coords = []
            rows = []
            for _, u in unassigned_in_cluster.iterrows():
                lat_km, lon_km = coords_to_km(u['latitude'], u['longitude'],
                                              office_lat, office_lon)
                coords.append([
                    lat_km, lon_km,
                    math.sin(math.radians(u['bearing_from_office'])),
                    math.cos(math.radians(u['bearing_from_office']))
                ])
                rows.append(u)
            coords = np.array(coords)

            # Safe fallback: if parts > len(points) just treat individually
            n_clusters = min(parts, len(coords))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            labels = kmeans.fit_predict(coords)

            # Assign each sub-part separately
            for p in range(n_clusters):
                part_indices = [
                    i for i, label in enumerate(labels) if label == p
                ]
                if not part_indices:
                    continue
                part_users = unassigned_in_cluster.iloc[part_indices]
                route = assign_best_driver_to_cluster(part_users,
                                                      available_drivers,
                                                      used_driver_ids,
                                                      office_lat, office_lon)
                if route:
                    routes.append(route)
                    assigned_user_ids.update(u['user_id']
                                             for u in route['assigned_users'])
            continue

        # Check bearing coherence and split if necessary
        bearings = unassigned_in_cluster['bearing_from_office'].values
        if len(bearings) > 1:
            bearing_diffs = []
            for i in range(len(bearings)):
                for j in range(i + 1, len(bearings)):
                    bearing_diffs.append(
                        bearing_difference(bearings[i], bearings[j]))

            if bearing_diffs and max(bearing_diffs) > MAX_BEARING_DIFFERENCE:
                logger.info(
                    f"  📍 Splitting sub-cluster {sub_cluster_id} due to bearing spread ({max(bearing_diffs):.1f}°)"
                )
                # Split into 2 sub-groups based on bearing with metric clustering
                split_routes = split_cluster_by_bearing_metric(
                    unassigned_in_cluster, available_drivers, used_driver_ids,
                    office_lat, office_lon)
                for route in split_routes:
                    if route:
                        routes.append(route)
                        assigned_user_ids.update(
                            u['user_id'] for u in route['assigned_users'])
                continue

        # Assign best driver to this cluster if not split
        route = assign_best_driver_to_cluster(unassigned_in_cluster,
                                              available_drivers,
                                              used_driver_ids, office_lat,
                                              office_lon)

        if route:
            routes.append(route)
            assigned_user_ids.update(u['user_id']
                                     for u in route['assigned_users'])

    # Apply route splitting for poor quality routes
    routes = apply_route_splitting(routes, available_drivers, used_driver_ids,
                                   office_lat, office_lon)

    logger.info(
        f"  ✅ Created {len(routes)} initial routes with priority-based assignment"
    )
    return routes, assigned_user_ids


def split_cluster_by_bearing_metric(cluster_users, available_drivers,
                                    used_driver_ids, office_lat, office_lon):
    """Split cluster using bearing-based K-means with metric coordinates"""
    split_routes = []

    # Convert to metric coordinates
    coords_km = []
    bearings = []
    for _, user in cluster_users.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'],
                                      office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
        bearings.append(user['bearing_from_office'])

    # Combine coordinates with bearing weights for clustering
    coords_with_bearing = np.column_stack([
        coords_km,
        np.sin(np.radians(bearings)),
        np.cos(np.radians(bearings))
    ])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    split_labels = kmeans.fit_predict(coords_with_bearing)

    # Process each split separately
    for split_id in range(2):
        split_users = cluster_users[split_labels == split_id]
        if len(split_users) > 0:
            route = assign_best_driver_to_cluster(split_users,
                                                  available_drivers,
                                                  used_driver_ids, office_lat,
                                                  office_lon)
            if route:
                split_routes.append(route)

    return split_routes


def assign_best_driver_to_cluster(cluster_users, available_drivers,
                                  used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver with mode-specific optimization"""
    logger = get_logger()
    cluster_size = len(cluster_users)

    best_driver = None
    min_cost = float('inf')
    best_sequence = None

    # Route efficiency weights (permanent)
    capacity_weight = _config.get('capacity_weight', 1.0)
    direction_weight = _config.get('direction_weight', 3.0)

    # Apply the requested change: replace logger.debug with temp_logger
    temp_logger = get_logger()
    temp_logger.debug(f"Assigning driver to cluster of {cluster_size} users")
    attempted_drivers = []

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            attempted_drivers.append((driver['driver_id'], "Already used"))
            continue

        # Strict capacity check for route efficiency
        if driver['capacity'] < cluster_size:
            attempted_drivers.append((
                driver['driver_id'],
                f"Insufficient capacity: {driver['capacity']} < {cluster_size}"
            ))
            continue

        # Calculate route cost
        route_cost, sequence, mean_turning_degrees = calculate_improved_route_cost(
            driver, cluster_users, office_lat, office_lon)

        # Priority penalty
        priority_penalty = driver['priority'] * 0.5

        # Route efficiency utilization calculation
        utilization = cluster_size / driver['capacity']

        # Light penalty for underutilization in efficiency mode
        utilization_bonus = utilization * capacity_weight * 0.5

        # Route efficiency zigzag penalty
        route_length_factor = max(1.0, cluster_size / 3.0)
        zigzag_penalty_weight = _config.get('zigzag_penalty_weight',
                                            3.0) * direction_weight

        # Very high penalty for zigzag in efficiency mode
        zigzag_penalty = mean_turning_degrees * zigzag_penalty_weight / 50.0

        total_cost = route_cost + priority_penalty - utilization_bonus + zigzag_penalty

        if total_cost < min_cost:
            min_cost = total_cost
            best_driver = driver
            best_sequence = sequence

    if best_driver is not None:
        used_driver_ids.add(best_driver['driver_id'])

        route = {
            'driver_id': str(best_driver['driver_id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': []
        }

        logger.info(
            f"Selected driver {best_driver['driver_id']} for {cluster_size} users"
        )

        # Add users to route in the optimal sequence
        users_to_assign = cluster_users[cluster_users['user_id'].isin(
            [u['user_id'] for u in best_sequence])]

        ordered_users_to_assign = []
        for seq_user in best_sequence:
            for _, cluster_user in users_to_assign.iterrows():
                if cluster_user['user_id'] == seq_user['user_id']:
                    ordered_users_to_assign.append(cluster_user)
                    break

        for user in ordered_users_to_assign:
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }

            if pd.notna(user.get('first_name')):
                user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                user_data['email'] = str(user['email'])

            route['assigned_users'].append(user_data)

        update_route_metrics_improved(route, office_lat, office_lon)
        return route

    # Log attempted drivers if no suitable driver found
    if not best_driver:
        logger.warning(
            f"No suitable driver found for cluster of {cluster_size} users. Attempted drivers: {attempted_drivers}"
        )

    return None


def calculate_improved_route_cost(driver, cluster_users, office_lat,
                                  office_lon):
    """Calculate route cost with mode-specific penalties"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence
    sequence = calculate_optimal_sequence_improved(driver_pos, cluster_users,
                                                   office_pos)

    # Calculate total route distance
    total_distance = 0
    bearing_differences = []

    # Driver to first pickup
    if sequence:
        first_user = sequence[0]
        total_distance += haversine_distance(driver_pos[0], driver_pos[1],
                                             first_user['latitude'],
                                             first_user['longitude'])

    # Between pickups - calculate bearing differences
    for i in range(len(sequence) - 1):
        current_user = sequence[i]
        next_user = sequence[i + 1]

        distance = haversine_distance(current_user['latitude'],
                                      current_user['longitude'],
                                      next_user['latitude'],
                                      next_user['longitude'])
        total_distance += distance

        # Calculate bearing difference between segments
        if i == 0:
            prev_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                             current_user['latitude'],
                                             current_user['longitude'])
        else:
            prev_pos = (sequence[i - 1]['latitude'],
                        sequence[i - 1]['longitude'])
            prev_bearing = calculate_bearing(prev_pos[0], prev_pos[1],
                                             current_user['latitude'],
                                             current_user['longitude'])

        next_bearing = calculate_bearing(current_user['latitude'],
                                         current_user['longitude'],
                                         next_user['latitude'],
                                         next_user['longitude'])

        bearing_diff = bearing_difference(prev_bearing, next_bearing)
        bearing_differences.append(bearing_diff)

    # Last pickup to office
    if sequence:
        last_user = sequence[-1]
        total_distance += haversine_distance(last_user['latitude'],
                                             last_user['longitude'],
                                             office_lat, office_lon)

    # Calculate mean turning angle
    mean_turning_degrees = sum(bearing_differences) / len(
        bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees


def calculate_optimal_sequence_improved(driver_pos, cluster_users, office_pos):
    """Calculate optimal pickup sequence using geodesic bearing projection with outlier detection"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(
            cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(
        cluster_users, 'to_dict') else list(cluster_users)

    # Calculate main route bearing (driver to office)
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                           office_pos[0], office_pos[1])

    # Sort users by geodesic projection along route axis, then by user_id for consistency
    def geodesic_projection_score(user):
        # Distance from driver to user
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      user['latitude'], user['longitude'])

        # Bearing from driver to user
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                         user['latitude'], user['longitude'])

        # Bearing difference from main route direction
        bearing_diff = normalize_bearing_difference(user_bearing -
                                                    main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))

        # Geodesic projection: distance * cos(bearing_difference)
        projection = distance * math.cos(bearing_diff_rad)

        # Add user_id as tiebreaker for consistency
        return (projection, user['user_id'])

    users_list.sort(key=geodesic_projection_score, reverse=True)

    # Apply improved 2-opt with strict direction constraints
    return apply_strict_direction_aware_2opt(users_list, driver_pos,
                                             office_pos, main_route_bearing)


def apply_strict_direction_aware_2opt(sequence, driver_pos, office_pos,
                                      main_bearing):
    """Apply 2-opt improvements with strict bearing-based constraints"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    max_iterations = 3
    iteration = 0

    # Strict turning angle threshold based on route characteristics
    max_turning_threshold = _config.get('MAX_TURNING_ANGLE', 35)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        best_distance = calculate_sequence_distance(sequence, driver_pos,
                                                    office_pos)
        best_turning_score = calculate_sequence_turning_score_improved(
            sequence, driver_pos, office_pos)

        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                # Try 2-opt swap
                new_sequence = sequence[:i +
                                        1] + sequence[i + 1:j +
                                                      1][::-1] + sequence[j +
                                                                          1:]

                # Calculate new metrics
                new_distance = calculate_sequence_distance(
                    new_sequence, driver_pos, office_pos)
                new_turning_score = calculate_sequence_turning_score_improved(
                    new_sequence, driver_pos, office_pos)

                # Strict acceptance criteria: both distance AND direction must improve or maintain quality
                distance_improved = new_distance < best_distance * 0.995  # At least 0.5% improvement
                turning_acceptable = new_turning_score <= max(
                    best_turning_score, max_turning_threshold)

                # For strict direction compliance, require combined improvement
                combined_improvement = (best_distance - new_distance) + (
                    best_turning_score - new_turning_score) * 0.1

                if distance_improved and turning_acceptable and combined_improvement > 0.01:
                    sequence = new_sequence
                    best_distance = new_distance
                    best_turning_score = new_turning_score
                    improved = True
                    break
            if improved:
                break

    return sequence


def calculate_sequence_turning_score_improved(sequence, driver_pos,
                                              office_pos):
    """Calculate average bearing difference for a sequence"""
    if len(sequence) <= 1:
        return 0

    bearing_differences = []
    prev_bearing = None

    for i in range(len(sequence)):
        if i == 0:
            # Bearing from driver to first user
            current_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                                sequence[i]['latitude'],
                                                sequence[i]['longitude'])
            if len(sequence) == 1:
                # Single user: bearing from user to office
                next_bearing = calculate_bearing(sequence[i]['latitude'],
                                                 sequence[i]['longitude'],
                                                 office_pos[0], office_pos[1])
                bearing_diff = bearing_difference(current_bearing,
                                                  next_bearing)
                bearing_differences.append(bearing_diff)
            prev_bearing = current_bearing
            continue
        elif i == len(sequence) - 1:
            # Last user: bearing from current to office
            current_bearing = calculate_bearing(sequence[i - 1]['latitude'],
                                                sequence[i - 1]['longitude'],
                                                sequence[i]['latitude'],
                                                sequence[i]['longitude'])
            next_bearing = calculate_bearing(sequence[i]['latitude'],
                                             sequence[i]['longitude'],
                                             office_pos[0], office_pos[1])
        else:
            # Between users
            current_bearing = calculate_bearing(sequence[i - 1]['latitude'],
                                                sequence[i - 1]['longitude'],
                                                sequence[i]['latitude'],
                                                sequence[i]['longitude'])
            next_bearing = calculate_bearing(sequence[i]['latitude'],
                                             sequence[i]['longitude'],
                                             sequence[i + 1]['latitude'],
                                             sequence[i + 1]['longitude'])

        if prev_bearing is not None:
            bearing_diff = bearing_difference(prev_bearing, current_bearing)
            if bearing_diff > 0:
                bearing_differences.append(bearing_diff)

        prev_bearing = current_bearing

    return sum(bearing_differences) / len(
        bearing_differences) if bearing_differences else 0


def calculate_sequence_distance(sequence, driver_pos, office_pos):
    """Calculate total distance for a pickup sequence"""
    if not sequence:
        return 0

    total = haversine_distance(driver_pos[0], driver_pos[1],
                               sequence[0]['latitude'],
                               sequence[0]['longitude'])

    for i in range(len(sequence) - 1):
        total += haversine_distance(sequence[i]['latitude'],
                                    sequence[i]['longitude'],
                                    sequence[i + 1]['latitude'],
                                    sequence[i + 1]['longitude'])

    total += haversine_distance(sequence[-1]['latitude'],
                                sequence[-1]['longitude'], office_pos[0],
                                office_pos[1])

    return total


def normalize_bearing_difference(diff):
    """Normalize bearing difference to [-180, 180] range"""
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def apply_route_splitting(routes, available_drivers, used_driver_ids,
                          office_lat, office_lon):
    """Split routes with poor quality metrics into better sub-routes"""
    split_routes = []

    turning_threshold = _config.get('route_split_turning_threshold', 35)
    consistency_threshold = _config.get('route_split_consistency_threshold',
                                        0.7)

    for route in routes:
        if len(route['assigned_users']) <= 2:
            split_routes.append(route)
            continue

        # Calculate route quality metrics
        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)

        turning_score = calculate_route_turning_score_improved(
            route['assigned_users'], driver_pos, office_pos)
        direction_consistency = calculate_direction_consistency_improved(
            route['assigned_users'], driver_pos, office_pos)

        # Check if route needs splitting
        needs_split = (turning_score > turning_threshold
                       or direction_consistency < consistency_threshold)

        if needs_split and len(route['assigned_users']) >= 4:
            logger = get_logger()
            logger.info(
                f"  🔄 Splitting route {route['driver_id']} - turning: {turning_score:.1f}°"
            )

            # Split route by bearing sectors
            sub_routes = split_route_by_bearing_improved(
                route, available_drivers, used_driver_ids, office_lat,
                office_lon)
            split_routes.extend(sub_routes)
        else:
            split_routes.append(route)

    return split_routes


def split_route_by_bearing_improved(route, available_drivers, used_driver_ids,
                                    office_lat, office_lon):
    """Split a route into bearing-consistent sub-routes with capacity-aware clustering"""
    users = route['assigned_users']
    if len(users) <= 2:
        return [route]

    # Calculate bearings from office to each user and cluster them
    user_bearings = []
    coords_km = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'],
                                    user['lng'])
        user_bearings.append((user, bearing))
        lat_km, lon_km = coords_to_km(user['lat'], user['lng'], office_lat,
                                      office_lon)
        coords_km.append([lat_km, lon_km])

    # Calculate bearing spread to determine split strategy
    bearings = [b for _, b in user_bearings]
    bearing_spread = max(bearings) - min(bearings)
    if bearing_spread > 180:  # Handle circular nature
        bearing_spread = 360 - bearing_spread

    # Strategy selection based on route characteristics
    if bearing_spread > 60:
        # Large bearing spread: split by direction
        sub_routes = split_by_bearing_clusters_improved(
            route, available_drivers, used_driver_ids, office_lat,
            office_lon, coords_km, bearings)
    else:
        # Small bearing spread: split by distance clusters
        sub_routes = split_by_distance_clusters_improved(
            route, available_drivers, office_lat, office_lon, coords_km)

    # Optimize sequences and update metrics for all split routes
    for sub_route in sub_routes:
        sub_route = optimize_route_sequence_improved(sub_route, office_lat,
                                                     office_lon)
        update_route_metrics_improved(sub_route, office_lat, office_lon)

    return sub_routes if len(sub_routes) > 1 else [route]


def create_sub_route_improved(original_route, users, available_drivers,
                              used_driver_ids, office_lat, office_lon):
    """Create a new sub-route from split users with improved driver selection"""
    # Try to find an available driver
    users_center = calculate_users_center_improved(users)
    best_driver = None
    best_score = float('inf')

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue
        if driver['capacity'] < len(users):
            continue

        # Calculate suitability score
        distance = haversine_distance(driver['latitude'], driver['longitude'],
                                      users_center[0], users_center[1])
        utilization = len(users) / driver['capacity']
        score = distance - (utilization * 2.0)  # Prefer higher utilization

        if score < best_score:
            best_score = score
            best_driver = driver

    if best_driver is not None:
        # Create new route
        new_route = {
            'driver_id': str(best_driver['driver_id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': users
        }

        # Mark driver as used
        used_driver_ids.add(best_driver['driver_id'])

        return new_route

    return None


def calculate_users_center_improved(users):
    """Calculate center point of a list of users"""
    if not users:
        return (0, 0)

    avg_lat = sum(u['lat'] for u in users) / len(users)
    avg_lng = sum(u['lng'] for u in users) / len(users)
    return (avg_lat, avg_lng)


# STEP 4: LOCAL OPTIMIZATION
def local_optimization(routes, office_lat, office_lon):
    """
    Step 4: Local optimization within routes and between nearby routes
    """
    logger = get_logger()
    logger.info("🔧 Step 4: Local optimization...")

    improved = True
    iterations = 0

    while improved and iterations < MAX_SWAP_ITERATIONS:
        improved = False
        iterations += 1

        # Sort routes by driver_id for deterministic processing
        routes_with_index = [(i, route) for i, route in enumerate(routes)]
        routes_with_index.sort(key=lambda x: x[1]['driver_id'])

        # Optimize user sequence within each route
        for i, route in routes_with_index:
            original_metrics = (calculate_route_turning_score_improved(
                route['assigned_users'],
                (route['latitude'], route['longitude']),
                (office_lat, office_lon)),
                                calculate_route_cost_improved(
                                    route, office_lat, office_lon))

            optimized_route = optimize_route_sequence_improved(
                route, office_lat, office_lon)
            routes[i] = optimized_route

            new_metrics = (calculate_route_turning_score_improved(
                optimized_route['assigned_users'],
                (optimized_route['latitude'], optimized_route['longitude']),
                (office_lat, office_lon)),
                           calculate_route_cost_improved(
                               optimized_route, office_lat, office_lon))

            if new_metrics != original_metrics:
                improved = True

        # Try swapping users between nearby routes with strict quality control
        # Sort routes by driver_id before swapping for consistency
        routes.sort(key=lambda r: r['driver_id'])
        if try_user_swap_improved(routes, office_lat, office_lon):
            improved = True

    logger.info(f"  ✅ Local optimization completed in {iterations} iterations")
    return routes


def optimize_route_sequence_improved(route, office_lat, office_lon):
    """Optimize pickup sequence using improved direction-aware sorting"""
    if not route['assigned_users'] or len(route['assigned_users']) <= 1:
        return route

    users = route['assigned_users'].copy()
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    # Convert users to consistent format for sequencing
    users_for_sequencing = []
    for user in users:
        users_for_sequencing.append({
            'latitude': user['lat'],
            'longitude': user['lng'],
            'user_id': user['user_id']
        })

    # Calculate optimal sequence using improved method
    optimized_sequence = calculate_optimal_sequence_improved(
        driver_pos, users_for_sequencing, office_pos)

    # Convert back to original format
    final_sequence = []
    for seq_user in optimized_sequence:
        for orig_user in users:
            if orig_user['user_id'] == seq_user['user_id']:
                final_sequence.append(orig_user)
                break

    route['assigned_users'] = final_sequence
    return route


def calculate_route_cost_improved(route, office_lat, office_lon):
    """Calculate cost of a route with improved penalty structure"""
    if not route['assigned_users']:
        return 0

    # Basic distance cost
    total_cost = 0
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    # Cost from driver to users (in sequence)
    current_pos = driver_pos
    for user in route['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        total_cost += haversine_distance(current_pos[0], current_pos[1],
                                         user_pos[0], user_pos[1])
        current_pos = user_pos

    # Cost from last user to office
    total_cost += haversine_distance(current_pos[0], current_pos[1],
                                     office_pos[0], office_pos[1])

    # Reduced penalty for low utilization (rebalanced)
    utilization = len(route['assigned_users']) / route['vehicle_type']
    if utilization < _config['LOW_UTILIZATION_THRESHOLD']:
        utilization_penalty = (_config['LOW_UTILIZATION_THRESHOLD'] -
                               utilization) * 2.0  # Reduced from 5.0
        total_cost += utilization_penalty

    return total_cost


def calculate_route_turning_score_improved(users, driver_pos, office_pos):
    """Calculate the average turning angle for a route using proper bearing calculations"""
    if len(users) <= 1:
        return 0

    # Validate user format
    for user in users:
        if 'lat' not in user or 'lng' not in user:
            raise ValueError(
                f"User missing required keys 'lat'/'lng': {user.keys()}")

    bearing_differences = []
    prev_bearing = None

    # Calculate bearing differences between consecutive segments
    for i in range(len(users) + 1):  # +1 to include office segment
        if i == 0:
            # Driver to first user
            current_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                                users[0]['lat'],
                                                users[0]['lng'])
            if len(users) == 1:
                # Single user: bearing from user to office
                next_bearing = calculate_bearing(users[i]['lat'],
                                                 users[i]['lng'],
                                                 office_pos[0], office_pos[1])
                bearing_diff = bearing_difference(current_bearing,
                                                  next_bearing)
                bearing_differences.append(bearing_diff)
            prev_bearing = current_bearing
            continue
        elif i == len(users):
            # Last user to office
            if len(users) > 0:
                current_bearing = calculate_bearing(users[i - 1]['lat'],
                                                    users[i - 1]['lng'],
                                                    office_pos[0],
                                                    office_pos[1])
            else:
                continue
        else:
            # Between users
            current_bearing = calculate_bearing(users[i - 1]['lat'],
                                                users[i - 1]['lng'],
                                                users[i]['lat'],
                                                users[i]['lng'])

        if prev_bearing is not None:
            bearing_diff = bearing_difference(prev_bearing, current_bearing)
            if bearing_diff > 0:
                bearing_differences.append(bearing_diff)

        prev_bearing = current_bearing

    return sum(bearing_differences) / len(
        bearing_differences) if bearing_differences else 0


def calculate_direction_consistency_improved(users, driver_pos, office_pos):
    """Calculate direction consistency using proper bearing analysis"""
    if len(users) <= 1:
        return 1.0

    # Calculate main route bearing (driver to office)
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                     office_pos[0], office_pos[1])

    consistent_segments = 0
    total_segments = 0

    # Check each segment against main direction
    for i in range(len(users) + 1):  # +1 to include office segment
        if i == 0:
            # Driver to first user
            segment_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                                users[0]['lat'],
                                                users[0]['lng'])
        elif i == len(users):
            # Last user to office
            if len(users) > 0:
                segment_bearing = calculate_bearing(users[i - 1]['lat'],
                                                    users[i - 1]['lng'],
                                                    office_pos[0],
                                                    office_pos[1])
            else:
                continue
        else:
            # Between users
            segment_bearing = calculate_bearing(users[i - 1]['lat'],
                                                users[i - 1]['lng'],
                                                users[i]['lat'],
                                                users[i]['lng'])

        bearing_diff = bearing_difference(segment_bearing, main_bearing)
        if bearing_diff <= 45:  # Within 45 degrees of main direction
            consistent_segments += 1
        total_segments += 1

    return consistent_segments / total_segments if total_segments > 0 else 1.0


def try_user_swap_improved(routes, office_lat, office_lon):
    """Try swapping users between routes with improved quality control"""
    improvements = 0
    threshold = _config.get('SWAP_IMPROVEMENT_THRESHOLD', 0.5)

    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1, route2 = routes[i], routes[j]

            if not route1['assigned_users'] or not route2['assigned_users']:
                continue

            # Calculate distance between route centers
            center1 = calculate_route_center_improved(route1)
            center2 = calculate_route_center_improved(route2)
            route_distance = haversine_distance(center1[0], center1[1],
                                                center2[0], center2[1])

            # Skip distant route swaps
            max_swap_distance = _config.get('MERGE_DISTANCE_KM', 1.5) * 2.0
            if route_distance > max_swap_distance:
                continue

            # Sort users by user_id for deterministic swap attempts
            sorted_users1 = sorted(route1['assigned_users'],
                                   key=lambda u: u['user_id'])

            # Try swapping each user from route1 to route2
            for user1 in sorted_users1:
                if (len(route2['assigned_users']) + 1 <= route2['vehicle_type']
                        and len(route1['assigned_users']) > 1):

                    # Calculate current metrics
                    cost1_before = calculate_route_cost_improved(
                        route1, office_lat, office_lon)
                    cost2_before = calculate_route_cost_improved(
                        route2, office_lat, office_lon)
                    turn1_before = calculate_route_turning_score_improved(
                        route1['assigned_users'],
                        (route1['latitude'], route1['longitude']),
                        (office_lat, office_lon))
                    turn2_before = calculate_route_turning_score_improved(
                        route2['assigned_users'],
                        (route2['latitude'], route2['longitude']),
                        (office_lat, office_lon))

                    # Perform swap
                    route1['assigned_users'].remove(user1)
                    route2['assigned_users'].append(user1)

                    # Recalculate with optimized sequences
                    route1_optimized = optimize_route_sequence_improved(
                        route1, office_lat, office_lon)
                    route2_optimized = optimize_route_sequence_improved(
                        route2, office_lat, office_lon)

                    cost1_after = calculate_route_cost_improved(
                        route1_optimized, office_lat, office_lon)
                    cost2_after = calculate_route_cost_improved(
                        route2_optimized, office_lat, office_lon)
                    turn1_after = calculate_route_turning_score_improved(
                        route1_optimized['assigned_users'],
                        (route1_optimized['latitude'],
                         route1_optimized['longitude']),
                        (office_lat, office_lon))
                    turn2_after = calculate_route_turning_score_improved(
                        route2_optimized['assigned_users'],
                        (route2_optimized['latitude'],
                         route2_optimized['longitude']),
                        (office_lat, office_lon))

                    # Combined improvement check with proper weighting
                    distance_improvement = (cost1_before + cost2_before) - (
                        cost1_after + cost2_after)
                    turning_improvement = (turn1_before + turn2_before) - (
                        turn1_after + turn2_after)

                    # Weight turning improvement more heavily (convert degrees to km-equivalent)
                    total_improvement = distance_improvement + (
                        turning_improvement * 0.05)  # ~5km per degree

                    if total_improvement > threshold:
                        improvements += 1
                        # Keep the optimized sequences
                        routes[i] = route1_optimized
                        routes[j] = route2_optimized
                    else:
                        # Revert swap
                        route2['assigned_users'].remove(user1)
                        route1['assigned_users'].append(user1)

    return improvements > 0


def calculate_route_center_improved(route):
    """Calculate the geometric center of users in a route"""
    if not route['assigned_users']:
        return (route['latitude'], route['longitude'])

    lats = [u['lat'] for u in route['assigned_users']]
    lngs = [u['lng'] for u in route['assigned_users']]
    return (np.mean(lats), np.mean(lngs))


def update_route_metrics_improved(route, office_lat, office_lon):
    """Update route metrics with improved calculations"""
    if route['assigned_users']:
        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)

        route['centroid'] = calculate_users_center_improved(
            route['assigned_users'])
        route['utilization'] = len(
            route['assigned_users']) / route['vehicle_type']
        route['turning_score'] = calculate_route_turning_score_improved(
            route['assigned_users'], driver_pos, office_pos)
        route['tortuosity_ratio'] = calculate_tortuosity_ratio_improved(
            route['assigned_users'], driver_pos, office_pos)
        route[
            'direction_consistency'] = calculate_direction_consistency_improved(
                route['assigned_users'], driver_pos, office_pos)
    else:
        route['centroid'] = [route['latitude'], route['longitude']]
        route['utilization'] = 0
        route['turning_score'] = 0
        route['tortuosity_ratio'] = 1.0
        route['direction_consistency'] = 1.0


def calculate_tortuosity_ratio_improved(users, driver_pos, office_pos):
    """Calculate improved tortuosity ratio"""
    if not users:
        return 1.0

    # Actual route distance
    actual_distance = calculate_sequence_distance([{
        'latitude': u['lat'],
        'longitude': u['lng']
    } for u in users], driver_pos, office_pos)

    # Straight line distance (driver to centroid to office)
    if users:
        centroid_lat = sum(u['lat'] for u in users) / len(users)
        centroid_lng = sum(u['lng'] for u in users) / len(users)

        straight_distance = (haversine_distance(driver_pos[0], driver_pos[1],
                                                centroid_lat, centroid_lng) +
                             haversine_distance(centroid_lat, centroid_lng,
                                                office_pos[0], office_pos[1]))
    else:
        straight_distance = haversine_distance(driver_pos[0], driver_pos[1],
                                               office_pos[0], office_pos[1])

    return actual_distance / straight_distance if straight_distance > 0 else 1.0


# STEP 5: GLOBAL OPTIMIZATION
def global_optimization(routes, user_df, assigned_user_ids, driver_df,
                        office_lat, office_lon):
    """
    Step 5: Global optimization with improved single-route fixing and route quality management
    """
    logger = get_logger()
    logger.info("🌍 Step 5: Enhanced Global optimization...")

    # PHASE 1: Fix single-user routes first (highest priority)
    logger.info("  🎯 Phase 1: Fixing single-user routes...")
    routes = fix_single_user_routes_improved(routes, user_df,
                                             assigned_user_ids, driver_df,
                                             office_lat, office_lon)

    # PHASE 2: Fill underutilized routes with strict quality checks
    logger.info("  📈 Phase 2: Quality-controlled route filling...")
    unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids
                                                           )].copy()
    routes = quality_controlled_route_filling(routes, unassigned_users_df,
                                              assigned_user_ids, office_lat,
                                              office_lon)

    # PHASE 3: Merge underutilized routes with strict quality preservation
    logger.info("  🔗 Phase 3: Quality-preserving route merging...")
    routes = quality_preserving_route_merging(routes, driver_df, office_lat,
                                              office_lon)

    # PHASE 4: Split poor quality routes with improved logic
    logger.info("  ✂️ Phase 4: Enhanced route splitting...")
    routes = enhanced_route_splitting(routes, driver_df, office_lat,
                                      office_lon)

    # PHASE 5: Outlier detection and reassignment
    logger.info("  🔍 Phase 5: Outlier detection and reassignment...")
    routes, failed_outlier_reassignments = outlier_detection_and_reassignment(
        routes, office_lat, office_lon)

    # Handle remaining unassigned users
    remaining_unassigned_users_df = user_df[~user_df['user_id'].
                                            isin(assigned_user_ids)]
    unassigned_list = handle_remaining_users_improved(
        remaining_unassigned_users_df, driver_df, routes, office_lat,
        office_lon)

    # Add failed outlier reassignments to unassigned list
    if failed_outlier_reassignments:
        logger.info(
            f"  📋 Adding {len(failed_outlier_reassignments)} failed outlier reassignments to unassigned list"
        )
        unassigned_list.extend(failed_outlier_reassignments)

    logger.info("  ✅ Enhanced global optimization completed")
    return routes, unassigned_list


# STEP 6: FINAL-PASS MERGE ALGORITHM
def final_pass_merge(routes, config, office_lat, office_lon):
    """
    Final-pass merge algorithm: Loop over all pairs of routes and merge compatible ones
    """
    logger = get_logger()
    logger.info("🔄 Step 6: Final-pass merge algorithm...")

    merged_routes = []
    used = set()
    MERGE_BEARING_THRESHOLD = 30  # degrees
    MERGE_DISTANCE_KM = config.get("MERGE_DISTANCE_KM", 3.0)

    for i, r1 in enumerate(routes):
        if i in used:
            continue

        best_merge = None
        best_merge_quality = float('inf')

        for j, r2 in enumerate(routes):
            if j <= i or j in used:
                continue

            # 1. Check direction similarity
            b1 = calculate_average_bearing_improved(r1, office_lat, office_lon)
            b2 = calculate_average_bearing_improved(r2, office_lat, office_lon)
            bearing_diff = bearing_difference(b1, b2)

            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # 2. Check centroid distance
            c1 = calculate_route_center_improved(r1)
            c2 = calculate_route_center_improved(r2)
            centroid_distance = haversine_distance(c1[0], c1[1], c2[0], c2[1])

            if centroid_distance > MERGE_DISTANCE_KM:
                continue

            # 3. Check combined capacity
            total_users = len(r1['assigned_users']) + len(r2['assigned_users'])
            max_capacity = max(r1['vehicle_type'], r2['vehicle_type'])

            if total_users > max_capacity:
                continue

            # 4. Test merge quality
            # Choose better positioned driver
            combined_center = calculate_combined_route_center(r1, r2)
            dist1 = haversine_distance(r1['latitude'], r1['longitude'],
                                       combined_center[0], combined_center[1])
            dist2 = haversine_distance(r2['latitude'], r2['longitude'],
                                       combined_center[0], combined_center[1])

            better_route = r1 if dist1 <= dist2 else r2

            # Create test merged route
            test_route = better_route.copy()
            test_route[
                'assigned_users'] = r1['assigned_users'] + r2['assigned_users']
            test_route['vehicle_type'] = max_capacity

            # Optimize sequence for merged route
            test_route = optimize_route_sequence_improved(
                test_route, office_lat, office_lon)

            # Calculate quality metrics
            turning_score = calculate_route_turning_score_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            tortuosity = calculate_tortuosity_ratio_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            direction_consistency = calculate_direction_consistency_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            # Quality thresholds for merging
            MAX_TURNING_ANGLE = config.get('MAX_TURNING_ANGLE', 35)
            MAX_TORTUOSITY = 1.3
            MIN_DIRECTION_CONSISTENCY = 0.7

            # Accept merge only if quality is acceptable
            if (turning_score <= MAX_TURNING_ANGLE
                    and tortuosity <= MAX_TORTUOSITY
                    and direction_consistency >= MIN_DIRECTION_CONSISTENCY):

                # Calculate combined quality score (lower is better)
                quality_score = turning_score + (tortuosity - 1.0) * 20 + (
                    1.0 - direction_consistency) * 50

                if quality_score < best_merge_quality:
                    best_merge_quality = quality_score
                    best_merge = (j, test_route)

        if best_merge is not None:
            j, merged_route = best_merge
            merged_routes.append(merged_route)
            used.add(i)
            used.add(j)
            temp_logger = get_logger()
            temp_logger.info(
                f"  ✅ Merged routes {r1['driver_id']} and {routes[j]['driver_id']} (quality: {best_merge_quality:.1f})"
            )
        else:
            merged_routes.append(r1)
            used.add(i)

    logger.info(
        f"  🔄 Final-pass merge: {len(routes)} → {len(merged_routes)} routes")
    return merged_routes


def calculate_combined_route_center(route1, route2):
    """Calculate the center point of users from two routes combined"""
    all_users = route1['assigned_users'] + route2['assigned_users']
    if not all_users:
        return (0, 0)

    avg_lat = sum(u['lat'] for u in all_users) / len(all_users)
    avg_lng = sum(u['lng'] for u in all_users) / len(all_users)
    return (avg_lat, avg_lng)


def fix_single_user_routes_improved(routes, user_df, assigned_user_ids,
                                    driver_df, office_lat, office_lon):
    """Aggressively fix single-user routes with improved quality control"""
    logger = get_logger()
    logger.info("    🎯 Intelligently fixing single-user routes...")

    single_user_routes = []
    multi_user_routes = []

    # Separate routes by user count
    for route in routes:
        if len(route['assigned_users']) == 1:
            single_user_routes.append(route)
        else:
            multi_user_routes.append(route)

    logger.info(
        f"    📊 Found {len(single_user_routes)} single-user routes to optimize"
    )

    # Strategy 1: Merge single users into compatible multi-user routes
    routes_to_keep = []
    reassigned_count = 0

    for route in multi_user_routes:
        if len(route['assigned_users']) >= route['vehicle_type']:
            routes_to_keep.append(route)
            continue

        route_center = calculate_route_center_improved(route)
        route_bearing = calculate_average_bearing_improved(
            route, office_lat, office_lon)

        # Find compatible single-user routes to merge
        for single_route in single_user_routes[:]:
            if len(route['assigned_users']) >= route['vehicle_type']:
                break

            single_user = single_route['assigned_users'][0]
            user_pos = (single_user['lat'], single_user['lng'])

            # Check compatibility with stricter thresholds
            distance = haversine_distance(route_center[0], route_center[1],
                                          user_pos[0], user_pos[1])
            max_distance = _config.get('MAX_FILL_DISTANCE_KM',
                                       5.0) * 1.2  # Slightly more lenient

            if distance <= max_distance:
                user_bearing = calculate_bearing(office_lat, office_lon,
                                                 user_pos[0], user_pos[1])
                bearing_diff = bearing_difference(user_bearing, route_bearing)
                max_bearing_diff = _config.get(
                    'MAX_BEARING_DIFFERENCE',
                    20) * 1.3  # Slightly more lenient

                if bearing_diff <= max_bearing_diff:
                    # Test merge quality with strict thresholds
                    test_route = route.copy()
                    test_route['assigned_users'] = route['assigned_users'] + [
                        single_user
                    ]
                    test_route = optimize_route_sequence_improved(
                        test_route, office_lat, office_lon)

                    turning_score = calculate_route_turning_score_improved(
                        test_route['assigned_users'],
                        (test_route['latitude'], test_route['longitude']),
                        (office_lat, office_lon))

                    # Stricter quality thresholds for single-user merges
                    if turning_score <= 40:  # Stricter than before
                        route['assigned_users'].append(single_user)
                        single_user_routes.remove(single_route)
                        reassigned_count += 1
                        logger.info(
                            f"    ✅ Merged single user {single_user['user_id']} into route {route['driver_id']}"
                        )

        # Re-optimize the route after merging
        route = optimize_route_sequence_improved(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)
        routes_to_keep.append(route)

    # Strategy 2: Merge compatible single-user routes
    remaining_singles = single_user_routes[:]
    merged_singles = []

    while len(remaining_singles) >= 2:
        route1 = remaining_singles.pop(0)
        best_merge = None
        best_score = float('inf')

        for i, route2 in enumerate(remaining_singles):
            # Check if they can be merged
            combined_capacity = max(route1['vehicle_type'],
                                    route2['vehicle_type'])
            if combined_capacity >= 2:
                user1_pos = (route1['assigned_users'][0]['lat'],
                             route1['assigned_users'][0]['lng'])
                user2_pos = (route2['assigned_users'][0]['lat'],
                             route2['assigned_users'][0]['lng'])
                distance = haversine_distance(user1_pos[0], user1_pos[1],
                                              user2_pos[0], user2_pos[1])

                # Check bearing compatibility
                user1_bearing = calculate_bearing(office_lat, office_lon,
                                                  user1_pos[0], user1_pos[1])
                user2_bearing = calculate_bearing(office_lat, office_lon,
                                                  user2_pos[0], user2_pos[1])
                bearing_diff = bearing_difference(user1_bearing, user2_bearing)

                # Combined score considering distance and bearing
                score = distance + (
                    bearing_diff * 0.1
                )  # Convert degrees to distance-like metric

                if (distance <= _config.get('MERGE_DISTANCE_KM', 3.0) * 1.5
                        and bearing_diff <= 30 and score < best_score):
                    best_score = score
                    best_merge = (i, route2)

        if best_merge is not None:
            i, route2 = best_merge
            remaining_singles.pop(i)

            # Create merged route using better positioned driver
            center1 = (route1['assigned_users'][0]['lat'],
                       route1['assigned_users'][0]['lng'])
            center2 = (route2['assigned_users'][0]['lat'],
                       route2['assigned_users'][0]['lng'])
            combined_center = ((center1[0] + center2[0]) / 2,
                               (center1[1] + center2[1]) / 2)

            dist1 = haversine_distance(route1['latitude'], route1['longitude'],
                                       combined_center[0], combined_center[1])
            dist2 = haversine_distance(route2['latitude'], route2['longitude'],
                                       combined_center[0], combined_center[1])

            better_route = route1 if dist1 <= dist2 else route2
            merged_route = better_route.copy()
            merged_route['assigned_users'] = route1['assigned_users'] + route2[
                'assigned_users']
            merged_route['vehicle_type'] = max(route1['vehicle_type'],
                                               route2['vehicle_type'])

            merged_route = optimize_route_sequence_improved(
                merged_route, office_lat, office_lon)
            update_route_metrics_improved(merged_route, office_lat, office_lon)
            merged_singles.append(merged_route)
            reassigned_count += 1
            logger.info(
                f"    ✅ Merged two single-user routes into optimized route")
        else:
            break

    # Add remaining unmerged single routes
    final_routes = routes_to_keep + merged_singles + remaining_singles

    logger.info(
        f"    ✅ Optimized {reassigned_count} single-user assignments")
    logger.info(
        f"    📊 Reduced single-user routes from {len(single_user_routes)} to {len([r for r in final_routes if len(r['assigned_users']) == 1])}"
    )

    return final_routes


def calculate_average_bearing_improved(route, office_lat, office_lon):
    """Calculate the average bearing of users in a route towards the office"""
    if not route['assigned_users']:
        return calculate_bearing(route['latitude'], route['longitude'],
                                 office_lat, office_lon)

    avg_lat = np.mean([u['lat'] for u in route['assigned_users']])
    avg_lng = np.mean([u['lng'] for u in route['assigned_users']])

    return calculate_bearing(avg_lat, avg_lng, office_lat, office_lon)


def quality_controlled_route_filling(routes, unassigned_users_df,
                                     assigned_user_ids, office_lat,
                                     office_lon):
    """Fill routes with unassigned users using mode-specific quality control"""
    if unassigned_users_df.empty:
        return routes

    logger = get_logger()
    logger.info(
        f"    📋 ROUTE EFFICIENCY mode: assignment of {len(unassigned_users_df)} unassigned users"
    )

    # Sort routes by utilization (prioritize underutilized routes)
    routes_by_util = sorted(routes, key=lambda r: r.get('utilization', 1.0))

    assignments_made = 0

    for route in routes_by_util:
        if len(route['assigned_users']
               ) >= route['vehicle_type'] or unassigned_users_df.empty:
            continue

        route_center = route.get('centroid',
                                 [route['latitude'], route['longitude']])
        route_bearing = calculate_average_bearing_improved(
            route, office_lat, office_lon)

        # Route efficiency compatible user finding
        compatible_users = []
        for _, user in unassigned_users_df.iterrows():
            if user['user_id'] in assigned_user_ids:
                continue

            distance = haversine_distance(route_center[0], route_center[1],
                                          user['latitude'], user['longitude'])

            # Stricter distance for route efficiency
            utilization = len(route['assigned_users']) / route['vehicle_type']
            max_distance = _config['MAX_FILL_DISTANCE_KM'] * (
                1.0 if utilization > 0.7 else 0.8)

            if distance <= max_distance:
                user_bearing = user.get(
                    'bearing_from_office',
                    calculate_bearing(office_lat, office_lon, user['latitude'],
                                      user['longitude']))
                bearing_diff = bearing_difference(user_bearing, route_bearing)

                # Strict bearing constraints for route efficiency
                max_bearing = _config['MAX_BEARING_DIFFERENCE'] * 0.8

                if bearing_diff <= max_bearing:
                    compatible_users.append((user, distance, bearing_diff))

        if not compatible_users:
            continue

        # Sort by quality score
        compatible_users.sort(key=lambda x: x[1] + (x[2] * 0.05))

        slots_available = route['vehicle_type'] - len(route['assigned_users'])
        users_added_this_route = 0

        for user, distance, bearing_diff in compatible_users:
            if users_added_this_route >= slots_available or user[
                    'user_id'] in assigned_user_ids:
                break

            # Test route quality with new user
            test_route = route.copy()
            test_route['assigned_users'] = route['assigned_users'] + [
                {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }
            ]
            test_route = optimize_route_sequence_improved(
                test_route, office_lat, office_lon)

            new_turning = calculate_route_turning_score_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))
            new_tortuosity = calculate_tortuosity_ratio_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            # Strict quality thresholds for route efficiency
            max_turning = 25  # Very strict
            max_tortuosity = 1.2

            if new_turning <= max_turning and new_tortuosity <= max_tortuosity:
                route['assigned_users'].append({
                    'user_id':
                    str(user['user_id']),
                    'lat':
                    float(user['latitude']),
                    'lng':
                    float(user['longitude']),
                    'office_distance':
                    float(user.get('office_distance', 0))
                })
                assigned_user_ids.add(user['user_id'])
                users_added_this_route += 1
                assignments_made += 1

                logger.info(
                    f"    ✅ ROUTE EFFICIENCY-assigned user {user['user_id']} to route {route['driver_id']} (turn: {new_turning:.1f}°)"
                )

                route = optimize_route_sequence_improved(
                    route, office_lat, office_lon)
                update_route_metrics_improved(route, office_lat, office_lon)

    logger.info(
        f"    📊 ROUTE EFFICIENCY mode: assigned {assignments_made} users")
    return routes


def quality_preserving_route_merging(routes, driver_df, office_lat,
                                     office_lon):
    """Merge routes while strictly preserving quality metrics"""
    current_routes = routes.copy()
    merged_count = 0
    max_passes = 2  # Reduced for conservative approach

    for pass_num in range(max_passes):
        merged_routes_this_pass = []
        used_route_indices = set()
        pass_merges = 0

        # Only consider underutilized routes
        underutilized_routes = [
            (i, r) for i, r in enumerate(current_routes)
            if r.get('utilization', 1) < 0.6 and len(r['assigned_users']) > 0
        ]

        for orig_i, route_a in underutilized_routes:
            if orig_i in used_route_indices:
                continue

            best_merge_candidate = None
            best_candidate_index = None
            best_quality_score = float('inf')

            for orig_j, route_b in underutilized_routes:
                if orig_j in used_route_indices or orig_j == orig_i:
                    continue

                # Strict compatibility check
                if strict_merge_compatibility_improved(route_a, route_b,
                                                       office_lat, office_lon):
                    # Calculate merge quality score
                    quality_score = calculate_merge_quality_score(
                        route_a, route_b, office_lat, office_lon)

                    if quality_score < best_quality_score:
                        best_quality_score = quality_score
                        best_merge_candidate = route_b
                        best_candidate_index = orig_j

            if best_merge_candidate is not None:
                # Perform merge with quality optimization
                merged_route = perform_quality_merge_improved(
                    route_a, best_merge_candidate, office_lat, office_lon)
                merged_routes_this_pass.append(merged_route)
                used_route_indices.add(orig_i)
                used_route_indices.add(best_candidate_index)
                pass_merges += 1
                merged_count += 1
            else:
                merged_routes_this_pass.append(route_a)
                used_route_indices.add(orig_i)

        # Add routes that weren't considered for merging
        for i, route in enumerate(current_routes):
            if i not in used_route_indices:
                merged_routes_this_pass.append(route)

        current_routes = merged_routes_this_pass

        if pass_merges == 0:
            break

    if merged_count > 0:
        logger = get_logger()
        logger.info(
            f"    🔗 Quality-preserving merges: {merged_count}, Final routes: {len(current_routes)}"
        )

    return current_routes


def strict_merge_compatibility_improved(route1, route2, office_lat,
                                        office_lon):
    """Very strict compatibility check for merging with quality preservation"""
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])

    if total_users > max_capacity:
        return False

    # Strict distance constraint
    center1 = calculate_route_center_improved(route1)
    center2 = calculate_route_center_improved(route2)
    distance = haversine_distance(center1[0], center1[1], center2[0],
                                  center2[1])

    if distance > _config.get('MERGE_DISTANCE_KM', 1.5) * 0.8:
        return False

    # Strict bearing constraint
    bearing1 = calculate_average_bearing_improved(route1, office_lat,
                                                  office_lon)
    bearing2 = calculate_average_bearing_improved(route2, office_lat,
                                                  office_lon)
    bearing_diff = bearing_difference(bearing1, bearing2)

    if bearing_diff > _config.get('MAX_BEARING_DIFFERENCE', 20) * 0.7:
        return False

    # Both routes must be significantly underutilized
    util1 = route1.get('utilization', 1)
    util2 = route2.get('utilization', 1)
    if util1 > 0.7 or util2 > 0.7:
        return False

    return True


def calculate_merge_quality_score(route1, route2, office_lat, office_lon):
    """Calculate quality score for potential merge (lower is better)"""
    # Simulate merge
    all_users = route1['assigned_users'] + route2['assigned_users']

    # Choose better positioned driver
    combined_center = calculate_combined_route_center(route1, route2)
    dist1 = haversine_distance(route1['latitude'], route1['longitude'],
                               combined_center[0], combined_center[1])
    dist2 = haversine_distance(route2['latitude'], route2['longitude'],
                               combined_center[0], combined_center[1])

    better_route = route1 if dist1 <= dist2 else route2

    # Create test merged route
    test_route = better_route.copy()
    test_route['assigned_users'] = all_users
    test_route['vehicle_type'] = max(route1['vehicle_type'],
                                     route2['vehicle_type'])

    # Calculate quality metrics
    turning_score = calculate_route_turning_score_improved(
        test_route['assigned_users'],
        (test_route['latitude'], test_route['longitude']),
        (office_lat, office_lon))
    tortuosity = calculate_tortuosity_ratio_improved(
        test_route['assigned_users'],
        (test_route['latitude'], test_route['longitude']),
        (office_lat, office_lon))

    direction_consistency = calculate_direction_consistency_improved(
        test_route['assigned_users'],
        (test_route['latitude'], test_route['longitude']),
        (office_lat, office_lon))

    # Combined quality score (lower is better)
    quality_score = turning_score + (tortuosity -
                                     1.0) * 20  # Normalize tortuosity penalty

    return quality_score


def perform_quality_merge_improved(route1, route2, office_lat, office_lon):
    """Perform merge with improved quality optimization"""
    # Choose better positioned driver
    all_users = route1['assigned_users'] + route2['assigned_users']
    combined_center = calculate_combined_route_center(route1, route2)

    dist1 = haversine_distance(route1['latitude'], route1['longitude'],
                               combined_center[0], combined_center[1])
    dist2 = haversine_distance(route2['latitude'], route2['longitude'],
                               combined_center[0], combined_center[1])

    better_route = route1 if dist1 <= dist2 else route2

    merged_route = better_route.copy()
    merged_route['assigned_users'] = all_users
    merged_route['vehicle_type'] = max(route1['vehicle_type'],
                                       route2['vehicle_type'])

    # Optimize and update metrics
    merged_route = optimize_route_sequence_improved(merged_route, office_lat,
                                                    office_lon)
    update_route_metrics_improved(merged_route, office_lat, office_lon)

    return merged_route


def enhanced_route_splitting(routes, driver_df, office_lat, office_lon):
    """Enhanced route splitting with intelligent clustering and driver allocation"""
    logger = get_logger()
    improved_routes = []
    available_drivers = driver_df[~driver_df['driver_id'].
                                  isin([r['driver_id']
                                        for r in routes])].copy()

    # Stricter thresholds for splitting
    turning_threshold = _config.get('route_split_turning_threshold',
                                    30)  # Reduced
    tortuosity_threshold = _config.get('max_tortuosity_ratio', 1.3)  # Reduced
    consistency_threshold = _config.get('route_split_consistency_threshold',
                                        0.8)  # Increased

    routes_split = 0

    for route in routes:
        if len(route['assigned_users']) < 3:  # Can't split small routes
            improved_routes.append(route)
            continue

        # Calculate current quality metrics
        driver_pos = (route['latitude'], route['longitude'])
        turning_score = route.get('turning_score', 0)
        tortuosity = route.get('tortuosity_ratio', 1.0)
        consistency = route.get('direction_consistency', 1.0)

        # Enhanced splitting criteria
        needs_split = (
            turning_score > turning_threshold
            or tortuosity > tortuosity_threshold
            or consistency < consistency_threshold
            or (len(route['assigned_users']) >= 5 and turning_score > 25
                )  # Split large routes with moderate issues
        )

        if needs_split and len(available_drivers) > 0:
            logger.info(
                f"    🔄 Enhanced splitting - users: {len(route['assigned_users'])}, turning: {turning_score:.1f}°"
            )
            split_routes = intelligent_route_splitting_improved(
                route, available_drivers, office_lat, office_lon)

            if len(split_routes) > 1:  # Successfully split
                improved_routes.extend(split_routes)
                routes_split += 1

                # Remove used drivers from available pool
                used_driver_ids = [
                    sr['driver_id'] for sr in split_routes
                    if sr['driver_id'] != route['driver_id']
                ]
                available_drivers = available_drivers[
                    ~available_drivers['driver_id'].isin(used_driver_ids)]
            else:
                improved_routes.append(route)  # Couldn't split, keep original
        else:
            improved_routes.append(route)

    if routes_split > 0:
        logger.info(
            f"    ✂️ Successfully split {routes_split} routes with enhanced logic"
        )

    return improved_routes


def intelligent_route_splitting_improved(route, available_drivers, office_lat,
                                       office_lon):
    """Intelligent route splitting using metric-based clustering"""
    users = route['assigned_users']
    if len(users) < 3:
        return [route]

    # Convert users to metric coordinates for clustering
    coords_km = []
    bearings = []
    for user in users:
        lat_km, lon_km = coords_to_km(user['lat'], user['lng'], office_lat,
                                      office_lon)
        coords_km.append([lat_km, lon_km])
        bearing = calculate_bearing(office_lat, office_lon, user['lat'],
                                    user['lng'])
        bearings.append(bearing)

    # Calculate bearing spread to determine split strategy
    bearing_spread = max(bearings) - min(bearings)
    if bearing_spread > 180:  # Handle circular nature
        bearing_spread = 360 - bearing_spread

    # Strategy selection based on route characteristics
    if bearing_spread > 60:
        # Large bearing spread: split by direction
        return split_by_bearing_clusters_improved(route, available_drivers,
                                                  office_lat, office_lon,
                                                  coords_km, bearings)
    else:
        # Small bearing spread: split by distance clusters
        return split_by_distance_clusters_improved(route, available_drivers,
                                                   office_lat, office_lon,
                                                   coords_km)


def split_by_bearing_clusters_improved(route, available_drivers, office_lat,
                                       office_lon, coords_km, bearings):
    """Split by bearing using metric coordinates"""
    users = route['assigned_users']

    # Combine coordinates with bearing features for clustering
    coords_with_bearing = np.column_stack([
        coords_km,
        [math.sin(math.radians(b)) * 2
         for b in bearings],  # Weight bearing more heavily
        [math.cos(math.radians(b)) * 2 for b in bearings]
    ])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_with_bearing)

    group1 = [users[i] for i in range(len(users)) if labels[i] == 0]
    group2 = [users[i] for i in range(len(users)) if labels[i] == 1]

    return create_split_routes_improved(route, [group1, group2],
                                        available_drivers, office_lat,
                                        office_lon)


def split_by_distance_clusters_improved(route, available_drivers, office_lat,
                                        office_lon, coords_km):
    """Split by distance using metric coordinates"""
    users = route['assigned_users']

    # Use K-means on metric coordinates
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_km)

    group1 = [users[i] for i in range(len(users)) if labels[i] == 0]
    group2 = [users[i] for i in range(len(users)) if labels[i] == 1]

    return create_split_routes_improved(route, [group1, group2],
                                        available_drivers, office_lat,
                                        office_lon)


def create_split_routes_improved(original_route, user_groups,
                                 available_drivers, office_lat, office_lon):
    """Create new routes from split user groups with improved driver matching"""
    split_routes = []

    for i, group in enumerate(user_groups):
        if not group:
            continue

        if i == 0:
            # Keep original driver for first group
            new_route = original_route.copy()
            new_route['assigned_users'] = group
            split_routes.append(new_route)
        else:
            # Find best driver for additional groups
            suitable_driver = find_best_driver_for_group(
                group, available_drivers, office_lat, office_lon)
            if suitable_driver is not None:
                new_route = {
                    'driver_id': str(suitable_driver['driver_id']),
                    'vehicle_id': str(suitable_driver.get('vehicle_id', '')),
                    'vehicle_type': int(suitable_driver['capacity']),
                    'latitude': float(suitable_driver['latitude']),
                    'longitude': float(suitable_driver['longitude']),
                    'assigned_users': group
                }
                split_routes.append(new_route)
            else:
                # If no driver available, merge back with first group
                if split_routes:
                    split_routes[0]['assigned_users'].extend(group)

    # Optimize sequences and update metrics for all split routes
    for route in split_routes:
        route = optimize_route_sequence_improved(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)

    return split_routes if len(split_routes) > 1 else [original_route]


def find_best_driver_for_group(user_group, available_drivers, office_lat,
                               office_lon):
    """Find the best driver for a group using improved scoring"""
    if available_drivers.empty or len(user_group) == 0:
        return None

    group_center = calculate_users_center_improved(user_group)
    group_bearing = calculate_bearing(office_lat, office_lon, group_center[0],
                                      group_center[1])

    best_driver = None
    best_score = float('inf')

    for _, driver in available_drivers.iterrows():
        if driver['capacity'] < len(user_group):
            continue

        # Calculate comprehensive score
        distance = haversine_distance(driver['latitude'], driver['longitude'],
                                      group_center[0], group_center[1])

        # Driver-cluster bearing alignment
        driver_bearing = calculate_bearing(office_lat, office_lon,
                                           driver['latitude'],
                                           driver['longitude'])
        bearing_diff = bearing_difference(driver_bearing, group_bearing)

        # Utilization bonus
        utilization = len(user_group) / driver['capacity']
        utilization_bonus = utilization * 2.0

        # Priority penalty
        priority_penalty = driver.get('priority', 1) * 0.5

        # Combined score
        score = distance + (
            bearing_diff * 0.05
        ) - utilization_bonus  # Convert degrees to distance-like metric

        if score < best_score:
            best_score = score
            best_driver = driver

    return best_driver


def outlier_detection_and_reassignment(routes, office_lat, office_lon):
    """Disabled outlier detection to prevent unassigned users - only optimize existing routes"""
    logger = get_logger()
    logger.info("    🔍 Optimizing route quality without removing users...")

    improved_routes = []

    for route in routes:
        # Just optimize existing routes without removing any users
        route = optimize_route_sequence_improved(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)
        improved_routes.append(route)

    logger.info(
        f"    📊 Optimized {len(improved_routes)} routes without removing any users"
    )
    return improved_routes, []  # No failed reassignments


def try_reassign_outlier(outlier_user, routes, office_lat, office_lon):
    """Try to reassign an outlier user to a compatible route"""
    outlier_pos = (outlier_user['lat'], outlier_user['lng'])
    outlier_bearing = calculate_bearing(office_lat, office_lon, outlier_pos[0],
                                        outlier_pos[1])

    best_route = None
    best_score = float('inf')

    for route in routes:
        # Check capacity
        if len(route['assigned_users']) >= route['vehicle_type']:
            continue

        # Check compatibility
        route_center = calculate_route_center_improved(route)
        route_bearing = calculate_average_bearing_improved(
            route, office_lat, office_lon)

        distance = haversine_distance(route_center[0], route_center[1],
                                      outlier_pos[0], outlier_pos[1])
        bearing_diff = bearing_difference(route_bearing, outlier_bearing)

        if distance <= _config[
                'MAX_FILL_DISTANCE_KM'] * 1.5 and bearing_diff <= 30:
            # Test quality impact
            test_route = route.copy()
            test_route['assigned_users'] = route['assigned_users'] + [
                outlier_user
            ]
            test_route = optimize_route_sequence_improved(
                test_route, office_lat, office_lon)

            test_turning = calculate_route_turning_score_improved(
                test_route['assigned_users'],
                (test_route['latitude'], test_route['longitude']),
                (office_lat, office_lon))

            # Score based on distance, bearing, and quality impact
            score = distance + (bearing_diff * 0.05) + (test_turning * 0.1)

            if test_turning <= 40 and score < best_score:  # Quality threshold
                best_score = score
                best_route = route

    if best_route is not None:
        # Add outlier to best compatible route
        best_route['assigned_users'].append(outlier_user)
        # Re-optimize the route
        best_route = optimize_route_sequence_improved(best_route, office_lat,
                                                      office_lon)
        update_route_metrics_improved(best_route, office_lat, office_lon)
        return True

    return False


def handle_remaining_users_improved(unassigned_users_df, driver_df, routes,
                                    office_lat, office_lon):
    """Handle remaining unassigned users by first trying existing routes, then creating new ones"""
    if unassigned_users_df.empty:
        return []

    logger = get_logger()
    remaining_users = unassigned_users_df.copy()
    unassigned_list = []
    new_routes_created = []

    logger.info(f"    Processing {len(remaining_users)} unassigned users")

    # PHASE 1: Try to assign to existing routes with available capacity (RELAXED THRESHOLDS)
    if routes is not None and len(routes) > 0:
        logger.info(
            f"    Phase 1: Checking existing {len(routes)} routes for capacity with relaxed constraints"
        )

        users_assigned_to_existing = 0
        users_to_remove = []

        # Process each unassigned user with RELAXED criteria
        for _, user in remaining_users.iterrows():
            if user['user_id'] in [u['user_id'] for u in users_to_remove]:
                continue  # Already assigned

            user_bearing = calculate_bearing(office_lat, office_lon,
                                             user['latitude'],
                                             user['longitude'])
            user_pos = (user['latitude'], user['longitude'])

            best_route = None
            best_score = float('inf')

            # Check each existing route for compatibility with RELAXED CONSTRAINTS
            for route in routes:
                # Check if route has capacity
                if len(route['assigned_users']) >= route['vehicle_type']:
                    continue

                # Calculate route center and bearing
                route_center = calculate_route_center_improved(route)
                route_bearing = calculate_average_bearing_improved(
                    route, office_lat, office_lon)

                # RELAXED distance compatibility - much more lenient for unassigned users
                distance = haversine_distance(route_center[0], route_center[1],
                                              user_pos[0], user_pos[1])
                max_distance = _config.get('MAX_FILL_DISTANCE_KM',
                                           5.0) * 2.5  # Much more lenient

                if distance > max_distance:
                    continue

                # RELAXED bearing compatibility - much more lenient for unassigned users
                bearing_diff = bearing_difference(user_bearing, route_bearing)
                max_bearing_diff = _config.get('MAX_BEARING_DIFFERENCE',
                                               20) * 3.0  # Much more lenient

                if bearing_diff > max_bearing_diff:
                    continue

                # Test route quality with this user - RELAXED quality check
                test_route = route.copy()
                test_user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }

                if pd.notna(user.get('first_name')):
                    test_user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    test_user_data['email'] = str(user['email'])

                test_route['assigned_users'] = route['assigned_users'] + [
                    test_user_data
                ]
                test_route = optimize_route_sequence_improved(
                    test_route, office_lat, office_lon)

                # Calculate quality metrics - VERY RELAXED for unassigned users
                new_turning = calculate_route_turning_score_improved(
                    test_route['assigned_users'],
                    (test_route['latitude'], test_route['longitude']),
                    (office_lat, office_lon))

                # RELAXED quality threshold - prioritize assignment over perfect quality
                if new_turning <= 80:  # Much more relaxed threshold
                    # Calculate combined score for route selection
                    score = distance + (bearing_diff * 0.02) + (
                        new_turning * 0.05)  # Reduce penalty weights

                    if score < best_score:
                        best_score = score
                        best_route = route

            # Assign user to best compatible route
            if best_route is not None:
                user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }

                if pd.notna(user.get('first_name')):
                    user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    user_data['email'] = str(user['email'])

                best_route['assigned_users'].append(user_data)
                users_to_remove.append({'user_id': user['user_id']})
                users_assigned_to_existing += 1

                # Re-optimize the route
                best_route = optimize_route_sequence_improved(
                    best_route, office_lat, office_lon)
                update_route_metrics_improved(best_route, office_lat,
                                              office_lon)

                logger.info(
                    f"    RELAXED-assigned user {user['user_id']} to existing route {best_route['driver_id']}"
                )

        # Remove assigned users from remaining list
        if users_to_remove:
            assigned_user_ids_in_phase1 = {
                u['user_id']
                for u in users_to_remove
            }
            remaining_users = remaining_users[
                ~remaining_users['user_id'].isin(assigned_user_ids_in_phase1)]
            logger.info(
                f"    Phase 1: Assigned {users_assigned_to_existing} users to existing routes with relaxed constraints"
            )

    # PHASE 2: Create new routes for remaining unassigned users
    if not remaining_users.empty and driver_df is not None:
        # Find available drivers
        assigned_driver_ids = {route['driver_id']
                               for route in routes} if routes else set()
        available_drivers = driver_df[~driver_df['driver_id'].
                                      isin(assigned_driver_ids)]

        if not available_drivers.empty:
            logger.info(
                f"    Phase 2: Creating new routes for {len(remaining_users)} remaining users"
            )

            # Convert to metric coordinates for clustering
            coords_km = []
            user_list = []
            for _, user in remaining_users.iterrows():
                lat_km, lon_km = coords_to_km(user['latitude'],
                                              user['longitude'], office_lat,
                                              office_lon)
                coords_km.append([lat_km, lon_km])
                user_list.append(user)

            if len(coords_km) > 1:
                # Use DBSCAN for intelligent grouping
                eps_km = _config.get('DBSCAN_EPS_KM', 1.5)
                dbscan = DBSCAN(eps=eps_km, min_samples=1)
                cluster_labels = dbscan.fit_predict(coords_km)

                # Group users by cluster
                clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(user_list[i])

                # Create routes for each cluster
                for cluster_users in clusters.values():
                    if not cluster_users:
                        continue

                    # Find best driver for this cluster
                    cluster_center = (np.mean([
                        u['latitude'] for u in cluster_users
                    ]), np.mean([u['longitude'] for u in cluster_users]))

                    best_driver = find_best_driver_for_cluster_improved(
                        cluster_users, available_drivers, cluster_center,
                        office_lat, office_lon)

                    if best_driver is not None and len(
                            cluster_users) <= best_driver['capacity']:
                        # Create new route
                        route_users = []
                        user_ids_to_remove = []

                        for user in cluster_users:
                            user_data = {
                                'user_id':
                                str(user['user_id']),
                                'lat':
                                float(user['latitude']),
                                'lng':
                                float(user['longitude']),
                                'office_distance':
                                float(user.get('office_distance', 0))
                            }

                            if pd.notna(user.get('first_name')):
                                user_data['first_name'] = str(
                                    user['first_name'])
                            if pd.notna(user.get('email')):
                                user_data['email'] = str(user['email'])

                            route_users.append(user_data)
                            user_ids_to_remove.append(user['user_id'])

                        # Create the route
                        new_route = {
                            'driver_id':
                            str(best_driver['driver_id']),
                            'vehicle_id':
                            str(best_driver.get('vehicle_id', '')),
                            'vehicle_type':
                            int(best_driver['capacity']),
                            'latitude':
                            float(best_driver['latitude']),
                            'longitude':
                            float(best_driver['longitude']),
                            'assigned_users': route_users
                        }

                        # Optimize sequence and update metrics
                        new_route = optimize_route_sequence_improved(
                            new_route, office_lat, office_lon)
                        update_route_metrics_improved(new_route, office_lat,
                                                      office_lon)

                        new_routes_created.append(new_route)

                        # Remove assigned users and driver
                        remaining_users = remaining_users[
                            ~remaining_users['user_id'].isin(user_ids_to_remove)]
                        available_drivers = available_drivers[
                            available_drivers['driver_id'] !=
                            best_driver['driver_id']]

                        logger.info(
                            f"    Created new route for driver {best_driver['driver_id']} with {len(route_users)} users"
                        )
            elif len(coords_km) == 1:
                # Single user - try to assign to best available driver
                user = user_list[0]
                best_driver = find_best_driver_for_cluster_improved(
                    [user], available_drivers,
                    (user['latitude'], user['longitude']), office_lat,
                    office_lon)

                if best_driver is not None:
                    user_data = {
                        'user_id': str(user['user_id']),
                        'lat': float(user['latitude']),
                        'lng': float(user['longitude']),
                        'office_distance':
                        float(user.get('office_distance', 0))
                    }

                    if pd.notna(user.get('first_name')):
                        user_data['first_name'] = str(user['first_name'])
                    if pd.notna(user.get('email')):
                        user_data['email'] = str(user['email'])

                    new_route = {
                        'driver_id': str(best_driver['driver_id']),
                        'vehicle_id': str(best_driver.get('vehicle_id', '')),
                        'vehicle_type': int(best_driver['capacity']),
                        'latitude': float(best_driver['latitude']),
                        'longitude': float(best_driver['longitude']),
                        'assigned_users': [user_data]
                    }

                    update_route_metrics_improved(new_route, office_lat,
                                                  office_lon)
                    new_routes_created.append(new_route)
                    remaining_users = remaining_users[
                        remaining_users['user_id'] != user['user_id']]

                    logger.info(
                        f"    Created single-user route for driver {best_driver['driver_id']}"
                    )

            # Update routes list
            if new_routes_created:
                routes.extend(new_routes_created)
                logger.info(
                    f"    Phase 2: Created {len(new_routes_created)} new routes"
                )

    # Convert remaining unassigned users to list format
    for _, user in remaining_users.iterrows():
        user_data = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0))
        }

        if pd.notna(user.get('first_name')):
            user_data['first_name'] = str(user['first_name'])
        if pd.notna(user.get('email')):
            user_data['email'] = str(user['email'])

        unassigned_list.append(user_data)

    if unassigned_list:
        logger.warning("    UNASSIGNED USERS REQUIRING ATTENTION")
        logger.warning(f"    {'─' * 54}")
        for i, user in enumerate(unassigned_list, 1):
            lat = user['lat']
            lng = user['lng']
            distance = user.get('office_distance', 0)
            logger.warning(
                f"       {i}. User {user['user_id']}: Location ({lat}, {lng}) | Office Distance: {distance} km"
            )

    return unassigned_list


def find_best_driver_for_cluster_improved(cluster_users, available_drivers,
                                          cluster_center, office_lat,
                                          office_lon):
    """Find the best driver for a cluster of users"""
    if available_drivers.empty:
        return None

    cluster_bearing = calculate_bearing(office_lat, office_lon,
                                        cluster_center[0], cluster_center[1])
    best_driver = None
    best_score = float('inf')

    for _, driver in available_drivers.iterrows():
        if driver['capacity'] < len(cluster_users):
            continue

        # Calculate comprehensive score
        distance = haversine_distance(driver['latitude'], driver['longitude'],
                                      cluster_center[0], cluster_center[1])

        # Driver-cluster bearing alignment
        driver_bearing = calculate_bearing(office_lat, office_lon,
                                           driver['latitude'],
                                           driver['longitude'])
        bearing_diff = bearing_difference(driver_bearing, cluster_bearing)

        # Utilization bonus
        utilization = len(cluster_users) / driver['capacity']
        utilization_bonus = utilization * 2.0

        # Priority penalty
        priority_penalty = driver.get('priority', 1) * 0.5

        # Combined score
        score = distance + (
            bearing_diff * 0.05
        ) - utilization_bonus  # 0.05 km per degree

        if score < best_score:
            best_score = score
            best_driver = driver

    return best_driver


# MAIN ASSIGNMENT FUNCTION
def run_assignment(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    Main assignment function that automatically routes to the appropriate algorithm
    based on ride_settings priority value from the API response:
    - Priority 1 → assign_capacity.py (Capacity Optimization)
    - Priority 2 → assign_balance.py (Balanced Optimization)
    - Priority 3 → assign_route.py (Road-Aware Routing)
    - Default → assignment.py (Route Efficiency)
    """
    start_time = time.time()

    # Clear any cached data files to ensure fresh assignment
    cache_files = [
        "drivers_and_routes.json",
        "drivers_and_routes_capacity.json",
        "drivers_and_routes_balance.json",
        "drivers_and_routes_road_aware.json"
    ]

    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)

    # Initialize logging and progress tracking
    logger = get_logger()
    progress = get_progress_tracker()

    logger.info(f"Starting assignment for source_id: {source_id}")
    logger.info(f"Parameters: {parameter}, String: {string_param}")

    try:
        # STAGE 1: Data Loading & API Response Analysis
        progress.start_stage("Data Loading & Algorithm Detection",
                             "Loading data from API and detecting algorithm...")
        data = load_env_and_fetch_data(source_id, parameter, string_param, choice)

        # Check safety flag first - this overrides all other algorithm selection
        safety_flag = data.get("_safety_flag", 0)

        if safety_flag == 1:
            logger.info("🔒 SAFETY FLAG ACTIVE - Running safety algorithm (override)")
            from algorithm.safety.safety import run_safety_assignment_simplified
            return run_safety_assignment_simplified(source_id, parameter, string_param, choice)

        # Get the algorithm priority from ride_settings (only used if safety_flag = 0)
        algorithm_priority = data.get("_algorithm_priority")

        # Route to appropriate algorithm based on priority (normal flow when safety = 0)
        if algorithm_priority == 1:
            logger.info("🎪 Routing to CAPACITY OPTIMIZATION (assign_capacity.py)")
            from algorithm.capacity.capacity import run_assignment_capacity
            return run_assignment_capacity(source_id, parameter, string_param, choice)
        elif algorithm_priority == 2:
            logger.info("⚖️ Routing to BALANCED OPTIMIZATION (assign_balance.py)")
            from algorithm.balance.balance import run_assignment_balance
            return run_assignment_balance(source_id, parameter, string_param, choice)
        elif algorithm_priority == 3:
            logger.info("🗺️ Routing to ROAD-AWARE ROUTING (assign_route.py)")
            from algorithm.road.road import run_road_aware_assignment
            return run_road_aware_assignment(source_id, parameter, string_param, choice)
        else:
            logger.info("🎯 Using default ROUTE EFFICIENCY algorithm (assignment.py)")
            # Continue with route efficiency algorithm (original assignment.py logic)
            return run_route_efficiency_assignment(source_id, parameter, string_param, choice)

    except Exception as e:
        logger.error(f"Error in algorithm routing: {e}", exc_info=True)
        # Fallback to route efficiency
        logger.info("🔄 Falling back to ROUTE EFFICIENCY algorithm")
        return run_route_efficiency_assignment(source_id, parameter, string_param, choice)


def run_route_efficiency_assignment(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    Route efficiency assignment function (original assignment.py logic)
    - Prioritizes straight routes with minimal zigzag
    - Strict quality control for route turning and tortuosity
    - Efficient user-to-driver matching based on bearing alignment
    """
    start_time = time.time()

    # Initialize logging and progress tracking
    logger = get_logger()
    progress = get_progress_tracker()

    progress.start_assignment(source_id, "ROUTE EFFICIENCY")
    logger.log_session_start()

    # Reload configuration for route efficiency
    global _config
    _config = load_and_validate_config()

    # Update global variables from new config
    global MAX_FILL_DISTANCE_KM, MERGE_DISTANCE_KM, MAX_BEARING_DIFFERENCE, UTILIZATION_PENALTY_PER_SEAT
    MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
    MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM']
    MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
    UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']

    logger.info(
        f"Starting ROUTE EFFICIENCY assignment for source_id: {source_id}")
    logger.info(f"Parameters: {parameter}, String: {string_param}")

    try:
        # STAGE 1: Data Loading & Validation
        progress.start_stage("Data Loading & Validation",
                             "Loading data from API...")
        data = load_env_and_fetch_data(source_id, parameter, string_param, choice)

        progress.update_stage_progress("Validating data structure...")

        # Edge case handling
        users = data.get('users', [])
        if not users:
            logger.warning("No users found in API response")
            progress.complete_stage("No users to assign")
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": [],
                "unassignedDrivers": _get_all_drivers_as_unassigned(data),
                "clustering_analysis": {
                    "method": "No Users",
                    "clusters": 0
                },
                "optimization_mode": "route_efficiency",
                "parameter": parameter,
            }

        # Get all drivers
        all_drivers = []
        if "drivers" in data:
            drivers_data = data["drivers"]
            all_drivers.extend(drivers_data.get("driversUnassigned", []))
            all_drivers.extend(drivers_data.get("driversAssigned", []))
        else:
            all_drivers.extend(data.get("driversUnassigned", []))
            all_drivers.extend(data.get("driversAssigned", []))

        if not all_drivers:
            logger.warning("No drivers available - all users unassigned")
            unassigned_users = _convert_users_to_unassigned_format(users)
            progress.complete_stage("No drivers available")
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": unassigned_users,
                "unassignedDrivers": [],
                "clustering_analysis": {
                    "method": "No Drivers",
                    "clusters": 0
                },
                "optimization_mode": "route_efficiency",
                "parameter": parameter,
            }

        logger.info(
            f"Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}"
        )
        progress.update_stage_progress("Data loaded successfully.")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        logger.info("Data validation passed")
        progress.update_stage_progress("Data validated.")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)

        logger.info(
            f"DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}"
        )
        progress.update_stage_progress("DataFrames prepared.")

        # STEP 1: Geographic clustering with proper distance metrics
        progress.start_stage("Clustering", "Creating geographic clusters...")
        user_df = create_geographic_clusters(user_df, office_lat, office_lon,
                                             _config)
        clustering_results = {
            "method": "metric_aware_" + _config['clustering_method'],
            "clusters": user_df['geo_cluster'].nunique()
        }
        logger.info(
            f"Created {clustering_results['clusters']} geographic clusters.")
        progress.complete_stage("Geographic clusters created.")

        # STEP 2: Capacity-based sub-clustering (direction-aware)
        progress.start_stage("Sub-Clustering",
                             "Creating capacity-based sub-clusters...")
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon,
                                              _config)
        logger.info("Created capacity-based sub-clusters.")
        progress.complete_stage("Sub-clustering done.")

        # STEP 3: Priority-based driver assignment (sequence-aware with bearing-based costs)
        progress.start_stage("Driver Assignment",
                             "Assigning drivers by priority...")
        routes, assigned_user_ids = assign_drivers_by_priority(
            user_df, driver_df, office_lat, office_lon)
        logger.info(f"Created {len(routes)} initial routes.")
        progress.complete_stage("Initial routes created.")

        # STEP 4: Local optimization (improved turning calculations and swap logic)
        progress.start_stage("Local Optimization",
                             "Optimizing within and between routes...")
        routes = local_optimization(routes, office_lat, office_lon)
        logger.info("Local optimization completed.")
        progress.complete_stage("Local optimization done.")

        # STEP 5: Global optimization (enhanced with outlier detection)
        progress.start_stage("Global Optimization",
                             "Performing global optimization...")
        routes, unassigned_users = global_optimization(routes, user_df,
                                                       assigned_user_ids,
                                                       driver_df, office_lat,
                                                       office_lon)
        logger.info("Global optimization completed.")
        progress.complete_stage("Global optimization done.")

        # STEP 6: Final-pass merge algorithm
        progress.start_stage("Final Merge",
                             "Applying final merge algorithm...")
        routes = final_pass_merge(routes, _config, office_lat, office_lon)
        logger.info("Final merge pass completed.")
        progress.complete_stage("Final merge applied.")

        # Filter out routes with no assigned users and move those drivers to unassigned
        filtered_routes = []
        empty_route_driver_ids = set()

        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                filtered_routes.append(route)
            else:
                empty_route_driver_ids.add(route['driver_id'])
                logger.warning(
                    f"Moving driver {route['driver_id']} with no users to unassigned drivers"
                )

        routes = filtered_routes

        # Build unassigned drivers list (including drivers from empty routes)
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].
                                          isin(assigned_driver_ids)]
        unassigned_drivers = []

        for _, driver in unassigned_drivers_df.iterrows():
            driver_data = {
                'driver_id': str(driver.get('driver_id', '')),
                'capacity': int(driver.get('capacity', 0)),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'latitude': float(driver.get('latitude', 0.0)),
                'longitude': float(driver.get('longitude', 0.0))
            }
            unassigned_drivers.append(driver_data)

        # Final metrics update for all routes
        for route in routes:
            update_route_metrics_improved(route, office_lat, office_lon)

        execution_time = time.time() - start_time

        # COMPREHENSIVE USER ACCOUNTING AND VALIDATION
        progress.start_stage("Final Merge & Validation",
                             "Performing final user accounting...")

        total_users_in_api = len(users)

        # Build comprehensive user tracking
        assigned_user_ids_final = set()
        for route in routes:
            for user in route['assigned_users']:
                assigned_user_ids_final.add(str(user['user_id']))

        unassigned_user_ids_final = {
            str(user['user_id'])
            for user in unassigned_users
        }

        # Check for duplicate assignments
        if len(assigned_user_ids_final) != sum(
                len(r['assigned_users']) for r in routes):
            logger.critical("DUPLICATE USER ASSIGNMENTS DETECTED!")
            duplicate_assignments = []
            all_assigned_users = []
            for route in routes:
                for user in route['assigned_users']:
                    user_id = str(user['user_id'])
                    if user_id in all_assigned_users:
                        duplicate_assignments.append(user_id)
                        logger.critical(
                            f"User {user_id} assigned multiple times")
                    all_assigned_users.append(user_id)

        users_assigned = len(assigned_user_ids_final)
        users_unassigned = len(unassigned_user_ids_final)
        users_accounted_for = users_assigned + users_unassigned

        # Check for overlap between assigned and unassigned
        overlap = assigned_user_ids_final.intersection(
            unassigned_user_ids_final)
        if overlap:
            logger.critical(
                f"USER OVERLAP DETECTED: {len(overlap)} users in both assigned and unassigned lists"
            )
            for user_id in overlap:
                logger.critical(f"Overlapping user: {user_id}")
                # Remove from unassigned list
                unassigned_users = [
                    u for u in unassigned_users if str(u['user_id']) != user_id
                ]
            users_unassigned = len(unassigned_users)
            users_accounted_for = users_assigned + users_unassigned

        progress.update_stage_progress(
            f"Accounted for {users_accounted_for}/{total_users_in_api} users..."
        )

        # Handle missing users with emergency fallback assignment
        discrepancy = total_users_in_api - users_accounted_for
        logger.log_accounting_check(total_users_in_api, users_assigned,
                                    unassigned_users, discrepancy, "assignment.py")

        if discrepancy != 0:
            logger.warning(
                f"Discrepancy found: {discrepancy} users unaccounted for. Attempting emergency fallback."
            )

            # Find missing users
            assigned_user_ids_final_for_fallback = set()
            for route in routes:
                for user in route['assigned_users']:
                    assigned_user_ids_final_for_fallback.add(
                        str(user['user_id']))
            unassigned_user_ids_final_for_fallback = {
                str(user['user_id'])
                for user in unassigned_users
            }
            all_accounted_user_ids = assigned_user_ids_final_for_fallback.union(
                unassigned_user_ids_final_for_fallback)
            original_user_ids = {str(user['id']) for user in users}
            missing_user_ids = original_user_ids - all_accounted_user_ids

            if missing_user_ids:
                logger.critical(f"Missing user IDs: {missing_user_ids}")
                logger.critical(
                    f"EMERGENCY FALLBACK: Attempting to force-assign missing users..."
                )

                emergency_assignments = 0

                # Try emergency assignment to routes with available capacity
                for user in users:
                    if str(user['id']) in missing_user_ids:
                        assigned = False

                        # Find any route with available capacity
                        for route in routes:
                            if len(route['assigned_users']
                                   ) < route['vehicle_type']:
                                user_data = {
                                    'user_id':
                                    str(user['id']),
                                    'lat':
                                    float(user['latitude']),
                                    'lng':
                                    float(user['longitude']),
                                    'office_distance':
                                    float(user.get('office_distance', 0))
                                }
                                if user.get('first_name'):
                                    user_data['first_name'] = str(
                                        user['first_name'])
                                if user.get('email'):
                                    user_data['email'] = str(user['email'])

                                route['assigned_users'].append(user_data)
                                route = optimize_route_sequence_improved(
                                    route, office_lat, office_lon)
                                update_route_metrics_improved(
                                    route, office_lat, office_lon)
                                assigned = True
                                emergency_assignments += 1
                                logger.info(
                                    f"EMERGENCY: Assigned user {user['id']} to route {route['driver_id']}"
                                )
                                break

                        # If no route capacity, try to create new route with available drivers
                        if not assigned:
                            assigned_driver_ids = {
                                route['driver_id']
                                for route in routes
                            }
                            available_drivers_fallback = driver_df[
                                ~driver_df['driver_id'].
                                isin(assigned_driver_ids)]

                            if not available_drivers_fallback.empty:
                                best_driver = available_drivers_fallback.iloc[
                                    0]  # Take first available

                                user_data = {
                                    'user_id':
                                    str(user['id']),
                                    'lat':
                                    float(user['latitude']),
                                    'lng':
                                    float(user['longitude']),
                                    'office_distance':
                                    float(user.get('office_distance', 0))
                                }
                                if user.get('first_name'):
                                    user_data['first_name'] = str(
                                        user['first_name'])
                                if user.get('email'):
                                    user_data['email'] = str(user['email'])

                                emergency_route = {
                                    'driver_id':
                                    str(best_driver['driver_id']),
                                    'vehicle_id':
                                    str(best_driver.get('vehicle_id', '')),
                                    'vehicle_type':
                                    int(best_driver['capacity']),
                                    'latitude':
                                    float(best_driver['latitude']),
                                    'longitude':
                                    float(best_driver['longitude']),
                                    'assigned_users': [user_data]
                                }

                                update_route_metrics_improved(
                                    emergency_route, office_lat, office_lon)
                                routes.append(emergency_route)
                                assigned = True
                                emergency_assignments += 1
                                logger.info(
                                    f"EMERGENCY: Created new route for user {user['id']} with driver {best_driver['driver_id']}"
                                )

                        # If still not assigned, add to unassigned list
                        if not assigned:
                            missing_user_data = {
                                'user_id':
                                str(user['id']),
                                'lat':
                                float(user['latitude']),
                                'lng':
                                float(user['longitude']),
                                'office_distance':
                                float(user.get('office_distance', 0))
                            }
                            if user.get('first_name'):
                                missing_user_data['first_name'] = str(
                                    user['first_name'])
                            if user.get('email'):
                                missing_user_data['email'] = str(user['email'])
                            unassigned_users.append(missing_user_data)

                logger = get_logger()
                logger.info(
                    f"Emergency assignments completed: {emergency_assignments} users"
                )

                # Update final stats
                assigned_user_ids_final = set()
                for route in routes:
                    for user in route['assigned_users']:
                        assigned_user_ids_final.add(str(user['user_id']))
                users_assigned = len(assigned_user_ids_final)
                users_unassigned = len(unassigned_users)

        temp_logger = get_logger()
        temp_logger.info(
            f"Final assignment stats - Total: {total_users_in_api}, Assigned: {users_assigned}, Unassigned: {users_unassigned}"
        )
        progress.complete_stage(
            f"Final accounting done - {users_assigned}/{total_users_in_api} users assigned"
        )

        # Final results assembly
        # Extract additional data for rich response
        company_info = data.get("company", {})
        shift_info = data.get("shift", {})

        # Enhance route data with rich information
        enhanced_routes = []
        for route in routes:
            enhanced_route = route.copy()

            # Add enhanced driver information
            driver_id = route['driver_id']
            driver_info = None

            # Find driver in original data
            if "drivers" in data:
                all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
            else:
                all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])

            for driver in all_drivers_data:
                if str(driver.get('id', driver.get('sub_user_id', ''))) == driver_id:
                    driver_info = driver
                    break

            if driver_info:
                # Add driver details directly to route level instead of nested driver_info
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

            # Enhance user information
            enhanced_users = []
            original_users = data.get('users', [])

            for user in route['assigned_users']:
                enhanced_user = user.copy()

                # Find user in original data
                for orig_user in original_users:
                    if str(orig_user.get('id', orig_user.get('sub_user_id', ''))) == user['user_id']:
                        enhanced_user.update({
                            'address': orig_user.get('address', ''),
                            'employee_shift': orig_user.get('employee_shift', ''),
                            'shift_type': orig_user.get('shift_type', ''),
                            'last_name': orig_user.get('last_name', '')
                        })
                        break

                enhanced_users.append(enhanced_user)

            enhanced_route['assigned_users'] = enhanced_users
            enhanced_routes.append(enhanced_route)

        # Enhance unassigned users
        enhanced_unassigned_users = []
        original_users = data.get('users', [])

        for user in unassigned_users:
            enhanced_user = user.copy()

            # Find user in original data
            for orig_user in original_users:
                if str(orig_user.get('id', orig_user.get('sub_user_id', ''))) == user['user_id']:
                    enhanced_user.update({
                        'address': orig_user.get('address', ''),
                        'employee_shift': orig_user.get('employee_shift', ''),
                        'shift_type': orig_user.get('shift_type', ''),
                        'last_name': orig_user.get('last_name', '')
                    })
                    break

            enhanced_unassigned_users.append(enhanced_user)

        # Enhance unassigned drivers
        enhanced_unassigned_drivers = []
        if "drivers" in data:
            all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
        else:
            all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])

        for driver in unassigned_drivers:
            enhanced_driver = driver.copy()

            # Find driver in original data
            for orig_driver in all_drivers_data:
                if str(orig_driver.get('id', orig_driver.get('sub_user_id', ''))) == driver['driver_id']:
                    enhanced_driver.update({
                        'first_name': orig_driver.get('first_name', ''),
                        'last_name': orig_driver.get('last_name', ''),
                        'email': orig_driver.get('email', ''),
                        'vehicle_name': orig_driver.get('vehicle_name', ''),
                        'vehicle_no': orig_driver.get('vehicle_no', ''),
                        'chasis_no': orig_driver.get('chasis_no', ''),
                        'color': orig_driver.get('color', ''),
                        'registration_no': orig_driver.get('registration_no', ''),
                        'shift_type_id': orig_driver.get('shift_type_id', '')
                    })
                    break

            enhanced_unassigned_drivers.append(enhanced_driver)

        # Apply optimal pickup ordering to routes if available
        if ORDERING_AVAILABLE and enhanced_routes:
            try:
                logger.info(f"Applying optimal pickup ordering to {len(enhanced_routes)} routes")

                # Extract dynamic office coordinates and db name from API response
                office_lat, office_lon = extract_office_coordinates(data)
                # Use source_id as db_name when available, otherwise try to extract from API response
                db_name = source_id if source_id and source_id != "1" else data.get("db", "default")

                logger.info(f"Using company coordinates: {office_lat}, {office_lon} for db: {db_name}")

                enhanced_routes = apply_route_ordering(enhanced_routes, office_lat, office_lon, db_name=db_name, algorithm_name="base")
                logger.info("Optimal pickup ordering applied successfully")
            except Exception as e:
                logger.error(f"Failed to apply optimal ordering: {e}")
                # Continue with routes without optimal ordering

        result = {
            "status": "true",
            "execution_time": execution_time,
            "company": company_info,
            "shift": shift_info,
            "data": enhanced_routes,
            "unassignedUsers": enhanced_unassigned_users,
            "unassignedDrivers": enhanced_unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "route_efficiency",
            "parameter": parameter,
        }

        progress.show_final_summary(result)
        logger.info("Assignment session completed successfully")
        return result

    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        progress.fail_assignment("API request failed")
        return {"status": "false", "details": str(req_err), "data": []}
    except ValueError as val_err:
        logger.error(f"Data validation error: {val_err}")
        progress.fail_assignment("Data validation error")
        return {"status": "false", "details": str(val_err), "data": []}
    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        progress.fail_assignment("An unexpected error occurred")
        return {"status": "false", "details": str(e), "data": []}


def _get_all_drivers_as_unassigned(data):
    """Helper to get all drivers in the unassigned format"""
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    unassigned_drivers = []
    for driver in all_drivers:
        unassigned_drivers.append({
            'driver_id':
            str(driver.get('id', '')),
            'capacity':
            int(driver.get('capacity', 0)),
            'vehicle_id':
            str(driver.get('vehicle_id', '')),
            'latitude':
            float(driver.get('latitude', 0.0)),
            'longitude':
            float(driver.get('longitude', 0.0))
        })
    return unassigned_drivers


def _convert_users_to_unassigned_format(users):
    """Helper to convert user data to unassigned format"""
    unassigned_users = []
    for user in users:
        unassigned_users.append({
            'user_id':
            str(user.get('id', '')),
            'lat':
            float(user.get('latitude', 0.0)),
            'lng':
            float(user.get('longitude', 0.0)),
            'office_distance':
            float(user.get('office_distance', 0.0)),
            'first_name':
            str(user.get('first_name', '')),
            'email':
            str(user.get('email', ''))
        })
    return unassigned_users


def analyze_assignment_quality(result):
    """Analyze the quality of the assignment with enhanced metrics"""
    if result["status"] != "true":
        return "Assignment failed"

    total_routes = len(result["data"])
    total_assigned = sum(
        len(route["assigned_users"]) for route in result["data"])
    total_unassigned = len(result["unassignedUsers"])

    utilizations = []
    distance_issues = []
    turning_scores = []
    tortuosity_ratios = []
    direction_consistencies = []

    for route in result["data"]:
        if route["assigned_users"]:
            util = len(route["assigned_users"]) / route["vehicle_type"]
            utilizations.append(util)

            # Check distances
            driver_pos = (route["latitude"], route["longitude"])
            for user in route["assigned_users"]:
                dist = haversine_distance(driver_pos[0], driver_pos[1],
                                          user["lat"], user["lng"])
                if dist > DISTANCE_ISSUE_THRESHOLD:
                    distance_issues.append({
                        "driver_id": route["driver_id"],
                        "user_id": user["user_id"],
                        "distance_km": round(dist, 2)
                    })

            # Collect quality metrics
            turning_scores.append(route.get('turning_score', 0))
            tortuosity_ratios.append(route.get('tortuosity_ratio', 1.0))
            direction_consistencies.append(
                route.get('direction_consistency', 1.0))

    analysis = {
        "total_routes":
        total_routes,
        "total_assigned_users":
        total_assigned,
        "total_unassigned_users":
        total_unassigned,
        "assignment_rate":
        round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if
        (total_assigned + total_unassigned) > 0 else 0,
        "avg_utilization":
        round(np.mean(utilizations) * 100, 1) if utilizations else 0,
        "min_utilization":
        round(np.min(utilizations) * 100, 1) if utilizations else 0,
        "max_utilization":
        round(np.max(utilizations) * 100, 1) if utilizations else 0,
        "routes_below_80_percent":
        sum(1 for u in utilizations if u < 0.8),
        "avg_turning_score":
        round(np.mean(turning_scores), 1) if turning_scores else 0,
        "avg_tortuosity":
        round(np.mean(tortuosity_ratios), 2) if tortuosity_ratios else 1.0,
        "avg_direction_consistency":
        round(np.mean(direction_consistencies) *
              100, 1) if direction_consistencies else 100.0,
        "distance_issues":
        distance_issues,
        "clustering_method":
        result.get("clustering_analysis", {}).get("method", "Unknown"),
        "routes_with_good_turning":
        sum(1 for t in turning_scores if t <= 35),
        "routes_with_poor_turning":
        sum(1 for t in turning_scores if t > 50)
    }

    return analysis


def validate_route_path_coherence(route, office_lat, office_lon, strict_mode=True):
    """Validate route path coherence (placeholder implementation)"""
    # Simple validation - check if route has users and reasonable metrics
    if not route['assigned_users']:
        return False

    turning_score = route.get('turning_score', 0)
    tortuosity = route.get('tortuosity_ratio', 1.0)

    # Basic thresholds for validation
    max_turning = 60 if not strict_mode else 45
    max_tortuosity = 2.0 if not strict_mode else 1.8

    return turning_score <= max_turning and tortuosity <= max_tortuosity


def reoptimize_route_with_road_awareness(route, office_lat, office_lon):
    """Reoptimize route with road awareness (placeholder implementation)"""
    # Simple reoptimization - just re-sequence the users
    optimized_route = optimize_route_sequence_improved(route, office_lat, office_lon)
    update_route_metrics_improved(optimized_route, office_lat, office_lon)
    return optimized_route


def analyze_assignment_quality(result):
    """Analyze the quality of the assignment with enhanced metrics"""
    if result["status"] != "true":
        return "Assignment failed"

    total_routes = len(result["data"])
    total_assigned = sum(
        len(route["assigned_users"]) for route in result["data"])
    total_unassigned = len(result["unassignedUsers"])

    utilizations = []
    distance_issues = []
    turning_scores = []
    tortuosity_ratios = []
    direction_consistencies = []

    for route in result["data"]:
        if route["assigned_users"]:
            util = len(route["assigned_users"]) / route["vehicle_type"]
            utilizations.append(util)

            # Check distances
            driver_pos = (route["latitude"], route["longitude"])
            for user in route["assigned_users"]:
                dist = haversine_distance(driver_pos[0], driver_pos[1],
                                          user["lat"], user["lng"])
                if dist > DISTANCE_ISSUE_THRESHOLD:
                    distance_issues.append({
                        "driver_id": route["driver_id"],
                        "user_id": user["user_id"],
                        "distance_km": round(dist, 2)
                    })

            # Collect quality metrics
            turning_scores.append(route.get('turning_score', 0))
            tortuosity_ratios.append(route.get('tortuosity_ratio', 1.0))
            direction_consistencies.append(
                route.get('direction_consistency', 1.0))

    analysis = {
        "total_routes":
        total_routes,
        "total_assigned_users":
        total_assigned,
        "total_unassigned_users":
        total_unassigned,
        "assignment_rate":
        round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if
        (total_assigned + total_unassigned) > 0 else 0,
        "avg_utilization":
        round(np.mean(utilizations) * 100, 1) if utilizations else 0,
        "min_utilization":
        round(np.min(utilizations) * 100, 1) if utilizations else 0,
        "max_utilization":
        round(np.max(utilizations) * 100, 1) if utilizations else 0,
        "routes_below_80_percent":
        sum(1 for u in utilizations if u < 0.8),
        "avg_turning_score":
        round(np.mean(turning_scores), 1) if turning_scores else 0,
        "avg_tortuosity":
        round(np.mean(tortuosity_ratios), 2) if tortuosity_ratios else 1.0,
        "avg_direction_consistency":
        round(np.mean(direction_consistencies) *
              100, 1) if direction_consistencies else 100.0,
        "distance_issues":
        distance_issues,
        "clustering_method":
        result.get("clustering_analysis", {}).get("method", "Unknown"),
        "routes_with_good_turning":
        sum(1 for t in turning_scores if t <= 35),
        "routes_with_poor_turning":
        sum(1 for t in turning_scores if t > 50)
    }

    return analysis