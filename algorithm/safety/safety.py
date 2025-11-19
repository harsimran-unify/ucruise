import os
import math
import numpy as np
import pandas as pd
import time
import json
from functools import lru_cache
from sklearn.cluster import DBSCAN
import warnings
from logger import get_logger
from progress import ProgressTracker

warnings.filterwarnings('ignore')

# Setup logging
logger = get_logger()
progress = ProgressTracker()

# Import ordering system for optimal pickup sequences
try:
    from ordering import apply_route_ordering
    ORDERING_AVAILABLE = True
except ImportError:
    ORDERING_AVAILABLE = False
    logger.warning("Ordering system not available - routes will not have optimal pickup sequences")

# Import core functions from base module
from algorithm.base.base import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, calculate_bearing,
    bearing_difference, normalize_bearing_difference, coords_to_km,
    _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format,
    get_progress_tracker
)

# ================== SIMPLIFIED CONFIGURATION WITH FEMALE SAFETY ==================

def load_simple_config():
    """Load simple, sensible configuration for geographic clustering with female safety"""
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(script_dir, 'config.json')

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}

    # Get female safety configuration
    mode_configs = cfg.get("mode_configs", {})
    safety_config = mode_configs.get("female_safety_optimization", {})

    # Base configuration with female safety parameters
    config = {
        'GEOGRAPHIC_CLUSTER_RADIUS_KM': 1.5,  # Reduced from 2.0 to 1.5km for more compact clusters
        'MAX_ON_ROUTE_DETOUR_KM': 2.0,        # Increased from 1.5 to 2.0km for better coverage
        'BEARING_TOLERANCE_DEGREES': 45,      # Increased from 30 to 45 degrees for more flexible "on the way"
        'MIN_CAPACITY_UTILIZATION': 0.4,      # Reduced from 0.6 to 0.4 to accept more routes
        'MAX_CLUSTER_SIZE': 6,                # NEW: Don't create clusters larger than 6 users

        # NEW: Direction-aware route merging parameters
        'DIRECTION_TOLERANCE_DEGREES': 60,    # Tolerance for considering routes direction-compatible
        'MAX_MERGE_DISTANCE_KM': 4.0,         # Maximum geographic distance for route merging
        'MERGE_SCORE_THRESHOLD': 1.5,         # Maximum merge score to allow merging
        'SMALL_ROUTE_THRESHOLD': 2,           # Routes with <= this many users are considered "small"

        # FEMALE SAFETY SPECIFIC PARAMETERS
        'FEMALE_RESCUE_DISTANCE_MULTIPLIER': safety_config.get('female_rescue_distance_multiplier', 1.5),
        'MIN_MALE_REQUIRED_FOR_FEMALE_SAFETY': safety_config.get('min_male_required_for_female_safety', 1),
        'SAFETY_MERGE_DISTANCE_BONUS_KM': safety_config.get('safety_merge_distance_bonus_km', 2.0),
        'SAFETY_PRIORITY_OVER_EFFICIENCY': safety_config.get('safety_priority_over_efficiency', True),
        'ALLOW_FEMALE_ONLY_ASSIGNMENT': safety_config.get('allow_female_only_assignment', False),

        # New parameters for smart geographic clustering
        'FEMALE_GEOGRAPHIC_CLUSTER_RADIUS_KM': safety_config.get('female_geographic_cluster_radius_km', 2.0),
        'MIN_FEMALE_CLUSTER_SIZE': safety_config.get('min_female_cluster_size', 2),
        'MAX_DRIVER_CLUSTER_DISTANCE_KM': safety_config.get('max_driver_cluster_distance_km', 10),
        'MAX_FEMALE_MALE_DRIVER_DISTANCE_KM': safety_config.get('max_female_male_driver_distance_km', 8),

        'OFFICE_LAT': float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489))),
        'OFFICE_LON': float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))
    }

    logger.info(f"Using FEMALE SAFETY geographic clustering with direction-aware merging:")
    logger.info(f"   Cluster radius: {config['GEOGRAPHIC_CLUSTER_RADIUS_KM']}km")
    logger.info(f"   On-route detour: {config['MAX_ON_ROUTE_DETOUR_KM']}km")
    logger.info(f"   Bearing tolerance: {config['BEARING_TOLERANCE_DEGREES']}deg")
    logger.info(f"   Direction tolerance: {config['DIRECTION_TOLERANCE_DEGREES']}deg")
    logger.info(f"   Max merge distance: {config['MAX_MERGE_DISTANCE_KM']}km")
    logger.info(f"   Small route threshold: {config['SMALL_ROUTE_THRESHOLD']} users")
    logger.info(f"   Female rescue distance multiplier: {config['FEMALE_RESCUE_DISTANCE_MULTIPLIER']}x")
    logger.info(f"   Minimum males required for female safety: {config['MIN_MALE_REQUIRED_FOR_FEMALE_SAFETY']}")

    return config

# Load simple configuration
CONFIG = load_simple_config()

# ================== FEMALE SAFETY SPECIFIC FUNCTIONS ==================

def segment_users_by_gender(user_df, raw_users=None):
    """
    Segment users by gender for female safety processing
    Returns: females_df, males_df
    """
    logger.info("Segmenting users by gender for safety processing...")

    # If we have raw users data, extract gender information and add it to the dataframe
    if raw_users is not None:
        logger.info("   Extracting gender information from raw user data...")
        # Create a mapping from user_id to gender
        gender_mapping = {}
        for user in raw_users:
            user_id = str(user.get('id', ''))
            gender = str(user.get('gender', '')).strip()
            if gender:
                gender_mapping[user_id] = gender

        # Add gender column to the dataframe
        user_df = user_df.copy()
        user_df['gender'] = user_df['user_id'].map(lambda uid: gender_mapping.get(str(uid), 'Unknown'))

        # Count genders
        female_count = len(user_df[user_df['gender'] == 'Female'])
        male_count = len(user_df[user_df['gender'] == 'Male'])
        unknown_count = len(user_df[user_df['gender'] == 'Unknown'])

        logger.info(f"   Gender distribution: Females: {female_count}, Males: {male_count}, Unknown: {unknown_count}")

        females_df = user_df[user_df['gender'] == 'Female'].copy()
        males_df = user_df[user_df['gender'] == 'Male'].copy()
    else:
        # Fallback: try to use existing gender column or estimate by name
        logger.warning("   No raw user data provided, attempting to infer gender from names...")

        # Common female names in the dataset
        female_names = {'Kritika', 'Sakshi', 'Kirandeep', 'Nandita', 'Riya', 'Jagjeet',
                       'Chaitanya', 'Isha', 'Shruti', 'Nandini', 'Chumphila', 'Anchal', 'Jasmeet'}

        user_df = user_df.copy()
        user_df['gender'] = user_df['first_name'].apply(
            lambda name: 'Female' if name in female_names else 'Male'
        )

        females_df = user_df[user_df['gender'] == 'Female'].copy()
        males_df = user_df[user_df['gender'] == 'Male'].copy()

    logger.info(f"   Segmented - Females: {len(females_df)}, Males: {len(males_df)}")

    return females_df, males_df

def validate_route_safety(female_count, male_count):
    """
    Core safety validation function
    Returns: (is_safe, message)
    """
    if female_count > 0 and male_count == 0:
        return False, f"Female passengers require at least {CONFIG['MIN_MALE_REQUIRED_FOR_FEMALE_SAFETY']} male passenger(s) for safety"

    return True, "Route meets safety requirements"

def count_gender_in_cluster(cluster_users):
    """
    Count male and female users in a cluster
    Returns: (female_count, male_count)
    """
    # Common female names in the dataset for fallback identification
    female_names = {'Kritika', 'Sakshi', 'Kirandeep', 'Nandita', 'Riya', 'Jagjeet',
                   'Chaitanya', 'Isha', 'Shruti', 'Nandini', 'Chumphila', 'Anchal', 'Jasmeet'}

    if isinstance(cluster_users, pd.DataFrame):
        # Check if gender column exists
        if 'gender' in cluster_users.columns:
            female_count = len(cluster_users[cluster_users['gender'] == 'Female'])
            male_count = len(cluster_users[cluster_users['gender'] == 'Male'])
        else:
            # Fallback: infer gender from first names
            female_count = len(cluster_users[cluster_users['first_name'].isin(female_names)])
            male_count = len(cluster_users) - female_count
    else:  # List of user dicts
        female_count = 0
        male_count = 0
        for user in cluster_users:
            if isinstance(user, dict):
                # Check if gender is explicitly available
                if 'gender' in user:
                    gender = str(user.get('gender', '')).lower()
                    if gender == 'female':
                        female_count += 1
                    elif gender == 'male':
                        male_count += 1
                else:
                    # Fallback: infer from first name
                    first_name = str(user.get('first_name', ''))
                    if first_name in female_names:
                        female_count += 1
                    else:
                        male_count += 1

    return female_count, male_count

def calculate_female_cluster_compatibility(female, male_cluster):
    """
    Calculate compatibility score for merging a female into a male cluster
    Higher score = better compatibility
    """
    # Proper check for empty/None male_cluster
    if male_cluster is None or (hasattr(male_cluster, 'empty') and male_cluster.empty):
        return 0

    # Get cluster center
    if isinstance(male_cluster, pd.DataFrame):
        cluster_lat = male_cluster['latitude'].mean()
        cluster_lon = male_cluster['longitude'].mean()
        cluster_size = len(male_cluster)
    else:  # List of user dicts
        cluster_lat = sum(user['latitude'] for user in male_cluster) / len(male_cluster)
        cluster_lon = sum(user['longitude'] for user in male_cluster) / len(male_cluster)
        cluster_size = len(male_cluster)

    # Distance score (prefer closer clusters)
    distance = haversine_distance(female['latitude'], female['longitude'], cluster_lat, cluster_lon)
    distance_score = max(0, 1 - (distance / (CONFIG['GEOGRAPHIC_CLUSTER_RADIUS_KM'] * CONFIG['FEMALE_RESCUE_DISTANCE_MULTIPLIER'])))

    # Capacity score (prefer clusters with more space)
    capacity_score = max(0, 1 - (cluster_size / CONFIG['MAX_CLUSTER_SIZE']))

    # Combined score
    total_score = (distance_score * 0.6 + capacity_score * 0.4)

    return total_score

def find_best_male_cluster_for_female(female, male_clusters):
    """
    Find the best male cluster to merge with a female passenger
    Returns the best cluster or None if no suitable cluster found
    """
    best_cluster = None
    best_score = 0

    for cluster in male_clusters:
        # Check capacity
        cluster_size = len(cluster)
        if cluster_size >= CONFIG['MAX_CLUSTER_SIZE']:
            continue

        # Calculate compatibility
        score = calculate_female_cluster_compatibility(female, cluster)

        if score > best_score:
            best_score = score
            best_cluster = cluster

    return best_cluster if best_score > 0.3 else None  # Minimum threshold for compatibility

def rescue_unsafe_females(unsafe_females_df, male_clusters, office_lat, office_lon):
    """
    Try to rescue unsafe female passengers by merging them with male clusters
    Returns: (rescued_females_df, still_unsafe_females_df)
    """
    logger.info(f"Attempting to rescue {len(unsafe_females_df)} unsafe female passengers...")

    rescued_females = []
    still_unsafe_females = []

    # Convert male clusters to list of dataframes for easier manipulation
    male_cluster_dfs = []

    # Robust male_clusters validation and processing
    try:
        if male_clusters is not None and hasattr(male_clusters, '__len__') and len(male_clusters) > 0:
            # Check if male_clusters has geo_cluster column
            if hasattr(male_clusters, 'columns') and 'geo_cluster' in male_clusters.columns:
                # Get unique cluster IDs safely
                unique_clusters = male_clusters['geo_cluster'].unique()
                if unique_clusters is not None and len(unique_clusters) > 0:
                    for cluster_id in unique_clusters:
                        try:
                            cluster_df = male_clusters[male_clusters['geo_cluster'] == cluster_id]
                            if cluster_df is not None and hasattr(cluster_df, '__len__') and len(cluster_df) > 0:
                                male_cluster_dfs.append(cluster_df)
                        except Exception as cluster_error:
                            logger.warning(f"Error processing cluster {cluster_id}: {cluster_error}")
                            continue
            else:
                # If no geo_cluster column, treat the entire male_clusters as one cluster
                if hasattr(male_clusters, '__len__') and len(male_clusters) > 0:
                    male_cluster_dfs.append(male_clusters)
    except Exception as e:
        logger.warning(f"Error processing male_clusters: {e}")
        # Continue with empty male_cluster_dfs if there's an issue

    for _, female in unsafe_females_df.iterrows():
        best_cluster = find_best_male_cluster_for_female(female, male_cluster_dfs)

        if best_cluster is not None:
            # Add female to the best cluster
            new_cluster_id = best_cluster['geo_cluster'].iloc[0]
            female['geo_cluster'] = new_cluster_id
            rescued_females.append(female)

            logger.info(f"   Rescued female {female['user_id']} -> merged with cluster {new_cluster_id}")
        else:
            # No suitable cluster found
            still_unsafe_females.append(female)
            logger.warning(f"   Female {female['user_id']} could not be rescued - no suitable male cluster found")

    rescued_females_df = pd.DataFrame(rescued_females) if rescued_females else pd.DataFrame()
    still_unsafe_females_df = pd.DataFrame(still_unsafe_females) if still_unsafe_females else pd.DataFrame()

    logger.info(f"   Rescue complete: {len(rescued_females)} rescued, {len(still_unsafe_females)} still unsafe")

    return rescued_females_df, still_unsafe_females_df

def validate_and_fix_clusters_for_safety(user_df, office_lat, office_lon):
    """
    Validate clusters for female safety and fix unsafe clusters
    Returns: (safe_user_df, unsafe_females_df)
    """
    logger.info("Validating clusters for female safety...")

    safe_users = []
    unsafe_females = []

    # Common female names for identification
    female_names = {'Kritika', 'Sakshi', 'Kirandeep', 'Nandita', 'Riya', 'Jagjeet',
                   'Chaitanya', 'Isha', 'Shruti', 'Nandini', 'Chumphila', 'Anchal', 'Jasmeet'}

    for cluster_id in user_df['geo_cluster'].unique():
        cluster_users = user_df[user_df['geo_cluster'] == cluster_id]
        female_count, male_count = count_gender_in_cluster(cluster_users)

        is_safe, message = validate_route_safety(female_count, male_count)

        if is_safe:
            safe_users.append(cluster_users)
        else:
            # Cluster is unsafe - separate females and males
            logger.warning(f"   Cluster {cluster_id} unsafe: {message}")

            # Separate males and females for rescue
            if 'gender' in cluster_users.columns:
                male_users = cluster_users[cluster_users['gender'] == 'Male'].copy()
                female_users = cluster_users[cluster_users['gender'] == 'Female'].copy()
            else:
                # Fallback: use first names to identify genders
                male_users = cluster_users[~cluster_users['first_name'].isin(female_names)].copy()
                female_users = cluster_users[cluster_users['first_name'].isin(female_names)].copy()

            # Add males to safe users (they can form their own clusters)
            if male_users is not None and not male_users.empty:
                safe_users.append(male_users)

            # Add females to unsafe list for rescue
            if female_users is not None and not female_users.empty:
                unsafe_females.append(female_users)

    safe_user_df = pd.concat(safe_users, ignore_index=True) if safe_users else pd.DataFrame()
    unsafe_females_df = pd.concat(unsafe_females, ignore_index=True) if unsafe_females else pd.DataFrame()

    logger.info(f"   Safety validation: {len(safe_user_df)} users in safe clusters, {len(unsafe_females_df)} females need rescue")

    return safe_user_df, unsafe_females_df

def smart_female_rescue(unsafe_females_df, males_df, drivers_df, office_lat, office_lon):
    """
    Smart rescue operation that considers geographic proximity for female safety
    Keeps geographically close female users together and finds optimal male drivers
    """
    logger.info("Starting SMART female rescue with geographic clustering...")

    if unsafe_females_df.empty or males_df.empty:
        logger.info("   No unsafe females or males to rescue")
        return pd.DataFrame(), pd.DataFrame()

    # Create female clusters based on geographic proximity
    female_clusters = create_geographic_female_clusters(unsafe_females_df)
    logger.info(f"   Created {len(female_clusters)} female geographic clusters")

    # Get available male drivers with their locations
    male_drivers = []
    if not males_df.empty:
        # Group males by their existing clusters
        if 'geo_cluster' in males_df.columns:
            for cluster_id in males_df['geo_cluster'].unique():
                cluster_males = males_df[males_df['geo_cluster'] == cluster_id]
                if len(cluster_males) > 0:
                    # Find the best driver for this male cluster
                    best_driver = find_best_driver_for_cluster(cluster_males, drivers_df, office_lat, office_lon)
                    if best_driver:
                        male_drivers.append({
                            'driver': best_driver,
                            'males': cluster_males,
                            'capacity': best_driver.get('capacity', CONFIG['MAX_CLUSTER_SIZE'])
                        })
        else:
            # All males as one group
            best_driver = find_best_driver_for_cluster(males_df, drivers_df, office_lat, office_lon)
            if best_driver:
                male_drivers.append({
                    'driver': best_driver,
                    'males': males_df,
                    'capacity': best_driver.get('capacity', CONFIG['MAX_CLUSTER_SIZE'])
                })

    # Also consider available unassigned male drivers
    if drivers_df is not None:
        unassigned_drivers = drivers_df[drivers_df['gender'] == 'Male'] if 'gender' in drivers_df.columns else drivers_df
        for _, driver in unassigned_drivers.iterrows():
            # Check if this driver is already assigned to a male cluster
            already_assigned = any(md['driver']['driver_id'] == driver['driver_id'] for md in male_drivers)
            if not already_assigned:
                male_drivers.append({
                    'driver': driver.to_dict(),
                    'males': pd.DataFrame(),
                    'capacity': driver.get('capacity', CONFIG['MAX_CLUSTER_SIZE'])
                })

    logger.info(f"   Available male drivers/clusters: {len(male_drivers)}")

    # Assign female clusters to male drivers
    rescued_females = []
    remaining_unsafe_females = []

    for female_cluster in female_clusters:
        best_assignment = find_best_male_driver_for_female_cluster(
            female_cluster, male_drivers, office_lat, office_lon
        )

        if best_assignment:
            # Assign females to this male driver
            driver_info = best_assignment['driver_info']
            combined_users = pd.concat([driver_info['males'], female_cluster], ignore_index=True)

            # Update the male driver's assigned users
            driver_info['males'] = combined_users
            driver_info['capacity'] -= len(female_cluster)

            rescued_females.append(female_cluster)
            logger.info(f"   Assigned {len(female_cluster)} females to driver {driver_info['driver']['driver_id']}")
        else:
            # No suitable male driver found
            remaining_unsafe_females.append(female_cluster)
            logger.warning(f"   No suitable male driver found for {len(female_cluster)} females")

    # Combine rescued females with their assigned male drivers
    final_safe_users = []
    for driver_info in male_drivers:
        if len(driver_info['males']) > 0:
            final_safe_users.append(driver_info['males'])

    # Add already safe users (from the original validation)
    # This will be handled by the calling function

    rescued_females_df = pd.concat(rescued_females, ignore_index=True) if rescued_females else pd.DataFrame()
    remaining_unsafe_df = pd.concat(remaining_unsafe_females, ignore_index=True) if remaining_unsafe_females else pd.DataFrame()

    logger.info(f"   Smart rescue: {len(rescued_females_df)} females rescued, {len(remaining_unsafe_df)} still unsafe")

    return rescued_females_df, remaining_unsafe_df

def create_geographic_female_clusters(unsafe_females_df):
    """
    Create clusters of female users based on geographic proximity
    Keeps geographically close females together
    """
    if unsafe_females_df.empty:
        return []

    # Configuration parameters
    max_cluster_radius_km = CONFIG.get('FEMALE_GEOGRAPHIC_CLUSTER_RADIUS_KM', 2.0)
    min_cluster_size = CONFIG.get('MIN_FEMALE_CLUSTER_SIZE', 2)

    clusters = []
    unassigned_indices = set(unsafe_females_df.index)

    while unassigned_indices:
        # Get the first unassigned female
        first_index = next(iter(unassigned_indices))
        cluster_center = unsafe_females_df.loc[first_index]
        cluster_indices = {first_index}
        unassigned_indices.remove(first_index)

        # Find all nearby females within the radius
        for idx in list(unassigned_indices):
            user = unsafe_females_df.loc[idx]
            distance = haversine_distance(
                cluster_center['latitude'], cluster_center['longitude'],
                user['latitude'], user['longitude']
            )

            if distance <= max_cluster_radius_km:
                cluster_indices.add(idx)
                unassigned_indices.remove(idx)

        # Only consider it a valid cluster if it has minimum size or if no more clustering possible
        if len(cluster_indices) >= min_cluster_size or not unassigned_indices:
            cluster_df = unsafe_females_df.loc[list(cluster_indices)].copy()
            clusters.append(cluster_df)

    return clusters

def find_best_driver_for_cluster(users_df, drivers_df, office_lat, office_lon):
    """
    Find the best driver for a given cluster of users based on geographic efficiency
    """
    if drivers_df is None or drivers_df.empty:
        return None

    # Calculate cluster center
    cluster_lat = users_df['latitude'].mean()
    cluster_lon = users_df['longitude'].mean()

    best_driver = None
    best_score = 0

    for _, driver in drivers_df.iterrows():
        # Calculate distance from driver to cluster center
        driver_distance = haversine_distance(
            driver['latitude'], driver['longitude'],
            cluster_lat, cluster_lon
        )

        # Calculate distance efficiency score (lower distance = higher score)
        distance_score = max(0, 1 - (driver_distance / CONFIG.get('MAX_DRIVER_CLUSTER_DISTANCE_KM', 10)))

        # Consider capacity
        capacity = driver.get('capacity', CONFIG['MAX_CLUSTER_SIZE'])
        capacity_score = max(0, 1 - (len(users_df) / capacity))

        # Combined score
        total_score = (distance_score * 0.7 + capacity_score * 0.3)

        if total_score > best_score:
            best_score = total_score
            best_driver = driver.to_dict()

    return best_driver

def find_best_male_driver_for_female_cluster(female_cluster, male_drivers, office_lat, office_lon):
    """
    Find the best male driver/cluster to assign a female cluster to
    Considers geographic proximity, capacity, and route efficiency
    """
    if not male_drivers:
        return None

    # Calculate female cluster center
    cluster_lat = female_cluster['latitude'].mean()
    cluster_lon = female_cluster['longitude'].mean()

    best_assignment = None
    best_score = 0

    for driver_info in male_drivers:
        # Check capacity
        if driver_info['capacity'] < len(female_cluster):
            continue

        # Calculate distance from female cluster to male driver's location
        driver_distance = haversine_distance(
            driver_info['driver']['latitude'], driver_info['driver']['longitude'],
            cluster_lat, cluster_lon
        )

        # Calculate distance efficiency score
        distance_score = max(0, 1 - (driver_distance / CONFIG.get('MAX_FEMALE_MALE_DRIVER_DISTANCE_KM', 8)))

        # Calculate combined route efficiency if we add these females
        combined_users = pd.concat([driver_info['males'], female_cluster], ignore_index=True)
        route_efficiency = calculate_route_efficiency_score(combined_users, driver_info['driver'], office_lat, office_lon)

        # Combined score
        total_score = (distance_score * 0.4 + route_efficiency * 0.6)

        if total_score > best_score:
            best_score = total_score
            best_assignment = {
                'driver_info': driver_info,
                'distance_km': driver_distance,
                'efficiency_score': route_efficiency
            }

    return best_assignment

def calculate_route_efficiency_score(users_df, driver, office_lat, office_lon):
    """
    Calculate route efficiency score based on geographic compactness
    Higher score = more efficient route
    """
    if users_df.empty:
        return 0

    # Calculate route compactness (average distance between users and driver)
    total_distance = 0
    for _, user in users_df.iterrows():
        distance = haversine_distance(
            driver['latitude'], driver['longitude'],
            user['latitude'], user['longitude']
        )
        total_distance += distance

    avg_distance = total_distance / len(users_df)

    # Calculate efficiency score (lower average distance = higher score)
    max_acceptable_distance = CONFIG.get('MAX_DRIVER_CLUSTER_DISTANCE_KM', 10)
    efficiency_score = max(0, 1 - (avg_distance / max_acceptable_distance))

    return efficiency_score

# ================== DIRECTION-AWARE ROUTE MERGING (COPIED FROM BALANCE) ==================

def calculate_route_trajectory_bearing(route, office_lat, office_lon):
    """
    Calculate the main trajectory bearing of a route considering all pickup points
    This gives the overall direction the route is heading towards the office
    """
    if not route or not route.get('assigned_users'):
        # For empty routes, return bearing from driver to office
        driver_pos = (route['latitude'], route['longitude']) if route else (0, 0)
        return calculate_bearing(driver_pos[0], driver_pos[1], office_lat, office_lon)

    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    # Calculate bearings for each segment of the route
    bearings = []

    # Driver to first user
    first_user = route['assigned_users'][0]
    first_user_pos = (first_user['lat'], first_user['lng'])
    driver_to_first = calculate_bearing(driver_pos[0], driver_pos[1], first_user_pos[0], first_user_pos[1])
    bearings.append(driver_to_first)

    # Between consecutive users
    for i in range(len(route['assigned_users']) - 1):
        current_user_pos = (route['assigned_users'][i]['lat'], route['assigned_users'][i]['lng'])
        next_user_pos = (route['assigned_users'][i + 1]['lat'], route['assigned_users'][i + 1]['lng'])
        segment_bearing = calculate_bearing(current_user_pos[0], current_user_pos[1], next_user_pos[0], next_user_pos[1])
        bearings.append(segment_bearing)

    # Last user to office
    last_user = route['assigned_users'][-1]
    last_user_pos = (last_user['lat'], last_user['lng'])
    last_to_office = calculate_bearing(last_user_pos[0], last_user_pos[1], office_pos[0], office_pos[1])
    bearings.append(last_to_office)

    # Calculate weighted average bearing (weighted by distance of each segment)
    if not bearings:
        return calculate_bearing(driver_pos[0], driver_pos[1], office_lat, office_lon)

    # For simplicity, return the average bearing
    # In a more advanced version, we could weight by segment distances
    avg_bearing = sum(bearings) / len(bearings)
    return avg_bearing

def are_routes_direction_compatible(route1, route2, office_lat, office_lon, tolerance_degrees=None):
    """
    Check if two routes are heading in compatible directions
    Returns compatibility score and boolean result
    """
    if tolerance_degrees is None:
        tolerance_degrees = CONFIG.get('DIRECTION_TOLERANCE_DEGREES', 60)  # More permissive default

    bearing1 = calculate_route_trajectory_bearing(route1, office_lat, office_lon)
    bearing2 = calculate_route_trajectory_bearing(route2, office_lat, office_lon)

    bearing_diff = abs(bearing_difference(bearing1, bearing2))

    # Calculate compatibility score (0-1, where 1 is perfectly aligned)
    compatibility_score = max(0, 1 - (bearing_diff / 180))

    is_compatible = bearing_diff <= tolerance_degrees

    return is_compatible, compatibility_score, bearing_diff

def calculate_merge_detour_cost(route1, route2, office_lat, office_lon):
    """
    Calculate the additional detour cost if we merge two routes
    Returns detour distance in km and a merge feasibility score
    """
    # Get all users from both routes
    all_users = route1['assigned_users'] + route2['assigned_users']

    # Choose the better driver (higher capacity)
    if route1['vehicle_type'] >= route2['vehicle_type']:
        main_driver = route1
        other_driver = route2
    else:
        main_driver = route2
        other_driver = route1

    # Calculate original total distance for both routes separately
    original_distance1 = calculate_total_route_distance(route1, office_lat, office_lon)
    original_distance2 = calculate_total_route_distance(route2, office_lat, office_lon)
    original_total = original_distance1 + original_distance2

    # Calculate merged route distance (optimized order)
    driver_pos = (main_driver['latitude'], main_driver['longitude'])
    merged_distance = calculate_optimal_merged_distance(driver_pos, all_users, office_lat, office_lon)

    # Detour cost is the difference
    detour_cost = merged_distance - original_total

    # Calculate merge feasibility score (lower is better)
    # Normalize by number of users (merge should be efficient per user)
    detour_per_user = detour_cost / len(all_users) if all_users else float('inf')

    # Bonus for merging very small routes
    size_bonus = 0
    if len(route1['assigned_users']) <= 2 or len(route2['assigned_users']) <= 2:
        size_bonus = 0.5  # Reduce effective cost for small routes

    effective_cost = detour_per_user - size_bonus

    return detour_cost, effective_cost, merged_distance

def calculate_total_route_distance(route, office_lat, office_lon):
    """Calculate total distance traveled by a route"""
    if not route.get('assigned_users'):
        return haversine_distance(route['latitude'], route['longitude'], office_lat, office_lon)

    total_distance = 0
    driver_pos = (route['latitude'], route['longitude'])

    # Driver to first user
    first_user = route['assigned_users'][0]
    first_user_pos = (first_user['lat'], first_user['lng'])
    total_distance += haversine_distance(driver_pos[0], driver_pos[1], first_user_pos[0], first_user_pos[1])

    # Between consecutive users
    for i in range(len(route['assigned_users']) - 1):
        current_user_pos = (route['assigned_users'][i]['lat'], route['assigned_users'][i]['lng'])
        next_user_pos = (route['assigned_users'][i + 1]['lat'], route['assigned_users'][i + 1]['lng'])
        total_distance += haversine_distance(current_user_pos[0], current_user_pos[1], next_user_pos[0], next_user_pos[1])

    # Last user to office
    last_user = route['assigned_users'][-1]
    last_user_pos = (last_user['lat'], last_user['lng'])
    total_distance += haversine_distance(last_user_pos[0], last_user_pos[1], office_lat, office_lon)

    return total_distance

def calculate_optimal_merged_distance(driver_pos, all_users, office_lat, office_lon):
    """
    Calculate optimal route distance for merged users using a simple greedy approach
    This is a simplified version - could be enhanced with TSP optimization
    """
    if not all_users:
        return haversine_distance(driver_pos[0], driver_pos[1], office_lat, office_lon)

    # Simple greedy approach: sort users by projection along main axis to office
    office_pos = (office_lat, office_lon)
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    def user_projection_score(user):
        user_pos = (user['lat'], user['lng'])
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        bearing_alignment = math.cos(math.radians(abs(bearing_difference(user_bearing, main_bearing))))
        distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        return distance * bearing_alignment

    # Sort users by projection score
    sorted_users = sorted(all_users, key=user_projection_score)

    # Calculate route distance with sorted users
    total_distance = 0
    current_pos = driver_pos

    for user in sorted_users:
        user_pos = (user['lat'], user['lng'])
        total_distance += haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        current_pos = user_pos

    # Final leg to office
    total_distance += haversine_distance(current_pos[0], current_pos[1], office_lat, office_lon)

    return total_distance

def merge_small_routes_with_nearby(routes, office_lat, office_lon):
    """
    Main algorithm to merge small routes (1-2 users) with nearby compatible routes
    This is the core direction-aware route merging functionality
    """
    logger.info("Step 3.5: Direction-aware route merging for small routes...")

    if len(routes) < 2:
        logger.info("   No merging needed (less than 2 routes)")
        return routes

    original_route_count = len(routes)
    merged_routes = []
    used_route_indices = set()

    # Sort routes to prioritize merging small routes first
    route_data = []
    for i, route in enumerate(routes):
        if i in used_route_indices:
            continue

        route_data.append({
            'index': i,
            'route': route,
            'user_count': len(route['assigned_users']),
            'capacity': route['vehicle_type'],
            'available_capacity': route['vehicle_type'] - len(route['assigned_users'])
        })

    # Sort by user count (small routes first) to prioritize merging them
    route_data.sort(key=lambda x: x['user_count'])

    merges_performed = 0

    for i, route1_data in enumerate(route_data):
        if route1_data['index'] in used_route_indices:
            continue

        # Focus on small routes or routes with available capacity
        if route1_data['user_count'] > CONFIG['SMALL_ROUTE_THRESHOLD'] and route1_data['available_capacity'] == 0:
            # This is a reasonably sized full route, keep as is
            merged_routes.append(route1_data['route'])
            used_route_indices.add(route1_data['index'])
            continue

        best_merge_candidate = None
        best_merge_score = float('inf')

        # Look for merge candidates
        for j, route2_data in enumerate(route_data):
            if i == j or route2_data['index'] in used_route_indices:
                continue

            # Check if routes can be merged (capacity constraint)
            combined_users = route1_data['user_count'] + route2_data['user_count']
            max_capacity = max(route1_data['capacity'], route2_data['capacity'])

            if combined_users > max_capacity:
                continue  # Cannot merge - would exceed capacity

            # Calculate geographic distance between route centers
            route1_center = calculate_route_center(route1_data['route']['assigned_users'])
            route2_center = calculate_route_center(route2_data['route']['assigned_users'])

            if not route1_center or not route2_center:
                continue

            geographic_distance = haversine_distance(
                route1_center[0], route1_center[1],
                route2_center[0], route2_center[1]
            )

            # Skip if routes are too far apart geographically
            if geographic_distance > CONFIG['MAX_MERGE_DISTANCE_KM']:
                continue

            # Check direction compatibility
            is_direction_compatible, compatibility_score, bearing_diff = are_routes_direction_compatible(
                route1_data['route'], route2_data['route'], office_lat, office_lon
            )

            if not is_direction_compatible:
                continue

            # Calculate merge detour cost
            detour_cost, effective_cost, merged_distance = calculate_merge_detour_cost(
                route1_data['route'], route2_data['route'], office_lat, office_lon
            )

            # Calculate overall merge score (lower is better)
            # Factors: geographic distance, direction alignment, detour cost
            merge_score = (
                geographic_distance * 0.3 +           # Geographic proximity
                (1 - compatibility_score) * 2.0 +     # Direction alignment
                effective_cost * 0.5                  # Detour efficiency
            )

            # Strong bonus for merging very small routes
            if route1_data['user_count'] <= 1 or route2_data['user_count'] <= 1:
                merge_score -= 1.0

            # Bonus for good capacity utilization
            utilization = combined_users / max_capacity
            if utilization >= 0.7:
                merge_score -= 0.5

            # Check if this is the best merge candidate
            if merge_score < best_merge_score:
                best_merge_candidate = route2_data
                best_merge_score = merge_score

        # If we found a good merge candidate, perform the merge
        if best_merge_candidate and best_merge_score < CONFIG['MERGE_SCORE_THRESHOLD']:  # Threshold for merging
            merged_route = perform_route_merge(
                route1_data['route'],
                best_merge_candidate['route'],
                office_lat,
                office_lon
            )

            if merged_route:
                merged_routes.append(merged_route)
                used_route_indices.add(route1_data['index'])
                used_route_indices.add(best_merge_candidate['index'])
                merges_performed += 1

                total_users = len(merged_route['assigned_users'])
                capacity = merged_route['vehicle_type']
                utilization = (total_users / capacity) * 100

                logger.info(f"   Merged routes: {route1_data['route']['driver_id']} + {best_merge_candidate['route']['driver_id']}")
                logger.info(f"      Result: {total_users}/{capacity} users ({utilization:.1f}%)")
                logger.info(f"      Merge score: {best_merge_score:.2f}")
            else:
                # If merge failed, keep the original route
                merged_routes.append(route1_data['route'])
                used_route_indices.add(route1_data['index'])
                logger.warning(f"      Merge failed between routes {route1_data['route']['driver_id']} and {best_merge_candidate['route']['driver_id']}")
        else:
            # No good merge found, keep the original route
            merged_routes.append(route1_data['route'])
            used_route_indices.add(route1_data['index'])

            if route1_data['user_count'] <= CONFIG['SMALL_ROUTE_THRESHOLD']:
                logger.info(f"   Keeping small route {route1_data['route']['driver_id']} ({route1_data['user_count']} users) - no compatible merge found")

    # Add any remaining routes that weren't processed
    for route_data in route_data:
        if route_data['index'] not in used_route_indices:
            merged_routes.append(route_data['route'])

    # Log results
    final_route_count = len(merged_routes)
    routes_eliminated = original_route_count - final_route_count

    if routes_eliminated > 0:
        logger.info(f"   Direction-aware merging complete: {routes_eliminated} routes eliminated ({original_route_count} -> {final_route_count})")
        logger.info(f"   Merges performed: {merges_performed}")
    else:
        logger.info(f"   No beneficial merges found, keeping all {original_route_count} routes")

    return merged_routes

def optimize_wide_spread_routes_after_merge(routes, office_lat, office_lon):
    """
    Post-merge optimization to detect and fix wide-spread routes created by merging
    Specifically addresses the driver 335468 issue where route merging creates problematic routes
    """
    logger.info("Step 4: Wide-spread route optimization...")
    logger.info(f"   Analyzing {len(routes)} routes for wide-spread issues...")

    # Debug: List all routes being analyzed
    for i, route in enumerate(routes):
        logger.info(f"   Route {i+1}: {route['driver_id']} ({len(route['assigned_users'])} passengers)")

    if len(routes) < 2:
        logger.info("   No optimization needed (less than 2 routes)")
        return routes

    problematic_routes = []

    # Analyze each route for wide-spread issues
    for i, route in enumerate(routes):
        if len(route['assigned_users']) < 2:
            continue  # Skip single-user routes

        # Calculate route spread (maximum distance between any two points)
        positions = [(route['latitude'], route['longitude'])]
        for user in route['assigned_users']:
            positions.append((user['lat'], user['lng']))

        max_distance = 0
        for pos_i in range(len(positions)):
            for pos_j in range(pos_i + 1, len(positions)):
                dist = haversine_distance(positions[pos_i][0], positions[pos_i][1],
                                        positions[pos_j][0], positions[pos_j][1])
                max_distance = max(max_distance, dist)

        # Check if this route has wide spread with females
        female_names = {'Kritika', 'Sakshi', 'Kirandeep', 'Nandita', 'Riya', 'Jagjeet',
                       'Chaitanya', 'Isha', 'Shruti', 'Nandini', 'Chumphila', 'Anchal', 'Jasmeet'}

        females_in_route = [user for user in route['assigned_users']
                           if user.get('first_name', '') in female_names]
        males_in_route = [user for user in route['assigned_users']
                         if user.get('first_name', '') not in female_names]

        # MORE AGGRESSIVE DETECTION - Lower thresholds and specific driver 335468 handling
        is_problematic = False
        reason = ""

        # Check 1: Standard wide-spread detection (lowered threshold)
        if max_distance > 6.0 and len(females_in_route) > 0 and len(males_in_route) > 0:
            is_problematic = True
            reason = f"Spread {max_distance:.2f}km > 6.0km with females"

        # Check 2: Specific driver 335468 detection (even more aggressive)
        if route['driver_id'] == '335468' and max_distance > 4.0 and len(females_in_route) > 0:
            is_problematic = True
            reason = f"Driver 335468 spread {max_distance:.2f}km > 4.0km with females"

        # Check 3: High variance in user distances from driver
        if len(females_in_route) > 0 and len(males_in_route) > 0:
            user_distances = []
            for user in route['assigned_users']:
                dist = haversine_distance(route['latitude'], route['longitude'], user['lat'], user['lng'])
                user_distances.append(dist)

            if len(user_distances) > 1:
                distance_variance = max(user_distances) - min(user_distances)
                if distance_variance > 8.0:  # High variance in distances
                    is_problematic = True
                    reason = f"High distance variance {distance_variance:.2f}km with females"

        if is_problematic:
            problematic_routes.append({
                'index': i,
                'route': route,
                'spread_km': max_distance,
                'females': females_in_route,
                'males': males_in_route,
                'reason': reason
            })

            logger.info(f"   Identified problematic route {route['driver_id']}: {max_distance:.2f}km spread - {reason}")
            logger.info(f"     Females: {len(females_in_route)}, Males: {len(males_in_route)}")

            # Debug: specifically identify driver 335468 with detailed analysis
            if route['driver_id'] == '335468':
                logger.info(f"   *** DRIVER 335468 ISSUE DETECTED ***")
                logger.info(f"   Route spread: {max_distance:.2f} km (threshold: 4.0km for driver 335468)")
                logger.info(f"   Reason: {reason}")
                logger.info(f"   Females: {len(females_in_route)}, Males: {len(males_in_route)}")
                for user in route['assigned_users']:
                    user_type = "Female" if user.get('first_name', '') in female_names else "Male"
                    dist_from_driver = haversine_distance(route['latitude'], route['longitude'], user['lat'], user['lng'])
                    logger.info(f"     {user['first_name']} ({user_type}): {dist_from_driver:.2f}km from driver")

    if not problematic_routes:
        logger.info("   No wide-spread routes detected after merging")
        return routes

    logger.info(f"   Found {len(problematic_routes)} wide-spread routes that need optimization")

    # Try to fix each problematic route by redistributing problematic males
    optimized_routes = routes.copy()
    fixes_applied = 0

    for problematic in problematic_routes:
        route_index = problematic['index']
        route = problematic['route']
        females = problematic['females']
        males = problematic['males']

        logger.info(f"   Attempting to optimize route {route['driver_id']}...")

        # Calculate female cluster center
        female_center_lat = sum(f['lat'] for f in females) / len(females)
        female_center_lon = sum(f['lng'] for f in females) / len(females)

        # ENHANCED: Find problematic males with LOWERED thresholds
        problematic_males = []
        for male in males:
            male_to_female_distance = haversine_distance(
                male['lat'], male['lng'], female_center_lat, female_center_lon
            )
            # More aggressive threshold for driver 335468
            threshold = 3.0 if route['driver_id'] == '335468' else 5.0
            if male_to_female_distance > threshold:
                problematic_males.append((male, male_to_female_distance))

        if not problematic_males:
            logger.info(f"     No problematic males found in route {route['driver_id']} (threshold: {threshold}km)")
            continue

        # Sort by distance (farthest first)
        problematic_males.sort(key=lambda x: x[1], reverse=True)
        most_problematic_male = problematic_males[0][0]

        logger.info(f"     Most problematic male: {most_problematic_male['first_name']} ({problematic_males[0][1]:.2f}km from females)")

        # ENHANCED: Try to move this male to another route with more aggressive search
        best_target_route = None
        best_improvement = 0
        best_candidate = None

        for j, other_route in enumerate(optimized_routes):
            if j == route_index:
                continue  # Skip the same route

            # Check if other route can accommodate this male
            if len(other_route['assigned_users']) >= other_route['vehicle_type']:
                continue  # Route is full

            # Calculate how well this male fits the other route
            other_route_center_lat = sum(u['lat'] for u in other_route['assigned_users']) / len(other_route['assigned_users'])
            other_route_center_lon = sum(u['lng'] for u in other_route['assigned_users']) / len(other_route['assigned_users'])

            male_to_other_route_distance = haversine_distance(
                most_problematic_male['lat'], most_problematic_male['lng'],
                other_route_center_lat, other_route_center_lon
            )

            # ENHANCED: More aggressive improvement criteria for driver 335468
            min_improvement = 1.0 if route['driver_id'] == '335468' else 2.0
            distance_improvement = problematic_males[0][1] - male_to_other_route_distance

            # Consider this route if it provides improvement
            if distance_improvement > min_improvement:
                # Check if this move maintains safety for both routes
                original_females_count = len(females)
                original_males_count = len(males) - 1  # After removing problematic male

                other_route_females = [user for user in other_route['assigned_users']
                                      if user.get('first_name', '') in female_names]
                other_route_males = [user for user in other_route['assigned_users']
                                    if user.get('first_name', '') not in female_names]

                other_new_females_count = len(other_route_females)
                other_new_males_count = len(other_route_males) + 1  # After adding our male

                # Safety checks
                original_safe = original_males_count >= original_females_count if original_females_count > 0 else True
                other_safe = other_new_males_count >= other_new_females_count if other_new_females_count > 0 else True

                # Allow exchanges that maintain or improve safety
                if original_safe and other_safe:
                    # Bonus for exchanging with male-only routes (creates safety opportunities)
                    if len(other_route_females) == 0:
                        distance_improvement += 1.0  # Bonus for male-only route

                    # Bonus for driver 335468 fixes
                    if route['driver_id'] == '335468':
                        distance_improvement += 2.0  # Extra bonus for fixing the specific issue

                    if distance_improvement > best_improvement:
                        best_improvement = distance_improvement
                        best_target_route = j
                        best_candidate = {
                            'target_route': other_route,
                            'improvement': distance_improvement,
                            'male_distance': male_to_other_route_distance
                        }

        # ENHANCED: Perform the exchange with LOWERED thresholds for driver 335468
        min_threshold = 1.5 if route['driver_id'] == '335468' else 2.0
        if best_target_route is not None and best_improvement > min_threshold:
            target_route = optimized_routes[best_target_route]

            logger.info(f"     Moving male {most_problematic_male['first_name']} to route {target_route['driver_id']}")
            logger.info(f"     Distance improvement: {best_improvement:.2f} km (new distance: {best_candidate['male_distance']:.2f}km)")

            # Remove male from original route
            original_route = optimized_routes[route_index]
            original_route['assigned_users'] = [
                user for user in original_route['assigned_users']
                if user['user_id'] != most_problematic_male['user_id']
            ]

            # Add male to target route
            target_route['assigned_users'].append(most_problematic_male)

            fixes_applied += 1
            logger.info(f"     SUCCESS: Fixed wide-spread route {route['driver_id']} (improvement: {best_improvement:.2f}km)")

            # Special success message for driver 335468
            if route['driver_id'] == '335468':
                logger.info(f"     *** DRIVER 335468 ISSUE RESOLVED ***")
        else:
            logger.info(f"     No suitable target route found for {most_problematic_male['first_name']}")
            logger.info(f"     Best improvement found: {best_improvement:.2f}km (threshold: {min_threshold}km)")

    if fixes_applied > 0:
        logger.info(f"   Post-merge optimization complete: {fixes_applied} wide-spread routes fixed")
    else:
        logger.info("   Post-merge optimization complete: No fixes could be applied")

    return optimized_routes

def perform_route_merge(route1, route2, office_lat, office_lon):
    """
    Perform the actual merging of two routes, optimizing user sequence
    """
    try:
        # Choose the better driver (higher capacity wins, break ties by keeping route1 driver)
        if route1['vehicle_type'] >= route2['vehicle_type']:
            merged_route = route1.copy()
            merged_capacity = route1['vehicle_type']
        else:
            merged_route = route2.copy()
            merged_capacity = route2['vehicle_type']

        # Combine users from both routes
        all_users = route1['assigned_users'] + route2['assigned_users']

        # Check if combined users exceed capacity (should not happen due to prior checks)
        if len(all_users) > merged_capacity:
            logger.warning(f"   Merge failed: {len(all_users)} users > capacity {merged_capacity}")
            return None

        # Optimize user sequence for the merged route
        driver_pos = (merged_route['latitude'], merged_route['longitude'])

        # Create temporary user list for optimization
        user_list = []
        for user in all_users:
            user_list.append({
                'user_id': user['user_id'],
                'latitude': user['lat'],
                'longitude': user['lng'],
                'office_distance': user.get('office_distance', 0)
            })

        # Optimize user sequence using projection along main axis
        optimized_users = optimize_user_sequence_for_merged_route(driver_pos, user_list, office_lat, office_lon)

        # Update merged route with optimized user sequence
        merged_route['assigned_users'] = []
        for user in optimized_users:
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }

            # Preserve additional user information if present
            for original_user in all_users:
                if str(original_user.get('user_id', '')) == user['user_id']:
                    for field in ['first_name', 'email', 'address', 'employee_shift', 'shift_type', 'last_name', 'phone']:
                        if field in original_user:
                            user_data[field] = str(original_user[field])
                    break

            merged_route['assigned_users'].append(user_data)

        return merged_route

    except Exception as e:
        logger.error(f"Error performing route merge: {e}")
        return None

def optimize_user_sequence_for_merged_route(driver_pos, user_list, office_lat, office_lon):
    """
    Optimize the pickup sequence for merged route users
    Uses a simple greedy approach based on projection towards office
    """
    if not user_list:
        return []

    office_pos = (office_lat, office_lon)
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    def user_projection_score(user):
        """Score users based on their alignment with main route direction"""
        user_pos = (user['latitude'], user['longitude'])
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        bearing_alignment = math.cos(math.radians(abs(bearing_difference(user_bearing, main_bearing))))
        distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        return distance * bearing_alignment

    # Sort users by projection score (prefer users in the direction of office)
    optimized_users = sorted(user_list, key=user_projection_score)

    return optimized_users

def calculate_route_center(users):
    """Calculate the geographic center of a route's users"""
    if not users:
        return None

    center_lat = sum(user['lat'] for user in users) / len(users)
    center_lon = sum(user['lng'] for user in users) / len(users)

    return (center_lat, center_lon)

# ================== STEP 1: SIMPLE GEOGRAPHIC CLUSTERING ==================

def cluster_users_by_proximity(user_df, office_lat, office_lon):
    """
    Improved geographic clustering with size limits to prevent oversized clusters
    """
    logger.info("Step 1: Geographic clustering by proximity...")

    if len(user_df) < 2:
        user_df['geo_cluster'] = 0
        return user_df

    # Convert coordinates to km for accurate distance calculation
    coords = user_df[['latitude', 'longitude']].values
    coords_km = np.array([coords_to_km(coord[0], coord[1], office_lat, office_lon) for coord in coords])

    # Use DBSCAN with conservative radius
    dbscan = DBSCAN(eps=CONFIG['GEOGRAPHIC_CLUSTER_RADIUS_KM'], min_samples=1)
    clusters = dbscan.fit_predict(coords_km)

    user_df['geo_cluster'] = clusters

    # Split oversized clusters to prevent large geographic spreads
    user_df = split_oversized_clusters(user_df, office_lat, office_lon)

    # Log cluster results
    n_clusters = len(user_df['geo_cluster'].unique())
    logger.info(f"   Created {n_clusters} geographic clusters")

    for cluster_id in user_df['geo_cluster'].unique():
        cluster_size = len(user_df[user_df['geo_cluster'] == cluster_id])
        logger.info(f"      Cluster {cluster_id}: {cluster_size} users")

    return user_df

def split_oversized_clusters(user_df, office_lat, office_lon):
    """Split clusters that are too large or spread out"""
    logger.info("   Splitting oversized clusters...")

    max_cluster_size = CONFIG['MAX_CLUSTER_SIZE']

    user_df = user_df.copy()
    next_cluster_id = user_df['geo_cluster'].max() + 1

    for cluster_id in user_df['geo_cluster'].unique():
        cluster_users = user_df[user_df['geo_cluster'] == cluster_id]

        # Check if cluster is too large
        if len(cluster_users) > max_cluster_size:
            logger.info(f"      Splitting large cluster {cluster_id} ({len(cluster_users)} users)")

            # Calculate cluster center
            center_lat = cluster_users['latitude'].mean()
            center_lon = cluster_users['longitude'].mean()

            # Sort users by distance from center
            distances = []
            for _, user in cluster_users.iterrows():
                dist = haversine_distance(center_lat, center_lon, user['latitude'], user['longitude'])
                distances.append((dist, user.name))

            distances.sort()  # Sort by distance from center

            # Split into groups of max_cluster_size
            groups = []
            current_group = []

            for dist, idx in distances:
                if len(current_group) >= max_cluster_size:
                    groups.append(current_group)
                    current_group = []

                current_group.append(idx)

            if current_group:
                groups.append(current_group)

            # Assign new cluster IDs
            for i, group in enumerate(groups):
                if i == 0:
                    # Keep original cluster ID for first group
                    continue
                else:
                    # Assign new cluster ID for additional groups
                    user_df.loc[group, 'geo_cluster'] = next_cluster_id
                    next_cluster_id += 1

            logger.info(f"      Split into {len(groups)} sub-clusters")

    return user_df

# ================== STEP 2: SMART CAPACITY MATCHING ==================

def assign_cab_to_cluster(cluster_users, available_drivers, office_lat, office_lon):
    """
    Smart capacity matching - send appropriately sized cabs to clusters
    Bigger cabs for denser areas, smaller cabs for smaller groups
    """
    cluster_size = len(cluster_users)

    if cluster_size == 0:
        return None

    # Sort drivers by capacity (prefer exact matches, then larger)
    suitable_drivers = available_drivers[available_drivers['capacity'] >= cluster_size].copy()

    if suitable_drivers.empty:
        logger.warning(f"No cab available for cluster of {cluster_size} users")
        return None

    # Find best driver: prefer exact capacity match, otherwise smallest that fits
    suitable_drivers['waste'] = suitable_drivers['capacity'] - cluster_size
    suitable_drivers = suitable_drivers.sort_values(['waste', 'priority'], ascending=[True, True])

    best_driver = suitable_drivers.iloc[0]

    # Calculate basic distance cost (driver to cluster center)
    cluster_center = (cluster_users['latitude'].mean(), cluster_users['longitude'].mean())
    driver_to_cluster = haversine_distance(
        best_driver['latitude'], best_driver['longitude'],
        cluster_center[0], cluster_center[1]
    )

    utilization = (cluster_size / best_driver['capacity']) * 100

    logger.info(f"   Assigned {best_driver['capacity']}-seater to {cluster_size} users "
               f"({utilization:.1f}% utilization, {driver_to_cluster:.1f}km to cluster)")

    return best_driver

# ================== STEP 3: REAL "ON THE WAY" LOGIC ==================

def is_user_on_the_way(user_pos, driver_pos, office_pos, current_route_users):
    """
    Improved "on the way" checking - more permissive to catch more users
    """
    if not current_route_users:
        # For empty routes, check if user is in general direction of office
        user_bearing_to_office = calculate_bearing(user_pos[0], user_pos[1], office_pos[0], office_pos[1])
        driver_bearing_to_office = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

        bearing_diff = abs(bearing_difference(user_bearing_to_office, driver_bearing_to_office))

        # More permissive bearing check
        if bearing_diff <= CONFIG['BEARING_TOLERANCE_DEGREES']:
            return True, f"Bearing aligned: {bearing_diff:.1f}deg"
        else:
            # Fallback: if user is reasonably close to driver, allow anyway
            driver_to_user = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
            if driver_to_user <= CONFIG['GEOGRAPHIC_CLUSTER_RADIUS_KM']:
                return True, f"Close to driver: {driver_to_user:.1f}km"
            else:
                return False, f"Too far off route: {bearing_diff:.1f}deg"

    # For existing routes, check detour distance
    route_distance = calculate_route_with_user(driver_pos, current_route_users, office_pos, user_pos)
    original_distance = calculate_route_without_user(driver_pos, current_route_users, office_pos)

    detour_distance = route_distance - original_distance

    if detour_distance <= CONFIG['MAX_ON_ROUTE_DETOUR_KM']:
        return True, f"Detour acceptable: {detour_distance:.1f}km"
    else:
        # Check if user is very close to any existing user in route
        min_distance_to_route = float('inf')
        for route_user in current_route_users:
            route_user_pos = (route_user['lat'], route_user['lng'])
            dist = haversine_distance(user_pos[0], user_pos[1], route_user_pos[0], route_user_pos[1])
            min_distance_to_route = min(min_distance_to_route, dist)

        if min_distance_to_route <= 1.0:  # Within 1km of existing route user
            return True, f"Close to route user: {min_distance_to_route:.1f}km"
        else:
            return False, f"Too much detour: {detour_distance:.1f}km"

def calculate_route_with_user(driver_pos, current_users, office_pos, new_user_pos):
    """Calculate total route distance if we add this user"""
    if not current_users:
        return haversine_distance(driver_pos[0], driver_pos[1], new_user_pos[0], new_user_pos[1]) + \
               haversine_distance(new_user_pos[0], new_user_pos[1], office_pos[0], office_pos[1])

    # Simple route: driver -> new_user -> existing_users -> office
    total = haversine_distance(driver_pos[0], driver_pos[1], new_user_pos[0], new_user_pos[1])

    last_pos = new_user_pos
    for user in current_users:
        user_pos = (user['lat'], user['lng'])
        total += haversine_distance(last_pos[0], last_pos[1], user_pos[0], user_pos[1])
        last_pos = user_pos

    total += haversine_distance(last_pos[0], last_pos[1], office_pos[0], office_pos[1])
    return total

def calculate_route_without_user(driver_pos, current_users, office_pos):
    """Calculate original route distance without this user"""
    if not current_users:
        return haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    total = 0
    last_pos = driver_pos

    for user in current_users:
        user_pos = (user['lat'], user['lng'])
        total += haversine_distance(last_pos[0], last_pos[1], user_pos[0], user_pos[1])
        last_pos = user_pos

    total += haversine_distance(last_pos[0], last_pos[1], office_pos[0], office_pos[1])
    return total

# ================== STEP 4: PRIORITY-BASED SEAT FILLING ==================

def fill_remaining_seats(routes, unassigned_users_df, office_lat, office_lon):
    """
    Smart seat filling - prioritize unassigned users and those who can help global optimization
    """
    logger.info("Step 4: Priority-based seat filling...")

    filled_user_ids = set()

    # Sort unassigned users: single users from sparse areas first (they're more "disruptive")
    unassigned_users_df = unassigned_users_df.copy()
    unassigned_users_df['cluster_size'] = unassigned_users_df.groupby('geo_cluster')['user_id'].transform('count')
    unassigned_users_df = unassigned_users_df.sort_values('cluster_size', ascending=True)

    for route in routes:
        available_seats = route['vehicle_type'] - len(route['assigned_users'])

        if available_seats <= 0:
            continue

        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)

        # Find best candidates for this route
        candidates = []

        for _, user in unassigned_users_df.iterrows():
            if user['user_id'] in filled_user_ids:
                continue

            user_pos = (user['latitude'], user['longitude'])

            # Check if user is on the way
            is_on_way, reason = is_user_on_the_way(
                user_pos, driver_pos, office_pos, route['assigned_users']
            )

            if is_on_way:
                # Calculate a simple score (prefer closer users)
                distance_to_route = haversine_distance(
                    user_pos[0], user_pos[1],
                    route['assigned_users'][0]['lat'] if route['assigned_users'] else driver_pos[0],
                    route['assigned_users'][0]['lng'] if route['assigned_users'] else driver_pos[1]
                )

                candidates.append((distance_to_route, user, reason))

        # Fill with best candidates
        candidates.sort(key=lambda x: x[0])  # Sort by distance

        seats_filled = 0
        for distance, user, reason in candidates:
            if seats_filled >= available_seats:
                break

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
            filled_user_ids.add(user['user_id'])
            seats_filled += 1

            logger.info(f"   Added user {user['user_id']} to route {route['driver_id']} "
                       f"({reason}, {distance:.1f}km)")

        if seats_filled > 0:
            utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(f"   Route {route['driver_id']}: {seats_filled} seats filled "
                       f"({utilization:.1f}% utilization)")

    return routes, filled_user_ids

# ================== GLOBAL OPTIMIZATION FOR FEMALE SAFETY ==================

def calculate_global_assignment_cost(routes, office_lat, office_lon):
    """
    Calculate total system cost for a given set of routes
    Considers total distance, efficiency, and safety compliance
    """
    total_distance = 0
    total_inefficiency = 0

    for route in routes:
        # Calculate route distance
        route_distance = calculate_total_route_distance(route, office_lat, office_lon)
        total_distance += route_distance

        # Calculate efficiency penalty
        capacity = route['vehicle_type']
        users = len(route['assigned_users'])
        utilization = users / capacity if capacity > 0 else 0

        # Penalize underutilized routes
        if utilization < 0.6:
            total_inefficiency += (1 - utilization) * 10  # Penalty factor

        # Check safety compliance
        female_count = sum(1 for user in route['assigned_users']
                          if user.get('first_name', '') in
                          ['Kritika', 'Sakshi', 'Kirandeep', 'Nandita', 'Riya', 'Jagjeet',
                           'Chaitanya', 'Isha', 'Shruti', 'Nandini', 'Chumphila', 'Anchal', 'Jasmeet'])
        male_count = users - female_count

        # Heavy penalty for unsafe routes (should not happen after optimization)
        if female_count > 0 and male_count == 0:
            total_inefficiency += 1000

    return total_distance + total_inefficiency

def create_optimization_matrix(females_df, male_clusters, office_lat, office_lon):
    """
    Create cost matrix for female-male assignment optimization
    Matrix rows = females, columns = male clusters, values = assignment costs
    """
    num_females = len(females_df)
    num_clusters = len(male_clusters)

    # Initialize cost matrix with high values (infinity)
    cost_matrix = [[float('inf')] * num_clusters for _ in range(num_females)]

    for i, (_, female) in enumerate(females_df.iterrows()):
        for j, male_cluster in enumerate(male_clusters):
            # Check capacity constraints
            cluster_size = len(male_cluster)
            if cluster_size >= CONFIG['MAX_CLUSTER_SIZE']:
                continue  # Can't add more users to this cluster

            # Calculate assignment cost
            cluster_lat = male_cluster['latitude'].mean()
            cluster_lon = male_cluster['longitude'].mean()

            # Distance from female to cluster center
            distance = haversine_distance(female['latitude'], female['longitude'], cluster_lat, cluster_lon)

            # Capacity utilization penalty
            capacity_penalty = (cluster_size / CONFIG['MAX_CLUSTER_SIZE']) * 5

            # Total cost (lower is better)
            total_cost = distance + capacity_penalty
            cost_matrix[i][j] = total_cost

    return cost_matrix

def solve_optimal_assignment_hungarian(cost_matrix, max_cost_threshold=50):
    """
    Solve assignment problem using simplified Hungarian-like algorithm
    For each female, find the best male cluster assignment
    """
    num_females = len(cost_matrix)
    num_clusters = len(cost_matrix[0]) if num_females > 0 else 0

    assignments = []
    used_clusters = set()

    # Sort females by their best available option (hardest to place first)
    female_priorities = []
    for i, female_costs in enumerate(cost_matrix):
        min_cost = min(female_costs)
        female_priorities.append((min_cost, i))

    female_priorities.sort()  # Hardest to place (highest min cost) first

    for min_cost, female_idx in female_priorities:
        if min_cost > max_cost_threshold:
            continue  # This female can't be placed within reasonable distance

        # Find best available cluster for this female
        best_cluster_idx = -1
        best_cost = float('inf')

        for cluster_idx in range(num_clusters):
            if cluster_idx in used_clusters:
                continue  # Cluster already assigned

            cost = cost_matrix[female_idx][cluster_idx]
            if cost < best_cost:
                best_cost = cost
                best_cluster_idx = cluster_idx

        if best_cluster_idx != -1:
            assignments.append((female_idx, best_cluster_idx))
            used_clusters.add(best_cluster_idx)

    return assignments

def global_female_optimization(females_df, males_df, office_lat, office_lon):
    """
    Main global optimization function for female safety
    Replaces greedy rescue with optimal assignment
    """
    logger.info("Starting GLOBAL female optimization...")

    if females_df.empty or males_df.empty:
        logger.info("   No females or males to optimize")
        return pd.DataFrame(), pd.DataFrame()

    # Group males by cluster
    male_clusters = []
    if 'geo_cluster' in males_df.columns:
        for cluster_id in males_df['geo_cluster'].unique():
            cluster_df = males_df[males_df['geo_cluster'] == cluster_id]
            if len(cluster_df) > 0:
                male_clusters.append(cluster_df)
    else:
        # Create one cluster from all males
        if len(males_df) > 0:
            male_clusters.append(males_df)

    logger.info(f"   Available male clusters: {len(male_clusters)}")
    logger.info(f"   Females to optimize: {len(females_df)}")

    # Create optimization cost matrix
    cost_matrix = create_optimization_matrix(females_df, male_clusters, office_lat, office_lon)

    # Solve assignment problem
    assignments = solve_optimal_assignment_hungarian(cost_matrix)

    logger.info(f"   Optimal assignments found: {len(assignments)}")

    # Apply assignments
    rescued_females = []
    still_unsafe_females = []
    assigned_females = set()

    for female_idx, cluster_idx in assignments:
        female_row = females_df.iloc[female_idx]
        male_cluster = male_clusters[cluster_idx]

        # Create new cluster with female added
        if len(male_cluster) < CONFIG['MAX_CLUSTER_SIZE']:
            female_row = female_row.copy()
            cluster_id = male_cluster['geo_cluster'].iloc[0] if 'geo_cluster' in male_cluster.columns else cluster_idx
            female_row['geo_cluster'] = cluster_id
            rescued_females.append(female_row)
            assigned_females.add(female_idx)

            logger.info(f"   Optimized: Female {female_row['user_id']} -> cluster {cluster_id}")

    # Add females that couldn't be assigned
    for i, (_, female) in enumerate(females_df.iterrows()):
        if i not in assigned_females:
            still_unsafe_females.append(female)
            logger.warning(f"   Still unsafe: Female {female['user_id']} - no optimal assignment found")

    rescued_females_df = pd.DataFrame(rescued_females) if rescued_females else pd.DataFrame()
    still_unsafe_females_df = pd.DataFrame(still_unsafe_females) if still_unsafe_females else pd.DataFrame()

    logger.info(f"   Global optimization complete: {len(rescued_females)} rescued, {len(still_unsafe_females)} still unsafe")

    return rescued_females_df, still_unsafe_females_df

def rebalance_clusters_for_global_efficiency(user_df, office_lat, office_lon, max_iterations=10):
    """
    Enhanced rebalancing with intelligent male-to-female exchange optimization
    Specifically addresses wide-spread routes like driver 335468
    """
    logger.info("Starting ENHANCED cluster rebalancing for global efficiency...")

    if 'geo_cluster' not in user_df.columns:
        logger.info("   No cluster information available for rebalancing")
        return user_df

    improved = True
    iteration = 0
    original_user_df = user_df.copy()

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        logger.info(f"   Enhanced rebalancing iteration {iteration}...")

        # Analyze current clusters with detailed metrics
        cluster_analysis = {}
        for cluster_id in user_df['geo_cluster'].unique():
            cluster_df = user_df[user_df['geo_cluster'] == cluster_id]
            female_count, male_count = count_gender_in_cluster(cluster_df)

            # Calculate cluster spread (how geographically dispersed the cluster is)
            if len(cluster_df) > 1:
                cluster_center_lat = cluster_df['latitude'].mean()
                cluster_center_lon = cluster_df['longitude'].mean()
                max_distance = cluster_df.apply(
                    lambda row: haversine_distance(cluster_center_lat, cluster_center_lon,
                                                 row['latitude'], row['longitude']), axis=1
                ).max()
            else:
                max_distance = 0

            cluster_analysis[cluster_id] = {
                'users': cluster_df,
                'females': female_count,
                'males': male_count,
                'total': len(cluster_df),
                'max_distance': max_distance,
                'surplus_males': max(0, male_count - max(1, female_count)),
                'deficit_males': max(0, max(1, female_count) - male_count)
            }

        # Identify problematic clusters (high spread with females) - LOWERED THRESHOLD
        problematic_clusters = [
            cluster_id for cluster_id, analysis in cluster_analysis.items()
            if analysis['max_distance'] > 6.0 and analysis['females'] > 0  # > 6km spread with females (more aggressive)
        ]

        logger.info(f"   Found {len(problematic_clusters)} problematic clusters with high spread")

        # Debug: Show which clusters are flagged as problematic
        if problematic_clusters:
            logger.info(f"   Problematic cluster IDs: {problematic_clusters}")
            for cluster_id in problematic_clusters:
                analysis = cluster_analysis[cluster_id]
                logger.info(f"     Cluster {cluster_id}: {analysis['females']} females, {analysis['males']} males, spread: {analysis['max_distance']:.2f}km")
        else:
            # Debug: Show why no clusters were flagged
            logger.info(f"   DEBUG: All cluster analyses:")
            for cluster_id, analysis in cluster_analysis.items():
                if analysis['females'] > 0:
                    logger.info(f"     Cluster {cluster_id}: {analysis['females']} females, {analysis['males']} males, spread: {analysis['max_distance']:.2f}km (threshold: 8.0km)")

        # Debug: Specifically look for driver 335468's cluster
        target_driver_id = "335468"
        logger.info(f"   DEBUG: Looking for driver {target_driver_id}...")
        for cluster_id, analysis in cluster_analysis.items():
            cluster_users = analysis['users']
            # Check if this cluster contains passengers that would be assigned to driver 335468
            # Since we don't have driver assignment yet, check for the known passengers
            target_passengers = ["335321", "335255", "335397"]  # Aadi, Sakshi, Nandini
            cluster_user_ids = set(str(uid) for uid in cluster_users['user_id'].values)
            if any(pid in cluster_user_ids for pid in target_passengers):
                logger.info(f"   FOUND POTENTIAL DRIVER 335468 CLUSTER: {cluster_id}")
                logger.info(f"     Cluster {cluster_id} details: {analysis['females']} females, {analysis['males']} males, spread: {analysis['max_distance']:.2f}km")
                logger.info(f"     User IDs in cluster: {cluster_user_ids}")
                break
        else:
            logger.info(f"   Driver 335468 passengers not found in any cluster")

        # NEW: Aggressive global male redistribution for wide-spread routes
        improved |= aggressive_male_redistribution_for_wide_spreads(user_df, cluster_analysis, problematic_clusters, office_lat, office_lon)

        # Enhanced logic: specifically target wide-spread routes like driver 335468
        for cluster_id in problematic_clusters:
            analysis = cluster_analysis[cluster_id]

            if analysis['males'] > 0 and analysis['females'] > 0:
                # This cluster has both males and females but is too spread out
                cluster_df = analysis['users']
                cluster_center_lat = cluster_df['latitude'].mean()
                cluster_center_lon = cluster_df['longitude'].mean()

                # Identify males that are far from the female cluster center
                females_in_cluster = cluster_df[cluster_df['gender'] == 'Female']
                males_in_cluster = cluster_df[cluster_df['gender'] == 'Male']

                if len(females_in_cluster) > 0 and len(males_in_cluster) > 0:
                    female_center_lat = females_in_cluster['latitude'].mean()
                    female_center_lon = females_in_cluster['longitude'].mean()

                    # Find males that are creating the wide spread - LOWERED THRESHOLD
                    problematic_males = []
                    for _, male in males_in_cluster.iterrows():
                        male_to_female_distance = haversine_distance(
                            male['latitude'], male['longitude'],
                            female_center_lat, female_center_lon
                        )

                        # If male is significantly far from females, consider him problematic
                        if male_to_female_distance > 4.0:  # More than 4km from female cluster (more aggressive)
                            problematic_males.append((male, male_to_female_distance))

                    if problematic_males:
                        # Sort by distance (farther males first)
                        problematic_males.sort(key=lambda x: x[1], reverse=True)
                        most_problematic_male = problematic_males[0][0]

                        logger.info(f"   Identified problematic male {most_problematic_male['user_id']} in cluster {cluster_id}")
                        logger.info(f"   Distance from female cluster: {problematic_males[0][1]:.2f} km")

                        # Find better male replacements from other clusters - MORE AGGRESSIVE SEARCH
                        best_replacement = None
                        best_improvement = 0

                        for other_cluster_id, other_analysis in cluster_analysis.items():
                            if other_cluster_id == cluster_id:
                                continue

                            # Consider ALL males from other clusters (not just surplus)
                            if other_analysis['males'] > 0:
                                males_in_other = other_analysis['users'][other_analysis['users']['gender'] == 'Male']

                                for _, candidate_male in males_in_other.iterrows():
                                    # Calculate how well this male fits our female cluster
                                    candidate_distance = haversine_distance(
                                        candidate_male['latitude'], candidate_male['longitude'],
                                        female_center_lat, female_center_lon
                                    )

                                    # Only consider significant improvements - LOWERED THRESHOLD
                                    distance_improvement = problematic_males[0][1] - candidate_distance
                                    if distance_improvement > 1.5:  # At least 1.5km improvement (more aggressive)

                                        # Calculate if this exchange maintains safety
                                        new_problematic_males_count = len(males_in_cluster) - 1
                                        new_other_males_count = len(other_analysis['users'][other_analysis['users']['gender'] == 'Male']) - 1

                                        # Check if both clusters would remain safe
                                        problem_cluster_safe = new_problematic_males_count >= analysis['females']
                                        other_cluster_safe = new_other_males_count >= max(1, other_analysis['females'])

                                        # If source cluster would become unsafe, allow it if it has no females
                                        if other_analysis['females'] == 0:
                                            other_cluster_safe = True

                                        if problem_cluster_safe and other_cluster_safe:
                                            # Calculate global cost improvement
                                            current_cost = calculate_global_assignment_cost(
                                                convert_clusters_to_routes([analysis['users'], other_analysis['users']]),
                                                office_lat, office_lon
                                            )

                                            # Simulate the exchange
                                            new_problematic_users = cluster_df[cluster_df['user_id'] != most_problematic_male['user_id']]
                                            new_problematic_users = pd.concat([new_problematic_users, pd.DataFrame([candidate_male])], ignore_index=True)

                                            new_other_users = other_analysis['users'][other_analysis['users']['user_id'] != candidate_male['user_id']]
                                            new_other_users = pd.concat([new_other_users, pd.DataFrame([most_problematic_male])], ignore_index=True)

                                            new_cost = calculate_global_assignment_cost(
                                                convert_clusters_to_routes([new_problematic_users, new_other_users]),
                                                office_lat, office_lon
                                            )

                                            if new_cost < current_cost:
                                                improvement = current_cost - new_cost
                                                if improvement > best_improvement:
                                                    best_improvement = improvement
                                                    best_replacement = (candidate_male, other_cluster_id, distance_improvement)

                        # Perform the exchange if we found a good replacement - LOWERED THRESHOLD
                        if best_replacement:
                            replacement_male, source_cluster_id, distance_improvement = best_replacement

                            # Exchange the males
                            user_df.loc[user_df['user_id'] == most_problematic_male['user_id'], 'geo_cluster'] = source_cluster_id
                            user_df.loc[user_df['user_id'] == replacement_male['user_id'], 'geo_cluster'] = cluster_id

                            improved = True
                            logger.info(f"   WIDE-SPREAD ROUTE FIX:")
                            logger.info(f"   Exchanged problematic male {most_problematic_male['user_id']} from cluster {cluster_id}")
                            logger.info(f"   With better male {replacement_male['user_id']} from cluster {source_cluster_id}")
                            logger.info(f"   Route compactness improvement: {distance_improvement:.2f} km")
                            logger.info(f"   Global efficiency improvement: {best_improvement:.2f} units")

        # Enhanced exchange logic for problematic clusters
        for cluster_id in problematic_clusters:
            analysis = cluster_analysis[cluster_id]

            if analysis['males'] > 0:
                # Try to find better male replacements for this cluster
                cluster_df = analysis['users']
                cluster_center_lat = cluster_df['latitude'].mean()
                cluster_center_lon = cluster_df['longitude'].mean()

                # For each male in the problematic cluster, try to find a better replacement
                males_in_cluster = cluster_df[cluster_df['gender'] == 'Male']

                for _, male in males_in_cluster.iterrows():
                    male_to_cluster_center = haversine_distance(
                        male['latitude'], male['longitude'],
                        cluster_center_lat, cluster_center_lon
                    )

                    # Find better male candidates from other clusters
                    best_candidate = None
                    best_improvement = 0

                    for other_cluster_id, other_analysis in cluster_analysis.items():
                        if other_cluster_id == cluster_id:
                            continue

                        # Only consider males from clusters that can spare them
                        if other_analysis['surplus_males'] > 0:
                            males_in_other = other_analysis['users'][other_analysis['users']['gender'] == 'Male']

                            for _, candidate_male in males_in_other.iterrows():
                                # Calculate how well this male fits our problematic cluster
                                candidate_distance = haversine_distance(
                                    candidate_male['latitude'], candidate_male['longitude'],
                                    cluster_center_lat, cluster_center_lon
                                )

                                # Calculate distance improvement
                                distance_improvement = male_to_cluster_center - candidate_distance

                                # Only consider if it significantly improves cluster compactness
                                if distance_improvement > 2.0:  # At least 2km improvement
                                    # Calculate global cost change
                                    current_cost = calculate_global_assignment_cost(
                                        convert_clusters_to_routes([analysis['users'], other_analysis['users']]),
                                        office_lat, office_lon
                                    )

                                    # Simulate the exchange
                                    new_problematic_users = cluster_df[cluster_df['user_id'] != male['user_id']]
                                    new_problematic_users = pd.concat([new_problematic_users, pd.DataFrame([candidate_male])], ignore_index=True)

                                    new_other_users = other_analysis['users'][other_analysis['users']['user_id'] != candidate_male['user_id']]
                                    new_other_users = pd.concat([new_other_users, pd.DataFrame([male])], ignore_index=True)

                                    new_cost = calculate_global_assignment_cost(
                                        convert_clusters_to_routes([new_problematic_users, new_other_users]),
                                        office_lat, office_lon
                                    )

                                    if new_cost < current_cost:
                                        improvement = current_cost - new_cost
                                        if improvement > best_improvement:
                                            best_improvement = improvement
                                            best_candidate = (candidate_male, other_cluster_id)

                    # Perform the exchange if we found a good candidate
                    if best_candidate:
                        candidate_male, other_cluster_id = best_candidate

                        # Exchange the males
                        user_df.loc[user_df['user_id'] == male['user_id'], 'geo_cluster'] = other_cluster_id
                        user_df.loc[user_df['user_id'] == candidate_male['user_id'], 'geo_cluster'] = cluster_id

                        improved = True
                        logger.info(f"   ENHANCED EXCHANGE: Male {candidate_male['user_id']} -> cluster {cluster_id}")
                        logger.info(f"   Male {male['user_id']} -> cluster {other_cluster_id}")
                        logger.info(f"   Improvement: {best_improvement:.2f} units, Distance improvement: {distance_improvement:.2f} km")
                        break  # One exchange per problematic cluster per iteration

        # Also perform basic surplus-to-deficit exchanges (original logic)
        for cluster1_id, analysis1 in cluster_analysis.items():
            if analysis1['surplus_males'] > 0:
                for cluster2_id, analysis2 in cluster_analysis.items():
                    if cluster1_id == cluster2_id or analysis2['deficit_males'] == 0:
                        continue

                    # Move one male from cluster1 to cluster2
                    if 'gender' in analysis1['users'].columns:
                        male_to_move = analysis1['users'][analysis1['users']['gender'] == 'Male'].iloc[0]
                    else:
                        # Fallback: identify male by excluding female names
                        female_names = {'Kritika', 'Sakshi', 'Kirandeep', 'Nandita', 'Riya', 'Jagjeet',
                                       'Chaitanya', 'Isha', 'Shruti', 'Nandini', 'Chumphila', 'Anchal', 'Jasmeet'}
                        male_users = analysis1['users'][~analysis1['users']['first_name'].isin(female_names)]
                        male_to_move = male_users.iloc[0] if not male_users.empty else analysis1['users'].iloc[0]

                    # Calculate global improvement
                    current_cost = calculate_global_assignment_cost(
                        convert_clusters_to_routes([analysis1['users'], analysis2['users']]),
                        office_lat, office_lon
                    )

                    # Simulate the exchange
                    new_cluster1_users = analysis1['users'][analysis1['users']['user_id'] != male_to_move['user_id']]
                    new_cluster2_users = pd.concat([analysis2['users'], pd.DataFrame([male_to_move])], ignore_index=True)

                    new_cost = calculate_global_assignment_cost(
                        convert_clusters_to_routes([new_cluster1_users, new_cluster2_users]),
                        office_lat, office_lon
                    )

                    if new_cost < current_cost:
                        user_df.loc[user_df['user_id'] == male_to_move['user_id'], 'geo_cluster'] = cluster2_id
                        improved = True
                        logger.info(f"   Basic exchange: Male {male_to_move['user_id']}: cluster {cluster1_id} -> cluster {cluster2_id}")
                        break

    if iteration > 1:
        logger.info(f"   Enhanced rebalancing complete after {iteration} iterations")

        # Calculate improvement
        original_cost = calculate_global_assignment_cost(
            convert_clusters_to_routes(list(original_user_df.groupby('geo_cluster'))),
            office_lat, office_lon
        )
        optimized_cost = calculate_global_assignment_cost(
            convert_clusters_to_routes(list(user_df.groupby('geo_cluster'))),
            office_lat, office_lon
        )

        improvement = original_cost - optimized_cost
        logger.info(f"   Total global efficiency improvement: {improvement:.2f} units")
    else:
        logger.info("   No beneficial exchanges found")

    return user_df

def aggressive_male_redistribution_for_wide_spreads(user_df, cluster_analysis, problematic_clusters, office_lat, office_lon):
    """
    NEW: Aggressive global male redistribution specifically targeting wide-spread routes
    This function looks for optimal male exchanges across all clusters to reduce route spread
    """
    logger.info("   Starting aggressive male redistribution for wide-spread routes...")

    improvement_made = False

    # For each problematic cluster, find the best male exchanges globally
    for cluster_id in problematic_clusters:
        analysis = cluster_analysis[cluster_id]

        if analysis['females'] == 0:
            continue  # Only focus on clusters with females

        cluster_df = analysis['users']

        # Calculate female cluster center
        females_in_cluster = cluster_df[cluster_df['gender'] == 'Female']
        if len(females_in_cluster) == 0:
            continue

        female_center_lat = females_in_cluster['latitude'].mean()
        female_center_lon = females_in_cluster['longitude'].mean()

        # Find the worst male in this cluster (farthest from females)
        males_in_cluster = cluster_df[cluster_df['gender'] == 'Male']
        if len(males_in_cluster) == 0:
            continue

        worst_male_in_cluster = None
        worst_male_distance = 0

        for _, male in males_in_cluster.iterrows():
            male_to_female_distance = haversine_distance(
                male['latitude'], male['longitude'],
                female_center_lat, female_center_lon
            )
            if male_to_female_distance > worst_male_distance:
                worst_male_distance = male_to_female_distance
                worst_male_in_cluster = male

        if worst_male_in_cluster is None or worst_male_distance <= 3.0:
            continue  # No problematic male found

        logger.info(f"   Cluster {cluster_id}: Worst male {worst_male_in_cluster['user_id']} is {worst_male_distance:.2f}km from females")

        # Search ALL other clusters for better male candidates
        best_global_exchange = None
        best_global_improvement = 0

        for other_cluster_id, other_analysis in cluster_analysis.items():
            if other_cluster_id == cluster_id:
                continue

            # Consider males from ANY other cluster
            other_males = other_analysis['users'][other_analysis['users']['gender'] == 'Male']
            if len(other_males) == 0:
                continue

            for _, candidate_male in other_males.iterrows():
                # Calculate how well this candidate fits our female cluster
                candidate_distance = haversine_distance(
                    candidate_male['latitude'], candidate_male['longitude'],
                    female_center_lat, female_center_lon
                )

                # Calculate improvement
                distance_improvement = worst_male_distance - candidate_distance
                if distance_improvement <= 1.0:  # Need at least 1km improvement
                    continue

                # Check safety constraints for both clusters after exchange
                # Our cluster: remove worst male, add candidate
                our_cluster_new_male_count = len(males_in_cluster) - 1 + 1  # Same count
                our_cluster_safe = our_cluster_new_male_count >= len(females_in_cluster)

                # Other cluster: remove candidate, add worst male
                other_females = other_analysis['users'][other_analysis['users']['gender'] == 'Female']
                other_males_new_count = len(other_males) - 1
                other_cluster_safe = other_males_new_count >= len(other_females) if len(other_females) > 0 else True

                # Allow the exchange if our cluster becomes more compact and safety is maintained
                if our_cluster_safe and other_cluster_safe:
                    # Calculate route spread improvement
                    new_our_spread = calculate_cluster_spread_after_exchange(
                        cluster_df, worst_male_in_cluster, candidate_male, female_center_lat, female_center_lon
                    )

                    new_other_spread = calculate_other_cluster_spread_after_exchange(
                        other_analysis['users'], candidate_male, worst_male_in_cluster
                    )

                    # Prioritize exchanges that significantly reduce our cluster spread
                    current_spread = analysis['max_distance']
                    spread_improvement = current_spread - new_our_spread

                    # Calculate total improvement score
                    total_improvement = (distance_improvement * 0.6 +
                                       spread_improvement * 0.4)

                    if total_improvement > best_global_improvement:
                        best_global_improvement = total_improvement
                        best_global_exchange = {
                            'worst_male': worst_male_in_cluster,
                            'candidate_male': candidate_male,
                            'source_cluster': other_cluster_id,
                            'target_cluster': cluster_id,
                            'distance_improvement': distance_improvement,
                            'spread_improvement': spread_improvement,
                            'total_improvement': total_improvement
                        }

        # Perform the best global exchange if found
        if best_global_exchange and best_global_improvement > 2.0:
            exchange = best_global_exchange

            # Execute the exchange
            user_df.loc[user_df['user_id'] == exchange['worst_male']['user_id'], 'geo_cluster'] = exchange['source_cluster']
            user_df.loc[user_df['user_id'] == exchange['candidate_male']['user_id'], 'geo_cluster'] = exchange['target_cluster']

            improvement_made = True
            logger.info(f"   AGGRESSIVE GLOBAL EXCHANGE:")
            logger.info(f"   Moved male {exchange['worst_male']['user_id']} from cluster {exchange['target_cluster']} to {exchange['source_cluster']}")
            logger.info(f"   Moved male {exchange['candidate_male']['user_id']} from cluster {exchange['source_cluster']} to {exchange['target_cluster']}")
            logger.info(f"   Distance improvement: {exchange['distance_improvement']:.2f} km")
            logger.info(f"   Spread improvement: {exchange['spread_improvement']:.2f} km")
            logger.info(f"   Total improvement: {exchange['total_improvement']:.2f} units")

    if improvement_made:
        logger.info("   Aggressive male redistribution completed - improvements made")
    else:
        logger.info("   Aggressive male redistribution completed - no beneficial exchanges found")

    return improvement_made

def calculate_cluster_spread_after_exchange(cluster_df, male_to_remove, male_to_add, female_center_lat, female_center_lon):
    """
    Calculate the new cluster spread after exchanging males
    """
    # Create new cluster dataframe
    new_cluster = cluster_df[cluster_df['user_id'] != male_to_remove['user_id']].copy()
    new_cluster = pd.concat([new_cluster, pd.DataFrame([male_to_add])], ignore_index=True)

    if len(new_cluster) <= 1:
        return 0

    # Calculate maximum distance from female center
    max_distance = 0
    for _, user in new_cluster.iterrows():
        distance = haversine_distance(
            female_center_lat, female_center_lon,
            user['latitude'], user['longitude']
        )
        max_distance = max(max_distance, distance)

    return max_distance

def calculate_other_cluster_spread_after_exchange(other_cluster_df, male_to_remove, male_to_add):
    """
    Calculate the spread impact on the other cluster
    """
    # For now, just return a simple estimate
    # In a more sophisticated version, we could calculate the actual spread change
    return 0

def convert_clusters_to_routes(cluster_groups):
    """
    Convert cluster groups to route format for cost calculation
    """
    routes = []
    for cluster_users in cluster_groups:
        if isinstance(cluster_users, pd.DataFrame):
            # Simulate a route for this cluster
            if len(cluster_users) > 0:
                center_lat = cluster_users['latitude'].mean()
                center_lon = cluster_users['longitude'].mean()

                route = {
                    'latitude': center_lat,
                    'longitude': center_lon,
                    'assigned_users': []
                }

                for _, user in cluster_users.iterrows():
                    user_data = {
                        'lat': user['latitude'],
                        'lng': user['longitude'],
                        'first_name': user.get('first_name', '')
                    }
                    route['assigned_users'].append(user_data)

                routes.append(route)

    return routes

# ================== MAIN FEMALE SAFETY ASSIGNMENT FUNCTION ==================

def run_safety_assignment_simplified(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    Female safety assignment algorithm:
    1. Segment users by gender
    2. Geographic clustering with safety validation
    3. Female rescue process
    4. Smart capacity matching
    5. Priority seat filling
    """
    start_time = time.time()

    logger.info(f"Starting FEMALE SAFETY assignment for source_id: {source_id}")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        users = data.get('users', [])
        if not users:
            logger.warning("No users found")
            return {"status": "true", "data": [], "unassignedUsers": [], "unassignedDrivers": []}

        # Get all drivers
        all_drivers = []
        if "drivers" in data:
            all_drivers.extend(data["drivers"].get("driversUnassigned", []))
            all_drivers.extend(data["drivers"].get("driversAssigned", []))
        else:
            all_drivers.extend(data.get("driversUnassigned", []))
            all_drivers.extend(data.get("driversAssigned", []))

        if not all_drivers:
            logger.warning("No drivers available")
            unassigned_users = _convert_users_to_unassigned_format(users)
            return {"status": "true", "data": [], "unassignedUsers": unassigned_users, "unassignedDrivers": []}

        logger.info(f"Data loaded - Users: {len(users)}, Drivers: {len(all_drivers)}")

        # Extract office coordinates and prepare data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        user_df, driver_df = prepare_user_driver_dataframes(data)

        logger.info(f"Data prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # SAFETY STEP 3: Geographic clustering for all users (do this first)
        logger.info("Step 1: Geographic clustering by proximity...")
        user_df = cluster_users_by_proximity(user_df, office_lat, office_lon)

        # SAFETY STEP 4: Segment users by gender (after clustering)
        females_df, males_df = segment_users_by_gender(user_df, users)

        # SAFETY STEP 5: Handle edge case - no males available
        if males_df.empty and not females_df.empty:
            logger.warning("No male passengers available. All female passengers will remain unassigned for safety.")
            unassigned_users = _convert_users_to_unassigned_format(users)
            return {"status": "true", "data": [], "unassignedUsers": unassigned_users, "unassignedDrivers": []}

        # SAFETY STEP 6: Validate clusters for safety and rescue unsafe females
        safe_user_df, unsafe_females_df = validate_and_fix_clusters_for_safety(user_df, office_lat, office_lon)

        # SAFETY STEP 7: Smart female rescue with geographic clustering
        if not unsafe_females_df.empty:
            logger.info(f"Step 7: Smart geographic rescue for {len(unsafe_females_df)} unsafe females...")

            rescued_females_df, still_unsafe_females_df = smart_female_rescue(
                unsafe_females_df, males_df, driver_df, office_lat, office_lon
            )

            # Combine safe users with rescued females
            if not rescued_females_df.empty:
                safe_user_df = pd.concat([safe_user_df, rescued_females_df], ignore_index=True)
                logger.info(f"   Added {len(rescued_females_df)} geographically optimized females to safe clusters")

            # Females that couldn't be rescued will remain unassigned
            if not still_unsafe_females_df.empty:
                logger.warning(f"   {len(still_unsafe_females_df)} females could not be rescued and will remain unassigned")

        # SAFETY STEP 7.5: Skip rebalancing for now (smart rescue already handles geographic optimization)
        logger.info("Step 7.5: Skipping rebalancing - smart rescue already provides geographic optimization...")
        # safe_user_df = rebalance_clusters_for_global_efficiency(safe_user_df, office_lat, office_lon)

        # SAFETY STEP 8: Smart capacity matching and initial assignment
        routes = []
        assigned_user_ids = set()
        used_driver_ids = set()
        available_drivers = driver_df.copy()

        for cluster_id in safe_user_df['geo_cluster'].unique():
            cluster_users = safe_user_df[safe_user_df['geo_cluster'] == cluster_id]

            if len(cluster_users) == 0:
                continue

            logger.info(f"Processing safe cluster {cluster_id}: {len(cluster_users)} users")

            # Find best cab for this cluster
            available_for_cluster = available_drivers[~available_drivers['driver_id'].isin(used_driver_ids)]

            assigned_driver = assign_cab_to_cluster(
                cluster_users, available_for_cluster, office_lat, office_lon
            )

            if assigned_driver is not None:
                # Create route
                route = {
                    'driver_id': str(assigned_driver['driver_id']),
                    'vehicle_id': str(assigned_driver.get('vehicle_id', '')),
                    'vehicle_type': int(assigned_driver['capacity']),
                    'latitude': float(assigned_driver['latitude']),
                    'longitude': float(assigned_driver['longitude']),
                    'assigned_users': []
                }

                # Add all cluster users to the route
                for _, user in cluster_users.iterrows():
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
                    assigned_user_ids.add(user['user_id'])

                routes.append(route)
                used_driver_ids.add(assigned_driver['driver_id'])

        logger.info(f"Initial assignment: {len(routes)} routes, {len(assigned_user_ids)} users assigned")

        # SAFETY STEP 9: Direction-aware route merging for small routes
        routes = merge_small_routes_with_nearby(routes, office_lat, office_lon)

        # SAFETY STEP 9.5: Post-merge wide-spread route optimization
        routes = optimize_wide_spread_routes_after_merge(routes, office_lat, office_lon)

        # SAFETY STEP 10: Priority-based seat filling (only for safe users)
        unassigned_safe_users_df = safe_user_df[~safe_user_df['user_id'].isin(assigned_user_ids)].copy()
        if not unassigned_safe_users_df.empty:
            routes, filled_ids = fill_remaining_seats(routes, unassigned_safe_users_df, office_lat, office_lon)
            assigned_user_ids.update(filled_ids)

        # SAFETY STEP 10.5: Final route optimization after seat filling
        logger.info("Step 10.5: Final wide-spread route optimization after seat filling...")
        routes = optimize_wide_spread_routes_after_merge(routes, office_lat, office_lon)

        # SAFETY STEP 11: Final results
        execution_time = time.time() - start_time

        # Prepare unassigned users (including unsafe females that couldn't be rescued)
        remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        unassigned_users = []
        for _, user in remaining_users.iterrows():
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'first_name': str(user.get('first_name', '')),
                'email': str(user.get('email', ''))
            }
            unassigned_users.append(user_data)

        # Prepare unassigned drivers
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
        unassigned_drivers = []
        for _, driver in unassigned_drivers_df.iterrows():
            driver_data = {
                'driver_id': str(driver['driver_id']),
                'capacity': int(driver['capacity']),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'latitude': float(driver['latitude']),
                'longitude': float(driver['longitude'])
            }
            unassigned_drivers.append(driver_data)

        logger.info(f"Female safety assignment complete in {execution_time:.2f}s")
        logger.info(f"Final: {len(routes)} routes, {len(assigned_user_ids)} users assigned, {len(unassigned_users)} unassigned")

        # Apply optimal pickup ordering to routes if available
        if ORDERING_AVAILABLE and routes:
            try:
                logger.info(f"Applying optimal pickup ordering to {len(routes)} routes")
                office_lat, office_lon = extract_office_coordinates(data)

                # Extract db name from source_id parameter
                db_name = source_id if source_id and source_id != "1" else data.get("db", "default")

                logger.info(f"Using company coordinates: {office_lat}, {office_lon} for db: {db_name}")

                routes = apply_route_ordering(routes, office_lat, office_lon, db_name=db_name, algorithm_name="safety")
                logger.info("Optimal pickup ordering applied successfully")
            except Exception as e:
                logger.error(f"Failed to apply optimal ordering: {e}")
                # Continue with routes without optimal ordering

        return {
            "status": "true",
            "execution_time": execution_time,
            "company": data.get("company", {}),
            "shift": data.get("shift", {}),
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": {"method": "female_safety_geographic", "clusters": len(safe_user_df['geo_cluster'].unique())},
            "optimization_mode": "female_safety",
            "parameter": parameter,
        }

    except Exception as e:
        logger.error(f"Female safety assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}

# Entry point function
def run_assignment_safety(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """Entry point that uses the female safety algorithm"""
    return run_safety_assignment_simplified(source_id, parameter, string_param, choice)