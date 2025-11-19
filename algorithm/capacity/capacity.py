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
from scipy.spatial import KDTree
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ordering system for optimal pickup sequences
try:
    from ordering import apply_route_ordering
    ORDERING_AVAILABLE = True
except ImportError:
    ORDERING_AVAILABLE = False
    logger.warning("Ordering system not available - routes will not have optimal pickup sequences")


# Load and validate configuration with capacity optimization settings
def load_and_validate_config():
    """Load configuration with capacity optimization settings"""
    # Find the config file relative to this script's location
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Go up from algorithm/capacity/ to project root
    config_path = os.path.join(script_dir, 'config.json')

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json from {config_path}, using defaults. Error: {e}")
        cfg = {}

    # Always use capacity mode
    current_mode = "capacity_optimization"

    # Get capacity optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("capacity_optimization", {})

    print(f"🎯 Using optimization mode: CAPACITY OPTIMIZATION WITH BLUEPRINT")
    
    # Validate and set configuration with mode-specific overrides
    config = {}

    # Distance configurations with mode overrides (more lenient for capacity filling)
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(mode_config.get("max_fill_distance_km", cfg.get("max_fill_distance_km", 8.0))))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(mode_config.get("merge_distance_km", cfg.get("merge_distance_km", 5.0))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.5)))
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 5.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 12.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 1.0)))

    # Blueprint-specific parameters - PROXIMITY-FIRST approach
    config['MICRO_CLUSTER_RADIUS_KM'] = 2.0  # Increased radius to keep nearby users together
    config['DIRECTIONAL_COHERENCE_THRESHOLD'] = 75  # More relaxed bearing spread for proximity priority
    config['PATH_INSERT_MAX_DETOUR_RATIO'] = 0.15  # 15% max detour for insertions (more lenient)
    config['PATH_INSERT_MAX_BEARING_DIFF'] = 45  # More relaxed bearing difference for insertions
    config['FINAL_MERGE_BEARING_THRESHOLD'] = 60  # More relaxed bearing threshold for final merges
    config['FINAL_MERGE_DISTANCE_THRESHOLD'] = 3.0  # Increased distance threshold for final merges
    config['MIN_UTILIZATION_IMPROVEMENT'] = 0.02  # 2% min improvement for merges (more aggressive)

    # Scoring weights for driver selection
    config['ALPHA_UTILIZATION'] = 1.0
    config['BETA_WASTE_PENALTY'] = 1.0
    config['GAMMA_DISTANCE_PENALTY'] = 0.5
    config['DELTA_BEARING_PENALTY'] = 0.3

    # Utilization thresholds (more aggressive for capacity)
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("min_util_threshold", 0.8))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.7))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(cfg.get("max_swap_iterations", 5)))
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(cfg.get("max_users_for_fallback", 5)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users", 3)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users", 10)))

    # Angle configurations with mode overrides (more lenient for capacity)
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(mode_config.get("max_bearing_difference", cfg.get("max_bearing_difference", 45)))))
    config['MAX_TURNING_ANGLE'] = max(0, min(180, float(mode_config.get("max_allowed_turning_score", cfg.get("max_allowed_turning_score", 60)))))

    # Cost penalties with mode overrides (prioritize capacity over route quality)
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(mode_config.get("utilization_penalty_per_seat", cfg.get("utilization_penalty_per_seat", 5.0))))

    # Office coordinates with environment variable fallbacks
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    # Validate coordinate bounds
    if not (-90 <= office_lat <= 90):
        print(f"Warning: Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        print(f"Warning: Invalid office longitude {office_lon}, using default")
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Capacity optimization parameters
    config['optimization_mode'] = "capacity_optimization"
    config['aggressive_merging'] = mode_config.get('aggressive_merging', True)
    config['capacity_weight'] = mode_config.get('capacity_weight', 5.0)
    config['direction_weight'] = mode_config.get('direction_weight', 1.0)

    # Clustering and optimization parameters with mode overrides
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', False)  # Less directional for capacity
    config['angular_sectors'] = cfg.get('angular_sectors', 6)  # Fewer sectors for larger groups
    config['max_users_per_initial_cluster'] = cfg.get('max_users_per_initial_cluster', 12)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 10)

    # Capacity optimization parameters
    config['zigzag_penalty_weight'] = mode_config.get('zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 0.5))  # Very low
    config['route_split_turning_threshold'] = cfg.get('route_split_turning_threshold', 80)  # Very high
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 2.0)  # Very lenient
    config['route_split_consistency_threshold'] = cfg.get('route_split_consistency_threshold', 0.3)  # Very low
    config['merge_tortuosity_improvement_required'] = cfg.get('merge_tortuosity_improvement_required', False)
    
    # Latitude conversion factor for distance normalization
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    print(f"   📊 Proximity-first micro-cluster radius: {config['MICRO_CLUSTER_RADIUS_KM']}km")
    print(f"   📊 Relaxed directional coherence: {config['DIRECTIONAL_COHERENCE_THRESHOLD']}°")
    print(f"   📊 Lenient max detour ratio: {config['PATH_INSERT_MAX_DETOUR_RATIO']*100}%")
    print(f"   📊 Capacity weight: {config['capacity_weight']}")
    print(f"   🎯 PRIORITY: Nearby users first, direction secondary")

    return config


# Import all other functions from assignment.py (keeping the same structure)
from algorithm.base.base import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, bearing_difference,
    calculate_bearing_vectorized, calculate_bearing, calculate_bearings_and_features,
    coords_to_km, dbscan_clustering_metric, kmeans_clustering_metric, estimate_clusters,
    create_geographic_clusters, sweep_clustering, polar_sector_clustering,
    create_capacity_subclusters, create_bearing_aware_subclusters, calculate_bearing_spread,
    normalize_bearing_difference, calculate_sequence_distance, calculate_sequence_turning_score_improved,
    apply_strict_direction_aware_2opt, calculate_optimal_sequence_improved,
    split_cluster_by_bearing_metric, apply_route_splitting, split_route_by_bearing_improved,
    create_sub_route_improved, calculate_users_center_improved, local_optimization,
    optimize_route_sequence_improved, calculate_route_cost_improved, calculate_route_turning_score_improved,
    calculate_direction_consistency_improved, try_user_swap_improved, calculate_route_center_improved,
    update_route_metrics_improved, calculate_tortuosity_ratio_improved, global_optimization,
    fix_single_user_routes_improved, calculate_average_bearing_improved, quality_controlled_route_filling,
    quality_preserving_route_merging, strict_merge_compatibility_improved, calculate_merge_quality_score,
    perform_quality_merge_improved, enhanced_route_splitting, intelligent_route_splitting_improved,
    split_by_bearing_clusters_improved, split_by_distance_clusters_improved, create_split_routes_improved,
    find_best_driver_for_group, outlier_detection_and_reassignment, try_reassign_outlier,
    handle_remaining_users_improved, find_best_driver_for_cluster_improved, final_pass_merge,
    calculate_combined_route_center, _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format,
    analyze_assignment_quality
)

# Load validated configuration - always capacity optimization
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


# =============================================================================
# BLUEPRINT IMPLEMENTATION - NEW FUNCTIONS
# =============================================================================

def micro_cluster_users(user_df, office_lat, office_lon, r_micro_km=1.2):
    """
    A. Micro-clustering: Create tight atomic clusters that should never be split
    PRIORITY: Nearby users first, regardless of direction
    """
    print(f"  🔬 Creating proximity-first micro-clusters with radius {r_micro_km}km...")
    
    if user_df.empty:
        return []
    
    # Prepare coordinates for DBSCAN
    coords = user_df[['latitude', 'longitude']].values
    
    # Convert to km using lat/lon scaling
    lat_to_km = 111.0
    lon_to_km = 111.0 * math.cos(math.radians(office_lat))
    
    coords_km = coords.copy()
    coords_km[:, 0] = coords[:, 0] * lat_to_km  # lat to km
    coords_km[:, 1] = coords[:, 1] * lon_to_km  # lon to km
    
    # Apply DBSCAN clustering with larger radius for proximity priority
    dbscan = DBSCAN(eps=r_micro_km, min_samples=1)  # min_samples=1 to avoid noise points
    cluster_labels = dbscan.fit_predict(coords_km)
    
    # Group users by cluster
    micro_clusters = []
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Noise points get individual clusters
            noise_indices = np.where(cluster_labels == cluster_id)[0]
            for idx in noise_indices:
                user_data = user_df.iloc[idx]
                micro_clusters.append([{
                    'user_id': user_data['user_id'],
                    'latitude': user_data['latitude'],
                    'longitude': user_data['longitude'],
                    'office_distance': user_data.get('office_distance', 0),
                    'first_name': user_data.get('first_name', ''),
                    'email': user_data.get('email', '')
                }])
        else:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_users = []
            for idx in cluster_indices:
                user_data = user_df.iloc[idx]
                cluster_users.append({
                    'user_id': user_data['user_id'],
                    'latitude': user_data['latitude'],
                    'longitude': user_data['longitude'],
                    'office_distance': user_data.get('office_distance', 0),
                    'first_name': user_data.get('first_name', ''),
                    'email': user_data.get('email', '')
                })
            micro_clusters.append(cluster_users)
    
    print(f"    📊 Created {len(micro_clusters)} proximity-first micro-clusters")
    for i, mc in enumerate(micro_clusters):
        print(f"      Micro-cluster {i}: {len(mc)} users")
    
    return micro_clusters


def aggregate_micro_clusters_into_groups(micro_clusters, drivers, office_lat, office_lon, config):
    """
    B. Cluster aggregation → capacity groups using PROXIMITY-FIRST Best-Fit Decreasing
    PRIORITY: Fill nearby clusters first, direction is secondary
    """
    print(f"  📦 Aggregating micro-clusters with PROXIMITY-FIRST approach...")
    
    if not micro_clusters or drivers.empty:
        return []
    
    # Sort micro-clusters by size (largest first) for efficient packing
    micro_sizes = sorted(micro_clusters, key=len, reverse=True)
    
    # Sort vehicles by capacity (largest first)
    vehicles = drivers.sort_values('capacity', ascending=False)
    
    groups = []  # Each group will be assigned to a vehicle
    
    for mc in micro_sizes:
        mc_size = len(mc)
        mc_center = calculate_micro_cluster_center(mc)
        mc_bearing = calculate_micro_cluster_bearing(mc, office_lat, office_lon)
        
        placed = False
        best_group = None
        best_proximity_score = float('inf')
        
        # PROXIMITY-FIRST: Try to place in existing groups based on proximity
        for group in groups:
            if mc_size <= group['remaining_capacity']:
                # Calculate proximity score (distance between cluster centers)
                group_center = calculate_group_center(group)
                proximity_distance = haversine_distance(
                    mc_center[0], mc_center[1], 
                    group_center[0], group_center[1]
                )
                
                # Relaxed directional check (only reject if extremely different)
                direction_ok = direction_compatible_relaxed(mc, group, 60, office_lat, office_lon)  # 60° threshold instead of 30°
                
                if direction_ok and proximity_distance < best_proximity_score:
                    best_proximity_score = proximity_distance
                    best_group = group
        
        if best_group is not None:
            # Add to the closest compatible group
            best_group['micro_clusters'].append(mc)
            best_group['total_users'].extend(mc)
            best_group['remaining_capacity'] -= mc_size
            best_group['bearings'].append(mc_bearing)
            placed = True
            print(f"    🎯 Added {mc_size} users to existing group (proximity: {best_proximity_score:.1f}km)")
        
        if not placed:
            # Create new group - find smallest vehicle that can fit this micro-cluster
            suitable_vehicle = None
            for _, vehicle in vehicles.iterrows():
                if vehicle['capacity'] >= mc_size:
                    suitable_vehicle = vehicle
                    break
            
            if suitable_vehicle is not None:
                new_group = {
                    'target_vehicle': suitable_vehicle,
                    'target_capacity': int(suitable_vehicle['capacity']),
                    'micro_clusters': [mc],
                    'total_users': list(mc),
                    'remaining_capacity': int(suitable_vehicle['capacity']) - mc_size,
                    'bearings': [mc_bearing]
                }
                groups.append(new_group)
                print(f"    🆕 Created new group for {mc_size} users")
            else:
                # No suitable vehicle - this shouldn't happen with proper data
                print(f"    ⚠️ Warning: No vehicle can fit micro-cluster of size {mc_size}")
    
    print(f"    📊 Created {len(groups)} proximity-optimized capacity groups")
    for i, group in enumerate(groups):
        utilization = len(group['total_users']) / group['target_capacity'] * 100
        print(f"      Group {i}: {len(group['total_users'])}/{group['target_capacity']} users ({utilization:.1f}%)")
    
    return groups


def calculate_micro_cluster_bearing(micro_cluster, office_lat, office_lon):
    """Calculate average bearing for a micro-cluster"""
    if not micro_cluster:
        return 0.0
    
    bearings = []
    for user in micro_cluster:
        bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
        bearings.append(bearing)
    
    # Calculate average bearing (handling circular nature)
    x_sum = sum(math.cos(math.radians(b)) for b in bearings)
    y_sum = sum(math.sin(math.radians(b)) for b in bearings)
    
    avg_bearing = math.degrees(math.atan2(y_sum, x_sum))
    if avg_bearing < 0:
        avg_bearing += 360
    
    return avg_bearing


def calculate_micro_cluster_center(micro_cluster):
    """Calculate geographic center of a micro-cluster"""
    if not micro_cluster:
        return (0, 0)
    
    avg_lat = sum(user['latitude'] for user in micro_cluster) / len(micro_cluster)
    avg_lon = sum(user['longitude'] for user in micro_cluster) / len(micro_cluster)
    return (avg_lat, avg_lon)


def calculate_group_center(group):
    """Calculate geographic center of a group"""
    if not group['total_users']:
        return (0, 0)
    
    avg_lat = sum(user['latitude'] for user in group['total_users']) / len(group['total_users'])
    avg_lon = sum(user['longitude'] for user in group['total_users']) / len(group['total_users'])
    return (avg_lat, avg_lon)


def direction_compatible_relaxed(micro_cluster, group, threshold_degrees, office_lat, office_lon):
    """
    C. Relaxed directional compatibility check - prioritizes proximity over strict direction
    """
    if not group['bearings']:
        return True  # Empty group is always compatible
    
    mc_bearing = calculate_micro_cluster_bearing(micro_cluster, office_lat, office_lon)
    
    # Check bearing spread if we add this micro-cluster
    all_bearings = group['bearings'] + [mc_bearing]
    
    # Calculate bearing spread
    max_diff = 0
    for i in range(len(all_bearings)):
        for j in range(i + 1, len(all_bearings)):
            diff = bearing_difference(all_bearings[i], all_bearings[j])
            max_diff = max(max_diff, diff)
    
    return max_diff <= threshold_degrees


def direction_compatible(micro_cluster, group, threshold_degrees, office_lat, office_lon):
    """
    C. Directional compatibility check (strict version for fallback)
    """
    return direction_compatible_relaxed(micro_cluster, group, threshold_degrees, office_lat, office_lon)


def score_driver_for_group(group, driver, office_lat, office_lon, config):
    """
    D. Driver selection scoring function
    """
    group_size = len(group['total_users'])
    vehicle_capacity = int(driver['capacity'])
    
    if group_size > vehicle_capacity:
        return -float('inf')  # Cannot fit
    
    # Scoring components
    utilization = group_size / vehicle_capacity
    utilization_score = 1000 * utilization
    
    waste_penalty = 200 * (vehicle_capacity - group_size)
    
    # Calculate route distance
    route_distance = calculate_group_route_distance(group['total_users'], driver, office_lat, office_lon)
    distance_penalty = route_distance * 1.0
    
    # Calculate bearing penalty
    group_bearing_spread = calculate_group_bearing_spread(group, office_lat, office_lon)
    bearing_penalty = max(0, group_bearing_spread - 20) * 5
    
    # Apply weights
    alpha = config['ALPHA_UTILIZATION']
    beta = config['BETA_WASTE_PENALTY']
    gamma = config['GAMMA_DISTANCE_PENALTY']
    delta = config['DELTA_BEARING_PENALTY']
    
    score = (alpha * utilization_score - 
             beta * waste_penalty - 
             gamma * distance_penalty - 
             delta * bearing_penalty)
    
    return score


def calculate_group_route_distance(users, driver, office_lat, office_lon):
    """Calculate total route distance for a group"""
    if not users:
        return 0
    
    # Simple approximation: driver to center of users + users to office
    center_lat = sum(u['latitude'] for u in users) / len(users)
    center_lon = sum(u['longitude'] for u in users) / len(users)
    
    driver_to_center = haversine_distance(driver['latitude'], driver['longitude'], center_lat, center_lon)
    center_to_office = haversine_distance(center_lat, center_lon, office_lat, office_lon)
    
    return driver_to_center + center_to_office


def calculate_group_bearing_spread(group, office_lat, office_lon):
    """Calculate bearing spread for a group"""
    if not group['bearings']:
        return 0
    
    max_diff = 0
    for i in range(len(group['bearings'])):
        for j in range(i + 1, len(group['bearings'])):
            diff = bearing_difference(group['bearings'][i], group['bearings'][j])
            max_diff = max(max_diff, diff)
    
    return max_diff


def path_insert_user_into_route(user, route, office_lat, office_lon, max_detour_ratio=0.07):
    """
    F. Path-aware insertion with strict detour limits
    """
    if route['vehicle_type'] <= len(route['assigned_users']):
        return False, route  # No capacity
    
    # Calculate current route distance
    current_distance = calculate_route_total_distance(route, office_lat, office_lon)
    
    # Calculate user bearing
    user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
    route_avg_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
    
    bearing_diff = bearing_difference(user_bearing, route_avg_bearing)
    
    # Check bearing compatibility first
    if bearing_diff > _config['PATH_INSERT_MAX_BEARING_DIFF']:
        return False, route
    
    # Try inserting user at different positions
    best_position = None
    best_distance = float('inf')
    
    current_users = route['assigned_users'].copy()
    
    for i in range(len(current_users) + 1):
        # Create test route with user inserted at position i
        test_users = current_users[:i] + [user] + current_users[i:]
        test_route = route.copy()
        test_route['assigned_users'] = test_users
        
        # Calculate new distance
        new_distance = calculate_route_total_distance(test_route, office_lat, office_lon)
        
        if new_distance < best_distance:
            best_distance = new_distance
            best_position = i
    
    # Check if detour is acceptable
    if current_distance > 0:
        detour_ratio = (best_distance - current_distance) / current_distance
        if detour_ratio <= max_detour_ratio:
            # Accept insertion
            new_route = route.copy()
            new_users = current_users[:best_position] + [user] + current_users[best_position:]
            new_route['assigned_users'] = new_users
            new_route = optimize_route_sequence_improved(new_route, office_lat, office_lon)
            return True, new_route
    
    return False, route


def calculate_route_total_distance(route, office_lat, office_lon):
    """Calculate total distance for a route"""
    if not route['assigned_users']:
        return 0
    
    total_distance = 0
    
    # Driver to first pickup
    first_user = route['assigned_users'][0]
    total_distance += haversine_distance(
        route['latitude'], route['longitude'],
        first_user['lat'], first_user['lng']
    )
    
    # Between pickups
    for i in range(len(route['assigned_users']) - 1):
        current_user = route['assigned_users'][i]
        next_user = route['assigned_users'][i + 1]
        total_distance += haversine_distance(
            current_user['lat'], current_user['lng'],
            next_user['lat'], next_user['lng']
        )
    
    # Last pickup to office
    last_user = route['assigned_users'][-1]
    total_distance += haversine_distance(
        last_user['lat'], last_user['lng'],
        office_lat, office_lon
    )
    
    return total_distance


def split_micro_cluster_by_bearing(micro_cluster, max_capacity):
    """
    E. Micro-cluster splitting by bearing (rare operation)
    """
    if len(micro_cluster) <= max_capacity:
        return [micro_cluster]
    
    print(f"    ⚠️ Splitting micro-cluster of size {len(micro_cluster)} (max capacity: {max_capacity})")
    
    # Calculate bearings for all users in micro-cluster
    user_bearings = []
    for user in micro_cluster:
        bearing = calculate_bearing(OFFICE_LAT, OFFICE_LON, user['latitude'], user['longitude'])
        user_bearings.append((bearing, user))
    
    # Sort by bearing
    user_bearings.sort(key=lambda x: x[0])
    
    # Split into chunks of max_capacity
    split_clusters = []
    for i in range(0, len(user_bearings), max_capacity):
        chunk = [user for _, user in user_bearings[i:i + max_capacity]]
        split_clusters.append(chunk)
    
    return split_clusters


# =============================================================================
# MAIN BLUEPRINT ASSIGNMENT FUNCTION
# =============================================================================

def assign_drivers_blueprint_approach(user_df, driver_df, office_lat, office_lon):
    """
    Main assignment function following the exact blueprint ordering
    """
    print("🚗 Step 3: BLUEPRINT capacity assignment with atomic micro-clusters...")
    
    routes = []
    used_driver_ids = set()
    
    # STEP 1: Micro-cluster users into tight atomic groups
    micro_clusters = micro_cluster_users(user_df, office_lat, office_lon, 
                                       _config['MICRO_CLUSTER_RADIUS_KM'])
    
    if not micro_clusters:
        print("  ⚠️ No micro-clusters created")
        return routes, set()
    
    # STEP 2: Split oversized micro-clusters if needed
    max_vehicle_capacity = driver_df['capacity'].max() if not driver_df.empty else 10
    final_micro_clusters = []
    
    for mc in micro_clusters:
        if len(mc) > max_vehicle_capacity:
            split_clusters = split_micro_cluster_by_bearing(mc, max_vehicle_capacity)
            final_micro_clusters.extend(split_clusters)
        else:
            final_micro_clusters.append(mc)
    
    print(f"  📊 After splitting: {len(final_micro_clusters)} micro-clusters")
    
    # STEP 3: Aggregate micro-clusters into capacity groups using bin-packing
    available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]
    capacity_groups = aggregate_micro_clusters_into_groups(
        final_micro_clusters, available_drivers, office_lat, office_lon, _config)
    
    # STEP 4: Match groups to drivers with capacity-aware scoring
    assigned_user_ids = set()
    
    for group in capacity_groups:
        if available_drivers.empty:
            break
        
        best_driver = None
        best_score = -float('inf')
        
        for _, driver in available_drivers.iterrows():
            if driver['driver_id'] in used_driver_ids:
                continue
            
            score = score_driver_for_group(group, driver, office_lat, office_lon, _config)
            
            if score > best_score:
                best_score = score
                best_driver = driver
        
        if best_driver is not None:
            # Create route
            route = create_route_from_group(group, best_driver, office_lat, office_lon)
            
            if route:
                routes.append(route)
                used_driver_ids.add(best_driver['driver_id'])
                
                for user in route['assigned_users']:
                    assigned_user_ids.add(user['user_id'])
                
                utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
                print(f"  ✅ Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization:.1f}%) - Atomic grouping")
        
        # Update available drivers
        available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]
    
    # STEP 5: Path-aware fill pass for leftover users
    print("  🛣️ Path-aware fill pass...")
    
    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    
    for _, user in remaining_users.iterrows():
        user_dict = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'latitude': float(user['latitude']),
            'longitude': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', ''))
        }
        
        inserted = False
        for route in routes:
            success, new_route = path_insert_user_into_route(
                user_dict, route, office_lat, office_lon, 
                _config['PATH_INSERT_MAX_DETOUR_RATIO'])
            
            if success:
                # Update the route in routes list
                route.clear()
                route.update(new_route)
                assigned_user_ids.add(user['user_id'])
                inserted = True
                print(f"    🎯 Inserted user {user['user_id']} into route {route['driver_id']}")
                break
        
        if not inserted:
            print(f"    ⏭️ Could not insert user {user['user_id']} - detour/bearing constraints")
    
    print(f"  ✅ Blueprint assignment complete: {len(routes)} routes")
    
    return routes, assigned_user_ids


def create_route_from_group(group, driver, office_lat, office_lon):
    """Create a route from a capacity group and driver"""
    route = {
        'driver_id': str(driver['driver_id']),
        'vehicle_id': str(driver.get('vehicle_id', '')),
        'vehicle_type': int(driver['capacity']),
        'latitude': float(driver['latitude']),
        'longitude': float(driver['longitude']),
        'assigned_users': []
    }
    
    # Add all users from the group
    for user in group['total_users']:
        user_data = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'lng': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0))
        }
        
        if user.get('first_name'):
            user_data['first_name'] = str(user['first_name'])
        if user.get('email'):
            user_data['email'] = str(user['email'])
        
        route['assigned_users'].append(user_data)
    
    # Optimize sequence
    route = optimize_route_sequence_improved(route, office_lat, office_lon)
    update_route_metrics_improved(route, office_lat, office_lon)
    
    return route


def final_merge_blueprint(routes, config, office_lat, office_lon):
    """
    G. Final merge with geographic proximity priority
    """
    print("🔄 Step 6: Blueprint final merge with geographic priority...")

    if len(routes) < 2:
        return routes

    merged_routes = []
    used = set()

    for i, r1 in enumerate(routes):
        if i in used:
            continue

        best_merge = None
        best_score = float('-inf')

        for j, r2 in enumerate(routes):
            if j <= i or j in used:
                continue

            # Check merge conditions
            total_users = len(r1['assigned_users']) + len(r2['assigned_users'])
            max_capacity = max(r1['vehicle_type'], r2['vehicle_type'])

            if total_users > max_capacity:
                continue

            # Calculate centroid distance (PRIORITY)
            c1 = calculate_route_center_improved(r1)
            c2 = calculate_route_center_improved(r2)
            centroid_distance = haversine_distance(c1[0], c1[1], c2[0], c2[1])

            if centroid_distance > config['FINAL_MERGE_DISTANCE_THRESHOLD']:
                continue

            # Check bearing difference (more lenient)
            b1 = calculate_average_bearing_improved(r1, office_lat, office_lon)
            b2 = calculate_average_bearing_improved(r2, office_lat, office_lon)
            bearing_diff = bearing_difference(b1, b2)

            if bearing_diff > config['FINAL_MERGE_BEARING_THRESHOLD']:
                continue

            # Calculate merge score prioritizing proximity and utilization
            merged_utilization = total_users / max_capacity
            distance_score = 1.0 / (1.0 + centroid_distance)  # Higher score for closer routes
            utilization_score = merged_utilization
            bearing_score = 1.0 - (bearing_diff / config['FINAL_MERGE_BEARING_THRESHOLD'])

            # Weighted score: 50% proximity, 40% utilization, 10% direction
            merge_score = (distance_score * 0.5 + utilization_score * 0.4 + bearing_score * 0.1)

            if merge_score > best_score:
                best_merge = j
                best_score = merge_score
        
        if best_merge is not None:
            # Perform merge
            r2 = routes[best_merge]
            better_route = r1 if r1['vehicle_type'] >= r2['vehicle_type'] else r2
            
            merged_route = better_route.copy()
            merged_route['assigned_users'] = r1['assigned_users'] + r2['assigned_users']
            merged_route = optimize_route_sequence_improved(merged_route, office_lat, office_lon)
            
            merged_routes.append(merged_route)
            used.add(i)
            used.add(best_merge)
            
            utilization_pct = len(merged_route['assigned_users']) / merged_route['vehicle_type'] * 100
            print(f"  🔗 Blueprint merge: routes {r1['driver_id']} + {r2['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%)")
        else:
            merged_routes.append(r1)
            used.add(i)
    
    total_seats = sum(r['vehicle_type'] for r in merged_routes)
    total_users = sum(len(r['assigned_users']) for r in merged_routes)
    overall_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0
    
    print(f"  🎯 Blueprint merge complete: {len(routes)} → {len(merged_routes)} routes")
    print(f"  📊 Final utilization: {total_users}/{total_seats} ({overall_utilization:.1f}%)")
    
    return merged_routes


# =============================================================================
# MAIN ASSIGNMENT FUNCTION FOR BLUEPRINT CAPACITY OPTIMIZATION
# =============================================================================

def run_assignment_capacity(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """Main entry point for blueprint capacity optimization assignment"""
    return run_assignment_capacity_internal(source_id, parameter, string_param, choice)

def run_assignment_capacity_internal(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    Main assignment function implementing the blueprint approach:
    - Micro-clustering for atomic groups
    - Capacity-aware group aggregation
    - Directional validation throughout
    - Path-aware insertion with strict limits
    """
    start_time = time.time()

    # Reload configuration for capacity optimization
    global _config
    _config = load_and_validate_config()
    
    print(f"🚀 Starting BLUEPRINT CAPACITY OPTIMIZATION for source_id: {source_id}")
    print(f"📋 Parameter: {parameter}, String parameter: {string_param}")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Edge case handling
        users = data.get('users', [])
        if not users:
            print("⚠️ No users found - returning empty assignment")
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": [],
                "unassignedDrivers": _get_all_drivers_as_unassigned(data),
                "clustering_analysis": {
                    "method": "Blueprint - No Users",
                    "clusters": 0
                },
                "optimization_mode": "blueprint_capacity_optimization",
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
            print("⚠️ No drivers available - all users unassigned")
            unassigned_users = _convert_users_to_unassigned_format(users)
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": unassigned_users,
                "unassignedDrivers": [],
                "clustering_analysis": {
                    "method": "Blueprint - No Drivers",
                    "clusters": 0
                },
                "optimization_mode": "blueprint_capacity_optimization",
                "parameter": parameter,
            }

        print(f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        print("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        
        print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # BLUEPRINT ASSIGNMENT EXECUTION
        
        # STEP 1: Blueprint assignment with micro-clustering
        routes, assigned_user_ids = assign_drivers_blueprint_approach(
            user_df, driver_df, office_lat, office_lon)

        clustering_results = {
            "method": "blueprint_micro_clustering",
            "clusters": len(routes)
        }

        # STEP 2: Handle remaining unassigned users with geographic-aware residual grouping
        unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        used_driver_ids = {route['driver_id'] for route in routes}
        available_drivers_df = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]

        if not unassigned_users_df.empty and not available_drivers_df.empty:
            print("  🔄 Geographic-aware residual grouping for remaining users...")

            # Try to insert remaining users into existing routes first (geographic priority)
            for _, user in unassigned_users_df.iterrows():
                user_dict = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'latitude': float(user['latitude']),
                    'longitude': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0)),
                    'first_name': str(user.get('first_name', '')),
                    'email': str(user.get('email', ''))
                }

                # Find closest route by geographic distance
                best_route = None
                min_distance = float('inf')

                for route in routes:
                    if len(route['assigned_users']) >= route['vehicle_type']:
                        continue

                    # Calculate distance to route center
                    route_center = calculate_group_center({'total_users': route['assigned_users']})
                    distance = haversine_distance(
                        user['latitude'], user['longitude'],
                        route_center[0], route_center[1]
                    )

                    if distance < min_distance and distance <= 3.0:  # Within 3km
                        min_distance = distance
                        best_route = route

                if best_route:
                    # Try to insert into closest route
                    success, new_route = path_insert_user_into_route(
                        user_dict, best_route, office_lat, office_lon, 
                        _config['PATH_INSERT_MAX_DETOUR_RATIO'])

                    if success:
                        best_route.clear()
                        best_route.update(new_route)
                        assigned_user_ids.add(user['user_id'])
                        print(f"    📍 Inserted user {user['user_id']} into geographic-nearest route")

            # Update unassigned users after geographic insertion
            unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]

            # Create new routes for truly remaining users
            if not unassigned_users_df.empty:
                remaining_routes, additional_assigned = handle_remaining_users_improved(
                    unassigned_users_df, available_drivers_df, routes, office_lat, office_lon)

                routes.extend(remaining_routes)
                assigned_user_ids.update(additional_assigned)

        # STEP 3: Blueprint final merge
        routes = final_merge_blueprint(routes, _config, office_lat, office_lon)

        # Build unassigned users list
        unassigned_users = []
        for _, user in user_df.iterrows():
            if user['user_id'] not in assigned_user_ids:
                unassigned_user = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }
                if pd.notna(user.get('first_name')):
                    unassigned_user['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    unassigned_user['email'] = str(user['email'])
                unassigned_users.append(unassigned_user)

        # Filter out routes with no assigned users
        filtered_routes = []
        empty_route_driver_ids = set()
        
        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                filtered_routes.append(route)
            else:
                empty_route_driver_ids.add(route['driver_id'])
                print(f"  📋 Moving driver {route['driver_id']} with no users to unassigned drivers")
        
        routes = filtered_routes
        
        # Build unassigned drivers list
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
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

        # Final user count verification
        total_users_in_api = len(users)
        users_assigned = sum(len(r['assigned_users']) for r in routes)
        users_unassigned = len(unassigned_users)
        users_accounted_for = users_assigned + users_unassigned
        
        print(f"✅ Blueprint capacity optimization complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"🎯 Users assigned: {users_assigned}")
        print(f"👥 Users unassigned: {users_unassigned}")
        print(f"📋 User accounting: {users_accounted_for}/{total_users_in_api} users")

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

        for user in unassigned_users:
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

        # Enhance unassigned drivers
        enhanced_unassigned_drivers = []
        if "drivers" in data:
            all_drivers_data = data["drivers"].get("driversUnassigned", []) + data["drivers"].get("driversAssigned", [])
        else:
            all_drivers_data = data.get("driversUnassigned", []) + data.get("driversAssigned", [])

        for driver in unassigned_drivers:
            enhanced_driver = driver.copy()

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

                # Extract dynamic office coordinates and db name from source_id parameter
                office_lat, office_lon = extract_office_coordinates(data)
                db_name = source_id if source_id and source_id != "1" else data.get("db", "default")

                logger.info(f"🔍 DEBUG: source_id='{source_id}', calculated db_name='{db_name}'")
                logger.info(f"Using company coordinates: {office_lat}, {office_lon} for db: {db_name}")

                enhanced_routes = apply_route_ordering(enhanced_routes, office_lat, office_lon, db_name=db_name, algorithm_name="capacity")
                logger.info("Optimal pickup ordering applied successfully")
            except Exception as e:
                logger.error(f"Failed to apply optimal ordering: {e}")
                # Continue with routes without optimal ordering

        return {
            "status": "true",
            "execution_time": execution_time,
            "company": company_info,
            "shift": shift_info,
            "data": enhanced_routes,
            "unassignedUsers": enhanced_unassigned_users,
            "unassignedDrivers": enhanced_unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "blueprint_capacity_optimization",
            "parameter": parameter,
        }

    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        return {"status": "false", "details": str(req_err), "data": []}
    except ValueError as val_err:
        logger.error(f"Data validation error: {val_err}")
        return {"status": "false", "details": str(val_err), "data": []}
    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}