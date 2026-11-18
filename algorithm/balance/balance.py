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

# Import algorithm-level caching system
try:
    from algorithm.algorithm_cache import get_algorithm_cache, is_cache_disabled
    # Check configuration to enable/disable cache
    if is_cache_disabled():
        ALGORITHM_CACHE_AVAILABLE = False
        logger.info("🚫 Algorithm caching is disabled via configuration - will run without caching")
    else:
        ALGORITHM_CACHE_AVAILABLE = True
except ImportError:
    ALGORITHM_CACHE_AVAILABLE = False
    logger.warning("Algorithm cache system not available - will run without caching")

# Import pricebook calculator
from algorithm.pricebook.pricebook_calculator import calculate_pricebook_id

# Import core functions from base module
from algorithm.base.base import (
    validate_input_data, load_env_and_fetch_data, extract_office_coordinates,
    prepare_user_driver_dataframes, haversine_distance, calculate_bearing,
    bearing_difference, normalize_bearing_difference, coords_to_km,
    _get_all_drivers_as_unassigned, _convert_users_to_unassigned_format,
    get_progress_tracker
)

# ================== SIMPLIFIED CONFIGURATION ==================

def load_simple_config():
    """Load simple, sensible configuration for geographic clustering"""
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(script_dir, 'config.json')

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}

    # Simple, sensible defaults - more conservative clustering based on test results
    config = {
        'GEOGRAPHIC_CLUSTER_RADIUS_KM': 1.2,  # Reduced from 1.5 to 1.2km for tighter initial clusters
        'MAX_ON_ROUTE_DETOUR_KM': 2.0,        # Increased from 1.5 to 2.0km for better coverage
        'BEARING_TOLERANCE_DEGREES': 45,      # Increased from 30 to 45 degrees for more flexible "on the way"
        'MIN_CAPACITY_UTILIZATION': 0.4,      # Reduced from 0.6 to 0.4 to accept more routes
        'MAX_CLUSTER_SIZE': 6,                # NEW: Don't create clusters larger than 6 users

        # NEW: Direction-aware route merging parameters
        'DIRECTION_TOLERANCE_DEGREES': 60,    # Tolerance for considering routes direction-compatible
        'MAX_MERGE_DISTANCE_KM': 4.0,         # Maximum geographic distance for route merging
        'MERGE_SCORE_THRESHOLD': 1.5,         # Maximum merge score to allow merging
        'SMALL_ROUTE_THRESHOLD': 2,           # Routes with <= this many users are considered "small"

        'OFFICE_LAT': float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489))),
        'OFFICE_LON': float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))
    }

    logger.info(f"🚀 Using enhanced geographic clustering with direction-aware merging:")
    logger.info(f"   📍 Cluster radius: {config['GEOGRAPHIC_CLUSTER_RADIUS_KM']}km")
    logger.info(f"   🧭 On-route detour: {config['MAX_ON_ROUTE_DETOUR_KM']}km")
    logger.info(f"   📐 Bearing tolerance: {config['BEARING_TOLERANCE_DEGREES']}°")
    logger.info(f"   🔀 Direction tolerance: {config['DIRECTION_TOLERANCE_DEGREES']}°")
    logger.info(f"   📏 Max merge distance: {config['MAX_MERGE_DISTANCE_KM']}km")
    logger.info(f"   🎯 Small route threshold: {config['SMALL_ROUTE_THRESHOLD']} users")

    return config

# Load simple configuration
CONFIG = load_simple_config()

# ================== DIRECTION-AWARE ROUTE MERGING ==================

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
        """Score users based on their alignment with main route direction"""
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
    logger.info("🔀 Step 3.5: Direction-aware route merging for small routes...")

    if len(routes) < 2:
        logger.info("   ✅ No merging needed (less than 2 routes)")
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

            logger.debug(f"      Merge candidate: Route {route1_data['route']['driver_id']} + {route2_data['route']['driver_id']}")
            logger.debug(f"         Geographic distance: {geographic_distance:.1f}km")
            logger.debug(f"         Bearing difference: {bearing_diff:.1f}°")
            logger.debug(f"         Detour cost: {detour_cost:.1f}km (effective: {effective_cost:.1f})")
            logger.debug(f"         Merge score: {merge_score:.2f}")

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

                logger.info(f"   🔗 Merged routes: {route1_data['route']['driver_id']} + {best_merge_candidate['route']['driver_id']}")
                logger.info(f"      Result: {total_users}/{capacity} users ({utilization:.1f}%)")
                logger.info(f"      Merge score: {best_merge_score:.2f}")
            else:
                # If merge failed, keep the original route
                merged_routes.append(route1_data['route'])
                used_route_indices.add(route1_data['index'])
                logger.warning(f"      ⚠️ Merge failed between routes {route1_data['route']['driver_id']} and {best_merge_candidate['route']['driver_id']}")
        else:
            # No good merge found, keep the original route
            merged_routes.append(route1_data['route'])
            used_route_indices.add(route1_data['index'])

            if route1_data['user_count'] <= CONFIG['SMALL_ROUTE_THRESHOLD']:
                logger.info(f"   📍 Keeping small route {route1_data['route']['driver_id']} ({route1_data['user_count']} users) - no compatible merge found")

    # Add any remaining routes that weren't processed
    for route_data in route_data:
        if route_data['index'] not in used_route_indices:
            merged_routes.append(route_data['route'])

    # Log results
    final_route_count = len(merged_routes)
    routes_eliminated = original_route_count - final_route_count

    if routes_eliminated > 0:
        logger.info(f"   ✅ Direction-aware merging complete: {routes_eliminated} routes eliminated ({original_route_count} → {final_route_count})")
        logger.info(f"   📊 Merges performed: {merges_performed}")
    else:
        logger.info(f"   ✅ No beneficial merges found, keeping all {original_route_count} routes")

    return merged_routes

def perform_route_merge(route1, route2, office_lat, office_lon):
    """
    Perform the actual merging of two routes, optimizing user sequence
    """
    try:
        # Choose the better driver (higher capacity wins, break ties by keeping route1 driver)
        if route1['vehicle_type'] >= route2['vehicle_type']:
            merged_route = route1.copy()
            merged_capacity = route1['vehicle_type']
            logger.debug(f"         Keeping driver {route1['driver_id']} (capacity {merged_capacity})")
        else:
            merged_route = route2.copy()
            merged_capacity = route2['vehicle_type']
            logger.debug(f"         Switching to driver {route2['driver_id']} (capacity {merged_capacity})")

        # Combine users from both routes
        all_users = route1['assigned_users'] + route2['assigned_users']

        # Check if combined users exceed capacity (should not happen due to prior checks)
        if len(all_users) > merged_capacity:
            logger.warning(f"         ⚠️ Merge failed: {len(all_users)} users > capacity {merged_capacity}")
            return None

        # Optimize user sequence for the merged route
        driver_pos = (merged_route['latitude'], merged_route['longitude'])

        # Create temporary user list for optimization
        user_list = []
        for user in all_users:
            user_list.append({
                'user_id': user['user_id'],
                'latitude': user['lat'],
                'lat': user['lat'],
                'longitude': user['lng'],
                'lng': user['lng'],
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
                'latitude': float(user['latitude']),
                'lng': float(user['longitude']),
                'longitude': float(user['longitude']),
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

# ================== STEP 1: SIMPLE GEOGRAPHIC CLUSTERING ==================

def cluster_users_by_proximity(user_df, office_lat, office_lon):
    """
    Improved geographic clustering with size limits to prevent oversized clusters
    """
    logger.info("🏘️ Step 1: Geographic clustering by proximity...")

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
    logger.info(f"   ✅ Created {n_clusters} geographic clusters")

    for cluster_id in user_df['geo_cluster'].unique():
        cluster_size = len(user_df[user_df['geo_cluster'] == cluster_id])
        logger.info(f"      Cluster {cluster_id}: {cluster_size} users")

    return user_df

def split_oversized_clusters(user_df, office_lat, office_lon):
    """Split clusters that are too large or spread out"""
    logger.info("   🔧 Splitting oversized clusters...")

    max_cluster_size = CONFIG['MAX_CLUSTER_SIZE']
    max_spread_km = CONFIG['GEOGRAPHIC_CLUSTER_RADIUS_KM'] * 1.5  # Max 2.25km spread

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
            current_distance = 0

            for dist, idx in distances:
                if len(current_group) >= max_cluster_size:
                    groups.append(current_group)
                    current_group = []
                    current_distance = 0

                current_group.append(idx)
                current_distance = max(current_distance, dist)

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
    Smart capacity matching with distance-sensitive scoring
    Score = distance_to_cluster * 0.7 + capacity_waste * 0.3
    """
    cluster_size = len(cluster_users)

    if cluster_size == 0:
        return None

    # Filter suitable drivers
    suitable_drivers = available_drivers[available_drivers['capacity'] >= cluster_size].copy()

    if suitable_drivers.empty:
        logger.warning(f"No cab available for cluster of {cluster_size} users")
        return None

    # Calculate cluster center
    cluster_center = (cluster_users['latitude'].mean(), cluster_users['longitude'].mean())

    # Calculate composite score for each driver
    best_driver = None
    best_score = float('inf')

    for _, driver in suitable_drivers.iterrows():
        driver_to_cluster = haversine_distance(
            driver['latitude'], driver['longitude'],
            cluster_center[0], cluster_center[1]
        )

        capacity_waste = driver['capacity'] - cluster_size

        # Composite score: 70% distance, 30% capacity waste
        score = (driver_to_cluster * 0.7) + (capacity_waste * 0.3)

        if score < best_score:
            best_score = score
            best_driver = driver

    if best_driver is None:
        best_driver = suitable_drivers.iloc[0]

    driver_to_cluster = haversine_distance(
        best_driver['latitude'], best_driver['longitude'],
        cluster_center[0], cluster_center[1]
    )
    utilization = (cluster_size / best_driver['capacity']) * 100

    logger.info(f"   🚗 Assigned {best_driver['capacity']}-seater to {cluster_size} users "
               f"({utilization:.1f}% utilization, {driver_to_cluster:.1f}km to cluster, score: {best_score:.2f})")

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
            return True, f"Bearing aligned: {bearing_diff:.1f}°"
        else:
            # Fallback: if user is reasonably close to driver, allow anyway
            driver_to_user = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
            if driver_to_user <= CONFIG['GEOGRAPHIC_CLUSTER_RADIUS_KM']:
                return True, f"Close to driver: {driver_to_user:.1f}km"
            else:
                return False, f"Too far off route: {bearing_diff:.1f}°"

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

def fill_remaining_seats_with_cluster_check(routes, unassigned_users_df, office_lat, office_lon):
    """
    Smart seat filling with cluster ownership check - don't break existing clusters
    """
    logger.info("🪑 Step 4: Priority-based seat filling with cluster ownership...")

    filled_user_ids = set()

    # Calculate route centers for cluster ownership
    route_centers = []
    for route in routes:
        if route['assigned_users']:
            center = calculate_route_center(route['assigned_users'])
            route_centers.append(center)
        else:
            route_centers.append((route['latitude'], route['longitude']))

    # Sort unassigned users: single users from sparse areas first
    unassigned_users_df = unassigned_users_df.copy()
    unassigned_users_df['cluster_size'] = unassigned_users_df.groupby('geo_cluster')['user_id'].transform('count')
    unassigned_users_df = unassigned_users_df.sort_values('cluster_size', ascending=True)

    for route_idx, route in enumerate(routes):
        available_seats = route['vehicle_type'] - len(route['assigned_users'])

        if available_seats <= 0:
            continue

        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)
        route_center = route_centers[route_idx]

        # Find best candidates for this route
        candidates = []

        for _, user in unassigned_users_df.iterrows():
            if user['user_id'] in filled_user_ids:
                continue

            user_pos = (user['latitude'], user['longitude'])

            # Check if user belongs to this route's cluster
            dist_to_route = haversine_distance(user_pos[0], user_pos[1], route_center[0], route_center[1])

            # Check distance to all other route centers
            belongs_to_other_cluster = False
            for other_idx, other_center in enumerate(route_centers):
                if other_idx == route_idx:
                    continue
                dist_to_other = haversine_distance(user_pos[0], user_pos[1], other_center[0], other_center[1])
                if dist_to_other < dist_to_route - 0.3:  # 300m threshold
                    belongs_to_other_cluster = True
                    break

            if belongs_to_other_cluster:
                logger.debug(f"      Skipping user {user['user_id']}: belongs to another cluster")
                continue

            # Check if user is on the way
            is_on_way, reason = is_user_on_the_way(
                user_pos, driver_pos, office_pos, route['assigned_users']
            )

            if is_on_way:
                candidates.append((dist_to_route, user, reason))

        # Fill with best candidates
        candidates.sort(key=lambda x: x[0])

        seats_filled = 0
        for distance, user, reason in candidates:
            if seats_filled >= available_seats:
                break

            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'latitude': float(user['latitude']),
                'lng': float(user['longitude']),
                'longitude': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }

            if pd.notna(user.get('first_name')):
                user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                user_data['email'] = str(user['email'])

            route['assigned_users'].append(user_data)
            filled_user_ids.add(user['user_id'])
            seats_filled += 1

            # Detailed logging for debugging potential overlaps
            logger.info(f"   ✅ Added user {user['user_id']} to route {route['driver_id']} "
                       f"({reason}, dist_to_route: {distance:.1f}km)")
            logger.debug(f"      User assignment details - on_way_reason: {reason}, dist_to_route_center: {distance:.1f}km")

        if seats_filled > 0:
            utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(f"   🪑 Route {route['driver_id']}: {seats_filled} seats filled "
                       f"({utilization:.1f}% utilization)")

    return routes, filled_user_ids

def fill_remaining_seats(routes, unassigned_users_df, office_lat, office_lon):
    """
    Smart seat filling - prioritize unassigned users and those who can help global optimization
    """
    logger.info("🪑 Step 4: Priority-based seat filling...")

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
                'latitude': float(user['latitude']),
                'lng': float(user['longitude']),
                'longitude': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }

            if pd.notna(user.get('first_name')):
                user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                user_data['email'] = str(user['email'])

            route['assigned_users'].append(user_data)
            filled_user_ids.add(user['user_id'])
            seats_filled += 1

            # Detailed logging for debugging potential overlaps
            logger.info(f"   ✅ Added user {user['user_id']} to route {route['driver_id']} "
                       f"({reason}, dist_to_route: {distance:.1f}km)")
            logger.debug(f"      User assignment details - on_way_reason: {reason}, dist_to_route_center: {distance:.1f}km")

        if seats_filled > 0:
            utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(f"   🪑 Route {route['driver_id']}: {seats_filled} seats filled "
                       f"({utilization:.1f}% utilization)")

    return routes, filled_user_ids

# ================== STEP 3: ADVANCED GEOGRAPHIC OPTIMIZATION ==================

def optimize_geographic_distribution(routes, office_lat, office_lon, driver_df):
    """
    Advanced optimization that reorganizes users between nearby routes for better geographic distribution
    This solves the problem where nearby routes have overlapping or suboptimal user assignments
    """
    logger.info("🗺️ Step 3: Advanced geographic optimization with user reallocation...")

    if len(routes) < 2:
        logger.info("   ✅ No optimization needed (less than 2 routes)")
        return routes

    original_route_count = len(routes)

    # Build route centers and identify nearby route groups
    route_groups = identify_nearby_route_groups(routes, office_lat, office_lon)

    optimized_routes = []

    for group_idx, route_group in enumerate(route_groups):
        if len(route_group) <= 1:
            # No nearby routes, keep as is
            if len(route_group) == 1:
                optimized_routes.append(route_group[0])
            continue

        logger.info(f"   🎯 Optimizing route group {group_idx + 1}: {len(route_group)} nearby routes")

        # Collect all users and available capacities from this group
        all_users_in_group = []
        route_info = []

        for route in route_group:
            all_users_in_group.extend(route['assigned_users'])
            route_info.append({
                'route': route,
                'capacity': route['vehicle_type'],
                'current_users': len(route['assigned_users']),
                'available_capacity': route['vehicle_type'] - len(route['assigned_users'])
            })

        logger.info(f"      Group has {len(all_users_in_group)} users across {len(route_group)} routes")

        # If total users exceed total capacity, keep original assignment
        total_capacity = sum(r['capacity'] for r in route_info)
        total_users = len(all_users_in_group)

        if total_users > total_capacity:
            logger.warning(f"      ⚠️ Cannot optimize: {total_users} users > {total_capacity} total capacity")
            for route in route_group:
                optimized_routes.append(route)
            continue

        # Reorganize users geographically using clustering
        reorganized_routes = reorganize_users_geographically(
            all_users_in_group, route_info, office_lat, office_lon
        )

        optimized_routes.extend(reorganized_routes)

    # Add any routes that weren't part of any group
    processed_route_ids = set()
    for group in route_groups:
        for route in group:
            processed_route_ids.add(route['driver_id'])

    for route in routes:
        if route['driver_id'] not in processed_route_ids:
            optimized_routes.append(route)

    # Log results
    final_route_count = len(optimized_routes)
    logger.info(f"   ✅ Geographic optimization complete: {original_route_count} → {final_route_count} routes")

    return optimized_routes

def swap_users_for_geographic_separation(routes):
    """
    Enhanced user optimization for nearby routes:
    1. Consolidate very close routes (< 1.5km) to prevent oscillation
    2. Traditional user swaps (exchange users between routes)
    3. Asymmetric moves (move users to routes with available capacity)
    4. Smart swap-out when target route is full

    This specifically targets the issue where nearby users are split between routes serving the same area
    """
    logger.info("🔄 Step 3.1: User swapping and reassignment for geographic separation...")

    if len(routes) < 2:
        logger.info("   ✅ No user swapping needed (less than 2 routes)")
        return routes

    routes_with_users = [route for route in routes if route.get('assigned_users')]
    if len(routes_with_users) < 2:
        logger.info("   ✅ No user swapping needed (less than 2 routes with users)")
        return routes

    routes_copy = [route.copy() for route in routes]
    swaps_made = 0
    moves_made = 0

    # PRE-STEP: Consolidate very close routes to prevent swap oscillation
    routes_copy = consolidate_very_close_routes(routes_copy)

    # Standardized proximity threshold for consistency
    proximity_threshold = 2.5  # km - standardized threshold

    # Multi-pass optimization - continue until convergence
    max_iterations = 5
    iteration = 0
    improvements_this_iteration = True

    while improvements_this_iteration and iteration < max_iterations:
        iteration += 1
        improvements_this_iteration = False
        iteration_swaps = 0
        iteration_moves = 0

        logger.info(f"   🔄 Optimization pass {iteration}/{max_iterations}")

        for i, route1 in enumerate(routes_copy):
            if not route1.get('assigned_users'):
                continue

            route1_center = calculate_route_center(route1['assigned_users'])
            if not route1_center:
                continue

            for j, route2 in enumerate(routes_copy):
                if i >= j or not route2.get('assigned_users'):
                    continue

                route2_center = calculate_route_center(route2['assigned_users'])
                if not route2_center:
                    continue

                # Check if routes are nearby
                distance = haversine_distance(
                    route1_center[0], route1_center[1],
                    route2_center[0], route2_center[1]
                )

                if distance <= proximity_threshold:
                    logger.info(f"   🎯 Analyzing nearby route pair: {route1['driver_id']} + {route2['driver_id']} ({distance:.1f}km apart)")

                    # Step 1: Try asymmetric moves first (highest priority for capacity optimization)
                    route1_capacity_available = route1['vehicle_type'] - len(route1['assigned_users'])
                    route2_capacity_available = route2['vehicle_type'] - len(route2['assigned_users'])

                    asymmetric_moves = []

                    # Check if route1 has capacity to accept users from route2
                    if route1_capacity_available > 0:
                        moves = find_asymmetric_move_candidates(route2, route1, route2_center, route1_center, route1_capacity_available)
                        for move in moves:
                            move['source_route'] = route2
                            move['target_route'] = route1
                            move['direction'] = f"{route2['driver_id']} → {route1['driver_id']}"
                        asymmetric_moves.extend(moves)
                    else:
                        # Target route is full - try smart swap-out
                        swap_out_moves = find_swap_out_candidates(route2, route1, route2_center, route1_center)
                        for move in swap_out_moves:
                            move['source_route'] = route2
                            move['target_route'] = route1
                            move['direction'] = f"{route2['driver_id']} ↔ {route1['driver_id']} (swap-out)"
                        asymmetric_moves.extend(swap_out_moves)

                    # Check if route2 has capacity to accept users from route1
                    if route2_capacity_available > 0:
                        moves = find_asymmetric_move_candidates(route1, route2, route1_center, route2_center, route2_capacity_available)
                        for move in moves:
                            move['source_route'] = route1
                            move['target_route'] = route2
                            move['direction'] = f"{route1['driver_id']} → {route2['driver_id']}"
                        asymmetric_moves.extend(moves)
                    else:
                        # Target route is full - try smart swap-out
                        swap_out_moves = find_swap_out_candidates(route1, route2, route1_center, route2_center)
                        for move in swap_out_moves:
                            move['source_route'] = route1
                            move['target_route'] = route2
                            move['direction'] = f"{route1['driver_id']} ↔ {route2['driver_id']} (swap-out)"
                        asymmetric_moves.extend(swap_out_moves)

                    # Sort asymmetric moves by benefit score (highest first)
                    asymmetric_moves.sort(key=lambda x: x['move_score'], reverse=True)

                    # Process ALL beneficial moves (not just top 3)
                    moves_this_pair = 0

                    for move in asymmetric_moves:
                        # Stop if no more improvement
                        if move.get('move_score', 0) <= 0:
                            break

                        # Re-check capacity before each move (previous moves may have changed it)
                        is_swap_out = 'swap_out_user' in move
                        if not is_swap_out:
                            target_capacity_available = move['target_route']['vehicle_type'] - len(move['target_route']['assigned_users'])
                            if target_capacity_available <= 0:
                                logger.debug(f"      Skipping move: target route {move['target_route']['driver_id']} is now full")
                                continue

                        if is_swap_out:
                            logger.info(f"      🔄 Attempting swap-out move {moves_this_pair + 1}: {move['user_id']} ↔ {move['swap_out_user']['user_id']} ({move['direction']})")
                        else:
                            logger.info(f"      🚀 Attempting asymmetric move {moves_this_pair + 1}: {move['user_id']} ({move['direction']})")
                        logger.info(f"         Route capacities: {move['source_route']['driver_id']} {len(move['source_route']['assigned_users'])}/{move['source_route']['vehicle_type']}, {move['target_route']['driver_id']} {len(move['target_route']['assigned_users'])}/{move['target_route']['vehicle_type']}")

                        success = False
                        if is_swap_out:
                            success = perform_swap_out_move(move['source_route'], move['target_route'], move)
                        else:
                            success = perform_asymmetric_user_move(move['source_route'], move['target_route'], move)

                        if success:
                            moves_made += 1
                            iteration_moves += 1
                            moves_this_pair += 1
                            improvements_this_iteration = True
                            if is_swap_out:
                                logger.info(f"      ✅ Swap-out complete: {move['user_id']} ↔ {move['swap_out_user']['user_id']}")
                            else:
                                logger.info(f"      ✅ Moved user {move['user_id']} from route {move['source_route']['driver_id']} to route {move['target_route']['driver_id']}")

                            # Recalculate route centers for next iteration
                            route1_center = calculate_route_center(route1['assigned_users']) if route1['assigned_users'] else route1_center
                            route2_center = calculate_route_center(route2['assigned_users']) if route2['assigned_users'] else route2_center
                        else:
                            logger.warning(f"      ❌ Failed to perform move")

                    # Step 2: Try traditional swaps (even if asymmetric moves were performed)
                    swap_candidates = find_user_swap_candidates(
                        route1, route2, route1_center, route2_center
                    )

                    if swap_candidates:
                        logger.info(f"      Found {len(swap_candidates)} swap candidates")

                    # Perform ALL beneficial swaps (not just top 3)
                    swaps_this_pair = 0

                    for swap in swap_candidates:
                        # Re-verify capacity constraints before each swap
                        route1_capacity_ok = len(route1['assigned_users']) <= route1['vehicle_type']
                        route2_capacity_ok = len(route2['assigned_users']) <= route2['vehicle_type']

                        if not (route1_capacity_ok and route2_capacity_ok):
                            logger.debug(f"      Skipping swap: capacity constraint would be violated")
                            continue

                        logger.info(f"      🔄 Attempting swap {swaps_this_pair + 1}: {swap['user1_id']} (route {route1['driver_id']}) ↔ {swap['user2_id']} (route {route2['driver_id']})")
                        logger.info(f"         Route capacities: {route1['driver_id']} {len(route1['assigned_users'])}/{route1['vehicle_type']}, {route2['driver_id']} {len(route2['assigned_users'])}/{route2['vehicle_type']}")

                        if perform_user_swap(route1, route2, swap):
                            swaps_made += 1
                            iteration_swaps += 1
                            swaps_this_pair += 1
                            improvements_this_iteration = True
                            logger.info(f"      ✅ Swapped user {swap['user1_id']} from route {route1['driver_id']} with user {swap['user2_id']} from route {route2['driver_id']}")

                            # Recalculate centers after each swap for next iteration
                            route1_center = calculate_route_center(route1['assigned_users']) if route1['assigned_users'] else route1_center
                            route2_center = calculate_route_center(route2['assigned_users']) if route2['assigned_users'] else route2_center
                        else:
                            logger.warning(f"      ❌ Failed to swap user {swap['user1_id']} from route {route1['driver_id']} with user {swap['user2_id']} from route {route2['driver_id']}")

        logger.info(f"   Pass {iteration}: {iteration_moves} moves, {iteration_swaps} swaps")

        if not improvements_this_iteration:
            logger.info(f"   ✅ Converged after {iteration} iterations")
            break

    total_optimizations = swaps_made + moves_made
    if total_optimizations > 0:
        logger.info(f"   ✅ User optimization complete: {moves_made} asymmetric moves, {swaps_made} swaps made across {iteration} iterations")
    else:
        logger.info(f"   ✅ No beneficial user optimizations found")

    return routes_copy

def consolidate_very_close_routes(routes):
    """
    Consolidate routes that are very close together (< 1.5km) to prevent swap oscillation
    This happens BEFORE swapping to create stable geographic boundaries
    """
    logger.info("   🔧 Pre-consolidating very close routes to prevent oscillation...")
    
    consolidated_count = 0
    very_close_threshold = 1.5  # km
    
    for i, route1 in enumerate(routes):
        if not route1.get('assigned_users'):
            continue
            
        route1_center = calculate_route_center(route1['assigned_users'])
        if not route1_center:
            continue
        
        for j, route2 in enumerate(routes):
            if i >= j or not route2.get('assigned_users'):
                continue
                
            route2_center = calculate_route_center(route2['assigned_users'])
            if not route2_center:
                continue
            
            # Check if routes are very close
            distance = haversine_distance(
                route1_center[0], route1_center[1],
                route2_center[0], route2_center[1]
            )
            
            if distance < very_close_threshold:
                # Try to consolidate users based on which route center they're actually closer to
                users_to_move_to_route2 = []
                users_to_move_to_route1 = []
                
                for user in route1['assigned_users']:
                    user_pos = (user['lat'], user['lng'])
                    dist_to_r1 = haversine_distance(user_pos[0], user_pos[1], route1_center[0], route1_center[1])
                    dist_to_r2 = haversine_distance(user_pos[0], user_pos[1], route2_center[0], route2_center[1])
                    
                    # If significantly closer to route2 AND route2 has capacity
                    if dist_to_r2 < dist_to_r1 - 0.2 and len(route2['assigned_users']) < route2['vehicle_type']:
                        users_to_move_to_route2.append(user)
                
                for user in route2['assigned_users']:
                    user_pos = (user['lat'], user['lng'])
                    dist_to_r1 = haversine_distance(user_pos[0], user_pos[1], route1_center[0], route1_center[1])
                    dist_to_r2 = haversine_distance(user_pos[0], user_pos[1], route2_center[0], route2_center[1])
                    
                    # If significantly closer to route1 AND route1 has capacity
                    if dist_to_r1 < dist_to_r2 - 0.2 and len(route1['assigned_users']) < route1['vehicle_type']:
                        users_to_move_to_route1.append(user)
                
                # Perform consolidation moves
                for user in users_to_move_to_route2:
                    route1['assigned_users'].remove(user)
                    route2['assigned_users'].append(user)
                    consolidated_count += 1
                    logger.info(f"      🔗 Consolidated user {user['user_id']} from route {route1['driver_id']} to {route2['driver_id']} (routes {distance:.1f}km apart)")
                
                for user in users_to_move_to_route1:
                    route2['assigned_users'].remove(user)
                    route1['assigned_users'].append(user)
                    consolidated_count += 1
                    logger.info(f"      🔗 Consolidated user {user['user_id']} from route {route2['driver_id']} to {route1['driver_id']} (routes {distance:.1f}km apart)")
    
    if consolidated_count > 0:
        logger.info(f"   ✅ Pre-consolidated {consolidated_count} users in very close routes")
    else:
        logger.info(f"   ✅ No pre-consolidation needed")
    
    return routes

def find_user_swap_candidates(route1, route2, center1, center2):
    """
    Find user swap candidates between two routes that would improve geographic cohesion
    Focus on creating cleaner geographic boundaries - very aggressive scoring
    
    Special handling: If routes are very close (<1.5km), avoid swaps to prevent oscillation
    """
    swap_candidates = []

    logger.debug(f"      Analyzing {len(route1['assigned_users'])} users in route {route1['driver_id']} and {len(route2['assigned_users'])} users in route {route2['driver_id']}")

    # Check if routes are very close together - if so, skip swapping to avoid oscillation
    route_center_distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])
    if route_center_distance < 1.5:
        logger.debug(f"      Routes are very close ({route_center_distance:.1f}km) - skipping swaps to avoid oscillation")
        return swap_candidates

    # Calculate distances for each user to both route centers AND to users in other route
    route1_user_analysis = []
    for user in route1['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        dist_to_center1 = haversine_distance(user_pos[0], user_pos[1], center1[0], center1[1])
        dist_to_center2 = haversine_distance(user_pos[0], user_pos[1], center2[0], center2[1])

        # Calculate distance to nearest user in route2
        min_dist_to_route2_users = float('inf')
        for other_user in route2['assigned_users']:
            other_pos = (other_user['lat'], other_user['lng'])
            dist = haversine_distance(user_pos[0], user_pos[1], other_pos[0], other_pos[1])
            min_dist_to_route2_users = min(min_dist_to_route2_users, dist)

        # User prefers other route if EITHER condition is true:
        # 1. Closer to other route's center
        # 2. Has a user in other route within 0.5km (very close neighbor)
        prefers_other = (dist_to_center2 < dist_to_center1) or (min_dist_to_route2_users < 0.5)

        improvement = dist_to_center1 - dist_to_center2
        route1_user_analysis.append({
            'user': user,
            'user_id': user['user_id'],
            'dist_to_own_center': dist_to_center1,
            'dist_to_other_center': dist_to_center2,
            'min_dist_to_other_route_users': min_dist_to_route2_users,
            'prefers_other_route': prefers_other,
            'improvement': improvement
        })

        if prefers_other:
            logger.debug(f"         User {user['user_id']} from route {route1['driver_id']} prefers route {route2['driver_id']} (center improvement: {improvement:.3f}km, nearest user: {min_dist_to_route2_users:.3f}km)")

    route2_user_analysis = []
    for user in route2['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        dist_to_center2 = haversine_distance(user_pos[0], user_pos[1], center2[0], center2[1])
        dist_to_center1 = haversine_distance(user_pos[0], user_pos[1], center1[0], center1[1])

        # Calculate distance to nearest user in route1
        min_dist_to_route1_users = float('inf')
        for other_user in route1['assigned_users']:
            other_pos = (other_user['lat'], other_user['lng'])
            dist = haversine_distance(user_pos[0], user_pos[1], other_pos[0], other_pos[1])
            min_dist_to_route1_users = min(min_dist_to_route1_users, dist)

        prefers_other = (dist_to_center1 < dist_to_center2) or (min_dist_to_route1_users < 0.5)

        improvement = dist_to_center2 - dist_to_center1
        route2_user_analysis.append({
            'user': user,
            'user_id': user['user_id'],
            'dist_to_own_center': dist_to_center2,
            'dist_to_other_center': dist_to_center1,
            'min_dist_to_other_route_users': min_dist_to_route1_users,
            'prefers_other_route': prefers_other,
            'improvement': improvement
        })

        if prefers_other:
            logger.debug(f"         User {user['user_id']} from route {route2['driver_id']} prefers route {route1['driver_id']} (center improvement: {improvement:.3f}km, nearest user: {min_dist_to_route1_users:.3f}km)")

    # Build all potential swaps with comprehensive scoring
    all_potential_swaps = []

    for user1_data in route1_user_analysis:
        for user2_data in route2_user_analysis:
            user1_pos = (user1_data['user']['lat'], user1_data['user']['lng'])
            user2_pos = (user2_data['user']['lat'], user2_data['user']['lng'])
            user_distance = haversine_distance(user1_pos[0], user1_pos[1], user2_pos[0], user2_pos[1])

            # Comprehensive scoring factors
            center_improvement = user1_data['improvement'] + user2_data['improvement']

            # Strong bonus for users who are very close to users in other route
            neighbor_bonus = 0
            if user1_data['min_dist_to_other_route_users'] < 0.5:
                neighbor_bonus += 2.0
            if user2_data['min_dist_to_other_route_users'] < 0.5:
                neighbor_bonus += 2.0

            # Bonus for users being close to each other (prevents fragmentation)
            proximity_bonus = max(0, (1.5 - user_distance) * 1.5)

            # Combined score
            total_score = center_improvement + neighbor_bonus + proximity_bonus

            # Create swap candidate
            swap = {
                'user1_id': user1_data['user_id'],
                'user1_data': user1_data['user'],
                'user2_id': user2_data['user_id'],
                'user2_data': user2_data['user'],
                'improvement_score': total_score,
                'center_improvement': center_improvement,
                'neighbor_bonus': neighbor_bonus,
                'proximity_bonus': proximity_bonus,
                'user_distance': user_distance,
                'route1_capacity_after': len(route1['assigned_users']),
                'route2_capacity_after': len(route2['assigned_users']),
                'swap_type': 'comprehensive'
            }

            all_potential_swaps.append(swap)

            if total_score > 0:
                logger.debug(f"         Swap candidate: {user1_data['user_id']} ↔ {user2_data['user_id']}")
                logger.debug(f"            Score: {total_score:.2f} (center: {center_improvement:.2f}, neighbor: {neighbor_bonus:.2f}, proximity: {proximity_bonus:.2f})")

    # Sort by total score and take best candidates
    all_potential_swaps.sort(key=lambda x: x['improvement_score'], reverse=True)

    # Add top 3 swaps with positive scores
    for swap in all_potential_swaps[:3]:
        if swap['improvement_score'] > 0:
            swap_candidates.append(swap)
            logger.info(f"         ✅ Swap candidate: {swap['user1_id']} ↔ {swap['user2_id']} (score: {swap['improvement_score']:.2f})")

    logger.debug(f"      Found {len(swap_candidates)} total swaps between routes {route1['driver_id']} and {route2['driver_id']}")

    # Sort by improvement score (highest first)
    swap_candidates.sort(key=lambda x: x['improvement_score'], reverse=True)

    return swap_candidates

def perform_user_swap(route1, route2, swap_candidate):
    """
    Perform the actual user swap between two routes
    """
    try:
        logger.debug(f"         Performing swap between routes {route1['driver_id']} ({len(route1['assigned_users'])}/{route1['vehicle_type']}) and {route2['driver_id']} ({len(route2['assigned_users'])}/{route2['vehicle_type']})")

        # For swaps, capacity should remain the same, so we don't need to check capacity beforehand
        # Each route gives up one user and takes one user

        # Find and remove users from their current routes
        user1_found = False
        user2_found = False
        original_route1_users = route1['assigned_users'].copy()
        original_route2_users = route2['assigned_users'].copy()

        # Remove user1 from route1
        route1['assigned_users'] = [u for u in route1['assigned_users'] if u['user_id'] != swap_candidate['user1_id']]
        if len(route1['assigned_users']) < len(original_route1_users):
            user1_found = True
            logger.debug(f"         Removed user {swap_candidate['user1_id']} from route {route1['driver_id']}")

        # Remove user2 from route2
        route2['assigned_users'] = [u for u in route2['assigned_users'] if u['user_id'] != swap_candidate['user2_id']]
        if len(route2['assigned_users']) < len(original_route2_users):
            user2_found = True
            logger.debug(f"         Removed user {swap_candidate['user2_id']} from route {route2['driver_id']}")

        if not user1_found or not user2_found:
            # Restore original users if swap failed
            logger.warning(f"         Swap failed: could not find users - user1_found: {user1_found}, user2_found: {user2_found}")
            route1['assigned_users'] = original_route1_users
            route2['assigned_users'] = original_route2_users
            return False

        # Add users to their new routes
        route1['assigned_users'].append(swap_candidate['user2_data'])
        route2['assigned_users'].append(swap_candidate['user1_data'])
        logger.debug(f"         Added user {swap_candidate['user2_id']} to route {route1['driver_id']}")
        logger.debug(f"         Added user {swap_candidate['user1_id']} to route {route2['driver_id']}")

        # Verify capacity constraints are still met (should be true for swaps)
        if len(route1['assigned_users']) > route1['vehicle_type'] or len(route2['assigned_users']) > route2['vehicle_type']:
            # Revert swap if capacity violated
            logger.warning(f"         Swap failed: capacity violation after swap")
            route1['assigned_users'] = original_route1_users
            route2['assigned_users'] = original_route2_users
            return False

        logger.debug(f"         Swap successful: {route1['driver_id']} now has {len(route1['assigned_users'])} users, {route2['driver_id']} now has {len(route2['assigned_users'])} users")
        return True

    except Exception as e:
        logger.error(f"Error performing user swap: {e}")
        return False

def find_swap_out_candidates(source_route, target_route, source_center, target_center):
    """
    Find smart swap-out opportunities when target route is full:
    - Move user from source to target
    - Move farthest user from target to make room
    """
    swap_out_candidates = []

    if not target_route['assigned_users']:
        return swap_out_candidates

    logger.debug(f"      Looking for swap-out opportunities: {source_route['driver_id']} → {target_route['driver_id']} (target full)")

    # Find users in source that want to move to target
    for source_user in source_route['assigned_users']:
        source_user_pos = (source_user['lat'], source_user['lng'])

        # Calculate distances
        dist_to_source_center = haversine_distance(source_user_pos[0], source_user_pos[1], source_center[0], source_center[1])
        dist_to_target_center = haversine_distance(source_user_pos[0], source_user_pos[1], target_center[0], target_center[1])

        # Only consider if user clearly prefers target
        if dist_to_target_center >= dist_to_source_center - 0.2:
            continue

        # Find the farthest user in target route to swap out
        best_swap_out_user = None
        best_swap_score = -float('inf')

        for target_user in target_route['assigned_users']:
            target_user_pos = (target_user['lat'], target_user['lng'])
            target_user_dist_to_center = haversine_distance(target_user_pos[0], target_user_pos[1], target_center[0], target_center[1])
            target_user_dist_to_source = haversine_distance(target_user_pos[0], target_user_pos[1], source_center[0], source_center[1])

            # Score: prefer users far from target center but close to source center
            swap_score = target_user_dist_to_center - target_user_dist_to_source

            if swap_score > best_swap_score:
                best_swap_score = swap_score
                best_swap_out_user = target_user

        if best_swap_out_user and best_swap_score > 0.1:
            improvement = (dist_to_source_center - dist_to_target_center) + best_swap_score

            swap_out_candidates.append({
                'user_id': source_user['user_id'],
                'user_data': source_user,
                'swap_out_user': best_swap_out_user,
                'distance_improvement': dist_to_source_center - dist_to_target_center,
                'swap_out_improvement': best_swap_score,
                'move_score': improvement
            })

            logger.debug(f"         Swap-out candidate: {source_user['user_id']} → target, swap out {best_swap_out_user['user_id']} (score: {improvement:.3f})")

    # Sort by improvement score
    swap_out_candidates.sort(key=lambda x: x['move_score'], reverse=True)

    return swap_out_candidates

def find_asymmetric_move_candidates(source_route, target_route, source_center, target_center, available_capacity):
    """
    Find users in source_route that would benefit from moving to target_route
    This handles the Amit/Krish scenario where one route has capacity and can accept users
    """
    move_candidates = []

    logger.debug(f"      Looking for asymmetric moves: {source_route['driver_id']} → {target_route['driver_id']} (capacity: {available_capacity})")

    # Analyze each user in the source route
    for user in source_route['assigned_users']:
        user_pos = (user['lat'], user['lng'])

        # Calculate distances to both route centers
        dist_to_source_center = haversine_distance(user_pos[0], user_pos[1], source_center[0], source_center[1])
        dist_to_target_center = haversine_distance(user_pos[0], user_pos[1], target_center[0], target_center[1])

        # Calculate improvement (positive = user is closer to target route)
        distance_improvement = dist_to_source_center - dist_to_target_center

        # Priority 1: Users significantly closer to target route (at least 0.1km closer)
        if distance_improvement > 0.1:
            # Calculate move score based on distance improvement and geographic proximity
            move_score = distance_improvement

            # Bonus for very close users (< 0.8km to target center)
            if dist_to_target_center < 0.8:
                move_score += 0.5  # Strong bonus for very close placement

            # Bonus for users who are very close to existing users in target route
            min_distance_to_target_users = float('inf')
            for target_user in target_route['assigned_users']:
                target_user_pos = (target_user['lat'], target_user['lng'])
                user_to_user_distance = haversine_distance(
                    user_pos[0], user_pos[1], target_user_pos[0], target_user_pos[1]
                )
                min_distance_to_target_users = min(min_distance_to_target_users, user_to_user_distance)

            if min_distance_to_target_users < 1.0:
                move_score += 0.3  # Bonus for being close to existing users in target route

            move_candidates.append({
                'user_id': user['user_id'],
                'user_data': user,
                'distance_improvement': distance_improvement,
                'dist_to_target_center': dist_to_target_center,
                'min_distance_to_target_users': min_distance_to_target_users,
                'move_score': move_score
            })

            logger.debug(f"         User {user['user_id']} prefers target route: improvement {distance_improvement:.3f}km, score {move_score:.3f}")

        # Priority 2: Users very close to target route center (< 0.5km) even if slightly further
        elif dist_to_target_center < 0.5:
            move_score = 0.2 - dist_to_target_center  # Small positive score for very close users

            move_candidates.append({
                'user_id': user['user_id'],
                'user_data': user,
                'distance_improvement': distance_improvement,
                'dist_to_target_center': dist_to_target_center,
                'min_distance_to_target_users': float('inf'),
                'move_score': move_score
            })

            logger.debug(f"         User {user['user_id']} very close to target route center: {dist_to_target_center:.3f}km, score {move_score:.3f}")

    # Sort by move score (highest first)
    move_candidates.sort(key=lambda x: x['move_score'], reverse=True)

    logger.debug(f"      Found {len(move_candidates)} asymmetric move candidates for {source_route['driver_id']} → {target_route['driver_id']}")

    return move_candidates

def perform_swap_out_move(source_route, target_route, move_candidate):
    """
    Perform swap-out move: move user from source to target, and swap-out user from target to source
    """
    try:
        logger.debug(f"         Performing swap-out move")

        original_source_users = source_route['assigned_users'].copy()
        original_target_users = target_route['assigned_users'].copy()

        # Remove users from their routes
        source_route['assigned_users'] = [u for u in source_route['assigned_users'] if u['user_id'] != move_candidate['user_id']]
        target_route['assigned_users'] = [u for u in target_route['assigned_users'] if u['user_id'] != move_candidate['swap_out_user']['user_id']]

        # Add users to new routes
        target_route['assigned_users'].append(move_candidate['user_data'])
        source_route['assigned_users'].append(move_candidate['swap_out_user'])

        # Verify capacity constraints
        if len(source_route['assigned_users']) > source_route['vehicle_type'] or len(target_route['assigned_users']) > target_route['vehicle_type']:
            logger.warning(f"         Swap-out failed: capacity violation")
            source_route['assigned_users'] = original_source_users
            target_route['assigned_users'] = original_target_users
            return False

        logger.debug(f"         Swap-out successful")
        return True

    except Exception as e:
        logger.error(f"Error performing swap-out move: {e}")
        return False

def perform_asymmetric_user_move(source_route, target_route, move_candidate):
    """
    Perform the actual asymmetric move: move user from source_route to target_route
    This handles the case where target_route has available capacity
    """
    try:
        logger.debug(f"         Performing asymmetric move: {source_route['driver_id']} → {target_route['driver_id']}")
        logger.debug(f"         Target route capacity: {len(target_route['assigned_users'])}/{target_route['vehicle_type']}")

        # Check if target route has capacity
        if len(target_route['assigned_users']) >= target_route['vehicle_type']:
            logger.warning(f"         Move failed: target route {target_route['driver_id']} is at full capacity")
            return False

        # Find and remove user from source route
        user_found = False
        original_source_users = source_route['assigned_users'].copy()
        original_target_users = target_route['assigned_users'].copy()

        # Remove user from source route
        source_route['assigned_users'] = [u for u in source_route['assigned_users'] if u['user_id'] != move_candidate['user_id']]
        if len(source_route['assigned_users']) < len(original_source_users):
            user_found = True
            logger.debug(f"         Removed user {move_candidate['user_id']} from source route {source_route['driver_id']}")
        else:
            logger.warning(f"         Move failed: could not find user {move_candidate['user_id']} in source route {source_route['driver_id']}")
            return False

        # Add user to target route
        target_route['assigned_users'].append(move_candidate['user_data'])
        logger.debug(f"         Added user {move_candidate['user_id']} to target route {target_route['driver_id']}")

        # Verify capacity constraints are still met
        if len(target_route['assigned_users']) > target_route['vehicle_type']:
            # Revert move if capacity violated
            logger.warning(f"         Move failed: capacity violation in target route {target_route['driver_id']}")
            source_route['assigned_users'] = original_source_users
            target_route['assigned_users'] = original_target_users
            return False

        logger.debug(f"         Asymmetric move successful: {source_route['driver_id']} now has {len(source_route['assigned_users'])} users, {target_route['driver_id']} now has {len(target_route['assigned_users'])} users")

        # Log the improvement
        logger.debug(f"         Distance improvement: {move_candidate['distance_improvement']:.3f}km")
        if move_candidate['min_distance_to_target_users'] != float('inf'):
            logger.debug(f"         Distance to nearest target route user: {move_candidate['min_distance_to_target_users']:.3f}km")

        return True

    except Exception as e:
        logger.error(f"Error performing asymmetric user move: {e}")
        return False

def apply_hard_consolidation(routes, office_lat, office_lon):
    """
    Final hard-consolidation step: ensure nearby users (within 0.5km) are in same route if possible
    """
    logger.info("🔧 Step 4.5: Final hard-consolidation for nearby users...")

    consolidation_threshold = 0.5  # km - users within this distance should be together
    changes_made = 0

    # Build user proximity graph
    all_users_with_routes = []
    for route_idx, route in enumerate(routes):
        for user in route['assigned_users']:
            all_users_with_routes.append({
                'user': user,
                'route_idx': route_idx,
                'pos': (user['lat'], user['lng'])
            })

    # Find users that are very close but in different routes
    for i, user1_info in enumerate(all_users_with_routes):
        for j, user2_info in enumerate(all_users_with_routes):
            if i >= j:
                continue

            # Skip if already in same route
            if user1_info['route_idx'] == user2_info['route_idx']:
                continue

            # Check distance
            distance = haversine_distance(
                user1_info['pos'][0], user1_info['pos'][1],
                user2_info['pos'][0], user2_info['pos'][1]
            )

            if distance <= consolidation_threshold:
                route1 = routes[user1_info['route_idx']]
                route2 = routes[user2_info['route_idx']]

                # Try to consolidate: move user2 to route1 if there's capacity
                if len(route1['assigned_users']) < route1['vehicle_type']:
                    # Remove from route2
                    route2['assigned_users'] = [u for u in route2['assigned_users'] if u['user_id'] != user2_info['user']['user_id']]
                    # Add to route1
                    route1['assigned_users'].append(user2_info['user'])
                    changes_made += 1
                    logger.info(f"   🔗 Consolidated user {user2_info['user']['user_id']} into route {route1['driver_id']} (distance: {distance:.3f}km)")
                    user2_info['route_idx'] = user1_info['route_idx']  # Update for future checks

                # Or try to move user1 to route2 if there's capacity
                elif len(route2['assigned_users']) < route2['vehicle_type']:
                    # Remove from route1
                    route1['assigned_users'] = [u for u in route1['assigned_users'] if u['user_id'] != user1_info['user']['user_id']]
                    # Add to route2
                    route2['assigned_users'].append(user1_info['user'])
                    changes_made += 1
                    logger.info(f"   🔗 Consolidated user {user1_info['user']['user_id']} into route {route2['driver_id']} (distance: {distance:.3f}km)")
                    user1_info['route_idx'] = user2_info['route_idx']  # Update for future checks

    if changes_made > 0:
        logger.info(f"   ✅ Hard-consolidation complete: {changes_made} nearby users consolidated")
    else:
        logger.info(f"   ✅ No consolidation needed - all nearby users already grouped")

    return routes

def identify_nearby_route_groups(routes, office_lat, office_lon, proximity_threshold=2.5):
    """
    Identify groups of routes that are geographically close to each other
    Returns groups of routes that should be optimized together
    """
    route_groups = []
    used_route_indices = set()

    for i, route1 in enumerate(routes):
        if i in used_route_indices:
            continue

        if len(route1['assigned_users']) == 0:
            continue

        # Calculate route1 center
        route1_center = calculate_route_center(route1['assigned_users'])

        # Start a new group with this route
        current_group = [route1]
        used_route_indices.add(i)

        # Find all nearby routes
        for j, route2 in enumerate(routes):
            if i == j or j in used_route_indices:
                continue

            if len(route2['assigned_users']) == 0:
                continue

            # Calculate route2 center
            route2_center = calculate_route_center(route2['assigned_users'])

            # Check if routes are nearby
            distance = haversine_distance(
                route1_center[0], route1_center[1],
                route2_center[0], route2_center[1]
            )

            if distance <= proximity_threshold:
                current_group.append(route2)
                used_route_indices.add(j)
                logger.info(f"      Found nearby routes: {route1['driver_id']} + {route2['driver_id']} ({distance:.1f}km apart)")

        if len(current_group) > 1:
            route_groups.append(current_group)
        else:
            # Single route group, won't be optimized
            used_route_indices.remove(i)

    return route_groups

def calculate_route_center(users):
    """Calculate the geographic center of a route's users"""
    if not users:
        return None

    center_lat = sum(user['lat'] for user in users) / len(users)
    center_lon = sum(user['lng'] for user in users) / len(users)

    return (center_lat, center_lon)

def reorganize_users_geographically(all_users, route_info, office_lat, office_lon):
    """
    Reorganize users geographically using capacity-constrained clustering
    This creates the most compact geographic distribution while respecting capacity limits
    """
    logger.info(f"      🔄 Reorganizing {len(all_users)} users among {len(route_info)} routes")

    if len(all_users) == 0 or len(route_info) == 0:
        return [r['route'] for r in route_info]

    # Sort routes by capacity (largest first) for better distribution
    route_info.sort(key=lambda x: x['capacity'], reverse=True)

    # Calculate total capacity
    total_capacity = sum(r['capacity'] for r in route_info)
    total_users = len(all_users)

    if total_users > total_capacity:
        logger.warning(f"      ⚠️ Cannot reorganize: {total_users} users > {total_capacity} total capacity")
        return [r['route'] for r in route_info]

    # Use capacity-constrained geographic clustering
    user_assignments = capacity_constrained_clustering(all_users, route_info, office_lat, office_lon)

    # Create new routes with assigned users
    optimized_routes = []

    for route_idx, route in enumerate(route_info):
        # Get users assigned to this route
        assigned_users = []
        for user_idx, assigned_route_idx in enumerate(user_assignments):
            if assigned_route_idx == route_idx:
                assigned_users.append(all_users[user_idx])

        # Verify capacity constraint
        if len(assigned_users) > route['capacity']:
            logger.error(f"      ❌ CAPACITY VIOLATION: Route {route['route']['driver_id']} has {len(assigned_users)}/{route['capacity']} users")
            # Fallback: keep original assignment
            assigned_users = route['route']['assigned_users']

        # Create new route with assigned users
        new_route = route['route'].copy()
        new_route['assigned_users'] = assigned_users

        # Calculate utilization
        utilization = len(assigned_users) / route['capacity'] * 100

        logger.info(f"      Route {route['route']['driver_id']}: {len(assigned_users)}/{route['capacity']} users ({utilization:.1f}%)")

        optimized_routes.append(new_route)

    return optimized_routes

def capacity_constrained_clustering(all_users, route_info, office_lat, office_lon):
    """
    Capacity-constrained clustering that respects route capacity limits
    Uses a greedy geographic assignment with capacity constraints
    """
    n_routes = len(route_info)
    user_positions = [(user['lat'], user['lng']) for user in all_users]

    # Sort routes by capacity (largest first) for better assignment
    route_indices = list(range(n_routes))
    route_indices.sort(key=lambda x: route_info[x]['capacity'], reverse=True)

    # Initialize assignments
    user_assignments = [-1] * len(all_users)  # -1 means unassigned
    route_current_load = [0] * n_routes

    # Calculate route centers for initial assignment
    route_centers = []
    for route in route_info:
        route_center = calculate_route_center(route['route']['assigned_users'])
        route_centers.append(route_center)

    # Assign users to routes respecting capacity constraints
    for user_idx, user_pos in enumerate(user_positions):
        best_route = -1
        best_score = float('inf')

        for route_idx in route_indices:
            capacity = route_info[route_idx]['capacity']
            current_load = route_current_load[route_idx]

            # Check if route has capacity
            if current_load >= capacity:
                continue

            # Calculate score based on distance to route center
            if route_idx < len(route_centers) and route_centers[route_idx]:
                distance = haversine_distance(
                    user_pos[0], user_pos[1],
                    route_centers[route_idx][0], route_centers[route_idx][1]
                )
            else:
                # Fallback: use office as reference
                distance = haversine_distance(
                    user_pos[0], user_pos[1], office_lat, office_lon
                )

            # Add capacity utilization bonus (prefer routes with better utilization)
            utilization = (current_load + 1) / capacity
            capacity_bonus = utilization * 0.5  # Prefer routes with higher utilization

            score = distance - capacity_bonus

            if score < best_score:
                best_score = score
                best_route = route_idx

        # Assign user to best route
        if best_route != -1:
            user_assignments[user_idx] = best_route
            route_current_load[best_route] += 1
        else:
            logger.warning(f"      ⚠️ Could not assign user {user_idx}: no route with available capacity")

    # Verify all users are assigned
    unassigned_count = user_assignments.count(-1)
    if unassigned_count > 0:
        logger.warning(f"      ⚠️ {unassigned_count} users could not be assigned due to capacity constraints")

    # Verify capacity constraints
    for route_idx in range(n_routes):
        assigned_count = user_assignments.count(route_idx)
        capacity = route_info[route_idx]['capacity']
        if assigned_count > capacity:
            logger.error(f"      ❌ Capacity constraint violated: Route {route_idx} has {assigned_count}/{capacity} users")
            # Emergency fix: remove excess users
            users_to_remove = assigned_count - capacity
            removed_count = 0
            for i, assigned_route in enumerate(user_assignments):
                if assigned_route == route_idx and removed_count < users_to_remove:
                    user_assignments[i] = -1
                    removed_count += 1

    logger.info(f"      ✅ Capacity-constrained assignment complete:")
    for route_idx, route in enumerate(route_info):
        assigned_count = user_assignments.count(route_idx)
        capacity = route['capacity']
        utilization = (assigned_count / capacity) * 100 if capacity > 0 else 0
        logger.info(f"         Route {route['route']['driver_id']}: {assigned_count}/{capacity} ({utilization:.1f}%)")

    return user_assignments

def k_means_clustering(positions, k, office_lat, office_lon, max_iterations=50):
    """
    Simple K-means clustering implementation for user positions
    Includes office location as a reference point for better geographic clustering
    """
    import random

    if len(positions) <= k:
        # Each user gets their own cluster
        return list(range(len(positions)))

    # Initialize k cluster centers using k-means++ style initialization
    # Start with office location as one center if k >= 2
    centers = []
    if k >= 2:
        centers.append((office_lat, office_lon))

    # Add remaining centers from user positions
    remaining_positions = positions.copy()
    if (office_lat, office_lon) in remaining_positions:
        remaining_positions.remove((office_lat, office_lon))

    for _ in range(k - len(centers)):
        if remaining_positions:
            centers.append(random.choice(remaining_positions))
            remaining_positions.remove(centers[-1])

    # Ensure we have exactly k centers
    while len(centers) < k and positions:
        for pos in positions:
            if pos not in centers:
                centers.append(pos)
                break

    if len(centers) < k:
        centers.extend([(office_lat, office_lon)] * (k - len(centers)))

    # K-means iterations
    for iteration in range(max_iterations):
        # Assign each position to nearest center
        clusters = []
        for pos in positions:
            min_dist = float('inf')
            best_cluster = 0

            for i, center in enumerate(centers):
                dist = haversine_distance(pos[0], pos[1], center[0], center[1])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = i

            clusters.append(best_cluster)

        # Update centers
        new_centers = []
        for i in range(k):
            cluster_positions = [positions[j] for j in range(len(positions)) if clusters[j] == i]

            if cluster_positions:
                new_center_lat = sum(pos[0] for pos in cluster_positions) / len(cluster_positions)
                new_center_lon = sum(pos[1] for pos in cluster_positions) / len(cluster_positions)
                new_centers.append((new_center_lat, new_center_lon))
            else:
                # Keep old center if cluster is empty
                new_centers.append(centers[i] if i < len(centers) else (office_lat, office_lon))

        # Check for convergence
        if new_centers == centers:
            break

        centers = new_centers

    return clusters

# ================== LEGACY GLOBAL CLUSTER MERGING (KEPT FOR REFERENCE) ==================

def merge_nearby_clusters_globally(routes, office_lat, office_lon, user_df, driver_df):
    """
    Global optimization: merge nearby clusters to create better geographic distribution
    Uses standardized 2.5km threshold for consistency
    """
    logger.info("🌐 Step 3: Global cluster merging optimization...")

    if len(routes) < 2:
        logger.info("   ✅ No merging needed (less than 2 routes)")
        return routes

    original_route_count = len(routes)
    merged_routes = []
    used_route_indices = set()

    # Convert routes to a more workable format
    route_data = []
    for i, route in enumerate(routes):
        if len(route['assigned_users']) == 0:
            continue

        # Calculate route center
        user_positions = [(user['lat'], user['lng']) for user in route['assigned_users']]
        center_lat = sum(pos[0] for pos in user_positions) / len(user_positions)
        center_lon = sum(pos[1] for pos in user_positions) / len(user_positions)

        route_data.append({
            'index': i,
            'route': route,
            'center': (center_lat, center_lon),
            'users': route['assigned_users'],
            'user_count': len(route['assigned_users']),
            'capacity': route['vehicle_type']
        })

    # Sort routes by capacity (largest first) to prioritize merging smaller routes into larger ones
    route_data.sort(key=lambda x: x['capacity'], reverse=True)

    # Try to merge routes globally
    for i, route1_data in enumerate(route_data):
        if route1_data['index'] in used_route_indices:
            continue

        best_merge_candidate = None
        best_merge_score = float('inf')

        # Look for merge candidates among other routes
        for j, route2_data in enumerate(route_data):
            if i == j or route2_data['index'] in used_route_indices:
                continue

            # Check if routes can be merged (capacity constraint)
            combined_users = route1_data['user_count'] + route2_data['user_count']
            max_capacity = max(route1_data['capacity'], route2_data['capacity'])

            if combined_users > max_capacity:
                continue  # Cannot merge - would exceed capacity

            # Calculate geographic distance between route centers
            center_distance = haversine_distance(
                route1_data['center'][0], route1_data['center'][1],
                route2_data['center'][0], route2_data['center'][1]
            )

            # Calculate merge desirability score (lower is better)
            # Factors: distance (closer is better), capacity utilization (higher is better)
            utilization = combined_users / max_capacity

            # Prefer merges that improve utilization and are geographically close
            merge_score = center_distance - (utilization * 2.0)  # Subtract utilization bonus

            # Additional bonus for merging very close routes (within 1.5km)
            if center_distance <= 1.5:
                merge_score -= 1.0  # Strong bonus for very close routes

            # Check if this is the best merge candidate
            if merge_score < best_merge_score:
                best_merge_candidate = route2_data
                best_merge_score = merge_score

        # If we found a good merge candidate, perform the merge
        if best_merge_candidate and best_merge_score < 1.0:  # Only merge if score is good
            merged_route = merge_two_routes(
                route1_data['route'],
                best_merge_candidate['route'],
                office_lat,
                office_lon
            )

            if merged_route:
                merged_routes.append(merged_route)
                used_route_indices.add(route1_data['index'])
                used_route_indices.add(best_merge_candidate['index'])

                total_users = len(merged_route['assigned_users'])
                capacity = merged_route['vehicle_type']
                utilization = (total_users / capacity) * 100

                logger.info(f"   🔗 Merged routes: {route1_data['route']['driver_id']} + {best_merge_candidate['route']['driver_id']} "
                           f"= {total_users}/{capacity} users ({utilization:.1f}%)")
            else:
                # If merge failed, keep the original route
                merged_routes.append(route1_data['route'])
                used_route_indices.add(route1_data['index'])
        else:
            # No good merge found, keep the original route
            merged_routes.append(route1_data['route'])
            used_route_indices.add(route1_data['index'])

    # Add any remaining routes that weren't merged
    for route_data in route_data:
        if route_data['index'] not in used_route_indices:
            merged_routes.append(route_data['route'])

    # Log results
    merges_made = original_route_count - len(merged_routes)
    if merges_made > 0:
        logger.info(f"   ✅ Global merging complete: {merges_made} merges made, {len(merged_routes)} final routes")
    else:
        logger.info(f"   ✅ No beneficial merges found, keeping all {original_route_count} routes")

    return merged_routes

def merge_two_routes(route1, route2, office_lat, office_lon):
    """Merge two routes into one, choosing the better driver and optimizing user sequence"""

    # Choose the better driver (higher capacity wins, break ties by keeping route1 driver)
    if route1['vehicle_type'] >= route2['vehicle_type']:
        merged_route = route1.copy()
        merged_capacity = route1['vehicle_type']
        logger.info(f"      Keeping driver {route1['driver_id']} (capacity {merged_capacity})")
    else:
        merged_route = route2.copy()
        merged_capacity = route2['vehicle_type']
        logger.info(f"      Switching to driver {route2['driver_id']} (capacity {merged_capacity})")

    # Combine users from both routes
    all_users = route1['assigned_users'] + route2['assigned_users']

    # Check if combined users exceed capacity
    if len(all_users) > merged_capacity:
        logger.warning(f"      ⚠️ Merge failed: {len(all_users)} users > capacity {merged_capacity}")
        return None

    # Re-optimize user sequence for the merged route
    driver_pos = (merged_route['latitude'], merged_route['longitude'])
    office_pos = (office_lat, office_lon)

    # Create temporary user list for optimization
    user_list = []
    for user in all_users:
        user_list.append({
            'user_id': user['user_id'],
            'latitude': user['lat'],
            'lat': user['lat'],
            'longitude': user['lng'],
            'lng': user['lng'],
            'office_distance': user.get('office_distance', 0)
        })

    # Simple optimization: sort users by projection along main axis
    main_axis_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    def projection_score(user):
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], user['latitude'], user['longitude'])
        bearing_alignment = math.cos(math.radians(abs(bearing_difference(user_bearing, main_axis_bearing))))
        distance = haversine_distance(driver_pos[0], driver_pos[1], user['latitude'], user['longitude'])
        return distance * bearing_alignment  # Closer points in the right direction get lower scores

    user_list.sort(key=projection_score)

    # Update merged route with optimized user sequence
    merged_route['assigned_users'] = []
    for user in user_list:
        user_data = {
            'user_id': str(user['user_id']),
            'lat': float(user['latitude']),
            'latitude': float(user['latitude']),
            'lng': float(user['longitude']),
            'longitude': float(user['longitude']),
            'office_distance': float(user.get('office_distance', 0))
        }

        # Preserve additional user information if present
        for original_user in all_users:
            if str(original_user.get('user_id', '')) == user['user_id']:
                if 'first_name' in original_user:
                    user_data['first_name'] = str(original_user['first_name'])
                if 'email' in original_user:
                    user_data['email'] = str(original_user['email'])
                if 'address' in original_user:
                    user_data['address'] = str(original_user['address'])
                if 'employee_shift' in original_user:
                    user_data['employee_shift'] = str(original_user['employee_shift'])
                if 'shift_type' in original_user:
                    user_data['shift_type'] = str(original_user['shift_type'])
                if 'last_name' in original_user:
                    user_data['last_name'] = str(original_user['last_name'])
                if 'phone' in original_user:
                    user_data['phone'] = str(original_user['phone'])
                break

        merged_route['assigned_users'].append(user_data)

    return merged_route

# ================== MAIN SIMPLIFIED ASSIGNMENT FUNCTION ==================

def run_balance_assignment_simplified(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """
    Simplified balance assignment:
    1. Geographic clustering (group nearby users)
    2. Smart capacity matching (send right-sized cabs)
    3. Priority seat filling (add on-way users)
    """
    start_time = time.time()

    logger.info(f"🚀 Starting SIMPLIFIED balance assignment for source_id: {source_id}")

    try:
        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param, choice)

        # Algorithm-level caching check
        db_name = source_id if source_id and source_id != "1" else data.get("db", "default")
        cached_result = None

        if ALGORITHM_CACHE_AVAILABLE:
            try:
                # Initialize cache for this algorithm
                cache = get_algorithm_cache(db_name, "balance")

                # Generate current data signature
                current_signature = cache.generate_data_signature(data, {
                    'parameter': parameter,
                    'string_param': string_param,
                    'choice': choice,
                    'algorithm': 'simplified_balance_assignment'
                })

                # Check for cached result
                cached_result = cache.get_cached_result(current_signature)

                if cached_result is not None:
                    logger.info("⚡ FAST RESPONSE: Using cached algorithm result")
                    cached_result['_execution_time'] = 0.001  # Cache hit time
                    cached_result['_cache_hit'] = True
                    return cached_result

            except Exception as e:
                logger.error(f"Cache system error: {e} - proceeding with algorithm execution")

        users = data.get('users', [])
        if not users:
            logger.warning("⚠️ No users found")
            from algorithm.response.response_builder import build_standard_response
            return build_standard_response(
                status="true",
                execution_time=0.0,
                routes=[],
                unassigned_users=[],
                unassigned_drivers=[],
                optimization_mode="simplified_balance",
                parameter=parameter
            )

        # Get all drivers
        all_drivers = []
        if "drivers" in data:
            all_drivers.extend(data["drivers"].get("driversUnassigned", []))
            all_drivers.extend(data["drivers"].get("driversAssigned", []))
        else:
            all_drivers.extend(data.get("driversUnassigned", []))
            all_drivers.extend(data.get("driversAssigned", []))

        if not all_drivers:
            logger.warning("⚠️ No drivers available")
            from algorithm.response.response_builder import build_standard_response
            unassigned_users = _convert_users_to_unassigned_format(users)
            return build_standard_response(
                status="true",
                execution_time=0.0,
                routes=[],
                unassigned_users=unassigned_users,
                unassigned_drivers=[],
                optimization_mode="simplified_balance",
                parameter=parameter
            )

        logger.info(f"📥 Data loaded - Users: {len(users)}, Drivers: {len(all_drivers)}")

        # Extract office coordinates and prepare data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        user_df, driver_df = prepare_user_driver_dataframes(data)

        logger.info(f"📊 Data prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # STEP 1: Geographic clustering
        user_df = cluster_users_by_proximity(user_df, office_lat, office_lon)

        # STEP 2: Smart capacity matching and initial assignment
        routes = []
        assigned_user_ids = set()
        used_driver_ids = set()
        available_drivers = driver_df.copy()

        for cluster_id in user_df['geo_cluster'].unique():
            cluster_users = user_df[user_df['geo_cluster'] == cluster_id]

            if len(cluster_users) == 0:
                continue

            logger.info(f"🏘️ Processing cluster {cluster_id}: {len(cluster_users)} users")

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
                    'capacity': int(assigned_driver['capacity']),
                    'latitude': float(assigned_driver['latitude']),
                    'longitude': float(assigned_driver['longitude']),
                    'assigned_users': []
                }

                # Add all cluster users to the route
                for _, user in cluster_users.iterrows():
                    user_data = {
                        'user_id': str(user['user_id']),
                        'lat': float(user['latitude']),
                        'latitude': float(user['latitude']),
                        'lng': float(user['longitude']),
                        'longitude': float(user['longitude']),
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

        logger.info(f"✅ Initial assignment: {len(routes)} routes, {len(assigned_user_ids)} users assigned")

        # STEP 3: Advanced geographic optimization with user reallocation
        routes = optimize_geographic_distribution(routes, office_lat, office_lon, driver_df)

        # STEP 3.1: User swapping for geographic separation
        routes = swap_users_for_geographic_separation(routes)

        # STEP 3.5: Direction-aware route merging for small routes
        routes = merge_small_routes_with_nearby(routes, office_lat, office_lon)

        # STEP 4: Priority-based seat filling with cluster ownership check
        unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
        if not unassigned_users_df.empty:
            routes, filled_ids = fill_remaining_seats_with_cluster_check(routes, unassigned_users_df, office_lat, office_lon)
            assigned_user_ids.update(filled_ids)

        # STEP 4.5: Final hard-consolidation for nearby users
        routes = apply_hard_consolidation(routes, office_lat, office_lon)

        # STEP 5: Final results
        execution_time = time.time() - start_time

        # Prepare unassigned users
        remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        unassigned_users = []
        for _, user in remaining_users.iterrows():
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'latitude': float(user['latitude']),
                'lng': float(user['longitude']),
                'longitude': float(user['longitude']),
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

        logger.info(f"✅ Simplified balance assignment complete in {execution_time:.2f}s")
        logger.info(f"📊 Final: {len(routes)} routes, {len(assigned_user_ids)} users assigned, {len(unassigned_users)} unassigned")

        # Apply optimal pickup ordering to routes if available
        if ORDERING_AVAILABLE and routes:
            try:
                logger.info(f"Applying optimal pickup ordering to {len(routes)} routes")
                office_lat, office_lon = extract_office_coordinates(data)

                # Extract db name from source_id parameter
                db_name = source_id if source_id and source_id != "1" else data.get("db", "default")

                logger.info(f"Using company coordinates: {office_lat}, {office_lon} for db: {db_name}")

                routes = apply_route_ordering(routes, office_lat, office_lon, db_name=db_name, algorithm_name="balance")
                logger.info("Optimal pickup ordering applied successfully")
            except Exception as e:
                logger.error(f"Failed to apply optimal ordering: {e}")
                # Continue with routes without optimal ordering

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

        # Save result to algorithm cache if available
        if ALGORITHM_CACHE_AVAILABLE and cached_result is None:
            try:
                cache = get_algorithm_cache(db_name, "balance")

                # Regenerate signature for cache storage
                current_signature = cache.generate_data_signature(data, {
                    'parameter': parameter,
                    'string_param': string_param,
                    'choice': choice,
                    'algorithm': 'simplified_balance_assignment'
                })

                # Save the complete result to cache
                cache_result = {
                    "status": "true",
                    "execution_time": execution_time,
                    "company": company_info,
                    "shift": shift_info,
                    "data": enhanced_routes,
                    "unassignedUsers": enhanced_unassigned_users,
                    "unassignedDrivers": enhanced_unassigned_drivers,
                    "clustering_analysis": {"method": "simplified_geographic", "clusters": len(user_df['geo_cluster'].unique())},
                    "optimization_mode": "simplified_balance",
                    "parameter": parameter,
                    "_cache_metadata": {
                        'cached': True,
                        'cache_timestamp': time.time(),
                        'data_signature': current_signature
                    }
                }

                cache.save_result_to_cache(cache_result, current_signature)
                logger.info("💾 Algorithm result saved to cache for future use")

            except Exception as e:
                logger.error(f"Failed to save result to cache: {e}")

        # Import the standardized response builder
        from algorithm.response.response_builder import (
            build_standard_response,
            save_standardized_response,
            log_response_metrics
        )

        # Build standardized response (clustering_analysis is removed as per new standards)
        result = build_standard_response(
            status="true",
            execution_time=execution_time,
            routes=enhanced_routes,
            unassigned_users=enhanced_unassigned_users,
            unassigned_drivers=enhanced_unassigned_drivers,
            optimization_mode="simplified_balance",
            parameter=parameter,
            company=company_info,
            shift=shift_info,
            string_param=string_param,
            choice=choice
        )

        # Save standardized response
        save_standardized_response(result, "drivers_and_routes.json")

        # Log metrics for monitoring
        log_response_metrics(result, "simplified_balance")

        return result

    except Exception as e:
        logger.error(f"Simplified assignment failed: {e}", exc_info=True)
        from algorithm.response.response_builder import create_error_response
        return create_error_response(
            error_message=f"Simplified assignment failed: {e}",
            execution_time=time.time() - start_time,
            optimization_mode="simplified_balance",
            parameter=parameter,
            string_param=string_param,
            choice=choice
        )

# Entry point function
def run_assignment_balance(source_id: str, parameter: int = 1, string_param: str = "", choice: str = ""):
    """Entry point that uses the simplified version"""
    return run_balance_assignment_simplified(source_id, parameter, string_param, choice)