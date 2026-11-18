"""
Google Maps API Integration for Optimal Route Ordering
Uses Google Directions API to calculate optimal pickup sequences
"""

import os
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from logger import get_logger

logger = get_logger()

# Load environment variables
load_dotenv()


class GoogleRouteOptimizer:
    """Google Maps API integration for optimal pickup sequencing"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.base_url = os.getenv("GOOGLE_DIRECTIONS_API_BASE_URL", "https://maps.googleapis.com/maps/api/directions/json")
        self.rate_limit = int(os.getenv("GOOGLE_API_RATE_LIMIT", "100"))
        self.last_request_time = 0
        self.request_count = 0

        if not self.api_key or self.api_key == "your_google_maps_api_key_here":
            logger.warning("Google Maps API key not configured properly in .env file")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Google Maps API optimizer initialized")

    def _rate_limit_delay(self):
        """Implement rate limiting to avoid API quota exceeded errors"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        # Ensure minimum delay between requests (1 second / rate_limit_per_second)
        min_delay = 1.0 / (self.rate_limit / 60.0)  # Convert per-minute rate to per-second

        if time_since_last_request < min_delay:
            sleep_time = min_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

    def optimize_pickup_sequence(self, driver_pos: Tuple[float, float],
                                user_positions: List[Tuple[float, float]],
                                office_pos: Tuple[float, float]) -> Dict[str, Any]:
        """
        Calculate optimal pickup sequence using Google Directions API
        Route: Office → User 1 → User 2 → ... → Office (round trip)

        Args:
            driver_pos: (latitude, longitude) of driver (not used for route calculation)
            user_positions: List of (latitude, longitude) for each user
            office_pos: (latitude, longitude) of office (start/end point)

        Returns:
            Dict containing ordered user indices, distances, durations, and route details
        """
        if not self.enabled:
            raise ValueError("Google Maps API not enabled or not configured")

        if len(user_positions) == 0:
            return {"ordered_indices": [], "total_distance_km": 0, "total_duration_min": 0}

        if len(user_positions) == 1:
            # Single user - simple route from office to user and back
            distance_km = self._haversine_distance(office_pos[0], office_pos[1],
                                                  user_positions[0][0], user_positions[0][1])
            distance_km += self._haversine_distance(user_positions[0][0], user_positions[0][1],
                                                  office_pos[0], office_pos[1])
            return {
                "ordered_indices": [0],
                "total_distance_km": round(distance_km, 2),
                "total_duration_min": round(distance_km / 0.5 * 60, 1),  # Assume 30 km/h average
                "route_details": []
            }

        self._rate_limit_delay()

        try:
            # Build waypoints for Google Directions API
            waypoints = []
            for lat, lng in user_positions:
                waypoints.append(f"{lat},{lng}")

            # Build request URL - Route starts and ends at OFFICE
            origin = f"{office_pos[0]},{office_pos[1]}"
            destination = f"{office_pos[0]},{office_pos[1]}"
            waypoints_str = "|".join(waypoints)

            url = (f"{self.base_url}?origin={origin}&destination={destination}"
                   f"&waypoints=optimize:true|{waypoints_str}"
                   f"&key={self.api_key}")

            logger.debug(f"Making Google API request with {len(user_positions)} waypoints")

            # Make API request
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "OK":
                error_msg = data.get("error_message", "Unknown error")
                logger.error(f"Google Directions API error: {data.get('status')} - {error_msg}")
                raise Exception(f"Google API error: {data.get('status')}")

            # Parse response
            route = data["routes"][0]
            leg = route["legs"][0]  # Since we optimize waypoints, we get one optimized leg

            total_distance_m = leg["distance"]["value"]
            total_duration_s = leg["duration"]["value"]
            total_distance_km = total_distance_m / 1000.0
            total_duration_min = total_duration_s / 60.0

            # Extract waypoint order from API response
            waypoint_order = route["waypoint_order"]

            # Build route details
            route_details = []
            steps = leg["steps"]
            for i, step in enumerate(steps):
                route_details.append({
                    "step": i,
                    "instruction": step["html_instructions"],
                    "distance_m": step["distance"]["value"],
                    "duration_s": step["duration"]["value"]
                })

            logger.info(f"Google API optimized route: {total_distance_km:.2f}km, {total_duration_min:.1f}min "
                       f"(order: {waypoint_order})")

            return {
                "ordered_indices": waypoint_order,
                "total_distance_km": round(total_distance_km, 2),
                "total_duration_min": round(total_duration_min, 1),
                "route_details": route_details,
                "api_response": data
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Google API request failed: {e}")
            raise Exception(f"Google API request failed: {e}")
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing Google API response: {e}")
            raise Exception(f"Error parsing Google API response: {e}")

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points (fallback)"""
        import math

        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Radius of earth in kilometers
        r = 6371
        return c * r

    def is_enabled(self) -> bool:
        """Check if Google API optimizer is enabled and configured"""
        return self.enabled

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "enabled": self.enabled,
            "api_key_configured": bool(self.api_key and self.api_key != "your_google_maps_api_key_here"),
            "request_count": self.request_count,
            "rate_limit": self.rate_limit
        }


# Global optimizer instance
_google_optimizer = None


def get_google_optimizer() -> GoogleRouteOptimizer:
    """Get global Google optimizer instance"""
    global _google_optimizer
    if _google_optimizer is None:
        _google_optimizer = GoogleRouteOptimizer()
    return _google_optimizer


def optimize_route_with_google(route: Dict[str, Any], office_lat: float, office_lon: float) -> Optional[Dict[str, Any]]:
    """
    Optimize a single route using Google Maps API

    Args:
        route: Route dictionary with driver_id and assigned_users
        office_lat, office_lon: Office coordinates

    Returns:
        Enhanced route with pickup_order and route details, or None if optimization fails
    """
    optimizer = get_google_optimizer()

    if not optimizer.is_enabled():
        logger.warning("Google optimizer not enabled, skipping route optimization")
        return None

    try:
        # Extract route information
        driver_id = route.get("driver_id")
        driver_lat = route.get("latitude")
        driver_lng = route.get("longitude")
        assigned_users = route.get("assigned_users", [])

        if not assigned_users:
            logger.warning(f"Route {driver_id} has no assigned users")
            return None

        # Prepare positions for API - Route starts and ends at OFFICE
        office_pos = (office_lat, office_lon)
        user_positions = [(user.get("lat"), user.get("lng")) for user in assigned_users]
        # Driver position not needed for route calculation, but keep for compatibility
        driver_pos = (driver_lat, driver_lng)

        # Call Google API - Office is both origin and destination
        result = optimizer.optimize_pickup_sequence(driver_pos, user_positions, office_pos)

        # Apply ordering to route
        enhanced_route = route.copy()
        ordered_indices = result["ordered_indices"]

        # Reorder users according to optimized sequence
        ordered_users = []
        for i, user_index in enumerate(ordered_indices):
            if 0 <= user_index < len(assigned_users):
                user_data = assigned_users[user_index].copy()
                user_data["pickup_order"] = i + 1
                ordered_users.append(user_data)

        enhanced_route["assigned_users"] = ordered_users
        enhanced_route["route_distance_km"] = result["total_distance_km"]
        enhanced_route["route_duration_min"] = result["total_duration_min"]
        enhanced_route["ordering_source"] = "google_api"
        enhanced_route["route_details"] = result.get("route_details", [])

        logger.info(f"Successfully optimized route {driver_id} with Google API "
                   f"({len(ordered_users)} users, {result['total_distance_km']:.2f}km)")

        return enhanced_route

    except Exception as e:
        logger.error(f"Failed to optimize route {route.get('driver_id')} with Google API: {e}")
        return None


def batch_optimize_routes(routes: List[Dict[str, Any]], office_lat: float, office_lon: float,
                         cache_results: bool = True, db_name: str = None, algorithm_name: str = "base") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Optimize multiple routes using Google API with caching

    Args:
        routes: List of route dictionaries
        office_lat, office_lon: Office coordinates
        cache_results: Whether to cache optimization results
        db_name: Company database name for caching
        algorithm_name: Algorithm name for cache separation

    Returns:
        Tuple of (optimized_routes, failed_routes)
    """
    from .route_cache import get_route_cache

    cache = get_route_cache(db_name, algorithm_name) if cache_results and db_name else None
    optimizer = get_google_optimizer()

    if not optimizer.is_enabled():
        logger.warning("Google optimizer not enabled, returning routes without optimization")
        return routes, []

    optimized_routes = []
    failed_routes = []

    logger.info(f"Starting batch optimization of {len(routes)} routes with Google API")

    for route in routes:
        try:
            optimized_route = optimize_route_with_google(route, office_lat, office_lon)

            if optimized_route:
                optimized_routes.append(optimized_route)

                # Cache the result
                if cache and cache_results:
                    driver_id = route.get("driver_id")
                    user_ids = [user.get("user_id") for user in route.get("assigned_users", [])]
                    ordered_user_ids = [user.get("user_id") for user in optimized_route.get("assigned_users", [])]

                    cache.cache_order(
                        driver_id=driver_id,
                        user_ids=user_ids,
                        ordered_user_ids=ordered_user_ids,
                        total_distance_km=optimized_route.get("route_distance_km", 0),
                        total_duration_min=optimized_route.get("route_duration_min", 0),
                        algorithm="google_directions_api"
                    )
            else:
                failed_routes.append(route)

        except Exception as e:
            logger.error(f"Unexpected error optimizing route {route.get('driver_id')}: {e}")
            failed_routes.append(route)

    logger.info(f"Batch optimization completed: {len(optimized_routes)} successful, {len(failed_routes)} failed")
    return optimized_routes, failed_routes