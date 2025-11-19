"""
Order Integration Module

Integrates smart route ordering with Google API and caching into all algorithms.
Provides a unified interface for applying optimal pickup sequences.
"""

import json
import os
from typing import List, Dict, Any, Tuple
from logger import get_logger

from .route_cache import RouteCache, apply_cached_orders
from .google_optimizer import get_google_optimizer, batch_optimize_routes

logger = get_logger()


class OrderIntegration:
    """Manages integration of smart ordering with route assignment algorithms"""

    def __init__(self, config: Dict[str, Any] = None, db_name: str = None, algorithm_name: str = "base"):
        self.config = config or self._load_config()
        self.db_name = db_name or "default"
        self.algorithm_name = algorithm_name
        self.cache_enabled = self.config.get("route_ordering", {}).get("enable_route_cache", True)
        self.google_enabled = self.config.get("route_ordering", {}).get("enable_google_optimization", True)
        self.fallback_enabled = self.config.get("route_ordering", {}).get("fallback_to_haversine", True)

        # Initialize components
        self.cache = RouteCache(self.db_name, self.algorithm_name) if self.cache_enabled else None
        self.google_optimizer = get_google_optimizer() if self.google_enabled else None

        logger.info(f"Order integration initialized for company '{self.db_name}' algorithm '{self.algorithm_name}' - Cache: {self.cache_enabled}, "
                   f"Google: {self.google_enabled}, Fallback: {self.fallback_enabled}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json"""
        try:
            # Find config file in project root directory
            # Get the directory containing the current file (ordering/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to project root
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, 'config.json')

            with open(config_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load config.json: {e}. Using defaults.")
            return {"route_ordering": {}}

    def apply_optimal_ordering(self, routes: List[Dict[str, Any]], office_lat: float, office_lon: float) -> List[Dict[str, Any]]:
        """
        Apply optimal pickup ordering to routes using Google API and/or cache

        Args:
            routes: List of route dictionaries with assigned_users
            office_lat, office_lon: Office coordinates

        Returns:
            Enhanced routes with pickup_order applied to each user
        """
        if not routes:
            logger.info("No routes to optimize")
            return routes

        logger.info(f"Applying optimal ordering to {len(routes)} routes for company '{self.db_name}'")

        # Step 1: Apply cached orders where available
        if self.cache_enabled:
            routes = apply_cached_orders(routes, self.db_name, self.algorithm_name)
            logger.info("Applied cached orders to routes")

        # Step 2: Identify routes that need Google API optimization
        routes_needing_optimization = [
            route for route in routes
            if not route.get("ordering_source") or route.get("ordering_source") != "cached"
        ]

        if not routes_needing_optimization:
            logger.info("All routes have cached orders, no Google API calls needed")
            return routes

        # Step 3: Apply Google API optimization if enabled
        if self.google_enabled and self.google_optimizer and self.google_optimizer.is_enabled():
            try:
                optimized_routes, failed_routes = batch_optimize_routes(
                    routes_needing_optimization, office_lat, office_lon,
                    cache_results=self.cache_enabled, db_name=self.db_name, algorithm_name=self.algorithm_name
                )

                logger.info(f"Google API optimization: {len(optimized_routes)} successful, {len(failed_routes)} failed")

                # Step 4: Apply fallback ordering for failed routes if enabled
                if self.fallback_enabled and failed_routes:
                    logger.info(f"Applying fallback ordering to {len(failed_routes)} failed routes")
                    fallback_routes = self._apply_fallback_ordering(failed_routes, office_lat, office_lon)
                    optimized_routes.extend(fallback_routes)

                # Combine cached routes and newly optimized routes
                cached_routes = [route for route in routes if route.get("ordering_source") == "cached"]
                final_routes = cached_routes + optimized_routes

                return final_routes

            except Exception as e:
                logger.error(f"Google API batch optimization failed: {e}")
                # Fall back to all routes with fallback ordering
                if self.fallback_enabled:
                    logger.info("Applying fallback ordering to all routes due to API failure")
                    return self._apply_fallback_ordering(routes, office_lat, office_lon)
                else:
                    logger.warning("No fallback available, returning routes without ordering")
                    return routes

        else:
            # Google API not enabled or not configured, use fallback
            if self.fallback_enabled:
                logger.info("Google API not enabled, applying fallback ordering to all routes")
                return self._apply_fallback_ordering(routes, office_lat, office_lon)
            else:
                logger.warning("Google API not enabled and fallback disabled, returning routes without ordering")
                return routes

    def _apply_fallback_ordering(self, routes: List[Dict[str, Any]], office_lat: float, office_lon: float) -> List[Dict[str, Any]]:
        """
        Apply fallback ordering using haversine distance calculations

        Args:
            routes: List of route dictionaries
            office_lat, office_lon: Office coordinates

        Returns:
            Routes with fallback pickup_order applied
        """
        logger.info(f"Applying fallback ordering to {len(routes)} routes")

        enhanced_routes = []

        for route in routes:
            try:
                enhanced_route = self._optimize_single_route_fallback(route, office_lat, office_lon)
                enhanced_routes.append(enhanced_route)
            except Exception as e:
                logger.error(f"Failed to apply fallback ordering to route {route.get('driver_id')}: {e}")
                # Return route without ordering
                route_copy = route.copy()
                route_copy["ordering_source"] = "failed"
                enhanced_routes.append(route_copy)

        return enhanced_routes

    def _optimize_single_route_fallback(self, route: Dict[str, Any], office_lat: float, office_lon: float) -> Dict[str, Any]:
        """
        Optimize a single route using fallback haversine-based ordering

        Uses nearest neighbor algorithm with haversine distances for reasonable pickup sequencing
        """
        driver_id = route.get("driver_id")
        driver_lat = route.get("latitude")
        driver_lng = route.get("longitude")
        assigned_users = route.get("assigned_users", [])

        if not assigned_users:
            return route

        if len(assigned_users) == 1:
            # Single user - simple ordering
            enhanced_route = route.copy()
            user = assigned_users[0].copy()
            user["pickup_order"] = 1
            enhanced_route["assigned_users"] = [user]
            enhanced_route["ordering_source"] = "fallback_single"
            return enhanced_route

        # Multiple users - apply nearest neighbor algorithm starting from OFFICE
        enhanced_route = route.copy()
        current_pos = (office_lat, office_lon)  # Start from OFFICE
        remaining_users = assigned_users.copy()
        ordered_users = []
        pickup_order = 1
        total_distance = 0

        while remaining_users:
            # Find nearest user to current position
            nearest_user = None
            nearest_distance = float('inf')
            nearest_index = -1

            for i, user in enumerate(remaining_users):
                user_pos = (user.get("lat"), user.get("lng"))
                distance = self._haversine_distance(
                    current_pos[0], current_pos[1],
                    user_pos[0], user_pos[1]
                )

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_user = user
                    nearest_index = i

            if nearest_user:
                # Add user to ordered list
                user_with_order = nearest_user.copy()
                user_with_order["pickup_order"] = pickup_order
                ordered_users.append(user_with_order)

                # Update current position and total distance
                total_distance += nearest_distance
                current_pos = (nearest_user.get("lat"), nearest_user.get("lng"))

                # Remove from remaining users
                remaining_users.pop(nearest_index)
                pickup_order += 1

        # Add distance from last user back to office (complete round trip)
        if ordered_users:
            total_distance += self._haversine_distance(
                current_pos[0], current_pos[1],
                office_lat, office_lon
            )

        # Update route with ordered users
        enhanced_route["assigned_users"] = ordered_users
        enhanced_route["route_distance_km"] = round(total_distance, 2)
        enhanced_route["route_duration_min"] = round(total_distance / 0.5 * 60, 1)  # Assume 30 km/h average
        enhanced_route["ordering_source"] = "fallback_nearest_neighbor"

        logger.debug(f"Fallback ordering for route {driver_id}: {len(ordered_users)} users, "
                    f"{total_distance:.2f}km")

        return enhanced_route

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
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

    def get_ordering_stats(self) -> Dict[str, Any]:
        """Get statistics about ordering system usage"""
        stats = {
            "cache_enabled": self.cache_enabled,
            "google_enabled": self.google_enabled,
            "fallback_enabled": self.fallback_enabled
        }

        if self.cache:
            stats["cache_stats"] = self.cache.get_cache_stats()

        if self.google_optimizer:
            stats["google_api_stats"] = self.google_optimizer.get_usage_stats()

        return stats


# Global integration instances per company and algorithm
_order_integrations = {}


def get_order_integration(config: Dict[str, Any] = None, db_name: str = None, algorithm_name: str = "base") -> OrderIntegration:
    """Get company and algorithm-specific order integration instance"""
    global _order_integrations
    db_name = db_name or "default"
    integration_key = f"{db_name}_{algorithm_name}"

    if integration_key not in _order_integrations:
        _order_integrations[integration_key] = OrderIntegration(config, db_name, algorithm_name)
    return _order_integrations[integration_key]


def apply_route_ordering(routes: List[Dict[str, Any]], office_lat: float, office_lon: float,
                        config: Dict[str, Any] = None, db_name: str = None, algorithm_name: str = "base") -> List[Dict[str, Any]]:
    """
    Convenience function to apply optimal ordering to routes

    Args:
        routes: List of route dictionaries
        office_lat, office_lon: Office coordinates
        config: Optional configuration dictionary
        db_name: Company database name for caching
        algorithm_name: Algorithm name for cache separation

    Returns:
        Enhanced routes with pickup_order applied
    """
    integration = get_order_integration(config, db_name, algorithm_name)
    return integration.apply_optimal_ordering(routes, office_lat, office_lon)