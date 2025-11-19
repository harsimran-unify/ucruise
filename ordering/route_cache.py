"""
Route Cache Management System for Smart Route Ordering
Handles caching of Google API optimized pickup sequences to reduce API costs
"""

import json
import hashlib
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from logger import get_logger

logger = get_logger()


class RouteCache:
    """Manages caching of optimized route pickup sequences with company and algorithm-specific persistent storage"""

    def __init__(self, db_name: str, algorithm_name: str = "base", cache_dir: str = "ordering/cache"):
        self.db_name = db_name
        self.algorithm_name = algorithm_name
        self.cache_dir = cache_dir
        self.company_cache_dir = os.path.join(cache_dir, db_name)
        self.algorithm_cache_dir = os.path.join(self.company_cache_dir, algorithm_name)
        self.cache_file = os.path.join(self.algorithm_cache_dir, "route_cache.json")

        # Create company and algorithm cache directories if they don't exist
        os.makedirs(self.company_cache_dir, exist_ok=True)
        os.makedirs(self.algorithm_cache_dir, exist_ok=True)

        self.cache_data = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from company-specific file or create new cache structure"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    logger.info(f"Loaded route cache for company '{self.db_name}' algorithm '{self.algorithm_name}' from {self.cache_file}")
                    return cache_data
            else:
                logger.info(f"Creating new route cache file for company '{self.db_name}' algorithm '{self.algorithm_name}'")
                return {"route_cache": {}, "metadata": {"last_cache_update": None, "total_cached_routes": 0, "company_db": self.db_name, "algorithm": self.algorithm_name}}
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error loading cache file {self.cache_file}: {e}. Creating new cache for company '{self.db_name}' algorithm '{self.algorithm_name}'.")
            return {"route_cache": {}, "metadata": {"last_cache_update": None, "total_cached_routes": 0, "company_db": self.db_name, "algorithm": self.algorithm_name}}

    def _save_cache(self):
        """Save cache to file"""
        try:
            self.cache_data["metadata"]["last_cache_update"] = datetime.now().isoformat()
            self.cache_data["metadata"]["total_cached_routes"] = len(self.cache_data["route_cache"])

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache file {self.cache_file}: {e}")

    def _generate_route_fingerprint(self, driver_id: str, user_ids: List[str]) -> str:
        """Generate unique fingerprint for a route composition"""
        # Sort user IDs to ensure consistent fingerprint regardless of user order
        sorted_user_ids = sorted(user_ids)
        route_string = f"{driver_id}_{'_'.join(sorted_user_ids)}"
        return hashlib.md5(route_string.encode()).hexdigest()

    def get_cached_order(self, driver_id: str, user_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached pickup order for a route if available (persistent cache - no expiration)"""
        fingerprint = self._generate_route_fingerprint(driver_id, user_ids)

        if fingerprint in self.cache_data["route_cache"]:
            cached_route = self.cache_data["route_cache"][fingerprint]
            logger.debug(f"Cache hit for route {driver_id} with {len(user_ids)} users")
            return cached_route

        return None

    def cache_order(self, driver_id: str, user_ids: List[str], ordered_user_ids: List[str],
                   total_distance_km: float, total_duration_min: float, algorithm: str = "google_directions_api"):
        """Cache optimized pickup order for a route"""
        fingerprint = self._generate_route_fingerprint(driver_id, user_ids)

        cache_entry = {
            "driver_id": driver_id,
            "user_ids": sorted(user_ids),
            "ordered_user_ids": ordered_user_ids,
            "pickup_sequence": [ordered_user_ids.index(uid) + 1 for uid in user_ids],
            "total_distance_km": round(total_distance_km, 2),
            "total_duration_min": round(total_duration_min, 1),
            "calculated_date": datetime.now().isoformat(),
            "algorithm": algorithm
        }

        self.cache_data["route_cache"][fingerprint] = cache_entry
        self._save_cache()

        logger.info(f"Cached optimized order for route {driver_id} with {len(user_ids)} users "
                   f"(distance: {total_distance_km:.2f}km, duration: {total_duration_min:.1f}min)")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information for monitoring"""
        total_routes = len(self.cache_data["route_cache"])
        last_update = self.cache_data["metadata"].get("last_cache_update")

        return {
            "company_db": self.db_name,
            "total_cached_routes": total_routes,
            "last_cache_update": last_update,
            "cache_file": self.cache_file,
            "cache_directory": self.company_cache_dir
        }

    
    def clear_cache(self):
        """Clear all cache entries for this company"""
        self.cache_data = {"route_cache": {}, "metadata": {"last_cache_update": None, "total_cached_routes": 0, "company_db": self.db_name}}
        self._save_cache()
        logger.info(f"Route cache cleared for company '{self.db_name}'")

    def get_changed_routes(self, current_routes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Compare current routes with cached routes and identify which ones need recalculation

        Returns:
            Tuple of (cached_routes, routes_to_recalculate)
        """
        cached_routes = []
        routes_to_recalculate = []

        for route in current_routes:
            driver_id = route.get("driver_id")
            user_ids = [user.get("user_id") for user in route.get("assigned_users", [])]

            cached_order = self.get_cached_order(driver_id, user_ids)

            if cached_order:
                # Route exists in cache, add cached ordering
                route_with_cache = route.copy()
                route_with_cache["cached_order"] = cached_order
                cached_routes.append(route_with_cache)
            else:
                # Route not in cache or expired, needs recalculation
                routes_to_recalculate.append(route)

        logger.info(f"Route comparison: {len(cached_routes)} cached, {len(routes_to_recalculate)} need recalculation")
        return cached_routes, routes_to_recalculate


# Global cache instances per company
_route_caches = {}


def get_route_cache(db_name: str, algorithm_name: str = "base") -> RouteCache:
    """Get company and algorithm-specific route cache instance"""
    global _route_caches
    cache_key = f"{db_name}_{algorithm_name}"
    if cache_key not in _route_caches:
        _route_caches[cache_key] = RouteCache(db_name, algorithm_name)
    return _route_caches[cache_key]


def apply_cached_orders(routes: List[Dict[str, Any]], db_name: str, algorithm_name: str = "base") -> List[Dict[str, Any]]:
    """Apply cached pickup orders to routes using company and algorithm-specific cache"""
    if not routes:
        return routes

    cache = get_route_cache(db_name, algorithm_name)
    cached_routes, routes_to_recalculate = cache.get_changed_routes(routes)

    # Apply cached orders to routes that have them
    enhanced_routes = []

    for route in cached_routes:
        if "cached_order" in route:
            enhanced_route = apply_order_to_route(route, route["cached_order"])
            enhanced_routes.append(enhanced_route)

    # Return routes without cached orders (will need Google API optimization)
    enhanced_routes.extend(routes_to_recalculate)

    return enhanced_routes


def apply_order_to_route(route: Dict[str, Any], cached_order: Dict[str, Any]) -> Dict[str, Any]:
    """Apply cached pickup order to a route"""
    enhanced_route = route.copy()
    ordered_user_ids = cached_order.get("ordered_user_ids", [])

    # Create a mapping of user_id to user data
    user_map = {user["user_id"]: user for user in route.get("assigned_users", [])}

    # Reorder users according to cached order
    ordered_users = []
    for i, user_id in enumerate(ordered_user_ids):
        if user_id in user_map:
            user_data = user_map[user_id].copy()
            user_data["pickup_order"] = i + 1
            ordered_users.append(user_data)

    enhanced_route["assigned_users"] = ordered_users
    enhanced_route["route_distance_km"] = cached_order.get("total_distance_km", 0)
    enhanced_route["route_duration_min"] = cached_order.get("total_duration_min", 0)
    enhanced_route["ordering_source"] = "cached"

    return enhanced_route