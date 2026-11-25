"""
RoutingChain — ordered engines with an explicit, observable fallback order.

Unlike the codebase's usual silent fallbacks, the chain records WHICH engine
answered every lookup (per-entry in the disk cache, plus aggregate stats), so
a degraded run ("everything is haversine") is visible instead of hidden.
"""

import threading
from typing import List, Optional, Tuple

from .base import HaversineEngine, RoutingEngine, haversine_km, validate_road_distance


class RoutingChain:
    def __init__(self, engines: List[RoutingEngine], cache=None,
                 max_ratio: float = 3.0):
        # Engines in priority order; a haversine engine must always be last.
        if not engines or not isinstance(engines[-1], HaversineEngine):
            engines = list(engines) + [HaversineEngine()]
        self.engines = engines
        self.cache = cache
        self.max_ratio = max_ratio
        self._lock = threading.RLock()

    # -- single pair ----------------------------------------------------------

    def point_distance_km(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """Road distance km for one ordered pair. Never returns None: the
        final fallback is haversine."""
        if self.cache is not None:
            hit = self.cache.get(lat1, lon1, lat2, lon2)
            if hit is not None:
                return hit["km"]

        straight = haversine_km(lat1, lon1, lat2, lon2)
        for engine in self.engines:
            km = engine.point_distance_km(lat1, lon1, lat2, lon2)
            if km is not None:
                km = validate_road_distance(km, straight, self.max_ratio)
                if self.cache is not None:
                    self.cache.put(lat1, lon1, lat2, lon2, km, engine.name)
                return km
        return straight

    # -- matrix -----------------------------------------------------------------

    def matrix_km(self, coords: List[Tuple[float, float]]) -> List[List[float]]:
        """NxN matrix. Cells the best available engine couldn't route fall
        back row-wise to the next engine, ultimately haversine."""
        n = len(coords)
        if n == 0:
            return []
        result = [[None] * n for _ in range(n)]
        for engine in self.engines:
            missing = [(i, j) for i in range(n) for j in range(n)
                       if result[i][j] is None]
            if not missing:
                break
            if engine.name == "haversine":
                matrix = engine.matrix_km(coords)
            else:
                # Only spend elements where nothing filled the cell yet —
                # but engines answer in full matrices, so just ask once and
                # accept the element cost for cells that get filled.
                matrix = engine.matrix_km(coords)
            for i in range(n):
                for j in range(n):
                    if result[i][j] is None and matrix[i][j] is not None:
                        result[i][j] = validate_road_distance(
                            matrix[i][j],
                            haversine_km(*coords[i], *coords[j]),
                            self.max_ratio)
        return result

    # -- introspection -----------------------------------------------------------

    def external_engines(self) -> List[RoutingEngine]:
        """Engines other than the haversine fallback."""
        return [e for e in self.engines if e.name != "haversine"]

    def has_real_engine(self) -> bool:
        """True if any non-haversine engine is up. Drives the provider choice:
        no external engine -> keep the GraphML road network."""
        return any(e.is_available() for e in self.external_engines())

    def best_engine_name(self) -> str:
        for engine in self.engines:
            if engine.name == "haversine" or engine.is_available():
                return engine.name
        return "haversine"

    def get_stats(self) -> List[dict]:
        return [e.get_stats() for e in self.engines]

    def save_cache(self):
        if self.cache is not None:
            self.cache.save()
