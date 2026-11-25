"""
Routing engine interface (Phase 1 of the solver-migration plan).

A RoutingEngine answers road-distance questions for the tricity area.
Engines are chained by RoutingChain (routing/chain.py): OSRM first, the
Google Routes matrix API second, haversine always last. All public methods
return kilometers; engines convert from their native units internally.
"""

import math
import threading
import time
from typing import List, Optional, Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers (same convention as the legacy
    algorithm.base.base.haversine_distance this module replaced)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def validate_road_distance(road_km: float, straight_km: float,
                           max_ratio: float = 3.0) -> float:
    """Reject road distances that are implausible vs the straight line.
    Mirrors RoadNetwork._validate_distance_result."""
    if road_km is None or road_km <= 0:
        return straight_km
    if straight_km > 0 and road_km / straight_km > max_ratio:
        return straight_km
    return road_km


class RoutingEngine:
    """Base class for road-distance engines."""

    name = "base"

    def __init__(self):
        self._available: Optional[bool] = None
        self._probed_at = 0.0
        self._probe_lock = threading.Lock()
        # Observability counters (thread-safe reads are approximate; exact
        # values only matter for reporting, not correctness).
        self._stats_lock = threading.Lock()
        self.stats = {"requests": 0, "successes": 0, "elements": 0, "failures": 0}

    # -- subclass contract -------------------------------------------------

    def _point_distance_km(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> Optional[float]:
        """Road distance km for one ordered pair, or None if unroutable."""
        raise NotImplementedError

    def _matrix_km(self, coords: List[Tuple[float, float]]) -> List[List[Optional[float]]]:
        """Full NxN distance matrix in km (None where unroutable)."""
        raise NotImplementedError

    def _probe_available(self) -> bool:
        """Cheap availability check, no caching required."""
        raise NotImplementedError

    # -- public API (availability probing + stats live here) ----------------

    def is_available(self, probe_ttl_s: float = 30.0) -> bool:
        now = time.time()
        if self._available is not None and now - self._probed_at < probe_ttl_s:
            return self._available
        with self._probe_lock:
            # Re-check under the lock: another thread may have probed already.
            if self._available is not None and time.time() - self._probed_at < probe_ttl_s:
                return self._available
            try:
                self._available = bool(self._probe_available())
            except Exception:
                self._available = False
            self._probed_at = time.time()
            return self._available

    def point_distance_km(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> Optional[float]:
        if not self.is_available():
            return None
        with self._stats_lock:
            self.stats["requests"] += 1
        try:
            km = self._point_distance_km(lat1, lon1, lat2, lon2)
        except Exception:
            km = None
        with self._stats_lock:
            if km is None:
                self.stats["failures"] += 1
            else:
                self.stats["successes"] += 1
        return km

    def matrix_km(self, coords: List[Tuple[float, float]]) -> List[List[Optional[float]]]:
        if not coords:
            return []
        if not self.is_available():
            return [[None] * len(coords) for _ in coords]
        with self._stats_lock:
            self.stats["requests"] += 1
            self.stats["elements"] += len(coords) * len(coords)
        try:
            return self._matrix_km(coords)
        except Exception:
            return [[None] * len(coords) for _ in coords]

    def get_stats(self) -> dict:
        with self._stats_lock:
            return {"name": self.name, **dict(self.stats)}


class HaversineEngine(RoutingEngine):
    """Straight-line fallback. Always available; not a road estimate."""

    name = "haversine"

    def _probe_available(self) -> bool:
        return True

    def _point_distance_km(self, lat1, lon1, lat2, lon2):
        return haversine_km(lat1, lon1, lat2, lon2)

    def _matrix_km(self, coords):
        return [
            [haversine_km(a[0], a[1], b[0], b[1]) for b in coords]
            for a in coords
        ]
