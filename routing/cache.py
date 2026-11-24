"""
Disk-backed point-distance cache for routing engines.

Follows the repo's existing JSON-cache pattern (ordering/cache/, routesCache/):
a JSON file under routing/cache/, loaded lazily, written through under a lock.
Values record which engine produced them so the chain can report honest stats
and so a later OSRM deployment doesn't silently reuse haversine fallbacks.

Thread-safe: road-aware mode queries the network from worker threads.
"""

import atexit
import json
import os
import threading
from typing import Optional, Tuple

CacheKey = Tuple[float, float, float, float]


class PointDistanceCache:
    def __init__(self, path: str, max_entries: int = 50000):
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._data = {}          # "lat1,lon1|lat2,lon2" -> {"km":..., "engine":...}
        self._dirty = False
        self._loaded = False
        atexit.register(self._flush_on_exit)

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self._data = raw
        except (OSError, json.JSONDecodeError) as e:
            # Cache is a convenience, never a dependency — start empty.
            print(f"routing cache: starting empty ({e})")
        self._loaded = True

    @staticmethod
    def _key(lat1, lon1, lat2, lon2) -> str:
        # 5 decimals ~= 1.1 m: stable across runs, fine for road estimates.
        return f"{lat1:.5f},{lon1:.5f}|{lat2:.5f},{lon2:.5f}"

    def get(self, lat1, lon1, lat2, lon2) -> Optional[dict]:
        with self._lock:
            self._ensure_loaded()
            return self._data.get(self._key(lat1, lon1, lat2, lon2))

    def put(self, lat1, lon1, lat2, lon2, km: float, engine: str):
        with self._lock:
            self._ensure_loaded()
            if len(self._data) >= self.max_entries:
                # Cheap eviction: drop the oldest half by insertion order.
                for k in list(self._data.keys())[: self.max_entries // 2]:
                    self._data.pop(k, None)
            self._data[self._key(lat1, lon1, lat2, lon2)] = {
                "km": round(float(km), 4), "engine": engine}
            self._dirty = True

    def save(self):
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._data, f)
                os.replace(tmp, self.path)
            except OSError as e:
                print(f"routing cache: save failed ({e})")

    def _flush_on_exit(self):
        try:
            self.save()
        except Exception:
            pass

    def size(self) -> int:
        with self._lock:
            self._ensure_loaded()
            return len(self._data)
