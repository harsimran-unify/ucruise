"""
Google Routes API engine — computeRouteMatrix (Distance Matrix v2).

Billed per ELEMENT (origin x destination pair), ~$5/1000 basic tier, with the
same GOOGLE_MAPS_API_KEY the project already uses for Directions. Every
element is counted in engine stats and bounded by an optional per-session
element budget so a misbehaving caller cannot burn quota silently.

Limits enforced here: <= 25 origins, <= 25 destinations, <= 100 elements per
request (Routes API matrix limits).
"""

import os
import threading
import time
from typing import List, Optional, Tuple

import requests

from .base import RoutingEngine

MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
FIELD_MASK = "distanceMeters,duration,status"
TRAVEL_MODE = "DRIVE"
MAX_ORIGINS = 25
MAX_DESTINATIONS = 25
MAX_ELEMENTS = 100


def _waypoint(lat: float, lon: float) -> dict:
    return {"waypoint": {"location": {"latLng": {
        "latitude": lat, "longitude": lon}}}}


class GoogleMatrixEngine(RoutingEngine):
    name = "google"

    def __init__(self, api_key: str, timeout_s: float = 10.0,
                 min_request_interval_s: float = 0.1,
                 element_budget: Optional[int] = 1000):
        super().__init__()
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.last_error: Optional[str] = None
        self._min_interval = min_request_interval_s
        self._last_request = 0.0
        self._rate_lock = threading.Lock()
        self._budget = element_budget
        self._elements_used = 0
        self._session = requests.Session()

    # -- internals ----------------------------------------------------------

    def _throttle(self):
        with self._rate_lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request = time.time()

    def _post(self, origins: List[dict], destinations: List[dict]) -> Optional[List[dict]]:
        for attempt in (0, 1):  # one retry on transient failures
            self._throttle()
            try:
                resp = self._session.post(
                    MATRIX_URL,
                    json={"origins": origins, "destinations": destinations,
                          "travelMode": TRAVEL_MODE},
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": self.api_key,
                        "X-Goog-FieldMask": FIELD_MASK,
                    },
                    timeout=self.timeout_s)
            except requests.RequestException as e:
                self.last_error = f"request failed: {e}"
                continue
            if resp.status_code == 200:
                return resp.json()
            self.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (429, 500, 503):
                time.sleep(0.5 * (attempt + 1))
                continue
            # 403/400: key or request problem — don't retry.
            return None
        return None

    def _probe_available(self) -> bool:
        if not self.api_key:
            return False
        return self._post([_waypoint(30.6810489, 76.7260711)],
                          [_waypoint(30.6820489, 76.7270711)]) is not None

    def _budget_ok(self, extra: int) -> bool:
        if self._budget is None:
            return True
        if self._elements_used + extra > self._budget:
            return False
        self._elements_used += extra
        return True

    def _parse_element(self, el: dict) -> Optional[float]:
        status = (el.get("status") or {})
        if status.get("code") in ("ROUTE_NOT_FOUND", "INVALID_ARGUMENT"):
            return None
        meters = el.get("distanceMeters")
        if meters is None:
            return None
        return meters / 1000.0

    # -- contract -----------------------------------------------------------

    def _point_distance_km(self, lat1, lon1, lat2, lon2) -> Optional[float]:
        if not self._budget_ok(1):
            return None
        elements = self._post([_waypoint(lat1, lon1)], [_waypoint(lat2, lon2)])
        if not elements:
            return None
        return self._parse_element(elements[0]) if elements else None

    def _matrix_km(self, coords: List[Tuple[float, float]]) -> List[List[Optional[float]]]:
        n = len(coords)
        result: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
        if not self._budget_ok(n * n):
            return result

        # Walk destinations in row-blocks so each request respects the three
        # limits simultaneously.
        dest_chunk = min(MAX_DESTINATIONS, MAX_ELEMENTS)
        for dst_start in range(0, n, dest_chunk):
            dst_idx = list(range(dst_start, min(dst_start + dest_chunk, n)))
            origins_per_req = min(MAX_ORIGINS, max(1, MAX_ELEMENTS // len(dst_idx)))
            for src_start in range(0, n, origins_per_req):
                src_idx = list(range(src_start, min(src_start + origins_per_req, n)))
                origins = [_waypoint(*coords[i]) for i in src_idx]
                destinations = [_waypoint(*coords[j]) for j in dst_idx]
                elements = self._post(origins, destinations)
                if not elements:
                    continue
                for el in elements:
                    try:
                        oi, di = el.get("originIndex", 0), el.get("destinationIndex", 0)
                        result[src_idx[oi]][dst_idx[di]] = self._parse_element(el)
                    except (IndexError, TypeError):
                        pass
        return result

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["elements_used_this_session"] = self._elements_used
        stats["element_budget"] = self._budget
        return stats


def engine_from_env(element_budget: Optional[int] = 1000) -> Optional[GoogleMatrixEngine]:
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    return GoogleMatrixEngine(api_key=key, element_budget=element_budget) if key else None
