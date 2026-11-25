"""
OSRM engine — self-hosted road distances via the OSRM HTTP API.

Setup (Docker), once a Chandigarh-region OSM extract exists, e.g. from
https://download.geofabrik.de/asia/india.html (or a smaller BBBike extract):

    docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract \
        -p /opt/car.lua /data/<region>.osm.pbf
    docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition \
        /data/<region>.osrm
    docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize \
        /data/<region>.osrm
    docker run -d -p 5001:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed \
        --algorithm mld /data/<region>.osrm

Then set OSRM_URL=http://localhost:5001 in .env (or rely on the default).
Free, local, milliseconds per query; no per-element billing.
"""

import os
import time
from typing import List, Optional, Tuple

import requests

from .base import RoutingEngine

# OSRM's default max-table-size is 100 coordinates per /table request.
MAX_TABLE_SIZE = 100


class OSRMEngine(RoutingEngine):
    name = "osrm"

    def __init__(self, base_url: str, timeout_s: float = 5.0):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    # -- internals ----------------------------------------------------------

    def _get(self, path: str) -> Optional[dict]:
        resp = self._session.get(self.base_url + path, timeout=self.timeout_s)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data.get("code") == "Ok" else None

    def _probe_available(self) -> bool:
        # Office coordinates — any two nearby points work; the request just
        # needs to prove the server answers.
        return self._get(
            "/route/v1/driving/76.7260711,30.6810489;76.7270711,30.6820489"
            "?overview=false&annotations=distance,duration") is not None

    def _point_distance_km(self, lat1, lon1, lat2, lon2) -> Optional[float]:
        data = self._get(
            f"/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
            "?overview=false&annotations=distance,duration")
        routes = (data or {}).get("routes") or []
        if not routes:
            return None
        return routes[0].get("distance", 0) / 1000.0

    def _matrix_km(self, coords: List[Tuple[float, float]]) -> List[List[Optional[float]]]:
        n = len(coords)
        result: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
        coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)

        # /table caps total coordinates per request; walk sources in chunks.
        chunk = max(1, MAX_TABLE_SIZE - n)
        for src_start in range(0, n, chunk):
            src_idx = list(range(src_start, min(src_start + chunk, n)))
            sources = ";".join(str(i) for i in range(len(src_idx)))
            destinations = ";".join(str(i) for i in range(n))
            data = self._get(
                f"/table/v1/driving/{coord_str}"
                f"?sources={sources}&destinations={destinations}"
                "&annotations=distance")
            distances = (data or {}).get("distances") or []
            for row, src in enumerate(src_idx):
                for dst in range(n):
                    try:
                        meters = distances[row][dst]
                        result[src][dst] = meters / 1000.0 if meters is not None else None
                    except (IndexError, TypeError):
                        pass
            time.sleep(0.05)  # be gentle with a small local box
        return result
