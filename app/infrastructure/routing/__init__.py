"""
routing — Phase 1 of the solver-migration plan (see docs/route-optimization-research.md).

One interface for road distances, engines in explicit fallback order:
    OSRM (self-hosted, free)  ->  Google Routes matrix (billed per element)
    ->  haversine (always available, last resort)

Config (config.json, "routing" block) decides engine order and cache path.
OSRM_URL / GOOGLE_MAPS_API_KEY come from the environment (.env).
"""

import json
import os
import threading
from typing import Optional

from .base import HaversineEngine, RoutingEngine, haversine_km, validate_road_distance
from .cache import PointDistanceCache
from .chain import RoutingChain

__all__ = [
    "RoutingEngine", "HaversineEngine", "RoutingChain", "PointDistanceCache",
    "haversine_km", "validate_road_distance", "get_routing_chain",
]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_chain_lock = threading.Lock()
_chain: Optional[RoutingChain] = None


def _load_routing_config() -> dict:
    defaults = {
        "engines": ["osrm", "google", "haversine"],
        "osrm_default_url": "http://localhost:5001",
        "osrm_timeout_s": 5.0,
        "google_element_budget": 1000,
        "google_min_request_interval_s": 0.1,
        "cache_enabled": True,
        "cache_max_entries": 50000,
        "max_reasonable_ratio": 3.0,
    }
    try:
        with open(os.path.join(_PROJECT_ROOT, "config.json"), "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        defaults.update(file_cfg.get("routing") or {})
    except (OSError, json.JSONDecodeError):
        pass
    return defaults


def get_routing_chain() -> RoutingChain:
    """Process-wide chain, built from config.json once."""
    global _chain
    if _chain is not None:
        return _chain
    with _chain_lock:
        if _chain is not None:
            return _chain

        cfg = _load_routing_config()
        engines: list = []
        for name in cfg["engines"]:
            if name == "osrm":
                from .osrm import OSRMEngine
                url = os.getenv("OSRM_URL", cfg["osrm_default_url"])
                engines.append(OSRMEngine(base_url=url,
                                          timeout_s=cfg["osrm_timeout_s"]))
            elif name == "google":
                from .google_matrix import engine_from_env
                engine = engine_from_env(element_budget=cfg["google_element_budget"])
                if engine is not None:
                    engine._min_interval = cfg["google_min_request_interval_s"]
                    engines.append(engine)
            elif name == "haversine":
                engines.append(HaversineEngine())

        cache = None
        if cfg["cache_enabled"]:
            cache = PointDistanceCache(
                path=os.path.join(_PROJECT_ROOT, "routing", "cache",
                                  "point_distances.json"),
                max_entries=cfg["cache_max_entries"])

        _chain = RoutingChain(engines, cache=cache,
                              max_ratio=cfg["max_reasonable_ratio"])
        return _chain
