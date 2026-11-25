"""
Central configuration for ucruise.

Single source of truth for:
  - project-root-anchored paths (config.json, .env, output files, logs,
    dashboard candidates) — every consumer resolves through here so the
    server behaves identically regardless of CWD,
  - .env credentials for the external data API,
  - the config.json document itself (cached),
  - office coordinates (API payload -> config.json -> documented fallback),
  - the solver block (enabled flag, runtime, seed, per-mode profiles).

Plain stdlib + python-dotenv on purpose: the project pins pydantic<2 and
pydantic-settings is a v2-only package.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# app/core/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT
MAIN_OUTPUT_FILE = OUTPUT_DIR / "drivers_and_routes.json"
SOLVER_OUTPUT_FMT = "drivers_and_routes_solver_{mode}.json"
LOGS_DIR = PROJECT_ROOT / "logs"
VISUALIZE_CANDIDATES = (
    PROJECT_ROOT / "localTesting" / "visualize.html",
    PROJECT_ROOT / "visualize.html",
)
OFFICE_FALLBACK = (30.6810489, 76.7260711)  # documented office fallback

SOLVER_DEFAULTS = {
    "enabled": True,
    "runtime_seconds": 3.0,
    "seed": 1,
    "drop_penalty_km": 40.0,
    "vehicle_fixed_cost_km": 6.0,
    "profiles": {},
}
SOLVER_PROFILE_DEFAULT = {"drop_penalty_km": 40.0, "vehicle_fixed_cost_km": 6.0}


def ensure_env_loaded() -> None:
    """Load .env from the project root (idempotent); raise if it is missing."""
    if not ENV_PATH.exists():
        raise FileNotFoundError(f".env file not found at {ENV_PATH}")
    load_dotenv(ENV_PATH)


def api_credentials() -> "tuple[str, str]":
    """(API_URL, API_AUTH_TOKEN) from .env; raises when either is unset."""
    ensure_env_loaded()
    url = os.getenv("API_URL")
    token = os.getenv("API_AUTH_TOKEN")
    if not url or not token:
        raise ValueError("Both API_URL and API_AUTH_TOKEN must be set in .env")
    return url, token


@lru_cache(maxsize=1)
def get_config() -> dict:
    """The parsed config.json document; {} when unreadable."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read config.json ({e}); using defaults")
        return {}


def office_coordinates(data: dict = None) -> "tuple[float, float]":
    """Office coordinates: API payload company -> config.json -> fallback."""
    company = (data or {}).get("company") or {}
    for lat_key, lon_key in (("latitude", "longitude"), ("lat", "lng")):
        try:
            lat = company.get(lat_key)
            lon = company.get(lon_key)
            if lat is not None and lon is not None:
                return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
    cfg = get_config()
    try:
        return float(cfg["office_latitude"]), float(cfg["office_longitude"])
    except (KeyError, TypeError, ValueError):
        return OFFICE_FALLBACK


def solver_config() -> dict:
    """The solver block with defaults applied."""
    cfg = dict(SOLVER_DEFAULTS)
    cfg.update(get_config().get("solver") or {})
    return cfg


def solver_profile(mode: str) -> dict:
    """Per-mode profile (drop penalty / vehicle fixed cost) with defaults."""
    profile = dict(SOLVER_PROFILE_DEFAULT)
    profile.update((solver_config().get("profiles") or {}).get(mode) or {})
    return profile


def solver_enabled() -> bool:
    return bool(solver_config().get("enabled", False))
