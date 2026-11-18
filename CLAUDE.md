# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI service (ucruise) that assigns office-commute users to drivers, optimizes the routes, and sequences pickups. Users/drivers come from an external API; fine-grained pickup ordering uses the Google Maps Directions API.

## Commands

The project uses **uv** (see `pyproject.toml`, `.python-version`; `requirements.txt` is legacy and unpinned — pyproject pins `pydantic<2` because the code uses the v1 API). All Python commands go through `uv run`:

```powershell
uv sync                                                # first setup / after dep changes
uv run uvicorn main:app --host 0.0.0.0 --port 5000 --reload   # from project root
```

Local end-to-end harness — run from inside `localTesting/` (it derives the project root as the parent of CWD). It starts the server, triggers an assignment, prints analytics, and opens the dashboard:

```powershell
cd localTesting
uv run python run_and_view.py [--source-id X --parameter 1 --string-param "Evening shift" --choice " "]
uv run python test_analysis.py    # direct in-process test, no HTTP layer
```

There is no test suite or linter. Verification is manual: inspect the API response, `logs/assignment_*.log`, and the dashboard at `http://localhost:5000/visualize` (served from `localTesting/visualize.html`).

Key endpoints: `POST /assign-drivers/{source_id}/{parameter}/{string_param}/{choice}` (main assignment), `GET /routes` (last saved result), `POST /pickup_order` and `POST /pickup_orderV2` (Google-based pickup reordering), `GET /health`.

## Configuration

- `.env` at project root (gitignored): `API_URL`, `API_AUTH_TOKEN` (external data API), `GOOGLE_MAPS_API_KEY`.
- `config.json` at project root is the central tuning file for all algorithms. Per-mode values in `mode_configs` override top-level defaults; road-network thresholds in it mirror the hardcoded `RoadAwareConfig` in `roadNetwork/config.py`.
- `disable_algorithm_cache: true` and `route_ordering.enable_route_cache: false` are both currently set — every run recomputes and hits Google fresh.

## Assignment flow (main.py → algorithm/)

1. The `/assign-drivers/...` endpoint fetches users/drivers from the external API via `load_env_and_fetch_data` in `algorithm/base/base.py` — a GET to `{API_URL}/{source_id}/{parameter}/{string_param}/{choice}` with a Bearer token.
2. Algorithm selection (implemented in both main.py's endpoint and `run_assignment` in base.py, same rules):
   - `safety == 1` in the API response overrides everything → `algorithm/safety/safety.py`
   - otherwise `ride_settings.pic_priority` (falling back to `drop_priority`): 1 → capacity, 2 → balanced, 3 → road-aware, anything else → route efficiency
   - any data-fetch error falls back to route efficiency
3. The result is shaped into the standardized payload defined by `algorithm/response/response_standards.py` and built by `response_builder.py` (`{status, execution_time, data, unassignedUsers, unassignedDrivers, optimization_mode, ...}`), written to `drivers_and_routes.json`, and returned. `GET /routes` reads that file back.

## The five algorithm modules are intentionally self-contained

`algorithm/base/base.py` (route efficiency), `capacity/capacity.py`, `balance/balance.py`, `road/road.py`, and `safety/safety.py` are each ~2k–4k-line standalone pipelines (clustering → driver assignment → sequence optimization → merging/splitting → outlier handling). They share patterns, not code: a behavioral fix usually has to be mirrored across the variants. `base.py` additionally hosts the shared helpers the others reuse (config loading, `load_env_and_fetch_data`, geometry utilities).

## Caches

- Algorithm-result cache (`algorithm/algorithm_cache.py`): whole-result caching per company + algorithm under `routesCache/`, keyed by a data signature.
- Pickup-order cache (`ordering/route_cache.py`): per company + algorithm under `ordering/cache/`.

## Pickup ordering (ordering/)

`OrderIntegration.apply_optimal_ordering` chains: cached orders → Google Directions batch optimization (`google_optimizer.py`, rate-limited) → haversine nearest-neighbor fallback. The fallback chains pickups starting from the office coordinates, not the driver's position.

## Road network (roadNetwork/)

`tricity_main_roads.graphml` (53 MB) is loaded into a NetworkX graph by `RoadNetwork` (`roadNetwork.py`). `RoadNetworkManager` (`router.py`) loads it in a background thread with retries; the road-aware algorithm uses `get_road_network(wait_timeout=...)`.

## Gotchas

- Run the server from the project root: `.env`, `config.json`, the GraphML file, and output files like `drivers_and_routes.json` are resolved relative to CWD in some places and via `__file__` in others.
- `string_param` can contain spaces (e.g. `Evening shift`) — URL-encode it when calling the API.
- Each assignment run deletes the `drivers_and_routes*.json` output files first; `run_and_view.py` also clears `logs/assignment_*.log` — logs are per-session only.
- The `algorithm/*` subpackages have no `__init__.py`; imports like `from algorithm.base.base import ...` rely on implicit namespace packages (Python 3.3+).
