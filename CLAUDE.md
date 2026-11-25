# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI service (ucruise) that assigns office-commute users to drivers, optimizes the routes, and sequences pickups. Users/drivers come from an external API; fine-grained pickup ordering uses the routing engine chain (solver TSP, with Google Directions as an optional fallback).

One assignment engine: a **PyVRP global solve** (per-mode profiles) replaces the former five standalone legacy pipelines. The legacy clustering→assignment→sequencing→merging pipelines, the roadNetwork GraphML module, and `requirements.txt` were removed; `algorithm/` and `roadNetwork/` no longer exist.

## Commands

The project uses **uv** (see `pyproject.toml`, `.python-version`). All Python commands go through `uv run`:

```powershell
uv sync                                                # first setup / after dep changes
uv run uvicorn main:app --host 0.0.0.0 --port 5000 --reload   # from project root
```

Local end-to-end harness — run from inside `localTesting/` (it derives the project root as the parent of CWD). It starts the server, triggers an assignment, prints analytics, and opens the dashboard:

```powershell
cd localTesting
uv run python run_and_view.py [--source-id X --parameter 1 --string-param "Evening shift" --choice " "]
uv run python test_analysis.py    # direct in-process test, no HTTP layer
uv run python baseline_metrics.py [--only <mode>]  # runs safety + all four solver profiles, writes baseline/*.json
uv run python validate_routing.py [--pairs 20]     # road-distance spot check per engine
uv run python test_solver_ordering.py              # unit test (ordering TSP)
uv run python test_solver_pipeline.py              # unit test (synthetic PyVRP instances)
uv run python test_road_adapter.py                 # unit test (MatrixRoadNetwork contract)
```

**Response-parity checking:** `localTesting/baseline/golden_*.json` hold legacy-pipeline reference payloads (gitignored — they contain user PII, never commit). Regenerate a golden by setting `solver.enabled: false` in config.json, running the server, POSTing `/assign-drivers/...`, and copying `drivers_and_routes.json`. Then compare:

```powershell
uv run python localTesting/diff_parity.py localTesting/baseline/golden_capacity.json drivers_and_routes_solver_capacity_optimization.json
```

(par files: golden_route_efficiency ↔ `drivers_and_routes_solver_route_efficiency.json`, golden_balance ↔ `..._balanced_optimization.json`, golden_road ↔ `..._road_aware_route_optimization.json`.)

There is no test suite or linter (an httpx dev-dependency enables FastAPI's TestClient for ad-hoc checks). Verification is manual: inspect the API response, `logs/assignment_*.log`, and the dashboard at `http://localhost:5000/visualize` (served from `localTesting/visualize.html` — its ALGORITHM sidebar loads `/routes-all`, switches per-mode results in memory, and triggers `/assign-drivers-all` on the local_server wrapper). Note: `localTesting/` is gitignored — dashboard/harness fixes don't reach the server copy automatically. `localTesting/compare_orderings.py --regenerate-basemap` is dead (it needed networkx + the deleted GraphML).

Key endpoints: `POST /assign-drivers/{source_id}/{parameter}/{string_param}/{choice}` (main assignment), `GET /routes` (last result), `POST /pickup_order` and `POST /pickup_orderV2` (pickup reordering), `GET /health`. Locally only: `localTesting/local_server.py` wraps `main:app` with `/assign-drivers-all` (runs safety + all four solver profiles on the same parameters) and `/routes-all` — run via `localTesting/run_and_view.py`; never deployed.

## Configuration

- `.env` at project root (gitignored): `API_URL`, `API_AUTH_TOKEN` (external data API), `GOOGLE_MAPS_API_KEY`.
- `config.json` at project root: `office_latitude/longitude`, `mode_configs.female_safety_optimization` (safety-only tuning), `solver` (enabled flag, runtime_seconds, seed, drop_penalty_km, vehicle_fixed_cost_km, `profiles` per mode), `routing` (engine chain), `route_ordering`. All paths/config resolve through `app/core/config.py` (project-root-anchored via `__file__`, so CWD no longer matters).
- `disable_algorithm_cache` and the legacy clustering/merging knobs are gone with the legacy pipelines.

## Assignment flow (main.py → app/features/assignment)

1. The `/assign-drivers/...` endpoint (in `app/features/assignment/routes.py`) fetches users/drivers once via `load_env_and_fetch_data` in `app/infrastructure/source_api.py` — a GET to `{API_URL}/{source_id}/{parameter}/{string_param}/{choice}` with a Bearer token.
2. Dispatch:
   - `safety == 1` in the API response overrides everything → `app/features/assignment/services/safety.py` (dedicated gender-pairing pipeline; not expressible in the VRP model)
   - otherwise `ride_settings.pic_priority` (falling back to `drop_priority`): 1 → `capacity_optimization`, 2 → `balanced_optimization`, 3 → `road_aware_route_optimization`, anything else → `route_efficiency` (endpoint relabels the default arm `route_efficiency_default`)
3. The result is shaped into the standardized payload defined by `app/features/assignment/helpers/response_standards.py` and built by `response_builder.py` (`{status, execution_time, data, unassignedUsers, unassignedDrivers, optimization_mode, company, shift, parameter, string_param, choice, metadata}`), written to `drivers_and_routes.json` (+ per-profile `drivers_and_routes_solver_{mode}.json`), and returned. `GET /routes` reads that file back.
4. Solver responses carry every field the legacy pipelines filled: `price_book_id` (pricebook nearest the farthest assigned user), driver vehicle details, user `shift_type`, top-level `company`/`shift`; `metadata` (engine/profile/solve_seconds/distance_source) is a solver addition. Output files are UTF-8 (older committed files were cp1252).

## Structure (MedicraftAI-style layout)

```
main.py                              FastAPI app + routers + lifespan (log session start); entry point "main:app"
app/core/config.py                   paths, .env credentials, config.json (cached), office coords, solver profiles
app/infrastructure/
  source_api.py                      load_env_and_fetch_data (external API client)
  logging/logger.py, progress.py     session logger (logs/assignment_*.log), progress tracker
  routing/                           OSRM → Google Routes matrix → haversine chain; cache/cache/point_distances.json
  ordering/                          pickup-order layer: order_integration, solver_optimizer (TSP),
                                     google_optimizer (Directions fallback), route_cache
app/features/assignment/
  routes.py                          /assign-drivers, /routes, /visualize
  services/solver_pipeline.py        PyVRP global solve + per-profile entry points
  services/safety.py                 dedicated female-safety pipeline
  helpers/                           response_standards, response_builder, geo, data_access, pricebook/
app/features/pickup_order/
  routes.py, models.py, helpers.py   /pickup_order, /pickup_orderV2 (pydantic v1 API — do not modernize)
```

No `__init__.py` at the `app/` top levels (implicit namespace packages); the moved `routing/`, `ordering/`, `pricebook/` packages keep theirs.

## Caches

- Pickup-order cache (`app/infrastructure/ordering/route_cache.py`): per company + algorithm under `app/infrastructure/ordering/cache/` (stale legacy-named subdirs were removed; `solver/` and `pickup_order_api/` remain).
- Routing distance cache: `app/infrastructure/routing/cache/point_distances.json` with per-entry engine stamps.
- The whole-result algorithm cache (`routesCache/`, `algorithm_cache.py`) was legacy-only and deleted; it was disabled in config anyway.

## Pickup ordering (app/infrastructure/ordering/)

`OrderIntegration.apply_optimal_ordering` chains: cached orders → primary optimizer → fallback. `route_ordering.optimizer` picks the primary: `"solver"` (current default — TSP over the routing chain: free, cached, starts from the driver, ends at the office) or `"google"` (Directions-ordered, per-call billed). The fallback is the solver TSP too; the old office-start nearest-neighbor is only kept as `_optimize_single_route_fallback` for direct callers.

## Solver pipeline (app/features/assignment/services/solver_pipeline.py)

One PyVRP global solve: drivers are vehicle types (capacity, start depot = home, end depot = office), users are optional clients with a drop penalty, distances from `app/infrastructure/routing/`. `solver.profiles` in config.json tunes each mode (fixed cost per used vehicle + drop penalty): capacity = high fixed cost (consolidate), balanced = low (spread), road_aware = same model as efficiency but road-truth matrix, route_efficiency = moderate. `solver.enabled` gates nothing anymore (the solver is the only non-safety engine; the flag remains as a config switch read by `app/core/config.py::solver_enabled`).

## Gotchas

- `uvicorn main:app` must stay the entry point (localTesting's run_and_view.py and deployment rely on it).
- `string_param` can contain spaces (e.g. `Evening shift`) — URL-encode it when calling the API.
- Each assignment run deletes the `drivers_and_routes*.json` output files first; `run_and_view.py` also clears `logs/assignment_*.log` — logs are per-session only (server session starts via FastAPI lifespan).
- `pydantic` is pinned `<2` — the pickup_order models use the v1 API (`@validator`, `class Config`); do not modernize them and do not add `pydantic-settings`.
- The old CWD gotcha is retired: output files, config, `.env`, logs, and caches are all resolved from the project root via `app/core/config.py` regardless of CWD.
- `localTesting/` is gitignored — the dashboard, goldens, and harness fixes never propagate to a deployed server copy automatically.
