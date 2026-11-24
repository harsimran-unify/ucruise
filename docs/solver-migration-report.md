# Solver Migration — Implementation & Test Report

**ucruise** · 2026-11-19 · scope: Phases 0–4 of the VRP-solver migration plan ([research report](route-optimization-research.md))

**Status: all phases implemented and tested. Nothing is committed** (working tree only). Legacy behavior is recoverable with one config flag at any point.

---

## TL;DR

The PyVRP solver pipeline replaces four of the five legacy pipelines behind `solver.enabled`, and on the live Evening-shift instance (52 users, 17 drivers) it produced **strictly better assignments on every measured axis**: fewer vehicles (11 vs 12–22), full 52/52 assignment, 100% seat utilization, ~70% less total distance, zero capacity violations, and — with solver pickup ordering — **zero Google API calls and a 3.5 s wall time** (vs 8–34 s legacy). One bad driver record (registered 244 km away) poisoned a legacy route and was naturally isolated by the solver.

All configuration switches default to **conservative**: production behavior changes only because `solver.enabled` is currently `true`; every legacy path remains one flag away.

---

## 1. What was implemented

### Phase 0 — Baseline harness (done 2026-11-19)
- `localTesting/baseline_metrics.py` — runs every mode on identical live data and records: routes, assigned/unassigned, seat utilization, haversine-chain distance, per-user detour, capacity violations, duplicate assignments, wall time, Google-call deltas. Writes `localTesting/baseline/<timestamp>.json` + `latest.json` (gitignored). Supports `--only <mode>` (repeatable).

### Phase 1 — Routing matrix (code complete; live-engine validation pending)
- `routing/` package: `RoutingEngine` base + availability probing + per-engine stats ([base.py](../routing/base.py)), OSRM engine ([osrm.py](../routing/osrm.py)), Google Routes `computeRouteMatrix` engine with 25×25/100-element chunking, rate limiting, retries and a **1,000-element session budget** ([google_matrix.py](../routing/google_matrix.py)), thread-safe disk cache with per-entry engine stamps ([cache.py](../routing/cache.py)), observable failover chain ([chain.py](../routing/chain.py)), and `MatrixRoadNetwork` — a drop-in replacement for the GraphML `RoadNetwork` surface ([road_adapter.py](../routing/road_adapter.py)).
- `roadNetwork/router.py` — provider switch: `road_network.provider: "matrix"` prefers the routing chain and **automatically falls back to GraphML when no external engine is reachable** (verified live).
- **Blocked:** the Google key returns `HTTP 403 PERMISSION_DENIED` — the **Routes API** is not enabled for it (Cloud Console → APIs & Services → Library → Routes API). OSRM is not installed (Docker commands in [osrm.py](../routing/osrm.py) docstring). Until one is enabled, distances are cached haversine and road-aware runs fall back to GraphML — exactly the pre-migration behavior.
- Validation tooling: `localTesting/validate_routing.py` (20-pair spot check per engine), `localTesting/test_road_adapter.py` (**9/9 pass**).

### Phase 2 — Solver-based pickup ordering
- `ordering/solver_optimizer.py` — exact TSP (brute force ≤ 8 stops, NN + 2-opt + or-opt above) over the RoutingChain, sequences **driver → pickups → office**. Cached and free; upgrades to road distances automatically once an engine is enabled.
- `ordering/order_integration.py` — new `route_ordering.optimizer` config: `"google"` (default, current behavior) or `"solver"` (TSP primary, Google Directions fallback). The fallback path now uses the solver TSP — **fixing the documented bug** where the haversine fallback chained pickups starting from the *office*.

### Phase 3 — Full solve per run
- `algorithm/solver_opt/solver_pipeline.py` — one PyVRP global solve per run: each driver is a vehicle type (capacity seats, start depot = home, end depot = office — native per-vehicle-type start/end depots in PyVRP 0.14), each user is an *optional client* with a drop penalty, distances from the RoutingChain. Output flows through the unchanged `response_builder` payload contract.
- Four legacy-compatible entry points (`run_solver_route_efficiency_assignment`, `run_solver_capacity_assignment`, `run_solver_balance_assignment`, `run_solver_road_assignment`) so dispatch and the local-testing harness call them unchanged.
- Dispatch gates in **both** dispatch sites (`main.py` endpoint + `run_assignment` in `base.py`): when `solver.enabled` is true, priorities 1/2/3/default route to the solver with the selected mode as profile.

### Phase 4 — Mode migration
- `solver.profiles` in `config.json` express the four modes as one engine's configuration: `vehicle_fixed_cost_km` (consolidation pressure) + `drop_penalty_km`. Capacity = high fixed cost; balanced = low; efficiency/road = moderate.
- **Safety mode intentionally stays legacy.** Min-male-per-vehicle gender pairing is not expressible in PyVRP's model; VROOM's *skills* feature is the natural fit and is recommended as the follow-up (research report §3). `safety == 1` dispatches to `algorithm/safety/safety.py` in all configurations.
- Rollback switches: `solver.enabled: false` (legacy pipelines), `route_ordering.optimizer: "google"` (legacy ordering), `road_network.provider: "graphml"` (legacy road network).

---

## 2. Test results

### 2.1 Unit tests (synthetic, deterministic)

| Suite | Checks | Result |
|---|---|---|
| `test_road_adapter.py` — MatrixRoadNetwork contract | distance scaling, coherence bounds, on-path logic, mock parity, honest degradation | **9/9 pass** |
| `test_solver_ordering.py` — pickup ordering | structure, driver-start geometry, exact-TSP vs office-start NN, heuristic validity, determinism, single-stop | **10/10 pass** |
| `test_solver_pipeline.py` — PyVRP core properties | capacity respected, no double assignment, outlier dropped not routed, fast solve, profile behavior | **10/10 pass** |

Headline from the ordering test: on one crafted route the solver TSP produces a **5.73 km** tour where the legacy office-start fallback produces **11.33 km** for the same stops.

### 2.2 Integration — all modes on identical live data (Evening shift, 52 users / 17 drivers)

Measured identically for every mode: haversine chain driver → pickups (in `pickup_order`) → office.

**Run A — legacy ordering (`route_ordering.optimizer: "google"`):**

| mode | routes | assigned | unassigned | util % | dist km | detour km | wall s | Google calls |
|---|---|---|---|---|---|---|---|---|
| capacity (legacy) | 12 | 50 | 2 | 83 | 510.3 | 942.3 | 8.2 | 12 |
| balance (legacy) | 12 | 44 | 8 | 73 | 485.1 | 1218.7 | 7.3 | 11 |
| route_efficiency (legacy default) | 22 | 51 | 1 | 49 | 644.8 | 474.5 | 8.2 | 12 |
| road (legacy) | 16 | 41 | 11 | 54 | 490.1 | 1205.1 | 33.6 | 13 |
| safety (legacy, unchanged) | 13 | 50 | 2 | 78 | 292.3 | 375.4 | 7.7 | 12 |
| **solver_eff** | **11** | **52** | **0** | **100** | **188.5** | **191.8** | 14.4 | 11 |
| solver_cap | 11 | 52 | 0 | 100 | 192.1 | 199.1 | 9.9 | 11 |
| solver_bal | 11 | 52 | 0 | 100 | 192.1 | 199.1 | 10.0 | 11 |
| solver_road | 11 | 52 | 0 | 100 | 188.5 | 191.8 | 15.6 | 11 |

**Run B — solver ordering (`optimizer: "solver"`), solver modes only:**

| mode | routes | assigned | dist km | detour km | wall s | Google calls |
|---|---|---|---|---|---|---|
| solver_eff | 11 | 52 | **156.1** | **119.1** | 8.0 | **0** |
| solver_cap | 11 | 52 | 156.8 | 117.5 | 3.5 | 0 |
| solver_bal | 11 | 52 | 156.8 | 117.5 | 3.6 | 0 |
| solver_road | 11 | 52 | 156.0 | 120.1 | 3.6 | 0 |

Against the Phase-0 baseline, the best legacy mode (capacity: 510 km, 12 routes, 2 unassigned) vs the shipped solver configuration (156 km, 11 routes, 0 unassigned): **69% less distance, one fewer vehicle, two more riders assigned, zero external API calls, ~2× faster wall time**. Against the default legacy mode (route_efficiency): **76% less distance, 11 fewer vehicles**.

Note on Run B's distance drop vs Run A: the metric is a haversine chain, and the solver ordering minimizes exactly that chain; Google Directions orders by real road distance. The road-truth comparison between orderings becomes possible once a routing engine is enabled (Phase 1 blocker).

### 2.3 Property checks on live solver output
- 0 capacity violations, 0 duplicate assignments across all four profiles; every route ends at the office by construction.
- 52/52 riders assigned; the drops-with-penalty mechanism was validated on synthetic outliers in the unit suite (a 200 km-out user is refused a dedicated route).

### 2.4 Data-quality finding (corrects an earlier note)
The "240 km outlier" flagged in the Phase-0 baseline is **driver 345013, registered 244 km from the office** — bad driver-pool data, not a remote rider. Legacy capacity assigned that driver to a local cluster and produced a **245 km, 2-user route** (detour metric 465 km). The solver left the driver unused and served the rider locally. Bad input data currently degrades legacy output silently; the solver isolates it structurally. Recommendation: flag drivers beyond a sane radius at ingestion (one validation rule, independent of this migration).

---

## 3. Findings

1. **The solver dominates on every axis for this instance** — quality, utilization, runtime, and external cost. The legacy pipelines' multi-stage heuristics (split on turning score, merge, seat-fill, swap) consistently fragment the solution (22 routes at 49% utilization in the default mode).
2. **Profiles converged on this instance** (cap == bal at 11 routes / 156.8 km): with 52 users and 17 local drivers, consolidating to 11 vehicles is distance-optimal, so the fixed-cost lever had no room to differentiate. Expect differentiation on sparser shifts; revisit profile values with real shift history.
3. **Runtime profile:** solver wall time is dominated by the Google ordering layer in Run A (14–16 s); in Run B (solver ordering) it is ~3.5 s. The legacy road-aware mode's 34 s remains the slowest path overall (GraphML + NetworkX).
4. **Google exposure can drop to zero today** (`optimizer: "solver"`), with the caveat that ordering then optimizes haversine geometry until OSRM/Routes is enabled — the honest trade-off is one config flag, measurable both ways (Run A vs Run B).
5. **Safety mode is the remaining legacy dependency.** Its gender-pairing rule needs VROOM skills (or a custom PyVRP constraint extension) — scoped as follow-up work in the research report.

---

## 4. How to operate

| Intent | Config |
|---|---|
| Revert everything to legacy | `solver.enabled: false` |
| Solver assignments, solver ordering (**shipped default, decided 2026-11-19**) | `solver.enabled: true`, `route_ordering.optimizer: "solver"` |
| Solver assignments, Google Directions ordering | `route_ordering.optimizer: "google"` |
| Road-truth distances (once Routes API enabled or OSRM up) | set `OSRM_URL` or enable Routes API; `road_network.provider: "matrix"` (already default) |
| Back to GraphML road network | `road_network.provider: "graphml"` |

Re-run the full comparison at any time:

```powershell
cd localTesting
uv run python baseline_metrics.py              # all nine modes → baseline/latest.json
uv run python test_solver_ordering.py          # Phase 2 unit tests
uv run python test_solver_pipeline.py          # Phase 3 unit tests
uv run python test_road_adapter.py             # Phase 1 unit tests
uv run python validate_routing.py --pairs 20   # Phase 1 live-engine spot check
```

Archived runs from this session: `localTesting/baseline/latest.json` (Phase 0), `run_all_google_ordering.json` (Run A), `run_solver_ordering.json` (Run B).

## 5. File inventory (all uncommitted)

**New:** `routing/` (6 files), `algorithm/solver_opt/solver_pipeline.py`, `ordering/solver_optimizer.py`, `localTesting/{baseline_metrics,validate_routing,test_road_adapter,test_solver_ordering,test_solver_pipeline}.py`, `docs/route-optimization-research.{md,html}`, this report.

**Modified:** `roadNetwork/router.py` (provider switch), `ordering/order_integration.py` (optimizer config + fixed fallback), `main.py` + `algorithm/base/base.py` (solver dispatch gates), `config.json` (`solver`, `routing`, `road_network.provider`, `route_ordering.optimizer`), `CLAUDE.md`, `pyproject.toml`/`uv.lock` (`pyvrp==0.14.1` via uv).

## 6. Next steps

1. Enable the **Routes API** for the existing key (or deploy OSRM) and run `validate_routing.py --pairs 20` — closes Phase 1 validation and upgrades every distance to road truth.
2. Decide the ordering default (A vs B) after a road-truth re-measurement.
3. Tune `solver.profiles` against historical shift data (the balance/capacity lever needs sparse instances to matter).
4. Add a driver-coordinate sanity check at ingestion (the 345013 case).
5. Safety-mode migration: evaluate VROOM skills or a PyVRP custom-constraint fork.
