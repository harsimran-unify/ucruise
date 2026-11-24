# Replacing the ridepool pipelines with a VRP solver

**Technical research report · ucruise** · 2026-11-18 · status: draft for team review

Published/shareable version: `docs/route-optimization-research.html` (also hosted as a claude.ai artifact from the session that produced this).

---

## TL;DR — Recommendation

**One solver, five profiles.** Adopt **PyVRP** (in-process, MIT, best measured solution quality) as the primary engine, with **OR-Tools** as the flexibility fallback, and serve road distances from a self-hosted **OSRM** matrix (keep the Google matrix API as the interim source). All five algorithm modes become configuration profiles of the same solve — the five pipelines retire behind a flag after a parity check, not before.

Highest-value first move, low risk: replace pickup sequencing (`ordering/`) with matrix + solver ordering. It touches one module, cuts Google calls per run, and improves every mode at once.

## 1. What the problem actually is

ucruise assigns office-commute users to drivers, groups them, and sequences pickups. That is a named problem in operations research — a **static capacitated, pickup-only, multi-depot open VRP**: every vehicle starts at its own depot (the driver's location) and ends at one common sink (the office), without returning home.

The data contract confirms the shape: users are `{id, latitude, longitude}`; drivers are `{id, capacity, latitude, longitude}` from the `driversUnassigned` + `driversAssigned` pools (`algorithm/base/base.py:221-328`). **There are no time-window fields**, so no VRPTW/DARP extension is needed today. If shift deadlines ever arrive as data, the same model extends to time windows — a profile change, not an architecture change.

| ucruise stage | Standard term | Solver primitive |
|---|---|---|
| User record | Customer node (pickup-only job) | Visit with demand 1 seat |
| Driver + capacity + home coords | Vehicle with distinct start depot | Per-vehicle start location (PyVRP, OR-Tools, VROOM all support this) |
| Office | Common end depot, open routes | Shared end depot; no return leg |
| Geographic clustering (`create_geographic_clusters`, sweep/polar/bearing variants) | Cluster-first heuristic | Unnecessary — solvers assign and sequence jointly |
| Driver→cluster assignment (`assign_drivers_by_priority`, `assign_cab_to_cluster`) | Assignment problem | Global solve (the Hungarian instinct was right — `safety.py` already implements it) |
| Pickup sequencing (geodesic ordering, direction-aware 2-opt, Google waypoint optimization) | Route ordering / TSP | Native in every solver |
| Splitting, merging, swaps, outlier reassignment, route filling (~40 functions per pipeline) | Inter-route local search | Built-in neighborhoods: relocate, swap, 2-opt*, ruin-and-recreate |
| Unassigned users/drivers output | Dropped customers | Optional visits with penalty — PyVRP prizes, OR-Tools disjunctions |
| Safety mode gender rules | Pairing constraint | VROOM skills; PyVRP penalties/groups; OR-Tools allowed-vehicle sets |
| Road-aware GraphML detour costs | Asymmetric travel costs | Cost matrix from OSRM `/table` |
| `mode_configs` (4 modes) | Objective weights / constraints | Solver profile — one engine, four configurations |

## 2. What the code does today

Five self-contained pipelines, each re-implementing the same six stages with different weights. Shared helpers exist in `algorithm/base/base.py`, but only safety and road import from it — capacity and balance duplicate geometry utilities locally. A behavioral fix has to be found and mirrored in up to five places.

| Pipeline | Size | Unique contribution |
|---|---|---|
| `base.py` route-efficiency | ~4.1k lines | Richest heuristic set: sweep/polar sectors, bearing-aware subclusters, tortuosity and turning-score scoring |
| `capacity/capacity.py` | ~1.9k | Seat filling *with eviction candidates*; hierarchical `linkage`/`fcluster` imported mid-file |
| `balance/balance.py` | ~2.7k | Its own k-means; geographic-separation swaps; consolidation passes |
| `road/road.py` | ~4.2k | GraphML shortest paths behind `MockRoadNetwork`/`_LazyRoadNetwork`; corridor-signature matching; a proposal-based generate/validate/apply local search — a hand-rolled metaheuristic shell |
| `safety/safety.py` | ~2.8k | Gender segmentation and min-male-per-female-cluster rules; female rescue; **a real Hungarian assignment** (`solve_optimal_assignment_hungarian`) |

Common stages across all five: clustering → capacity subclustering → driver assignment → sequencing + 2-opt → splitting → merging → seat filling → outlier reassignment → swaps.

**Where runtime and cost actually go:**

- **NetworkX shortest paths** over a 53 MB GraphML (`roadNetwork/tricity_main_roads.graphml`) — and no builder script exists in the repo, so the graph's provenance is unknown and unrefreshable.
- **Google Directions calls**, rate-limited, per route (`ordering/google_optimizer.py`), with a haversine nearest-neighbor fallback that chains pickups from the office coordinates rather than the driver's position.
- Config sprawl: `config.json` carries ~60 tuning knobs with a `mode_configs` override layer mirroring hardcoded thresholds in `roadNetwork/config.py`.

At ucruise's scale — tens to low hundreds of riders, tens of drivers, ≤5 seats per vehicle — **every solver in §3 solves the whole problem in well under a second**. The pipeline stages exist because each heuristic was written locally; a solver replaces all of them with one global search whose quality is measured against published best-known solutions rather than against intuition.

## 3. Solver options

| Engine | License / stack | Method | Solution quality | At ucruise scale | Fit notes |
|---|---|---|---|---|---|
| **PyVRP** | MIT · Python pkg | Hybrid genetic search | **Best** — ~0.23% from best-known [1][2] | Sub-second | Multi-depot, per-vehicle start/end depots, VRPTW, heterogeneous fleet, optional clients (unassigned-with-penalty), client groups, multi-trip. In-process pip install — the natural FastAPI fit. Safety pairing modeled via penalties/groups rather than a skills feature. |
| **VROOM** | BSD-2 · C++20 service | Metaheuristic + insertion | Strong | Milliseconds | Per-vehicle custom start/end, **skills** — the most direct encoding of gender-safety rules — plus shipments. Runs as a service (vroom-express) or C++ lib; pairs with OSRM/Valhalla natively. Maintained by Verso [3]. |
| **OR-Tools** (routing) | Apache-2 · C++/Python | Guided local search on the routing model (not CP-SAT) | ~4% gap [2] | Sub-second | Broadest feature set: disjunctions (unassigned-with-penalty), arbitrary per-vehicle start/end, time windows. The flexibility fallback if a constraint defeats PyVRP's model. |
| **jsprit** | Apache-2 · Java/JVM | Ruin-and-recreate | Moderate | Fine, but JVM | Mature and flexible; requires a JVM sidecar for a Python stack. No advantage over the top three here. |
| **Timefold** | Apache-2 · Java (+Python alpha) | Constraint solver (OptaPlanner successor; Solver 2.0, Apr 2026) [4] | Good | Python alpha | Excellent custom-constraint modeling, but the Python bindings are not production-ready. Revisit if rich business rules outgrow a VRP-shaped model. |
| **Google Route Optimization API** | Managed · per-shipment | Google fleet routing (OptimizeTours) | Strong | Managed | GA, **$30 per 1,000 shipments**, 1,000 free stops/mo, volume discounts [5][6]. Replaces all optimization code but keeps per-run Google cost and dependency — contradicts the cost criterion at scale. |

The quality column matters most for the route-efficiency mode: PyVRP's measured gap to best-known solutions is roughly twenty times smaller than OR-Tools' on standard benchmarks [1][2], and Routific's public benchmark comparing VROOM, PyVRP and OR-Tools across 264 instances is a ready-made methodology if the team wants independent confirmation [7].

**Dead ends worth naming:** MILP via Gurobi/CPLEX scales poorly for this use and adds license cost; writing another custom metaheuristic is exactly the situation being exited.

## 4. Road distances: the other half of the stack

Every solver consumes a cost matrix. Today that role is split between haversine math, a 53 MB GraphML with NetworkX shortest paths, and rate-limited Google calls. A routing engine replaces all three with one honest source of road distances and times.

- **OSRM (recommended)** — self-hosted Docker, Chandigarh-region OSM extract; `/table` serves the full driver×user×user matrix in milliseconds, `/route` returns geometry for the dashboard. No per-call cost.
- **Valhalla** — strong time-dependent matrix API and costing profiles; the alternative if turn penalties or costing flexibility matter.
- **GraphHopper / openrouteservice** — viable, heavier to self-host; ORS already embeds VROOM as its optimization engine.
- **Keep Google (interim)** — Routes API `computeRouteMatrix` can serve as the matrix source during migration; it removes the rate-limit-sensitive Directions loop but keeps per-element cost.

Caveat to check early: OSM coverage quality for the tricity area. Spot-check 20 origin–destination pairs against Google before switching the default.

## 5. Target architecture

Five forks become one pipeline with profile configuration. Everything above the solve and below it stays: the external-API fetch, validation, response contract, caches, and the dashboard.

1. **Fetch + validate** — unchanged (`load_env_and_fetch_data`, `validate_input_data`)
2. **Matrix service** — one `RoutingMatrix` interface; OSRM first, Google matrix as interim/fallback; results cached per data signature
3. **Solver** — PyVRP model built from users + drivers; mode passed as a profile; solve in-process; warm-start from cached result
4. **Response** — unchanged (`response_builder` keeps the existing payload contract, including `unassignedUsers`/`unassignedDrivers`)
5. **Observability** — solver objective, gap, runtime, matrix cache hits logged per run

| Mode today | Becomes |
|---|---|
| route_efficiency | minimize total distance |
| capacity_optimization | distance + seat-utilization penalty |
| balanced_optimization | distance + load-spread penalty (minimax on route length/load) |
| road-aware | same model, costs from the OSRM matrix instead of haversine |
| safety | gender pairing constraints (VROOM skills / PyVRP penalties) layered on any profile |

Side effect worth wanting: `config.json` shrinks from ~60 pipeline knobs to roughly 15 solver parameters (weights, unassigned penalty, max detour, time limit, matrix source), and the duplicate dispatch logic between `main.py` and `base.run_assignment` collapses to one entry point.

## 6. Migration roadmap

Each phase ships value on its own and is validated against the previous phase's baseline. No phase requires a big-bang switch.

**Phase 0 — Baseline** *(measurement only)*
Run `localTesting/run_and_view.py` for every mode and record: total distance, vehicles used, seat-utilization mean/max, unassigned counts, end-to-end runtime, Google calls per run.
Touches: `localTesting/test_analysis.py` (extend analytics output to a baseline JSON). No product code changes.

**Phase 1 — Routing matrix** *(kills GraphML + haversine costs)*
Introduce a `RoutingMatrix` interface with two implementations: OSRM (self-hosted, Chandigarh extract) and the Google matrix API. Point the road-aware costs at the interface; stop querying NetworkX shortest paths for optimization costs.
Touches: new `routing/` module; `roadNetwork/` demoted to optional. Validate: 20-pair spot check vs Google; Phase-0 metrics unchanged or better.

**Phase 2 — Solver-based pickup ordering** *(best value-per-risk)*
Replace the Google waypoint optimizer and its haversine fallback with matrix + solver ordering per route. Every mode's final sequences improve at once, and Google calls per run drop sharply.
Touches: `ordering/` (new solver optimizer alongside `google_optimizer.py`, flag-controlled). Validate: sequence lengths vs Phase-1 baseline on identical data.

**Phase 3 — Full solve per run** *(one engine replaces five pipelines)*
A solver pipeline builds the PyVRP model from users + drivers and produces all routes in one solve — clustering, assignment, sequencing, merging and splitting are no longer separate code paths. Output flows through the existing response builder; the old pipelines stay selectable behind a config flag.
Touches: new `assignment/solver_pipeline.py`; `main.py` dispatch. Validate: shadow runs current-vs-solver on the same fetch; property checks (§7).

**Phase 4 — Mode migration + retirement** *(config becomes the product)*
Port each mode's semantics as solver profiles, including the safety rules (min-male-per-female-cluster as constraints/penalties). Parity-check mode by mode, then flip the default and freeze the old pipelines for rollback.
Touches: profile definitions in `config.json`; mode selection. Validate: per-mode parity report; safety-mode rule audit on real assignments.

## 7. Verifying parity

- **Metrics** (identical fetch, current vs solver): total distance, vehicle count, seat-utilization mean and std-dev, unassigned riders, runtime, external API calls.
- **Shadow mode**: run the solver pipeline in parallel with the current selection and log both results before anything changes for users.
- **Golden cases**: small instances (10 riders, 2–3 drivers) with hand-checkable optimal solutions, run in CI-style tests.
- **Property checks**: no route exceeds capacity; every rider appears in ≤ 1 route; every route ends at the office; safety mode produces no rule-violating cluster.
- **Trade-off visibility**: where distance and balance conflict, report both rather than one weighted number, so profile weights are tuned on evidence [7].

**Open questions for the team**

- Do shift deadlines belong in the model as soft arrival targets? (Data has no time fields today — adding them upgrades the profile, not the architecture.)
- Who hosts OSRM, and is a Docker host available? (The one new ops dependency in the plan.)
- Confirm typical fleet size per run — it sets OSRM matrix sizing and solver time limits, though both are comfortable at any plausible scale.

## Sources

1. [PyVRP — published benchmarks vs reference solvers](https://pyvrp.org/setup/benchmarks.html)
2. [PyVRP issue #453 — OR-Tools benchmark comparison (~0.23% vs ~4% gap)](https://github.com/PyVRP/PyVRP/issues/453)
3. [VROOM-Project/vroom — active under Verso, C++20](https://github.com/VROOM-Project/vroom)
4. [Timefold Solver 2.0 discussion — Python bindings status](https://github.com/TimefoldAI/timefold-solver/discussions/2036)
5. [Google Route Optimization API — usage and billing](https://developers.google.com/maps/documentation/route-optimization/usage-and-billing)
6. [Google Maps Platform pricing list](https://developers.google.com/maps/billing-and-pricing/pricing)
7. [Routific route-quality benchmark — VROOM / PyVRP / OR-Tools across 264 instances](https://github.com/routific/route-quality-benchmark)
8. [OSRM — routing engine and /table matrix service](https://project-osrm.org)
9. [Google OR-Tools — routing solver documentation](https://developers.google.com/optimization)

External claims were verified 2026-11-18; the "5–23% ILP improvement" figure circulating in earlier notes remains unverified and is deliberately not cited as evidence.
