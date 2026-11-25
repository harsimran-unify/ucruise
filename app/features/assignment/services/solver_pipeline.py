"""
Solver pipeline (Phase 3 of the solver-migration plan) — one engine, many
profiles. See docs/route-optimization-research.md and
docs/solver-migration-report.md.

Replaces the legacy cluster -> assign -> sequence -> merge -> split stages
with a single global PyVRP solve:

  - Every driver is a vehicle type with capacity (seats), starting at their
    own depot (home coordinates) and ending at the shared office depot —
    PyVRP 0.14+ supports per-vehicle-type start/end depots natively.
  - Every user is an OPTIONAL client with demand 1 seat and a prize equal to
    the profile's drop penalty: the solver assigns everyone it reasonably
    can and explicitly drops genuine outliers (e.g. a user 240 km out)
    instead of building them a dedicated route.
  - Distances come from the RoutingChain (routing/): OSRM or Google when an
    engine is enabled, cached haversine otherwise. This makes "road-aware"
    a matrix source, not a separate algorithm.
  - Profiles (config.json "solver".profiles) differ by vehicle fixed cost
    and drop penalty, replacing the four legacy mode pipelines:
      route_efficiency   - moderate consolidation, pure distance focus
      capacity_optimization - high fixed cost -> fewer, fuller vehicles
      balanced_optimization - low fixed cost -> more vehicles, lighter loads
      road_aware_route_optimization - same model; matrix gives road truth

The safety mode intentionally stays on the legacy pipeline: min-male-per-
vehicle gender pairing is not expressible in PyVRP's model (VROOM's skills
feature is the natural fit — see the research report's Phase 4 notes).

Entry points mirror the legacy naming so dispatch, localTesting's
ALGORITHM_MODES table, and the baseline harness can call them unchanged.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from app.core import config
from app.infrastructure.logging.logger import get_logger
from app.infrastructure.ordering.order_integration import get_order_integration
from app.infrastructure.source_api import load_env_and_fetch_data
from app.features.assignment.helpers.data_access import validate_input_data
from app.features.assignment.helpers.pricebook.pricebook_calculator import calculate_pricebook_id
from app.features.assignment.helpers.response_builder import (
    build_standard_response,
    create_error_response,
    log_response_metrics,
    save_standardized_response,
)

logger = get_logger()


def _driver_seats(driver: Dict[str, Any]) -> int:
    raw = driver.get("capacity")
    try:
        seats = int(raw) if raw is not None else 4
    except (TypeError, ValueError):
        seats = 4
    return max(1, seats)


def _user_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    users = []
    for u in data.get("users") or []:
        try:
            users.append({
                "user_id": str(u.get("id", "")),
                "latitude": float(u["latitude"]),
                "longitude": float(u["longitude"]),
                "office_distance": float(u.get("office_distance") or 0.0),
                "first_name": u.get("first_name"),
                "last_name": u.get("last_name"),
                "email": u.get("email"),
                "phone": u.get("phone"),
                "address": u.get("address"),
                "employee_shift": u.get("employee_shift"),
                "shift_type": u.get("shift_type"),
            })
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Skipping user with unusable fields: {u.get('id')}")
    return users


def _driver_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "drivers" in data:
        pool = (data["drivers"].get("driversUnassigned") or [])
    else:
        pool = data.get("driversUnassigned") or []
    drivers = []
    seen = set()
    for d in pool:
        try:
            did = str(d.get("id", ""))
            if did in seen:
                continue  # duplicates are allowed upstream; one vehicle each
            seen.add(did)
            drivers.append({
                "driver_id": did,
                "vehicle_id": str(d.get("vehicle_id", "") or ""),
                "latitude": float(d["latitude"]),
                "longitude": float(d["longitude"]),
                "capacity": _driver_seats(d),
                "first_name": d.get("first_name"),
                "last_name": d.get("last_name"),
                "email": d.get("email"),
                "vehicle_name": d.get("vehicle_name"),
                "vehicle_no": d.get("vehicle_no"),
                "chasis_no": d.get("chasis_no"),
                "color": d.get("color"),
                "registration_no": d.get("registration_no"),
                "shift_type_id": d.get("shift_type_id"),
                "office_distance": d.get("office_distance"),
                "phone": d.get("phone"),
                "address": d.get("address"),
                "employee_shift": d.get("employee_shift"),
                "shift_type": d.get("shift_type"),
            })
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Skipping driver with unusable fields: {d.get('id')}")
    return drivers


def solve_instance(users: List[Dict[str, Any]],
                   drivers: List[Dict[str, Any]],
                   office: Tuple[float, float],
                   profile: Dict[str, Any],
                   runtime_seconds: float,
                   seed: int) -> Tuple[List[List[int]], List[int]]:
    """Global solve. Returns (routes_as_client_index_lists, dropped_client_idxs).

    Each route is ordered pickups for the driver whose vehicle-type index
    matches the route's position in the returned list (index = driver idx).
    """
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    n_users = len(users)
    if not users or not drivers:
        return [], list(range(n_users))

    drop_penalty_m = int(round(profile.get("drop_penalty_km",
                                            config.SOLVER_PROFILE_DEFAULT["drop_penalty_km"]) * 1000))
    fixed_cost_m = int(round(profile.get("vehicle_fixed_cost_km",
                                         config.SOLVER_PROFILE_DEFAULT["vehicle_fixed_cost_km"]) * 1000))

    # Distance matrix over users + drivers + office, via the RoutingChain
    # (cached haversine now; OSRM/Google road distances when enabled).
    from app.infrastructure.routing import get_routing_chain
    chain = get_routing_chain()
    coords = [(u["latitude"], u["longitude"]) for u in users] + \
             [(d["latitude"], d["longitude"]) for d in drivers] + [office]
    km_matrix = chain.matrix_km(coords)
    matrix = [[0 if i == j else max(1, int(round((km or 0.0) * 1000)))
               for j, km in enumerate(row)] for i, row in enumerate(km_matrix)]
    office_idx = n_users + len(drivers)

    mod = Model()
    locs = [mod.add_location(lon, lat) for lat, lon in coords]  # x=lon, y=lat

    office_depot = mod.add_depot(location=locs[office_idx], name="office")
    home_depots = [mod.add_depot(location=locs[n_users + d], name=f"home_{d}")
                   for d in range(len(drivers))]

    clients = []
    for i, u in enumerate(users):
        clients.append(mod.add_client(
            location=locs[i], delivery=1, required=False,
            prize=drop_penalty_m, name=f"u_{u['user_id']}"))

    for d, depot in enumerate(home_depots):
        mod.add_vehicle_type(
            num_available=1, capacity=drivers[d]["capacity"],
            start_depot=depot, end_depot=office_depot,
            fixed_cost=fixed_cost_m, name=f"d_{drivers[d]['driver_id']}")

    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            if i == j:
                mod.add_edge(a, b, distance=0)
            else:
                mod.add_edge(a, b, distance=matrix[i][j])

    res = mod.solve(stop=MaxRuntime(runtime_seconds), seed=seed, display=False)

    routes: List[List[int]] = [[] for _ in drivers]
    visited = set()
    for route in res.best.routes():
        driver_idx = route.vehicle_type()
        if 0 <= driver_idx < len(routes):
            stops = [a.idx for a in route if a.is_client()]
            routes[driver_idx] = stops
            visited.update(stops)
    dropped = [i for i in range(n_users) if i not in visited]
    return routes, dropped


def run_solver_assignment(source_id: str,
                          parameter: int = 1,
                          string_param: str = "",
                          choice: str = "",
                          mode: str = "route_efficiency",
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One solver run for the given profile mode; standardized response out.

    `data` may carry an already-fetched API payload (the HTTP dispatch fetches
    once for mode selection); when omitted the payload is fetched here.
    """
    start_time = time.time()
    label = mode  # labels == config solver.profiles keys (canonical mode names)
    logger.info(f"🧩 Starting SOLVER assignment ({mode}) for {source_id}")

    try:
        if data is None:
            data = load_env_and_fetch_data(source_id, parameter, string_param, choice)
        if not data:
            raise ValueError("No data returned by the external API")
        validate_input_data(data)

        cfg = config.solver_config()
        profile = config.solver_profile(mode)

        users = _user_records(data)
        drivers = _driver_records(data)
        office = config.office_coordinates(data)
        if not users or not drivers:
            raise ValueError(f"Empty instance: {len(users)} users, {len(drivers)} drivers")

        t0 = time.time()
        routes_by_driver, dropped = solve_instance(
            users, drivers, office, profile,
            float(cfg.get("runtime_seconds", 3.0)), int(cfg.get("seed", 1)))
        solve_seconds = time.time() - t0
        logger.info(f"🧩 PyVRP solved in {solve_seconds:.2f}s: "
                    f"{len(users) - len(dropped)} assigned, {len(dropped)} dropped")

        routes: List[Dict[str, Any]] = []
        used_drivers = set()
        for d_idx, stops in enumerate(routes_by_driver):
            if not stops:
                continue
            d = drivers[d_idx]
            used_drivers.add(d_idx)
            assigned = []
            for rank, c_idx in enumerate(stops, 1):
                u = dict(users[c_idx])
                u["pickup_order"] = rank
                assigned.append(u)
            route = {
                "driver_id": d["driver_id"],
                "vehicle_id": d["vehicle_id"],
                "latitude": d["latitude"],
                "longitude": d["longitude"],
                "capacity": d["capacity"],
                "first_name": d.get("first_name"),
                "last_name": d.get("last_name"),
                "email": d.get("email"),
                "vehicle_name": d.get("vehicle_name"),
                "vehicle_no": d.get("vehicle_no"),
                "chasis_no": d.get("chasis_no"),
                "color": d.get("color"),
                "registration_no": d.get("registration_no"),
                "shift_type_id": d.get("shift_type_id"),
                "assigned_users": assigned,
            }
            # Same pricebook rule as the legacy pipelines: pricebook nearest
            # the farthest assigned user (membership, not stop order).
            route["price_book_id"] = calculate_pricebook_id(
                route=route, data=data,
                company_lat=office[0], company_lon=office[1], logger=logger)
            routes.append(route)

        unassigned_users = [dict(users[i]) for i in dropped]
        unassigned_drivers = [dict(drivers[d_idx])
                              for d_idx in range(len(drivers))
                              if d_idx not in used_drivers]

        # Road-truth ordering on top of the solver's sequences (Google
        # Directions when enabled; solver sequences survive as fallback).
        try:
            integration = get_order_integration(db_name=source_id,
                                                algorithm_name="solver")
            routes = integration.apply_optimal_ordering(
                routes, office[0], office[1])
        except Exception as order_error:
            logger.warning(f"Ordering layer skipped ({order_error}); "
                           "keeping solver sequences")

        execution_time = time.time() - start_time
        response = build_standard_response(
            status="true",
            execution_time=execution_time,
            routes=routes,
            unassigned_users=unassigned_users,
            unassigned_drivers=unassigned_drivers,
            optimization_mode=label,
            parameter=parameter,
            company=data.get("company"),
            shift=data.get("shift"),
            string_param=string_param,
            choice=choice,
            metadata={
                "engine": "pyvrp",
                "profile": profile,
                "solve_seconds": round(solve_seconds, 3),
                "distance_source": "routing_chain",
            },
        )
        save_standardized_response(response,
                                   f"drivers_and_routes_solver_{mode}.json")
        log_response_metrics(response, label)
        return response

    except Exception as e:
        logger.error(f"Solver assignment failed: {e}", exc_info=True)
        return create_error_response(
            error_message=f"Solver assignment failed: {e}",
            execution_time=time.time() - start_time,
            optimization_mode=mode,
            parameter=parameter,
            string_param=string_param,
            choice=choice,
        )


def solver_enabled() -> bool:
    return config.solver_enabled()


# ---- Entry points (one per mode) -------------------------------------------

def run_solver_route_efficiency_assignment(source_id: str, parameter: int = 1,
                                           string_param: str = "",
                                           choice: str = "",
                                           data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return run_solver_assignment(source_id, parameter, string_param, choice,
                                 mode="route_efficiency", data=data)


def run_solver_capacity_assignment(source_id: str, parameter: int = 1,
                                   string_param: str = "",
                                   choice: str = "",
                                   data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return run_solver_assignment(source_id, parameter, string_param, choice,
                                 mode="capacity_optimization", data=data)


def run_solver_balance_assignment(source_id: str, parameter: int = 1,
                                  string_param: str = "",
                                  choice: str = "",
                                  data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return run_solver_assignment(source_id, parameter, string_param, choice,
                                 mode="balanced_optimization", data=data)


def run_solver_road_assignment(source_id: str, parameter: int = 1,
                               string_param: str = "",
                               choice: str = "",
                               data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return run_solver_assignment(source_id, parameter, string_param, choice,
                                 mode="road_aware_route_optimization", data=data)
