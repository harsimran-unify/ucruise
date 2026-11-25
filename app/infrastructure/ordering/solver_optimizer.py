"""
Solver-based pickup ordering (Phase 2 of the solver-migration plan).

Replaces two weaknesses of the legacy ordering stack:

1. The haversine nearest-neighbor fallback in order_integration.py chains
   pickups starting from the OFFICE, not the driver's position — the wrong
   geometry for a commute that ends at the office.
2. Google Directions waypoint optimization is rate-limited and per-call
   billed; a proper TSP over a cached distance matrix is free, instant, and
   optimal for route-sized problems.

Sequences are computed over the RoutingChain (routing/), so they upgrade
automatically: haversine today, OSRM/Google road distances once an engine is
enabled — no code change.

For route-sized stops (<= exact_max_stops) the tour is solved EXACTLY by
brute-force permutation; larger stops get nearest-neighbor + 2-opt + or-opt.
Routes never exceed max_users_per_cluster in practice, so the exact path is
the one that runs.
"""

import itertools
import threading
from typing import Any, Dict, List, Optional, Tuple

from app.infrastructure.logging.logger import get_logger

logger = get_logger()

# Route sizes beyond this use the heuristic instead of brute force.
EXACT_MAX_STOPS = 8


def _tsp_order(costs):
    """Return (order, total) minimizing travel: costs is a (n+1)x(n+2) matrix
    over [start] + stops + [end]. Exact for small n, NN+2-opt+or-opt above."""
    n = len(costs) - 2

    if n <= EXACT_MAX_STOPS:
        best_order, best_total = None, None
        for perm in itertools.permutations(range(n)):
            total = costs[0][perm[0] + 1]
            for a, b in zip(perm, perm[1:]):
                total += costs[a + 1][b + 1]
            total += costs[perm[-1] + 1][n + 1]
            if best_total is None or total < best_total:
                best_order, best_total = list(perm), total
        return best_order, best_total

    # Nearest neighbor from start
    remaining = set(range(n))
    current = 0
    order = []
    while remaining:
        nxt = min(remaining, key=lambda j: costs[current][j + 1])
        order.append(nxt)
        remaining.discard(nxt)
        current = nxt + 1
    total = _tour_cost(costs, order)

    # 2-opt + or-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                cand_total = _tour_cost(costs, cand)
                if cand_total < total - 1e-9:
                    order, total, improved = cand, cand_total, True
            # or-opt: relocate single stop
            for j in range(len(order)):
                if j == i:
                    continue
                cand = order.copy()
                stop = cand.pop(i)
                cand.insert(j if j < i else j, stop)
                cand_total = _tour_cost(costs, cand)
                if cand_total < total - 1e-9:
                    order, total, improved = cand, cand_total, True
    return order, total


def _tour_cost(costs, order):
    if not order:
        return 0.0
    total = costs[0][order[0] + 1]
    for a, b in zip(order, order[1:]):
        total += costs[a + 1][b + 1]
    return total + costs[order[-1] + 1][len(costs) - 2 + 1]


class SolverPickupOptimizer:
    """Orders each route's pickups with an exact/heuristic TSP over the
    routing chain (driver start -> pickups -> office)."""

    def __init__(self):
        self._lock = threading.Lock()
        try:
            from app.infrastructure.routing import get_routing_chain
            self.chain = get_routing_chain()
        except Exception as e:
            logger.warning(f"RoutingChain unavailable ({e}); ordering falls "
                           "back to plain haversine")
            self.chain = None

    def _distance_km(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        if self.chain is not None:
            return self.chain.point_distance_km(a[0], a[1], b[0], b[1])
        from app.infrastructure.routing.base import haversine_km
        return haversine_km(a[0], a[1], b[0], b[1])

    def optimize_pickup_sequence(self, driver_pos: Tuple[float, float],
                                 users: List[Dict[str, Any]],
                                 office_pos: Tuple[float, float]):
        """Reorder users for one route.

        Returns (ordered_users, info_dict). Users keep all fields;
        pickup_order is set 1..n and route_distance_km reflects the chain
        cost of driver -> stops -> office.
        """
        n = len(users)
        if n <= 1:
            if n == 1:
                users[0]["pickup_order"] = 1
            return users, {"ordering_source": "solver_single",
                           "route_distance_km": None}

        # cost matrix over [start] + n stops + [end]
        nodes = [tuple(driver_pos)] + [
            (float(u["latitude"]), float(u["longitude"])) for u in users
        ] + [tuple(office_pos)]
        size = len(nodes)
        costs = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i != j:
                    costs[i][j] = self._distance_km(nodes[i], nodes[j])

        order, total = _tsp_order(costs)

        ordered = []
        for rank, user_idx in enumerate(order, 1):
            user = users[user_idx].copy()
            user["pickup_order"] = rank
            ordered.append(user)
        info = {
            "ordering_source": "solver_tsp",
            "route_distance_km": round(total, 3),
            "route_duration_min": round(total / 0.5 * 60, 1),  # ~30 km/h
            "sequence_km_saved_vs_input": round(
                _tour_cost(costs, list(range(n))) - total, 3),
        }
        return ordered, info

    def optimize_route(self, route: Dict[str, Any],
                       office_lat: float, office_lon: float) -> Dict[str, Any]:
        """Order one route dict (legacy pipeline shape); never raises."""
        try:
            driver_lat = route.get("latitude")
            driver_lon = route.get("longitude")
            users = route.get("assigned_users") or []
            if driver_lat is None or driver_lon is None or not users:
                out = route.copy()
                out["ordering_source"] = "solver_skipped"
                return out

            ordered, info = self.optimize_pickup_sequence(
                (float(driver_lat), float(driver_lon)),
                users, (office_lat, office_lon))
            out = route.copy()
            out["assigned_users"] = ordered
            out.update(info)
            return out
        except Exception as e:
            logger.warning(f"Solver ordering failed for route "
                           f"{route.get('driver_id')}: {e} - keeping order")
            out = route.copy()
            out["ordering_source"] = "solver_failed"
            return out

    def optimize_routes(self, routes: List[Dict[str, Any]],
                        office_lat: float, office_lon: float) -> List[Dict[str, Any]]:
        with self._lock:
            return [self.optimize_route(r, office_lat, office_lon) for r in routes]

    def save_cache(self):
        if self.chain is not None:
            self.chain.save_cache()


_optimizer: Optional[SolverPickupOptimizer] = None
_optimizer_lock = threading.Lock()


def get_solver_optimizer() -> SolverPickupOptimizer:
    global _optimizer
    if _optimizer is None:
        with _optimizer_lock:
            if _optimizer is None:
                _optimizer = SolverPickupOptimizer()
    return _optimizer
