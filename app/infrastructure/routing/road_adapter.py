"""
MatrixRoadNetwork — a road-distance adapter over the RoutingChain.

Implements the RoadNetwork surface that consumers expect:

    get_road_distance(lat1, lon1, lat2, lon2)              -> km
    get_route_coherence_score(driver, users, office)       -> 0..1
    is_user_on_route_path(driver, users, user, office, ...) -> bool
    find_nearest_road_node(lat, lon)                       -> (None, None)
    simplify_path_nodes(path, max_nodes=10)                -> path

Coherence and detour checks follow the former GraphML-backed RoadNetwork
implementation, with shortest-path distances replaced by the RoutingChain.
One behavioural difference is deliberate: there are no road *nodes*, so
find_nearest_road_node returns (None, None) — matching the historical
MockRoadNetwork contract — and the graph-path-proximity check inside
is_user_on_route_path is replaced by a distance-to-segment check against
the driver->office line.
"""

import math
from typing import List, Optional, Tuple

from .base import haversine_km
from .chain import RoutingChain

class RoadAwareConfig:
    """Road-awareness tuning constants (formerly roadNetwork/config.py;
    only the attributes this adapter consumes were kept)."""

    MAX_REASONABLE_RATIO = 3.0
    DEFAULT_NEAR_PATH_THRESHOLD_KM = 1.0
    COHERENCE_TOLERANCE_DEFAULT = 0.1
    COHERENCE_TOLERANCE_COVERAGE = 0.2
    DIRECTIONAL_PENALTY_WEIGHT = 0.3
    BACKTRACK_PENALTY_WEIGHT = 0.4
    CORRIDOR_BONUS_WEIGHT = 0.2
    SEQUENCE_PENALTY_WEIGHT = 0.1
    MAX_BACKTRACK_PENALTY = 0.5
    BACKTRACK_PENALTY_SCALE = 5.0


class MatrixRoadNetwork:
    def __init__(self, chain: RoutingChain):
        self.chain = chain
        self.provider = "matrix"

    # -- core distances ------------------------------------------------------

    def get_road_distance(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        return self.chain.point_distance_km(lat1, lon1, lat2, lon2)

    def find_nearest_road_node(self, lat: float, lon: float):
        # No node concept in a matrix provider; road.py's fallbacks handle
        # (None, None) (that is what its own MockRoadNetwork returns).
        return None, None

    def simplify_path_nodes(self, path, max_nodes: int = 10):
        return path

    # -- coherence (ported from RoadNetwork._calculate_coherence_internal) ----

    def get_route_coherence_score(self, driver_pos: Tuple[float, float],
                                  user_positions: List[Tuple[float, float]],
                                  office_pos: Tuple[float, float]) -> float:
        if not user_positions:
            return 1.0

        route_points = [driver_pos] + list(user_positions) + [office_pos]

        total_road = total_straight = 0.0
        for (lat1, lon1), (lat2, lon2) in zip(route_points, route_points[1:]):
            total_road += self.chain.point_distance_km(lat1, lon1, lat2, lon2)
            total_straight += haversine_km(lat1, lon1, lat2, lon2)
        if total_road <= 0:
            return 1.0
        base_efficiency = min(1.0, total_straight / total_road)

        directional_penalty = self._directional_penalty(route_points) * RoadAwareConfig.DIRECTIONAL_PENALTY_WEIGHT
        backtrack_penalty = self._backtrack_penalty(route_points, office_pos) * RoadAwareConfig.BACKTRACK_PENALTY_WEIGHT
        corridor_bonus = self._corridor_bonus(route_points, office_pos) * RoadAwareConfig.CORRIDOR_BONUS_WEIGHT
        sequence_penalty = self._sequence_penalty(user_positions, office_pos) * RoadAwareConfig.SEQUENCE_PENALTY_WEIGHT

        score = base_efficiency - directional_penalty - backtrack_penalty \
            + corridor_bonus - sequence_penalty
        return max(0.0, min(1.0, score))

    # -- path membership ------------------------------------------------------

    def is_user_on_route_path(self, driver_pos: Tuple[float, float],
                              current_user_positions: List[Tuple[float, float]],
                              user_pos: Tuple[float, float],
                              office_pos: Tuple[float, float],
                              max_detour_ratio: float = 2.5,
                              route_type: str = "default") -> bool:
        # 1) Detour ratio via road distances (ported _check_detour_ratio_simple)
        driver_office = self.chain.point_distance_km(*driver_pos, *office_pos)
        if driver_office > 0:
            driver_user = self.chain.point_distance_km(*driver_pos, *user_pos)
            user_office = self.chain.point_distance_km(*user_pos, *office_pos)
            if (driver_user + user_office) / driver_office <= max_detour_ratio:
                return True

        # 2) Proximity to the driver->office line (replaces the graph-path
        #    proximity check; same DEFAULT_NEAR_PATH_THRESHOLD_KM idea)
        if self._near_segment(user_pos, driver_pos, office_pos,
                              RoadAwareConfig.DEFAULT_NEAR_PATH_THRESHOLD_KM):
            return True

        # 3) Coherence impact (ported _check_coherence_impact_relaxed)
        coverage = route_type == "coverage"
        tolerance = (RoadAwareConfig.COHERENCE_TOLERANCE_COVERAGE if coverage
                     else RoadAwareConfig.COHERENCE_TOLERANCE_DEFAULT)
        current = self.get_route_coherence_score(driver_pos, current_user_positions or [], office_pos)
        updated = self.get_route_coherence_score(
            driver_pos, list(current_user_positions or []) + [user_pos], office_pos)
        return updated >= current - tolerance

    # -- geometry helpers (pure math, ported from RoadNetwork) ------------------

    @staticmethod
    def _calculate_bearing(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    @staticmethod
    def _normalize_bearing_difference(diff: float) -> float:
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    @staticmethod
    def _near_segment(point, seg_a, seg_b, threshold_km) -> bool:
        """Distance from point to the lat/lon segment in local-metric projection."""
        lat_r = math.radians(point[0])
        lat1, lon1 = seg_a
        lat2, lon2 = seg_b
        to_m = 111320.0
        px = (point[1] - lon1) * to_m * math.cos(lat_r)
        py = (point[0] - lat1) * to_m
        qx = (lon2 - lon1) * to_m * math.cos(lat_r)
        qy = (lat2 - lat1) * to_m
        seg_len2 = qx * qx + qy * qy
        if seg_len2 <= 0:
            return math.hypot(px, py) / 1000.0 <= threshold_km
        t = max(0.0, min(1.0, (px * qx + py * qy) / seg_len2))
        dist_m = math.hypot(px - t * qx, py - t * qy)
        return dist_m / 1000.0 <= threshold_km

    def _directional_penalty(self, route_points) -> float:
        if len(route_points) < 3:
            return 0.0
        bearings = [self._calculate_bearing(a[0], a[1], b[0], b[1])
                    for a, b in zip(route_points, route_points[1:])]
        if len(bearings) < 2:
            return 0.0
        total_change = 0.0
        for b1, b2 in zip(bearings, bearings[1:]):
            change = abs(self._normalize_bearing_difference(b2 - b1))
            if change > 45:
                total_change += (change - 45) / 135.0
        return min(1.0, total_change / max(1, len(bearings) - 1))

    def _backtrack_penalty(self, route_points, office_pos) -> float:
        if len(route_points) < 2:
            return 0.0
        penalty = 0.0
        prev_distance = haversine_km(*route_points[0], *office_pos)
        for pos in route_points[1:-1]:
            current_distance = haversine_km(*pos, *office_pos)
            if current_distance > prev_distance + 0.5:
                penalty += min(RoadAwareConfig.MAX_BACKTRACK_PENALTY,
                               (current_distance - prev_distance) / RoadAwareConfig.BACKTRACK_PENALTY_SCALE)
            prev_distance = current_distance
        return min(1.0, penalty)

    def _corridor_bonus(self, route_points, office_pos) -> float:
        if len(route_points) < 3:
            return 0.0
        main_bearing = self._calculate_bearing(*route_points[0], *office_pos)
        consistent = total = 0
        for a, b in zip(route_points, route_points[1:]):
            segment_bearing = self._calculate_bearing(*a, *b)
            if abs(self._normalize_bearing_difference(segment_bearing - main_bearing)) <= 45:
                consistent += 1
            total += 1
        return (consistent / total) * 0.3 if total else 0.0

    def _sequence_penalty(self, user_positions, office_pos) -> float:
        if len(user_positions) <= 1:
            return 0.0
        office_distances = [haversine_km(*pos, *office_pos) for pos in user_positions]
        violations = sum(
            1 for d1, d2 in zip(office_distances, office_distances[1:])
            if d1 < d2 - 1.0)
        return (violations / (len(office_distances) - 1)) * 0.2
