"""
Transform helpers for the pickup-order API: pydantic models <-> the internal
route-dict format the ordering layer (app/infrastructure/ordering) expects.
"""

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.features.pickup_order.models import DriverWithRoutes, RouteV2


def transform_v2_to_internal_format(routes_v2: List[RouteV2]) -> List[Dict[str, Any]]:
    """
    Transform V2 simplified models to internal format expected by Google API.
    Maps driver_location.lat/lng -> latitude/longitude at route level.
    """
    transformed_routes = []
    for route in routes_v2:
        route_dict = {
            'driver_id': str(route.driver_id),
            'latitude': route.driver_location.latitude,
            'longitude': route.driver_location.longitude,
            'route_no': route.route_no,
            'vehicle_id': None,  # Default value for V2
            'capacity': None,  # Default value for V2
            'assigned_users': []
        }

        # Transform assigned_users
        for user in route.assigned_users:
            user_dict = {
                'user_id': str(user.user_id),
                'latitude': user.latitude,
                'longitude': user.longitude,
                'pickup_order': user.pickup_order,
                'route_no': user.route_no
            }
            route_dict['assigned_users'].append(user_dict)

        transformed_routes.append(route_dict)

    return transformed_routes


def transform_internal_to_v2_format(internal_routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform internal format back to V2 simplified format.
    Extracts only the required fields for V2 response.
    """
    v2_routes = []
    for route in internal_routes:
        v2_route = {
            'driver_id': int(route['driver_id']),
            'route_no': route['route_no'],
            'driver_location': {
                'latitude': route['latitude'],
                'longitude': route['longitude']
            },
            'assigned_users': []
        }

        # Transform assigned_users back to V2 format
        for user in route.get('assigned_users', []):
            v2_user = {
                'user_id': int(user['user_id']),
                'latitude': user['latitude'],
                'longitude': user['longitude'],
                'pickup_order': user.get('pickup_order'),
                'route_no': user['route_no']
            }
            v2_route['assigned_users'].append(v2_user)

        v2_routes.append(v2_route)

    return v2_routes


def transform_routes_to_google_format(routes: List[DriverWithRoutes]) -> List[Dict[str, Any]]:
    """
    Transform Pydantic route models to dictionary format expected by Google API
    """
    transformed_routes = []
    for route in routes:
        route_dict = route.dict()

        # Transform assigned_users to expected format
        transformed_users = []
        for user in route.assigned_users:
            user_dict = user.dict()
            # Ensure lat/lng fields are consistent with what Google API expects
            transformed_users.append(user_dict)

        route_dict['assigned_users'] = transformed_users
        # Ensure vehicle_type is present for compatibility
        if 'vehicle_type' not in route_dict and route_dict.get('capacity'):
            route_dict['vehicle_type'] = route_dict['capacity']

        transformed_routes.append(route_dict)

    return transformed_routes


def apply_google_ordering_to_routes(routes: List[Dict[str, Any]],
                                   office_lat: float,
                                   office_lon: float,
                                   company_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Apply Google Maps API ordering to routes and update pickup_order
    """
    try:
        from app.infrastructure.ordering.order_integration import OrderIntegration

        # Initialize OrderIntegration with company ID for caching
        db_name = str(company_id) if company_id else "default"
        order_integration = OrderIntegration(db_name=db_name, algorithm_name="pickup_order_api")

        # Apply Google ordering
        optimized_routes = order_integration.apply_optimal_ordering(
            routes, office_lat, office_lon
        )

        # Ensure pickup_order is properly set
        for route in optimized_routes:
            for idx, user in enumerate(route.get('assigned_users', [])):
                user['pickup_order'] = idx + 1

        return optimized_routes

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Google Maps ordering system not available: {str(e)}"
        )
    except Exception as e:
        # Handle Google API errors
        error_msg = str(e).lower()
        if "api key" in error_msg:
            raise HTTPException(
                status_code=503,
                detail="Google Maps API key not configured or invalid"
            )
        elif "quota" in error_msg or "rate limit" in error_msg:
            raise HTTPException(
                status_code=503,
                detail="Google Maps API quota exceeded or rate limited"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Google Maps API error: {str(e)}"
            )
