"""
Pickup-order endpoints: recalculate pickup sequences for already-built
routes via the ordering layer (app/infrastructure/ordering).
"""

import traceback

from fastapi import APIRouter, HTTPException

from app.features.pickup_order.helpers import (
    apply_google_ordering_to_routes,
    transform_internal_to_v2_format,
    transform_routes_to_google_format,
    transform_v2_to_internal_format,
)
from app.features.pickup_order.models import (
    DataPayload,
    DriverWithRoutes,
    RecalculatePickupOrderRequest,
    RecalculatePickupOrderRequestV2,
    RecalculatePickupOrderResponse,
    RecalculatePickupOrderResponseV2,
    RouteV2,
)

router = APIRouter()


@router.post(
    "/pickup_order",
    summary="Pickup Order",
    description="Recalculate pickup orders for drivers with assigned users using Google Maps API"
)
async def pickup_order(request: RecalculatePickupOrderRequest):
    """
    Recalculate pickup orders for drivers with assigned users using Google Maps API.
    Updates pickup_order field while preserving all other data.
    """
    try:
        print(f"🚗 Starting pickup order recalculation for {len(request.data.routes)} drivers")

        # Validate input
        if not request.data.routes:
            raise HTTPException(status_code=400, detail="No routes provided in request")

        # Extract office coordinates from company data
        office_lat = request.data.company.latitude
        office_lon = request.data.company.longitude

        print(f"📍 Office coordinates: {office_lat}, {office_lon}")

        # Transform Pydantic models to dictionary format
        transformed_routes = transform_routes_to_google_format(request.data.routes)

        # Apply Google Maps API ordering
        optimized_routes = apply_google_ordering_to_routes(
            transformed_routes,
            office_lat,
            office_lon,
            request.data.company.id
        )

        # Convert back to Pydantic models
        response_routes = []
        for route_dict in optimized_routes:
            # Remove ordering_source field if present
            route_dict.pop('ordering_source', None)

            # Create DriverWithRoutes model
            route_model = DriverWithRoutes(**route_dict)
            response_routes.append(route_model)

        # Create response data payload
        response_data = DataPayload(
            company=request.data.company,
            routes=response_routes
        )

        # Generate optimization summary
        total_users = sum(len(route.assigned_users) for route in response_routes)
        optimization_summary = {
            "total_drivers": len(response_routes),
            "total_users": total_users,
            "google_api_used": True,
            "office_coordinates": {
                "latitude": office_lat,
                "longitude": office_lon
            }
        }

        print(f"✅ Pickup order recalculation successful")
        print(f"   - Total drivers: {len(response_routes)}")
        print(f"   - Total users: {total_users}")

        return RecalculatePickupOrderResponse(
            status="true",
            message="Pickup orders recalculated successfully using Google Maps API",
            data=response_data,
            optimization_summary=optimization_summary
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error in pickup order recalculation: {str(e)}")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/pickup_orderV2",
    summary="Pickup Order V2 (Simplified)",
    description="Recalculate pickup orders using Google Maps API with simplified data structure"
)
async def pickup_order_v2(request: RecalculatePickupOrderRequestV2):
    """
    Recalculate pickup orders using Google Maps API with simplified data structure.
    V2 API with fewer fields - only essential data required for optimization.
    Flat structure without nested 'data' field.
    """
    try:
        print(f"🚗 Starting pickup order V2 recalculation for {len(request.routes)} drivers")

        # Validate input
        if not request.routes:
            raise HTTPException(status_code=400, detail="No routes provided in request")

        # Extract office coordinates from company data
        office_lat = request.company.latitude
        office_lon = request.company.longitude

        print(f"📍 Office coordinates: {office_lat}, {office_lon}")

        # Transform V2 format to internal format
        transformed_routes = transform_v2_to_internal_format(request.routes)

        # Apply Google Maps API ordering (use None for company_id as V2 doesn't have it)
        optimized_routes = apply_google_ordering_to_routes(
            transformed_routes,
            office_lat,
            office_lon,
            None  # No company_id in V2
        )

        # Transform back to V2 simplified format
        v2_routes = transform_internal_to_v2_format(optimized_routes)

        # Create response routes
        response_routes = [RouteV2(**route) for route in v2_routes]

        # Log summary
        total_users = sum(len(route.assigned_users) for route in response_routes)
        print(f"✅ Pickup order V2 recalculation successful")
        print(f"   - Total drivers: {len(response_routes)}")
        print(f"   - Total users: {total_users}")

        return RecalculatePickupOrderResponseV2(
            status="true",
            message="Pickup orders recalculated successfully using Google Maps API (V2)",
            company=request.company,
            routes=response_routes
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error in pickup order V2 recalculation: {str(e)}")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
