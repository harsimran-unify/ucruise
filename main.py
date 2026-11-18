from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import os
import json
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssignmentRequest(BaseModel):
    source_id: str


# Pydantic models for pickup order recalculation API
class UserWithPickupOrder(BaseModel):
    user_id: str
    latitude: float = Field(..., description="User latitude - accepts string or float")
    longitude: float = Field(..., description="User longitude - accepts string or float")
    office_distance: Optional[float] = None
    first_name: Optional[str] = None
    last_name: Optional[Any] = None  # Can be null
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    employee_shift: Optional[str] = None
    shift_type: Optional[str] = None
    route_no: Optional[int] = None
    pickup_order: Optional[int] = None

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class DriverWithRoutes(BaseModel):
    driver_id: str
    vehicle_id: Optional[str] = None
    latitude: float = Field(..., description="Driver latitude - accepts string or float")
    longitude: float = Field(..., description="Driver longitude - accepts string or float")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    vehicle_name: Optional[str] = None
    vehicle_no: Optional[str] = None
    capacity: Optional[int] = None
    chasis_no: Optional[str] = None
    color: Optional[str] = None
    registration_no: Optional[str] = None
    shift_type_id: Optional[int] = None
    route_no: Optional[int] = None
    assigned_users: List[UserWithPickupOrder] = []

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class CompanyInfo(BaseModel):
    id: int
    address: Optional[str] = None
    latitude: float = Field(..., description="Company latitude - accepts string or float")
    longitude: float = Field(..., description="Company longitude - accepts string or float")

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class DataPayload(BaseModel):
    company: CompanyInfo
    routes: List[DriverWithRoutes]


class RecalculatePickupOrderRequest(BaseModel):
    data: DataPayload


class RecalculatePickupOrderResponse(BaseModel):
    status: str
    message: str
    data: DataPayload
    optimization_summary: Optional[Dict[str, Any]] = None


# ============== V2 API Models (Simplified) ==============

class DriverLocationV2(BaseModel):
    latitude: float
    longitude: float


class AssignedUserV2(BaseModel):
    user_id: int
    latitude: float
    longitude: float
    pickup_order: Optional[int] = None
    route_no: int

    @validator('user_id', 'route_no', pre=True)
    def parse_integers(cls, v):
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"Invalid integer value: {v}")
        return v

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class RouteV2(BaseModel):
    driver_id: int
    route_no: int
    driver_location: DriverLocationV2
    assigned_users: List[AssignedUserV2] = []

    @validator('driver_id', 'route_no', pre=True)
    def parse_integers(cls, v):
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"Invalid integer value: {v}")
        return v

    class Config:
        extra = "allow"


class CompanyInfoV2(BaseModel):
    latitude: float
    longitude: float

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class RecalculatePickupOrderRequestV2(BaseModel):
    company: CompanyInfoV2
    routes: List[RouteV2]  # No nesting - flat structure


class RecalculatePickupOrderResponseV2(BaseModel):
    status: str
    message: str
    company: CompanyInfoV2
    routes: List[RouteV2]


# Helper functions for pickup order recalculation

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
        from ordering.order_integration import OrderIntegration

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


@app.post(
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


@app.post(
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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}/{choice}")
def assign_drivers(source_id: str, parameter: int, string_param: str,
                   choice: str):
    try:
        print(
            f"🚗 Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}, choice: {choice}"
        )

        # Clear any existing route files to ensure fresh assignment
        route_files = [
            "drivers_and_routes.json", "drivers_and_routes_capacity.json",
            "drivers_and_routes_balance.json",
            "drivers_and_routes_road_aware.json"
        ]
        for filename in route_files:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️ Cleared cached file: {filename}")

        # Import here to avoid circular imports
        from algorithm.base.base import load_env_and_fetch_data

        # Handle empty choice parameter
        if not choice or choice.strip() == "":
            choice = " "  # Use space instead of empty string
            
        # First, get the data to determine which algorithm to use
        try:
            data = load_env_and_fetch_data(source_id, parameter, string_param,
                                           choice)

            # Check safety flag first - this overrides all other algorithm selection
            safety_flag = data.get("safety", 0)

            if safety_flag == 1:
                print("🔒 SAFETY FLAG DETECTED (safety=1) - Running safety algorithm (override)")
                from algorithm.safety.safety import run_safety_assignment_simplified
                result = run_safety_assignment_simplified(source_id, parameter, string_param, choice)
                result["optimization_mode"] = "safety_optimization"
            else:
                # Safety flag is 0 or missing - use ride settings as before
                print(f"📊 Using ride_settings for algorithm selection (safety={safety_flag})")

                # Get algorithm priority from ride_settings
                ride_settings = data.get("ride_settings", {})
                pic_priority = ride_settings.get("pic_priority")
                drop_priority = ride_settings.get("drop_priority")

                # Use pic_priority first, then drop_priority, then default to None
                algorithm_priority = pic_priority
                if algorithm_priority is None:
                    algorithm_priority = drop_priority

                print(f"🤖 Detected algorithm priority: {algorithm_priority}")
                print(f"🔍 Raw ride_settings: {ride_settings}")

                # Route to appropriate algorithm based on priority
                if algorithm_priority == 1:
                    print("🎪 Using CAPACITY OPTIMIZATION (Priority 1)")
                    from algorithm.capacity.capacity import run_assignment_capacity
                    result = run_assignment_capacity(source_id, parameter,
                                                     string_param, choice)
                    result["optimization_mode"] = "capacity_optimization"

                elif algorithm_priority == 2:
                    print("⚖️ Using BALANCED OPTIMIZATION (Priority 2)")
                    from algorithm.balance.balance import run_assignment_balance
                    result = run_assignment_balance(source_id, parameter,
                                                    string_param, choice)
                    result["optimization_mode"] = "balanced_optimization"

                elif algorithm_priority == 3:
                    print("🗺️ Using ROAD-AWARE ROUTING (Priority 3)")
                    from algorithm.road.road import run_road_aware_assignment
                    result = run_road_aware_assignment(source_id, parameter,
                                                       string_param, choice)
                    result["optimization_mode"] = "road_aware_route_optimization"

                else:
                    print(
                        f"🎯 Using ROUTE EFFICIENCY (Default) - priority was: {algorithm_priority}"
                    )
                    from algorithm.base.base import run_assignment
                    result = run_assignment(source_id, parameter, string_param,
                                            choice)
                    result["optimization_mode"] = "route_efficiency_default"

        except Exception as api_error:
            print(f"⚠️ API error, using default algorithm: {api_error}")
            from algorithm.base.base import run_assignment
            result = run_assignment(source_id, parameter, string_param, choice)
            result["optimization_mode"] = "route_efficiency_default"

        # Ensure result has proper structure
        if not isinstance(result, dict):
            result = {
                "status": "false",
                "details": "Invalid result format",
                "data": []
            }

        # Add parameters to result
        result["parameter"] = parameter
        result["string_param"] = string_param
        result["choice"] = choice

        # Ensure data is always an array
        if "data" not in result:
            result["data"] = []
        elif result["data"] is None:
            result["data"] = []
        elif not isinstance(result["data"], list):
            result["data"] = []

        if result["status"] == "true":
            print(f"✅ Assignment successful. Routes: {len(result['data'])}")

            # Save the complete standardized response to the main file for /routes endpoint
            try:
                with open("drivers_and_routes.json", "w") as f:
                    json.dump(result, f, indent=2)
                print("💾 Results saved to drivers_and_routes.json")
            except Exception as save_error:
                print(f"⚠️ Failed to save results: {save_error}")
        else:
            print(
                f"❌ Assignment failed: {result.get('details', 'Unknown error')}"
            )

        return result

    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "false",
            "details": f"Server error: {str(e)}",
            "data": [],
            "parameter": parameter,
            "string_param": string_param,
            "choice": choice,
            "optimization_mode": "error"
        }


@app.get("/routes")
def get_routes():
    # Check for the main route file first
    if os.path.exists("drivers_and_routes.json"):
        try:
            with open("drivers_and_routes.json", 'r') as f:
                data = json.load(f)
                # Handle both old format (list) and new standardized format (object with data)
                if isinstance(data, list):
                    # Old format - return directly
                    return FileResponse("drivers_and_routes.json",
                                        media_type="application/json")
                elif isinstance(data, dict) and "data" in data:
                    # New standardized format - return the data array
                    return {
                        "status": "true",
                        "data": data["data"] if data["data"] else [],
                        "message": "Data loaded from drivers_and_routes.json"
                    }
                else:
                    return {
                        "status": "true",
                        "data": [],
                        "message": "No valid data found in drivers_and_routes.json"
                    }
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading drivers_and_routes.json: {e}")

    # Check other route files as backup
    route_files = [
        "drivers_and_routes_capacity.json", "drivers_and_routes_balance.json",
        "drivers_and_routes_road_aware.json"
    ]

    for filename in route_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return FileResponse(filename,
                                            media_type="application/json")
                    else:
                        return {
                            "status": "true",
                            "data": data if data else [],
                            "message": f"Data loaded from {filename}"
                        }
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading {filename}: {e}")
                continue

    # Return empty but valid response
    return {
        "status": "false",
        "message": "No valid routes data available. Run assignment first.",
        "data": []
    }


@app.get("/visualize", response_class=HTMLResponse)
def get_visualization():
    # Check for visualize.html in localTesting folder first, then root
    local_testing_path = os.path.join("localTesting", "visualize.html")
    if os.path.exists(local_testing_path):
        return FileResponse(local_testing_path)
    elif os.path.exists("visualize.html"):
        return FileResponse("visualize.html")
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>Visualization Not Available</title></head>
        <body style='font-family: Arial, sans-serif; text-align: center; padding: 50px;'>
            <h1>🚗 Driver Routes Dashboard</h1>
            <p>Visualization file not found. Please run assignment algorithm first.</p>
            <p><a href="/routes">View Route Data</a> | <a href="/health">Health Check</a></p>
        </body>
        </html>
        """)


@app.get("/")
def root():
    return {
        "message":
        "Driver Assignment API with Multiple Optimization Modes",
        "endpoints": [
            "/assign-drivers/{source_id}/{parameter}/{string_param}/{choice}",
            "/pickup_order",
            "/pickup_orderV2",
            "/routes", "/visualize", "/health"
        ],
        "optimization_modes": {
            "automatic_detection":
            "System automatically selects algorithm based on API ride_settings priority",
            "priority_1":
            "Capacity Optimization (assign_capacity.py) - Maximizes seat utilization",
            "priority_2":
            "Balanced Optimization (assign_balance.py) - 50/50 route efficiency + capacity",
            "priority_3":
            "Road-Aware Routing (assign_route.py) - Uses road network data",
            "default":
            "Route Efficiency (assignment.py) - Prioritizes straight routes"
        },
        "pickup_order_api": {
            "endpoint": "/pickup_order",
            "method": "POST",
            "description": "Recalculates pickup orders using Google Maps API",
            "note": "Accepts company data with routes and returns optimized pickup sequences"
        },
        "pickup_order_api_v2": {
            "endpoint": "/pickup_orderV2",
            "method": "POST",
            "description": "Recalculates pickup orders using Google Maps API (Simplified)",
            "note": "V2 with simplified data structure - only essential fields required",
            "fields": {
                "company": ["latitude", "longitude"],
                "route": ["driver_id", "route_no", "driver_location", "assigned_users"],
                "assigned_user": ["user_id", "latitude", "longitude", "pickup_order", "route_no"]
            }
        },
        "usage":
        "Algorithm is automatically selected from API response _algorithm_priority value"
    }
