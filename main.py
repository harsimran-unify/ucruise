from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json

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

            # Save to the main file for /routes endpoint
            try:
                with open("drivers_and_routes.json", "w") as f:
                    json.dump(result["data"], f, indent=2)
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
                # Ensure data is valid
                if isinstance(data, list):
                    return FileResponse("drivers_and_routes.json",
                                        media_type="application/json")
                else:
                    return {
                        "status": "true",
                        "data": data if data else [],
                        "message": "Data loaded from drivers_and_routes.json"
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
        "usage":
        "Algorithm is automatically selected from API response _algorithm_priority value"
    }
