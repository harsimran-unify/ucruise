"""
Assignment endpoints: /assign-drivers (safety override, then solver profiles
by ride-settings priority), /routes (last saved result), /visualize.
"""

import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from app.core import config
from app.infrastructure.logging.logger import get_logger
from app.infrastructure.source_api import load_env_and_fetch_data
from app.features.assignment.services import solver_pipeline

logger = get_logger()
router = APIRouter()

# ride_settings priority -> solver profile mode
PRIORITY_MODES = {
    1: "capacity_optimization",
    2: "balanced_optimization",
    3: "road_aware_route_optimization",
}


@router.post("/assign-drivers/{source_id}/{parameter}/{string_param}/{choice}")
def assign_drivers(source_id: str, parameter: int, string_param: str,
                   choice: str):
    try:
        print(
            f"🚗 Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}, choice: {choice}"
        )

        # Handle empty choice parameter
        if not choice or choice.strip() == "":
            choice = " "  # Use space instead of empty string

        # Fetch once for mode selection; only data-fetch failures fall back
        # to the default profile run (which re-fetches and surfaces the
        # error as a standard failed response).
        try:
            data = load_env_and_fetch_data(source_id, parameter, string_param,
                                           choice)
        except Exception as api_error:
            print(f"⚠️ API error, using default algorithm: {api_error}")
            data = None

        if data is not None and data.get("safety", 0) == 1:
            # Safety flag overrides everything; gender pairing is not
            # expressible in the VRP model, so this mode keeps its own
            # dedicated pipeline.
            print("🔒 SAFETY FLAG DETECTED (safety=1) - Running safety algorithm (override)")
            from app.features.assignment.services.safety import run_safety_assignment_simplified
            result = run_safety_assignment_simplified(source_id, parameter,
                                                      string_param, choice)
            result["optimization_mode"] = "safety_optimization"
        else:
            priority = None
            if data is not None:
                ride_settings = data.get("ride_settings", {}) or {}
                priority = ride_settings.get("pic_priority")
                if priority is None:
                    priority = ride_settings.get("drop_priority")

            print(f"🤖 Detected algorithm priority: {priority}")
            mode = PRIORITY_MODES.get(priority, "route_efficiency")
            print(f"🧩 Using SOLVER profile: {mode}")

            runner = {
                "capacity_optimization": solver_pipeline.run_solver_capacity_assignment,
                "balanced_optimization": solver_pipeline.run_solver_balance_assignment,
                "road_aware_route_optimization": solver_pipeline.run_solver_road_assignment,
            }.get(mode, solver_pipeline.run_solver_route_efficiency_assignment)
            result = runner(source_id, parameter, string_param, choice, data=data)

            # Preserve the historical endpoint label for the default arm
            if mode == "route_efficiency":
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
                with open(config.MAIN_OUTPUT_FILE, "w", encoding="utf-8") as f:
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


@router.get("/routes")
def get_routes():
    # Read the main route file written by the last assignment run
    if os.path.exists(config.MAIN_OUTPUT_FILE):
        try:
            with open(config.MAIN_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both old format (list) and new standardized format (object with data)
                if isinstance(data, list):
                    # Old format - return directly
                    return FileResponse(str(config.MAIN_OUTPUT_FILE),
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

    # Return empty but valid response
    return {
        "status": "false",
        "message": "No valid routes data available. Run assignment first.",
        "data": []
    }


@router.get("/visualize", response_class=HTMLResponse)
def get_visualization():
    # Check for visualize.html in localTesting folder first, then root
    for candidate in config.VISUALIZE_CANDIDATES:
        if candidate.exists():
            return FileResponse(str(candidate))
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
