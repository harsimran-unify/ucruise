"""
ucruise entry point — FastAPI app, router registration, logging session.

Run from the project root:  uv run uvicorn main:app --host 0.0.0.0 --port 5000
"""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.infrastructure.logging.logger import start_session
from app.features.assignment.routes import router as assignment_router
from app.features.pickup_order.routes import router as pickup_order_router

# Force UTF-8 stdio: this codebase prints emoji, which crashes with
# UnicodeEncodeError on Windows when stdout is a pipe (cp1252 fallback)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One log session per server run (clears logs/assignment_*.log, starts
    # the session file) — previously done at legacy-module import time.
    start_session()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assignment_router)
app.include_router(pickup_order_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}


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
            "System automatically selects the solver profile based on API ride_settings priority; safety=1 always runs the dedicated safety pipeline",
            "priority_1":
            "Capacity Optimization (solver profile: capacity_optimization) - high vehicle fixed cost consolidates into fewer, fuller vehicles",
            "priority_2":
            "Balanced Optimization (solver profile: balanced_optimization) - low fixed cost spreads load across more vehicles",
            "priority_3":
            "Road-Aware Routing (solver profile: road_aware_route_optimization) - road-distance matrix via the routing engine chain",
            "default":
            "Route Efficiency (solver profile: route_efficiency) - moderate consolidation, distance focus"
        },
        "pickup_order_api": {
            "endpoint": "/pickup_order",
            "method": "POST",
            "description": "Recalculates pickup orders using the ordering layer (solver TSP / Google Directions)",
            "note": "Accepts company data with routes and returns optimized pickup sequences"
        },
        "pickup_order_api_v2": {
            "endpoint": "/pickup_orderV2",
            "method": "POST",
            "description": "Recalculates pickup orders using the ordering layer (Simplified)",
            "note": "V2 with simplified data structure - only essential fields required",
            "fields": {
                "company": ["latitude", "longitude"],
                "route": ["driver_id", "route_no", "driver_location", "assigned_users"],
                "assigned_user": ["user_id", "latitude", "longitude", "pickup_order", "route_no"]
            }
        },
        "usage":
        "Solver profile is automatically selected from API ride_settings priority value"
    }
