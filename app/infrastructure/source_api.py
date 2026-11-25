"""
External data API client.

Fetches the users/drivers payload for one company/shift from the external
API (GET {API_URL}/{source_id}/{parameter}/{string_param}/{choice} with a
Bearer token from .env). This was previously `load_env_and_fetch_data` in
the legacy algorithm/base/base.py; behavior is unchanged.
"""

import json

import requests

from app.core.config import api_credentials
from app.infrastructure.logging.logger import get_logger


def load_env_and_fetch_data(source_id: str,
                            parameter: int = 1,
                            string_param: str = "",
                            choice: str = ""):
    """Load .env credentials and fetch data from the external API."""
    logger = get_logger()

    BASE_API_URL, API_AUTH_TOKEN = api_credentials()

    # Send all parameters along with source_id in the API URL
    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{string_param}/{choice}"
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    logger.info(f"📡 Making API call to: {API_URL}")
    resp = requests.get(API_URL, headers=headers)
    resp.raise_for_status()

    # Check if response body is empty
    if len(resp.text.strip()) == 0:
        raise ValueError(
            f"API returned empty response body. "
            f"Status: {resp.status_code}, "
            f"Content-Type: {resp.headers.get('content-type', 'unknown')}, "
            f"URL: {API_URL}")

    try:
        payload = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(
            f"API returned invalid JSON. "
            f"Status: {resp.status_code}, "
            f"Content-Type: {resp.headers.get('content-type', 'unknown')}, "
            f"Response body: '{resp.text[:200]}...', "
            f"JSON Error: {str(e)}")

    if not payload.get("status") or "data" not in payload:
        raise ValueError(
            "Unexpected response format: 'status' or 'data' missing")

    # Use the provided parameters
    data = payload["data"]
    data["_parameter"] = parameter
    data["_string_param"] = string_param
    data["_choice"] = choice

    # Handle nested drivers structure
    if "drivers" in data:
        drivers_data = data["drivers"]
        data["driversUnassigned"] = drivers_data.get("driversUnassigned", [])
        data["driversAssigned"] = drivers_data.get("driversAssigned", [])
    else:
        data["driversUnassigned"] = data.get("driversUnassigned", [])
        data["driversAssigned"] = data.get("driversAssigned", [])

    # Extract safety flag first - this overrides ride settings
    safety_flag = data.get("safety", 0)

    # Extract ride_settings to determine algorithm
    ride_settings = data.get("ride_settings", {})
    pic_priority = ride_settings.get("pic_priority")
    drop_priority = ride_settings.get("drop_priority")

    # Determine algorithm priority based on safety flag and ride_settings
    if safety_flag == 1:
        # Safety flag is set to 1 - force safety algorithm regardless of ride settings
        algorithm_priority = 2  # Safety algorithm priority
        logger.info(f"🔒 SAFETY FLAG DETECTED (safety=1) - Forcing safety algorithm regardless of ride settings")
    else:
        # Safety flag is 0 or missing - use ride settings as before
        algorithm_priority = pic_priority if pic_priority is not None else drop_priority
        logger.info(f"📊 Using ride_settings for algorithm selection (safety={safety_flag})")

    data["_algorithm_priority"] = algorithm_priority
    data["_safety_flag"] = safety_flag

    # Log the data structure for debugging
    logger.info(f"📊 API Response structure:")
    logger.info(f"   - users: {len(data.get('users', []))}")
    logger.info(
        f"   - driversUnassigned: {len(data.get('driversUnassigned', []))}")
    logger.info(
        f"   - driversAssigned: {len(data.get('driversAssigned', []))}")
    logger.info(f"   - ride_settings: {ride_settings}")
    logger.info(f"   - safety flag: {safety_flag}")
    logger.info(f"   - final algorithm priority: {algorithm_priority} ({'SAFETY ALGORITHM' if safety_flag == 1 else 'REGULAR ALGORITHM'})")

    return data
