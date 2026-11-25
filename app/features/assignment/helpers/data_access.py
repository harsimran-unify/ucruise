"""
Input validation and record preparation for the assignment feature.

Extracted from the legacy algorithm/base/base.py; office-coordinate
extraction now delegates to app.core.config (same precedence: API payload
company coords -> config.json -> documented fallback).
"""

import pandas as pd

from app.core.config import office_coordinates
from app.infrastructure.logging.logger import get_logger


def validate_input_data(data):
    """Comprehensive data validation with bounds checking"""
    logger = get_logger()
    if not isinstance(data, dict):
        raise ValueError("API response must be a dictionary")

    # Check for users
    users = data.get("users", [])
    if not users:
        raise ValueError("No users found in API response")

    if not isinstance(users, list):
        raise ValueError("Users must be a list")

    # Special handling for empty users
    if len(users) == 0:
        raise ValueError("Empty users list")

    # Validate each user comprehensively
    for i, user in enumerate(users):
        if not isinstance(user, dict):
            raise ValueError(f"User {i} must be a dictionary")

        required_fields = ["id", "latitude", "longitude"]
        for field in required_fields:
            if field not in user:
                raise ValueError(f"User {i} missing required field: {field}")
            if user[field] is None or user[field] == "":
                raise ValueError(f"User {i} has null/empty {field}")

        # Validate coordinate bounds
        try:
            lat = float(user["latitude"])
            lon = float(user["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(
                    f"User {i} invalid latitude: {lat} (must be -90 to 90)")
            if not (-180 <= lon <= 180):
                raise ValueError(
                    f"User {i} invalid longitude: {lon} (must be -180 to 180)")
        except (ValueError, TypeError) as e:
            raise ValueError(f"User {i} invalid coordinates: {e}")

    # Get all drivers from both sources
    all_drivers = []

    # Check nested format first
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))

    # Check flat format
    if not all_drivers:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    if not all_drivers:
        raise ValueError("No drivers found in API response")

    # Validate drivers comprehensively (allow duplicates for pick/drop scenarios)
    duplicate_driver_count = 0
    for i, driver in enumerate(all_drivers):
        if not isinstance(driver, dict):
            raise ValueError(f"Driver {i} must be a dictionary")

        required_fields = ["id", "capacity", "latitude", "longitude"]
        for field in required_fields:
            if field not in driver:
                raise ValueError(f"Driver {i} missing required field: {field}")
            if driver[field] is None or driver[field] == "":
                raise ValueError(f"Driver {i} has null/empty {field}")

        # Count duplicates but don't error (legitimate for pick/drop scenarios)
        driver_id = str(driver["id"])
        duplicate_count = sum(1 for d in all_drivers
                              if str(d.get("id", "")) == driver_id)
        if duplicate_count > 1:
            duplicate_driver_count += 1

        # Validate driver coordinates
        try:
            lat = float(driver["latitude"])
            lon = float(driver["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(f"Driver {i} invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Driver {i} invalid longitude: {lon}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid coordinates: {e}")

        # Validate capacity
        try:
            capacity = int(driver["capacity"])
            if capacity <= 0:
                raise ValueError(
                    f"Driver {i} invalid capacity: {capacity} (must be > 0)")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid capacity: {e}")

    if duplicate_driver_count > 0:
        logger.info(
            f"ℹ️ INFO: Found {duplicate_driver_count} duplicate driver entries (normal for pick/drop scenarios)"
        )

    logger.info(
        f"✅ Input data validation passed - {len(users)} users, {len(all_drivers)} drivers"
    )


def extract_office_coordinates(data):
    """Extract dynamic office coordinates from API data"""
    return office_coordinates(data)


def _safe_coordinate(value):
    """Return a float coordinate, or None when missing/null/invalid.

    Never coerce to 0.0 — a missing coordinate silently placed at (0, 0)
    corrupts clustering and distance math for the whole run.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prepare_user_driver_dataframes(data):
    """Prepare user and driver dataframes from API data"""
    logger = get_logger()

    # Prepare user DataFrame
    users = data.get("users", [])
    user_data = []
    for user in users:
        lat = _safe_coordinate(user.get('latitude'))
        lon = _safe_coordinate(user.get('longitude'))
        if lat is None or lon is None:
            logger.warning(
                f"⚠️ Skipping user {user.get('id', '?')}: missing or invalid coordinates "
                f"({user.get('latitude')!r}, {user.get('longitude')!r})"
            )
            continue
        office_distance = _safe_coordinate(user.get('office_distance'))
        user_data.append({
            'user_id':
            str(user.get('id', '')),
            'latitude':
            lat,
            'longitude':
            lon,
            'first_name':
            str(user.get('first_name', '')),
            'email':
            str(user.get('email', '')),
            'office_distance':
            office_distance if office_distance is not None else 0.0
        })

    user_df = pd.DataFrame(
        user_data,
        columns=['user_id', 'latitude', 'longitude', 'first_name', 'email',
                 'office_distance'])

    # Prepare driver DataFrame
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    driver_data = []
    for i, driver in enumerate(all_drivers):
        lat = _safe_coordinate(driver.get('latitude'))
        lon = _safe_coordinate(driver.get('longitude'))
        if lat is None or lon is None:
            logger.warning(
                f"⚠️ Skipping driver {driver.get('id', '?')}: missing or invalid coordinates "
                f"({driver.get('latitude')!r}, {driver.get('longitude')!r})"
            )
            continue
        raw_capacity = driver.get('capacity')
        try:
            capacity = int(raw_capacity) if raw_capacity is not None else 1
        except (TypeError, ValueError):
            logger.warning(
                f"⚠️ Driver {driver.get('id', '?')}: invalid capacity "
                f"{raw_capacity!r}, defaulting to 1"
            )
            capacity = 1
        driver_data.append({
            'driver_id': str(driver.get('id', '')),
            'latitude': lat,
            'longitude': lon,
            'capacity': capacity,
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'priority': i + 1  # Simple priority based on order
        })

    driver_df = pd.DataFrame(
        driver_data,
        columns=['driver_id', 'latitude', 'longitude', 'capacity',
                 'vehicle_id', 'priority'])

    return user_df, driver_df


def _convert_users_to_unassigned_format(users):
    """Helper to convert user data to unassigned format"""
    unassigned_users = []
    for user in users:
        unassigned_users.append({
            'user_id':
            str(user.get('id', '')),
            'lat':
            float(user.get('latitude', 0.0)),
            'lng':
            float(user.get('longitude', 0.0)),
            'office_distance':
            float(user.get('office_distance', 0.0)),
            'first_name':
            str(user.get('first_name', '')),
            'email':
            str(user.get('email', ''))
        })
    return unassigned_users
