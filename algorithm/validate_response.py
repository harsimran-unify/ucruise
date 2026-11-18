"""
Response Validator - Validate algorithm outputs against standardized format
"""

from algorithm.response.response_standards import (
    STATUS, EXECUTION_TIME, DATA, UNASSIGNED_USERS, UNASSANGED_DRIVERS,
    OPTIMIZATION_MODE, PARAMETER, LATITUDE, LONGITUDE,
    DRIVER_ID, USER_ID, VEHICLE_ID, CAPACITY,
    validate_required_fields, validate_route_structure
)
import logging

logger = logging.getLogger(__name__)


def validate_algorithm_response(response_data, algorithm_name="unknown"):
    """
    Validate that an algorithm response conforms to the standard format

    Args:
        response_data (dict): Response data to validate
        algorithm_name (str): Name of the algorithm for logging

    Returns:
        tuple: (is_valid, validation_errors, standardized_response)
    """
    if not isinstance(response_data, dict):
        return False, ["Response must be a dictionary"], None

    validation_errors = []

    # Check for required fields
    required_fields_errors = validate_required_fields(response_data)
    if required_fields_errors:
        validation_errors.extend(required_fields_errors)

    # Validate data structure
    if DATA in response_data:
        routes = response_data[DATA]
        if not isinstance(routes, list):
            validation_errors.append("data must be a list")
        else:
            for i, route in enumerate(routes):
                route_errors = validate_route_structure(route)
                if route_errors:
                    validation_errors.extend([f"data[{i}].{error}" for error in route_errors])

    # Validate unassigned users
    if UNASSIGNED_USERS in response_data:
        unassigned_users = response_data[UNASSIGNED_USERS]
        if not isinstance(unassigned_users, list):
            validation_errors.append("unassignedUsers must be a list")
        else:
            for i, user in enumerate(unassigned_users):
                user_errors = validate_user_structure(user)
                if user_errors:
                    validation_errors.extend([f"unassignedUsers[{i}].{error}" for error in user_errors])

    # Validate unassigned drivers
    if UNASSANGED_DRIVERS in response_data:
        unassigned_drivers = response_data[UNASSANGED_DRIVERS]
        if not isinstance(unassigned_drivers, list):
            validation_errors.append("unassignedDrivers must be a list")
        else:
            for i, driver in enumerate(unassigned_drivers):
                driver_errors = validate_driver_structure(driver)
                if driver_errors:
                    validation_errors.extend([f"unassignedDrivers[{i}].{error}" for error in driver_errors])

    # Log validation results
    if validation_errors:
        logger.warning(f"Validation failed for {algorithm_name}: {validation_errors}")
    else:
        logger.info(f"Validation passed for {algorithm_name}")

    is_valid = len(validation_errors) == 0
    return is_valid, validation_errors, None


def validate_user_structure(user_data):
    """
    Validate user object structure

    Args:
        user_data (dict): User data to validate

    Returns:
        list: List of validation errors
    """
    errors = []

    if not isinstance(user_data, dict):
        return ["User must be a dictionary"]

    # Required fields
    if USER_ID not in user_data and 'id' not in user_data:
        errors.append("user_id or id is required")

    # Coordinate validation
    if LATITUDE in user_data:
        try:
            lat = float(user_data[LATITUDE])
            if not (-90 <= lat <= 90):
                errors.append("latitude must be between -90 and 90")
        except (ValueError, TypeError):
            errors.append("latitude must be a number")
    elif 'lat' in user_data:
        try:
            lat = float(user_data['lat'])
            if not (-90 <= lat <= 90):
                errors.append("lat must be between -90 and 90")
        except (ValueError, TypeError):
            errors.append("lat must be a number")
    else:
        errors.append("latitude or lat is required")

    if LONGITUDE in user_data:
        try:
            lng = float(user_data[LONGITUDE])
            if not (-180 <= lng <= 180):
                errors.append("longitude must be between -180 and 180")
        except (ValueError, TypeError):
            errors.append("longitude must be a number")
    elif 'lng' in user_data:
        try:
            lng = float(user_data['lng'])
            if not (-180 <= lng <= 180):
                errors.append("lng must be between -180 and 180")
        except (ValueError, TypeError):
            errors.append("lng must be a number")
    else:
        errors.append("longitude or lng is required")

    return errors


def validate_driver_structure(driver_data):
    """
    Validate driver object structure

    Args:
        driver_data (dict): Driver data to validate

    Returns:
        list: List of validation errors
    """
    errors = []

    if not isinstance(driver_data, dict):
        return ["Driver must be a dictionary"]

    # Required fields
    if DRIVER_ID not in driver_data and 'id' not in driver_data:
        errors.append("driver_id or id is required")

    if CAPACITY not in driver_data:
        errors.append("capacity is required")
    else:
        try:
            capacity = int(driver_data[CAPACITY])
            if capacity < 1:
                errors.append("capacity must be at least 1")
        except (ValueError, TypeError):
            errors.append("capacity must be an integer")

    # Coordinate validation (same as user)
    if LATITUDE in driver_data:
        try:
            lat = float(driver_data[LATITUDE])
            if not (-90 <= lat <= 90):
                errors.append("latitude must be between -90 and 90")
        except (ValueError, TypeError):
            errors.append("latitude must be a number")
    elif 'lat' in driver_data:
        try:
            lat = float(driver_data['lat'])
            if not (-90 <= lat <= 90):
                errors.append("lat must be between -90 and 90")
        except (ValueError, TypeError):
            errors.append("lat must be a number")

    if LONGITUDE in driver_data:
        try:
            lng = float(driver_data[LONGITUDE])
            if not (-180 <= lng <= 180):
                errors.append("longitude must be between -180 and 180")
        except (ValueError, TypeError):
            errors.append("longitude must be a number")
    elif 'lng' in driver_data:
        try:
            lng = float(driver_data['lng'])
            if not (-180 <= lng <= 180):
                errors.append("lng must be between -180 and 180")
        except (ValueError, TypeError):
            errors.append("lng must be a number")

    return errors


def validate_route_consistency(route_data):
    """
    Validate consistency within a route object

    Args:
        route_data (dict): Route data to validate

    Returns:
        list: List of validation errors
    """
    errors = []

    if not isinstance(route_data, dict):
        return ["Route must be a dictionary"]

    # Check if assigned_users field exists and is last (as per standards)
    if 'assigned_users' not in route_data:
        errors.append("assigned_users field is missing")

    # Validate driver capacity vs assigned users
    if CAPACITY in route_data and 'assigned_users' in route_data:
        try:
            capacity = int(route_data[CAPACITY])
            assigned_count = len(route_data['assigned_users'])
            if assigned_count > capacity:
                errors.append(f"Assigned users ({assigned_count}) exceed capacity ({capacity})")
        except (ValueError, TypeError):
            pass  # Capacity already validated in validate_driver_structure

    # Validate coordinates consistency between route and users
    if LATITUDE in route_data and LONGITUDE in route_data and 'assigned_users' in route_data:
        route_lat = float(route_data[LATITUDE])
        route_lng = float(route_data[LONGITUDE])

        for user in route_data['assigned_users']:
            if isinstance(user, dict):
                user_lat = user.get(LATITUDE) or user.get('lat')
                user_lng = user.get(LONGITUDE) or user.get('lng')

                if user_lat is not None and user_lng is not None:
                    try:
                        user_lat = float(user_lat)
                        user_lng = float(user_lng)

                        # Check if user coordinates are reasonable (not too far from route)
                        # Using a rough check: users should be within 100km of route start
                        from math import radians, cos, sin, asin, sqrt

                        def haversine_distance(lat1, lon1, lat2, lon2):
                            """Calculate haversine distance between two points"""
                            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                            dlat = lat2 - lat1
                            dlon = lon2 - lon1
                            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                            return 2 * asin(sqrt(a)) * 6371  # Earth radius in km

                        distance = haversine_distance(route_lat, route_lng, user_lat, user_lng)
                        if distance > 100:  # 100km threshold
                            errors.append(f"User is {distance:.1f}km from route start (threshold: 100km)")

                    except (ValueError, TypeError):
                        errors.append("Invalid user coordinates")

    return errors


def validate_field_ordering(data_structure, expected_order, field_type="object"):
    """
    Validate that fields are in the expected order

    Args:
        data_structure (dict): Object to check
        expected_order (list): Expected field order
        field_type (str): Type of object for error messages

    Returns:
        list: List of field ordering issues
    """
    issues = []

    if not isinstance(data_structure, dict):
        return [f"{field_type} must be a dictionary"]

    actual_fields = list(data_structure.keys())

    # Check if all expected fields are present
    missing_fields = [field for field in expected_order if field not in actual_fields]
    if missing_fields:
        issues.append(f"Missing required fields: {missing_fields}")

    # Check field order (only warn, not error)
    expected_fields_present = [field for field in expected_order if field in actual_fields]
    present_field_indices = {field: i for i, field in enumerate(actual_fields)}

    for i, field in enumerate(expected_fields_present):
        actual_index = present_field_indices[field]
        if actual_index != i:
            issues.append(f"Field order warning: {field} is at position {actual_index}, expected {i}")

    return issues


def comprehensive_validation(response_data, algorithm_name="unknown"):
    """
    Perform comprehensive validation of algorithm response

    Args:
        response_data (dict): Response data to validate
        algorithm_name (str): Name of the algorithm

    Returns:
        dict: Validation results with all issues found
    """
    validation_result = {
        "algorithm": algorithm_name,
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "field_order_issues": [],
        "coordinate_issues": [],
        "consistency_issues": []
    }

    # Basic structure validation
    is_valid, errors, _ = validate_algorithm_response(response_data, algorithm_name)
    validation_result["is_valid"] = is_valid
    validation_result["errors"] = errors

    if not is_valid:
        return validation_result

    # Field ordering validation
    from algorithm.response.response_standards import ROUTE_FIELD_ORDER

    if DATA in response_data:
        for i, route in enumerate(response_data[DATA]):
            ordering_issues = validate_field_ordering(route, ROUTE_FIELD_ORDER, f"route[{i}]")
            validation_result["field_order_issues"].extend(ordering_issues)

    # Consistency validation
    if DATA in response_data:
        for i, route in enumerate(response_data[DATA]):
            consistency_issues = validate_route_consistency(route)
            if consistency_issues:
                validation_result["consistency_issues"].extend([f"route[{i}]: {issue}" for issue in consistency_issues])

    # Check for coordinate naming issues
    coordinate_issues = check_coordinate_naming(response_data)
    validation_result["coordinate_issues"] = coordinate_issues

    # Convert warnings to errors for critical issues
    critical_issues = [issue for issue in validation_result["consistency_issues"]
                      if "exceed capacity" in issue or "Invalid coordinates" in issue]

    if critical_issues:
        validation_result["errors"].extend(critical_issues)
        validation_result["is_valid"] = False

    # Log summary
    total_issues = (len(validation_result["errors"]) +
                   len(validation_result["warnings"]) +
                   len(validation_result["field_order_issues"]) +
                   len(validation_result["coordinate_issues"]) +
                   len(validation_result["consistency_issues"]))

    if total_issues > 0:
        logger.warning(f"Comprehensive validation found {total_issues} issues for {algorithm_name}")
        logger.warning(f"  Errors: {len(validation_result['errors'])}")
        logger.warning(f"  Warnings: {len(validation_result['warnings'])}")
        logger.warning(f"  Field order issues: {len(validation_result['field_order_issues'])}")
        logger.warning(f"  Coordinate issues: {len(validation_result['coordinate_issues'])}")
        logger.warning(f"  Consistency issues: {len(validation_result['consistency_issues'])}")
    else:
        logger.info(f"Comprehensive validation passed for {algorithm_name}")

    return validation_result


def check_coordinate_naming(response_data):
    """
    Check for coordinate naming inconsistencies

    Args:
        response_data (dict): Response data to check

    Returns:
        list: List of coordinate naming issues
    """
    issues = []

    def check_object(obj, path):
        if isinstance(obj, dict):
            # Check for mixed coordinate naming
            has_lat = 'lat' in obj
            has_lng = 'lng' in obj
            has_latitude = LATITUDE in obj
            has_longitude = LONGITUDE in obj

            if has_lat and has_latitude:
                issues.append(f"{path}: Has both 'lat' and 'latitude'")
            if has_lng and has_longitude:
                issues.append(f"{path}: Has both 'lng' and 'longitude'")

            # Recursively check nested objects
            if 'assigned_users' in obj and isinstance(obj['assigned_users'], list):
                for i, user in enumerate(obj['assigned_users']):
                    check_object(user, f"{path}.assigned_users[{i}]")

    # Check routes
    if DATA in response_data and isinstance(response_data[DATA], list):
        for i, route in enumerate(response_data[DATA]):
            check_object(route, f"data[{i}]")

    # Check unassigned users
    if UNASSIGNED_USERS in response_data and isinstance(response_data[UNASSIGNED_USERS], list):
        for i, user in enumerate(response_data[UNASSIGNED_USERS]):
            check_object(user, f"unassignedUsers[{i}]")

    # Check unassigned drivers
    if UNASSANGED_DRIVERS in response_data and isinstance(response_data[UNASSANGED_DRIVERS], list):
        for i, driver in enumerate(response_data[UNASSANGED_DRIVERS]):
            check_object(driver, f"unassignedDrivers[{i}]")

    return issues