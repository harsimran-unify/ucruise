"""
Response Standards - Field definitions and helper functions for unified algorithm output
"""

# Core response field constants
STATUS = "status"
EXECUTION_TIME = "execution_time"
COMPANY = "company"
SHIFT = "shift"
DATA = "data"
UNASSIGNED_USERS = "unassignedUsers"
UNASSIGNED_DRIVERS = "unassignedDrivers"
OPTIMIZATION_MODE = "optimization_mode"
PARAMETER = "parameter"
STRING_PARAM = "string_param"
CHOICE = "choice"

# Route object field constants (ordered as required)
DRIVER_ID = "driver_id"
VEHICLE_ID = "vehicle_id"
PRICEBOOK_ID = "price_book_id"
LATITUDE = "latitude"
LONGITUDE = "longitude"
FIRST_NAME = "first_name"
LAST_NAME = "last_name"
EMAIL = "email"
VEHICLE_NAME = "vehicle_name"
VEHICLE_NO = "vehicle_no"
CAPACITY = "capacity"
CHASIS_NO = "chasis_no"
COLOR = "color"
REGISTRATION_NO = "registration_no"
SHIFT_TYPE_ID = "shift_type_id"
ASSIGNED_USERS = "assigned_users"

# User field constants
USER_ID = "user_id"
OFFICE_DISTANCE = "office_distance"
ADDRESS = "address"
EMPLOYEE_SHIFT = "employee_shift"
SHIFT_TYPE = "shift_type"
PICKUP_ORDER = "pickup_order"
PHONE = "phone"
ROUTE_NO = "route_no"

# Standard field order for route objects
ROUTE_FIELD_ORDER = [
    DRIVER_ID,
    VEHICLE_ID,
    PRICEBOOK_ID,
    LATITUDE,
    LONGITUDE,
    FIRST_NAME,
    LAST_NAME,
    EMAIL,
    VEHICLE_NAME,
    VEHICLE_NO,
    CAPACITY,
    CHASIS_NO,
    COLOR,
    REGISTRATION_NO,
    SHIFT_TYPE_ID,
    ROUTE_NO,
    ASSIGNED_USERS
]

# Standard field order for user objects (in assigned_users) - NO vehicle fields
USER_FIELD_ORDER = [
    USER_ID,
    LATITUDE,
    LONGITUDE,
    OFFICE_DISTANCE,
    FIRST_NAME,
    LAST_NAME,
    EMAIL,
    PHONE,
    ADDRESS,
    EMPLOYEE_SHIFT,
    PICKUP_ORDER,
    ROUTE_NO
]

# Standard field order for unassigned users - NO vehicle fields
UNASSIGNED_USER_FIELD_ORDER = [
    USER_ID,
    LATITUDE,
    LONGITUDE,
    OFFICE_DISTANCE,
    FIRST_NAME,
    LAST_NAME,
    EMAIL,
    PHONE,
    ADDRESS,
    EMPLOYEE_SHIFT,
    SHIFT_TYPE,
    ROUTE_NO
]

# Standard field order for unassigned drivers
UNASSIGNED_DRIVER_FIELD_ORDER = [
    DRIVER_ID,
    LATITUDE,
    LONGITUDE,
    OFFICE_DISTANCE,
    CAPACITY,
    VEHICLE_ID,
    VEHICLE_NAME,
    VEHICLE_NO,
    CHASIS_NO,
    COLOR,
    REGISTRATION_NO,
    SHIFT_TYPE_ID,
    FIRST_NAME,
    LAST_NAME,
    EMAIL,
    PHONE,
    ADDRESS,
    EMPLOYEE_SHIFT,
    SHIFT_TYPE,
    ROUTE_NO
]


def standardize_coordinates(obj):
    """
    Convert coordinate names to standard format (latitude/longitude)
    Handles both lat/lng and latitude/longitude variations
    """
    if not isinstance(obj, dict):
        return obj

    # Create a copy to avoid modifying the original
    standardized = obj.copy()

    # Convert lat -> latitude
    if 'lat' in standardized and LATITUDE not in standardized:
        standardized[LATITUDE] = float(standardized.pop('lat'))
    elif 'lat' in standardized and LATITUDE in standardized:
        # Remove lat if both exist
        standardized.pop('lat')

    # Convert lng -> longitude
    if 'lng' in standardized and LONGITUDE not in standardized:
        standardized[LONGITUDE] = float(standardized.pop('lng'))
    elif 'lng' in standardized and LONGITUDE in standardized:
        # Remove lng if both exist
        standardized.pop('lng')

    # Ensure latitude and longitude are floats if they exist
    if LATITUDE in standardized:
        standardized[LATITUDE] = float(standardized[LATITUDE])
    if LONGITUDE in standardized:
        standardized[LONGITUDE] = float(standardized[LONGITUDE])

    return standardized


def create_standard_user(user_data, include_user_id=True):
    """
    Create a standardized user object with all required fields
    """
    user = standardize_coordinates(user_data)

    # Ensure all standard fields exist with default values
    standardized_user = {}

    # Use appropriate field order and ID field
    field_order = UNASSIGNED_USER_FIELD_ORDER if include_user_id else USER_FIELD_ORDER
    id_field = USER_ID if include_user_id else USER_ID

    for field in field_order:
        if field in user:
            standardized_user[field] = user[field]
        elif field not in [id_field, PICKUP_ORDER]:  # Don't add defaults for ID or pickup_order
            standardized_user[field] = None

    # Add ID field if present
    if include_user_id and id_field in user:
        standardized_user[id_field] = user[id_field]

    # Add pickup_order only if it exists
    if PICKUP_ORDER in user:
        standardized_user[PICKUP_ORDER] = user[PICKUP_ORDER]

    return standardized_user


def create_standard_driver(driver_data):
    """
    Create a standardized driver object with all required fields
    """
    driver = standardize_coordinates(driver_data)

    # Ensure all standard fields exist with default values
    standardized_driver = {}

    for field in UNASSIGNED_DRIVER_FIELD_ORDER:
        if field in driver:
            standardized_driver[field] = driver[field]
        elif field not in [DRIVER_ID]:  # Don't add default for ID
            standardized_driver[field] = None

    # Add driver_id if present
    if DRIVER_ID in driver:
        standardized_driver[DRIVER_ID] = driver[DRIVER_ID]

    return standardized_driver


def create_standard_route(route_data):
    """
    Create a standardized route object with all required fields
    """
    route = standardize_coordinates(route_data)

    # Build standardized route with correct field order
    standardized_route = {}

    # Add all route-level fields in correct order
    for field in ROUTE_FIELD_ORDER:
        if field == ASSIGNED_USERS:
            # Handle assigned_users separately
            continue
        elif field in route:
            standardized_route[field] = route[field]
        elif field not in [DRIVER_ID, VEHICLE_ID]:  # Don't add defaults for required IDs
            standardized_route[field] = None

    # Add required IDs if present
    if DRIVER_ID in route:
        standardized_route[DRIVER_ID] = route[DRIVER_ID]
    if VEHICLE_ID in route:
        standardized_route[VEHICLE_ID] = route[VEHICLE_ID]

    # Standardize assigned_users
    assigned_users = route.get(ASSIGNED_USERS, [])
    if not isinstance(assigned_users, list):
        assigned_users = []

    standardized_assigned_users = []
    for user_data in assigned_users:
        standardized_user = create_standard_user(user_data, include_user_id=True)
        standardized_assigned_users.append(standardized_user)

    # Add assigned_users as the last field
    standardized_route[ASSIGNED_USERS] = standardized_assigned_users

    return standardized_route


def validate_required_fields(response_data):
    """
    Validate that all required fields are present in the response
    Returns a list of missing fields
    """
    if not isinstance(response_data, dict):
        return ["Response must be a dictionary"]

    required_fields = [
        STATUS,
        EXECUTION_TIME,
        DATA,
        UNASSIGNED_USERS,
        UNASSIGNED_DRIVERS,
        OPTIMIZATION_MODE,
        PARAMETER
    ]

    missing_fields = []
    for field in required_fields:
        if field not in response_data:
            missing_fields.append(field)

    # Validate data structure
    if DATA in response_data:
        routes = response_data[DATA]
        if not isinstance(routes, list):
            missing_fields.append(f"{DATA} must be a list")
        else:
            for i, route in enumerate(routes):
                route_errors = validate_route_structure(route)
                if route_errors:
                    missing_fields.extend([f"{DATA}[{i}].{error}" for error in route_errors])

    return missing_fields


def validate_route_structure(route_data):
    """
    Validate route object structure
    Returns a list of missing or invalid fields
    """
    if not isinstance(route_data, dict):
        return ["Route must be a dictionary"]

    required_fields = [DRIVER_ID, VEHICLE_ID, LATITUDE, LONGITUDE]
    missing_fields = []

    for field in required_fields:
        if field not in route_data:
            missing_fields.append(field)
        elif field in [LATITUDE, LONGITUDE] and not isinstance(route_data[field], (int, float)):
            missing_fields.append(f"{field} must be a number")

    # Validate assigned_users
    if ASSIGNED_USERS in route_data:
        assigned_users = route_data[ASSIGNED_USERS]
        if not isinstance(assigned_users, list):
            missing_fields.append(f"{ASSIGNED_USERS} must be a list")

    return missing_fields


def remove_clustering_analysis(response_data):
    """
    Remove clustering_analysis field from response as per new standards
    """
    if not isinstance(response_data, dict):
        return response_data

    # Create a copy to avoid modifying original
    cleaned = response_data.copy()

    # Remove clustering_analysis if present
    if 'clustering_analysis' in cleaned:
        del cleaned['clustering_analysis']

    return cleaned