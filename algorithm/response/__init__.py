"""
Algorithm Response Standardization Package

This package provides standardized response formatting for all clustering algorithms.
"""

from .response_standards import (
    # Field constants
    STATUS, EXECUTION_TIME, COMPANY, SHIFT, DATA, UNASSIGNED_USERS, UNASSIGNED_DRIVERS,
    OPTIMIZATION_MODE, PARAMETER, STRING_PARAM, CHOICE,

    # Object field constants
    DRIVER_ID, VEHICLE_ID, LATITUDE, LONGITUDE, FIRST_NAME, LAST_NAME, EMAIL,
    VEHICLE_NAME, VEHICLE_NO, CAPACITY, CHASIS_NO, COLOR, REGISTRATION_NO,
    SHIFT_TYPE_ID, ASSIGNED_USERS, USER_ID, OFFICE_DISTANCE, ADDRESS, EMPLOYEE_SHIFT,
    SHIFT_TYPE, PICKUP_ORDER,

    # Field orderings
    ROUTE_FIELD_ORDER, USER_FIELD_ORDER, UNASSIGNED_USER_FIELD_ORDER,
    UNASSIGNED_DRIVER_FIELD_ORDER,

    # Helper functions
    standardize_coordinates, create_standard_user, create_standard_driver,
    create_standard_route, validate_required_fields, remove_clustering_analysis
)

from .response_builder import (
    build_standard_response,
    standardize_algorithm_response,
    create_error_response,
    validate_response_structure,
    save_standardized_response,
    log_response_metrics
)

__all__ = [
    # Field constants
    'STATUS', 'EXECUTION_TIME', 'COMPANY', 'SHIFT', 'DATA', 'UNASSIGNED_USERS', 'UNASSIGNED_DRIVERS',
    'OPTIMIZATION_MODE', 'PARAMETER', 'STRING_PARAM', 'CHOICE',

    # Object field constants
    'DRIVER_ID', 'VEHICLE_ID', 'LATITUDE', 'LONGITUDE', 'FIRST_NAME', 'LAST_NAME', 'EMAIL',
    'VEHICLE_NAME', 'VEHICLE_NO', 'CAPACITY', 'CHASIS_NO', 'COLOR', 'REGISTRATION_NO',
    'SHIFT_TYPE_ID', 'ASSIGNED_USERS', 'USER_ID', 'OFFICE_DISTANCE', 'ADDRESS', 'EMPLOYEE_SHIFT',
    'SHIFT_TYPE', 'PICKUP_ORDER',

    # Field orderings
    'ROUTE_FIELD_ORDER', 'USER_FIELD_ORDER', 'UNASSIGNED_USER_FIELD_ORDER',
    'UNASSIGNED_DRIVER_FIELD_ORDER',

    # Helper functions
    'standardize_coordinates', 'create_standard_user', 'create_standard_driver',
    'create_standard_route', 'validate_required_fields', 'remove_clustering_analysis',

    # Response builder functions
    'build_standard_response', 'standardize_algorithm_response', 'create_error_response',
    'validate_response_structure', 'save_standardized_response', 'log_response_metrics'
]