"""
Response Builder - Shared functions for building standardized algorithm responses
"""

from .response_standards import (
    # Core fields
    STATUS, EXECUTION_TIME, COMPANY, SHIFT, DATA, UNASSIGNED_USERS, UNASSIGNED_DRIVERS,
    OPTIMIZATION_MODE, PARAMETER, STRING_PARAM, CHOICE,

    # Object creators
    create_standard_route, create_standard_user, create_standard_driver,
    remove_clustering_analysis,

    # Validation
    validate_required_fields,

    # Field constants
    ASSIGNED_USERS, ROUTE_NO
)

import json
import logging
import os

from app.core.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def build_standard_response(
    status,
    execution_time,
    routes,
    unassigned_users,
    unassigned_drivers,
    optimization_mode,
    parameter,
    company=None,
    shift=None,
    string_param=None,
    choice=None,
    metadata=None
):
    """
    Build a standardized response dictionary

    Args:
        status (str): "true" or "false"
        execution_time (float): Algorithm execution time in seconds
        routes (list): List of route dictionaries
        unassigned_users (list): List of unassigned user dictionaries
        unassigned_drivers (list): List of unassigned driver dictionaries
        optimization_mode (str): Optimization mode identifier
        parameter (int): Parameter value
        company (dict, optional): Company information
        shift (dict, optional): Shift information
        string_param (str, optional): String parameter
        choice (str, optional): Choice parameter
        metadata (dict, optional): Additional metadata

    Returns:
        dict: Standardized response dictionary
    """

    # Standardize all data structures with route numbering
    standardized_routes = []
    for i, route in enumerate(routes if routes else [], 1):
        standardized_route = create_standard_route(route)
        # Add route number (starting from 1)
        standardized_route[ROUTE_NO] = i
        # Add route number to assigned users
        if ASSIGNED_USERS in standardized_route:
            for user in standardized_route[ASSIGNED_USERS]:
                user[ROUTE_NO] = i
        standardized_routes.append(standardized_route)

    standardized_unassigned_users = []
    for user in unassigned_users if unassigned_users else []:
        standardized_user = create_standard_user(user, include_user_id=True)
        # Unassigned users have no route number
        standardized_user[ROUTE_NO] = None
        standardized_unassigned_users.append(standardized_user)

    standardized_unassigned_drivers = []
    for driver in unassigned_drivers if unassigned_drivers else []:
        standardized_driver = create_standard_driver(driver)
        # Unassigned drivers have no route number
        standardized_driver[ROUTE_NO] = None
        standardized_unassigned_drivers.append(standardized_driver)

    # Build response with all required fields
    response = {
        STATUS: status,
        EXECUTION_TIME: float(execution_time),
        DATA: standardized_routes,
        UNASSIGNED_USERS: standardized_unassigned_users,
        UNASSIGNED_DRIVERS: standardized_unassigned_drivers,
        OPTIMIZATION_MODE: optimization_mode,
        PARAMETER: int(parameter)
    }

    # Add optional fields if provided
    if company is not None:
        response[COMPANY] = company
    if shift is not None:
        response[SHIFT] = shift
    if string_param is not None:
        response[STRING_PARAM] = string_param
    if choice is not None:
        response[CHOICE] = choice
    if metadata is not None:
        response['metadata'] = metadata

    # Validate the response
    missing_fields = validate_required_fields(response)
    if missing_fields:
        logger.warning(f"Response validation warnings: {missing_fields}")

    return response


def standardize_algorithm_response(response_data, optimization_mode, parameter, string_param=None, choice=None):
    """
    Convert an existing algorithm response to the standardized format

    Args:
        response_data (dict): Existing algorithm response
        optimization_mode (str): Optimization mode identifier
        parameter (int): Parameter value
        string_param (str, optional): String parameter
        choice (str, optional): Choice parameter

    Returns:
        dict: Standardized response dictionary
    """

    if not isinstance(response_data, dict):
        return build_standard_response(
            status="false",
            execution_time=0.0,
            routes=[],
            unassigned_users=[],
            unassigned_drivers=[],
            optimization_mode=optimization_mode,
            parameter=parameter,
            string_param=string_param,
            choice=choice
        )

    # Remove clustering_analysis as per new standards
    cleaned_response = remove_clustering_analysis(response_data)

    # Extract data from existing response
    return build_standard_response(
        status=cleaned_response.get(STATUS, "false"),
        execution_time=cleaned_response.get(EXECUTION_TIME, 0.0),
        routes=cleaned_response.get(DATA, []),
        unassigned_users=cleaned_response.get(UNASSIGNED_USERS, []),
        unassigned_drivers=cleaned_response.get(UNASSIGNED_DRIVERS, []),
        optimization_mode=optimization_mode,
        parameter=parameter,
        company=cleaned_response.get(COMPANY),
        shift=cleaned_response.get(SHIFT),
        string_param=string_param or cleaned_response.get(STRING_PARAM),
        choice=choice or cleaned_response.get(CHOICE)
    )


def create_error_response(
    error_message,
    execution_time=0.0,
    optimization_mode="unknown",
    parameter=0,
    string_param=None,
    choice=None,
    company=None,
    shift=None
):
    """
    Create a standardized error response

    Args:
        error_message (str): Error description
        execution_time (float): Execution time before error
        optimization_mode (str): Optimization mode being used
        parameter (int): Parameter value
        string_param (str, optional): String parameter
        choice (str, optional): Choice parameter
        company (dict, optional): Company information
        shift (dict, optional): Shift information

    Returns:
        dict: Standardized error response
    """

    logger.error(f"Algorithm error: {error_message}")

    return build_standard_response(
        status="false",
        execution_time=execution_time,
        routes=[],
        unassigned_users=[],
        unassigned_drivers=[],
        optimization_mode=optimization_mode,
        parameter=parameter,
        company=company,
        shift=shift,
        string_param=string_param,
        choice=choice,
        # Add error metadata in metadata field
        metadata={
            'error_type': 'algorithm_error',
            'error_message': error_message
        }
    )


def validate_response_structure(response_data):
    """
    Validate that a response conforms to the standard structure

    Args:
        response_data (dict): Response to validate

    Returns:
        tuple: (is_valid, error_messages)
    """

    if not isinstance(response_data, dict):
        return False, ["Response must be a dictionary"]

    errors = validate_required_fields(response_data)

    # Additional validation checks
    if STATUS in response_data and response_data[STATUS] not in ["true", "false"]:
        errors.append("status must be 'true' or 'false'")

    if EXECUTION_TIME in response_data:
        try:
            float(response_data[EXECUTION_TIME])
        except (ValueError, TypeError):
            errors.append("execution_time must be a number")

    if PARAMETER in response_data:
        try:
            int(response_data[PARAMETER])
        except (ValueError, TypeError):
            errors.append("parameter must be an integer")

    return len(errors) == 0, errors


def save_standardized_response(response_data, filename="drivers_and_routes.json"):
    """
    Save standardized response to file

    Args:
        response_data (dict): Standardized response
        filename (str): Output filename
    """

    try:
        # Relative filenames always land in the project root regardless of CWD
        path = filename if os.path.isabs(filename) else os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Standardized response saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save response to {filename}: {e}")


def log_response_metrics(response_data, algorithm_name):
    """
    Log metrics about the response for monitoring

    Args:
        response_data (dict): Standardized response
        algorithm_name (str): Name of the algorithm
    """

    if not isinstance(response_data, dict):
        return

    try:
        route_count = len(response_data.get(DATA, []))
        unassigned_user_count = len(response_data.get(UNASSIGNED_USERS, []))
        unassigned_driver_count = len(response_data.get(UNASSIGNED_DRIVERS, []))
        execution_time = response_data.get(EXECUTION_TIME, 0)
        status = response_data.get(STATUS, "unknown")

        logger.info(f"Algorithm {algorithm_name} metrics:")
        logger.info(f"  Status: {status}")
        logger.info(f"  Execution time: {execution_time:.3f}s")
        logger.info(f"  Routes: {route_count}")
        logger.info(f"  Unassigned users: {unassigned_user_count}")
        logger.info(f"  Unassigned drivers: {unassigned_driver_count}")

        # Calculate assignment rate
        total_users = unassigned_user_count
        assigned_users = sum(len(route.get('assigned_users', [])) for route in response_data.get(DATA, []))
        total_users += assigned_users

        if total_users > 0:
            assignment_rate = (assigned_users / total_users) * 100
            logger.info(f"  Assignment rate: {assignment_rate:.1f}%")

    except Exception as e:
        logger.error(f"Error logging metrics for {algorithm_name}: {e}")