import logging
from .pricebook_utils import (
    extract_pricebook_list,
    find_farthest_user_from_company,
    find_closest_pricebook
)

logger = logging.getLogger(__name__)


def calculate_pricebook_id(route, data, company_lat, company_lon, logger=None):
    """
    Calculate pricebook_id for a route based on farthest assigned user.

    Args:
        route (dict): Route dictionary containing assigned_users
        data (dict): Complete input data containing price_book_list
        company_lat (float): Company latitude
        company_lon (float): Company longitude
        logger: Optional logger instance

    Returns:
        int: pricebook_id (defaults to 1 if no pricebook found or on error)
    """
    try:
        # Check if route has assigned_users
        assigned_users = route.get('assigned_users', [])
        if not assigned_users:
            return 1

        # Extract pricebook list
        pricebook_list = extract_pricebook_list(data)
        if not pricebook_list:
            return 1

        # Find farthest user from company
        farthest_user = find_farthest_user_from_company(
            assigned_users, company_lat, company_lon
        )
        if not farthest_user:
            return 1

        # Get user coordinates
        user_lat = float(farthest_user['latitude'])
        user_lon = float(farthest_user['longitude'])

        # Find closest pricebook
        pricebook_id = find_closest_pricebook(user_lat, user_lon, pricebook_list)
        if pricebook_id is None:
            return 1

        return pricebook_id

    except Exception as e:
        if logger:
            logger.error(f"Error calculating pricebook_id: {e}")
        return 1  # Always return default on error
