import logging
import math

logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(
        dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371
    return c * r


def extract_pricebook_list(data):
    """Safely extract price_book_list from input data."""
    pricebook_list = data.get('price_book_list', [])
    if not isinstance(pricebook_list, list):
        logger.warning("price_book_list is not a list")
        return []
    return pricebook_list


def validate_pricebook_entry(entry):
    """Validate pricebook entry has required coordinates."""
    if not isinstance(entry, dict):
        return False

    lat = entry.get('latitude')
    lon = entry.get('longitude')

    # Skip null or empty coordinates
    if lat is None or lon is None:
        return False

    # Skip empty strings
    if lat == '' or lon == '':
        return False

    return True


def find_farthest_user_from_company(assigned_users, company_lat, company_lon):
    """
    Find the assigned user with maximum distance from company.

    Returns:
        dict or None: User object with latitude/longitude
    """
    farthest_user = None
    max_distance = -1

    for user in assigned_users:
        user_lat = user.get('latitude')
        user_lon = user.get('longitude')

        # Skip users with invalid coordinates
        if not user_lat or not user_lon:
            continue

        try:
            # Convert to float if strings
            user_lat = float(user_lat)
            user_lon = float(user_lon)

            distance = haversine_distance(
                company_lat, company_lon,
                user_lat, user_lon
            )

            if distance > max_distance:
                max_distance = distance
                farthest_user = user
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid user coordinates: {e}")
            continue

    return farthest_user


def find_closest_pricebook(user_lat, user_lon, pricebook_list):
    """
    Find the closest pricebook entry to given coordinates.

    Returns:
        int or None: pricebook_id
    """
    closest_pricebook_id = None
    min_distance = float('inf')

    for entry in pricebook_list:
        if not validate_pricebook_entry(entry):
            continue

        try:
            pb_lat = float(entry['latitude'])
            pb_lon = float(entry['longitude'])

            distance = haversine_distance(
                user_lat, user_lon,
                pb_lat, pb_lon
            )

            if distance < min_distance:
                min_distance = distance
                closest_pricebook_id = entry['id']

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Invalid pricebook entry: {e}")
            continue

    return closest_pricebook_id
