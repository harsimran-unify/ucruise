import logging
import os
from datetime import datetime
import json
import glob

def clear_logs(log_dir="logs"):
    """Clear all existing log files before starting a new session"""
    import sys
    # Handle Windows console encoding issues
    safe_print = lambda msg: print(msg.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

    if os.path.exists(log_dir):
        log_files = glob.glob(os.path.join(log_dir, "assignment_*.log"))
        removed_count = 0
        for log_file in log_files:
            try:
                os.remove(log_file)
                removed_count += 1
            except OSError as e:
                safe_print(f"Warning: Could not remove {log_file}: {e}")

        if removed_count > 0:
            safe_print(f"[CLEARED] Removed {removed_count} old log files from {log_dir}/")
        else:
            safe_print(f"[INFO] No existing log files found in {log_dir}/")
    else:
        safe_print(f"[INFO] Log directory {log_dir}/ doesn't exist yet")

class RouteAssignmentLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamp for this session with milliseconds for uniqueness
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Setup main logger
        self.logger = logging.getLogger('route_assignment')
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create file handler with UTF-8 encoding
        log_file = os.path.join(log_dir, f"assignment_{self.session_timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8', delay=True)
        file_handler.setLevel(logging.DEBUG)

        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(funcName)20s:%(lineno)4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # All tracking will go to the main log file
        self.tracking_file = log_file

        self.log_session_start()

    def log_session_start(self):
        self.logger.info("="*80)
        self.logger.info("ROUTE ASSIGNMENT SESSION STARTED")
        self.logger.info(f"Session ID: {self.session_timestamp}")
        self.logger.info("="*80)

    def info(self, message, file_context=None):
        """Add info method with optional file context"""
        if file_context:
            self.logger.info(f"[{file_context}] {message}")
        else:
            self.logger.info(message)

    def warning(self, message, file_context=None):
        """Add warning method with optional file context"""
        if file_context:
            self.logger.warning(f"[{file_context}] {message}")
        else:
            self.logger.warning(message)

    def error(self, message, file_context=None, exc_info=False):
        """Add error method with optional file context"""
        if file_context:
            self.logger.error(f"[{file_context}] {message}", exc_info=exc_info)
        else:
            self.logger.error(message, exc_info=exc_info)

    def critical(self, message, file_context=None):
        """Add critical method with optional file context"""
        if file_context:
            self.logger.critical(f"[{file_context}] {message}")
        else:
            self.logger.critical(message)

    def debug(self, message, file_context=None):
        """Add debug method with optional file context"""
        if file_context:
            self.logger.debug(f"[{file_context}] {message}")
        else:
            self.logger.debug(message)

    def step_start(self, step_name, file_context):
        """Log the start of a major step"""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 STARTING: {step_name} | FILE: {file_context}")
        self.logger.info("=" * 80)

    def step_complete(self, step_name, file_context, details=""):
        """Log the completion of a major step"""
        self.logger.info(f"✅ COMPLETED: {step_name} | FILE: {file_context} {details}")
        self.logger.info("-" * 60)

    def file_operation(self, operation, file_context, details=""):
        """Log file-specific operations"""
        self.logger.info(f"🔧 [{file_context}] {operation} {details}")

    def assignment_mode(self, mode, file_context):
        """Log the assignment mode being used"""
        self.logger.info("=" * 60)
        self.logger.info(f"🎯 ASSIGNMENT MODE: {mode} | FILE: {file_context}")
        self.logger.info("=" * 60)

    def log_data_validation(self, users_count, drivers_count, office_coords, file_context):
        self.info(f"DATA VALIDATION - Users: {users_count}, Drivers: {drivers_count}", file_context)
        self.info(f"Office coordinates: {office_coords}", file_context)

    def log_clustering_decision(self, method, user_count, cluster_count, details, file_context):
        self.info(f"CLUSTERING - Method: {method}, Users: {user_count}, Clusters: {cluster_count}", file_context)
        self.debug(f"Clustering details: {details}", file_context)

    def log_route_creation(self, driver_id, users, reason, quality_metrics, file_context):
        self.info(f"ROUTE CREATED - Driver: {driver_id}, Users: {len(users)}", file_context)
        self.info(f"Creation reason: {reason}", file_context)
        for user in users:
            self.debug(f"  User {user.get('user_id', 'N/A')}", file_context)
        self.debug(f"Quality metrics: {quality_metrics}", file_context)

    def log_route_rejection(self, driver_id, users, reason, file_context):
        self.warning(f"ROUTE REJECTED - Driver: {driver_id}, Reason: {reason}", file_context)
        for user in users:
            self.debug(f"  Rejected user {user.get('user_id', 'N/A')}", file_context)

    def log_user_assignment(self, user_id, driver_id, route_details, file_context):
        self.info(f"USER ASSIGNED - User: {user_id} -> Driver: {driver_id}", file_context)
        self.debug(f"Route details: {route_details}", file_context)

    def log_user_unassigned(self, user_id, reason, attempted_drivers, file_context):
        self.warning(f"USER UNASSIGNED - User: {user_id}, Reason: {reason}", file_context)
        self.debug(f"Attempted drivers: {attempted_drivers}", file_context)

        # Additional tracking info in main log
        self.info(f"TRACKING | UNASSIGNED_USER | {user_id} | {reason}", file_context)

    def log_driver_unused(self, driver_id, reason, capacity, location, file_context):
        self.warning(f"DRIVER UNUSED - Driver: {driver_id}, Reason: {reason}", file_context)
        self.debug(f"Capacity: {capacity}, Location: {location}", file_context)

        # Additional tracking info in main log
        self.info(f"TRACKING | UNUSED_DRIVER | {driver_id} | {reason} | {capacity} | {location}", file_context)

    def log_optimization_step(self, step_name, before_state, after_state, changes, file_context):
        self.info(f"OPTIMIZATION - {step_name}", file_context)
        self.debug(f"Before: {before_state}", file_context)
        self.debug(f"After: {after_state}", file_context)
        self.info(f"Changes made: {changes}", file_context)

    def log_final_summary(self, total_users, assigned_users, unassigned_users, 
                         total_drivers, used_drivers, unused_drivers, routes, file_context):
        self.info("="*80, file_context)
        self.info("FINAL ASSIGNMENT SUMMARY", file_context)
        self.info(f"Users: {assigned_users}/{total_users} assigned ({len(unassigned_users)} unassigned)", file_context)
        self.info(f"Drivers: {used_drivers}/{total_drivers} used ({len(unused_drivers)} unused)", file_context)
        self.info(f"Routes created: {len(routes)}", file_context)

        # Detailed unassigned analysis
        if unassigned_users:
            self.warning(f"UNASSIGNED USERS ANALYSIS ({len(unassigned_users)} users):", file_context)
            for user in unassigned_users:
                self.warning(f"  User {user.get('user_id', 'N/A')}", file_context)

        if unused_drivers:
            self.warning(f"UNUSED DRIVERS ANALYSIS ({len(unused_drivers)} drivers):", file_context)
            for driver in unused_drivers:
                self.warning(f"  Driver {driver.get('driver_id', 'N/A')} capacity {driver.get('capacity', 'N/A')}", file_context)

        self.info("="*80, file_context)

    def log_accounting_check(self, total_api_users, final_assigned, final_unassigned, discrepancy, file_context):
        """Log comprehensive user accounting check"""
        self.critical("USER ACCOUNTING CHECK", file_context)
        self.critical(f"API Users: {total_api_users}", file_context)
        self.critical(f"Final Assigned: {final_assigned}", file_context)

        # Handle final_unassigned as either int (count) or list (actual users)
        if isinstance(final_unassigned, list):
            unassigned_count = len(final_unassigned)
            self.critical(f"Final Unassigned: {unassigned_count}", file_context)
            total_accounted = final_assigned + unassigned_count
        else:
            unassigned_count = final_unassigned
            self.critical(f"Final Unassigned: {unassigned_count}", file_context)
            total_accounted = final_assigned + unassigned_count

        self.critical(f"Total Accounted: {total_accounted}", file_context)
        self.critical(f"Discrepancy: {discrepancy}", file_context)

        if discrepancy != 0:
            self.critical(f"WARNING: {discrepancy} users unaccounted for!", file_context)
        else:
            self.critical("✅ User accounting verified", file_context)

    def log_route_verification(self, routes, file_context):
        """Log detailed route verification for debugging map mismatches"""
        self.info("🔍 DETAILED ROUTE VERIFICATION:", file_context)
        self.info("="*60, file_context)

        for i, route in enumerate(routes, 1):
            self.info(f"Route {i} - Driver {route['driver_id']}:", file_context)
            self.info(f"  📍 Driver coordinates: ({route['latitude']:.6f}, {route['longitude']:.6f})", file_context)
            self.info(f"  🚗 Vehicle type/capacity: {route['vehicle_type']}", file_context)
            self.info(f"  🆔 Vehicle ID: {route.get('vehicle_id', 'N/A')}", file_context)
            self.info(f"  👥 Number of users: {len(route['assigned_users'])}", file_context)

            if route['assigned_users']:
                self.info(f"  📋 User details:", file_context)
                for j, user in enumerate(route['assigned_users'], 1):
                    office_dist = user.get('office_distance', 'N/A')
                    self.info(f"    {j}. User {user['user_id']}", file_context)
                    self.info(f"       📍 Coordinates: ({user['lat']:.6f}, {user['lng']:.6f})", file_context)
                    self.info(f"       🏢 Office distance: {office_dist}km", file_context)
                    self.info(f"       👤 Name: {user.get('first_name', 'N/A')}", file_context)
                    self.info(f"       ✉️ Email: {user.get('email', 'N/A')}", file_context)
            else:
                self.warning(f"  ❌ NO USERS ASSIGNED TO THIS ROUTE!", file_context)

            utilization = len(route['assigned_users']) / route['vehicle_type'] if route['vehicle_type'] > 0 else 0
            self.info(f"  📊 Utilization: {utilization*100:.1f}%", file_context)
            self.info(f"  📏 Total distance: {route.get('total_distance', 'N/A')}km", file_context)
            self.info("", file_context)

    def log_driver_assignment_summary(self, all_drivers, used_driver_ids, unused_drivers, file_context):
        """Log comprehensive driver assignment summary"""
        self.info("🚗 DRIVER ASSIGNMENT SUMMARY:", file_context)
        self.info("="*60, file_context)

        self.info(f"📊 Driver statistics:", file_context)
        self.info(f"  Total drivers available: {len(all_drivers)}", file_context)
        self.info(f"  Drivers used: {len(used_driver_ids)}", file_context)
        self.info(f"  Drivers unused: {len(unused_drivers)}", file_context)

        self.info(f"📋 Used driver details:", file_context)
        for driver_id in sorted(list(used_driver_ids)):
            # Find the driver details
            driver_details = None
            for driver in all_drivers:
                if str(driver['id']) == driver_id:
                    driver_details = driver
                    break

            if driver_details:
                self.info(f"  ✅ Driver {driver_id}: capacity {driver_details['capacity']}, location ({driver_details['latitude']:.6f}, {driver_details['longitude']:.6f})", file_context)
            else:
                self.info(f"  ❓ Driver {driver_id}: details not found", file_context)

        if unused_drivers:
            self.info(f"📋 Unused driver details:", file_context)
            for driver in unused_drivers:
                self.info(f"  ❌ Driver {driver['driver_id']}: capacity {driver['capacity']}, reason: {driver.get('reason', 'Unknown')}", file_context)

# Global logger instance and session tracking
route_logger = None
current_session_id = None

def get_logger():
    global route_logger
    if route_logger is None:
        route_logger = RouteAssignmentLogger()
    return route_logger

def reset_logger():
    """Reset logger for new session - only if not in an active session"""
    global route_logger, current_session_id

    # Don't reset if we're in the middle of a session
    if current_session_id is not None:
        return route_logger

    if route_logger is not None:
        # Close existing handlers before resetting
        for handler in route_logger.logger.handlers[:]:
            handler.close()
            route_logger.logger.removeHandler(handler)
    route_logger = None
    return None

def start_session():
    """Start a new logging session"""
    global current_session_id

    # Clear old logs before starting new session
    clear_logs()

    current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return get_logger()

def end_session():
    """End the current logging session"""
    global current_session_id
    current_session_id = None