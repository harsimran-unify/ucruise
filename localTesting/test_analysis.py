import os
import json
import requests
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to Python path to import logger and other modules
sys.path.append('..')
from algorithm.base.base import run_assignment
from logger import get_logger

logger = get_logger()

def capture_api_response(source_id, parameter, string_param):
    """
    Capture the raw API response when fetching data
    """
    logger.info("CAPTURING API REQUEST/RESPONSE")
    logger.info("="*60)
    
    # Load environment variables
    env_path = ".env" if os.path.exists(".env") else "../.env"
    if not os.path.exists(env_path):
        logger.error(".env file not found!")
        return None

    load_dotenv(env_path)
    BASE_API_URL = os.getenv("API_URL")
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
    
    if not BASE_API_URL or not API_AUTH_TOKEN:
        logger.error("Missing API_URL or API_AUTH_TOKEN in .env file")
        return None
    
    # Construct API URL
    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{string_param}"
    
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    logger.info(f"API Request Details:")
    logger.info(f"URL: {API_URL}")
    logger.debug(f"Headers: {json.dumps(headers, indent=2)}")
    logger.info(f"Method: GET")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        logger.info("Making API request...")
        response = requests.get(API_URL, headers=headers, timeout=30)
        
        logger.info(f"API Response Details:")
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response Time: {response.elapsed.total_seconds():.2f} seconds")
        logger.debug(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        logger.debug(f"Content-Length: {len(response.text)} characters")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                logger.debug(f"JSON Structure: Valid")
                
                # Analyze response structure
                if isinstance(response_data, dict):
                    logger.debug(f"Top-level keys: {list(response_data.keys())}")
                    
                    # Check for users
                    users = response_data.get('users', [])
                    logger.debug(f"Users count: {len(users)}")
                    
                    # Check for drivers structure
                    if 'drivers' in response_data:
                        drivers = response_data['drivers']
                        drivers_unassigned = drivers.get('driversUnassigned', [])
                        drivers_assigned = drivers.get('driversAssigned', [])
                        logger.debug(f"Drivers Unassigned: {len(drivers_unassigned)}")
                        logger.debug(f"Drivers Assigned: {len(drivers_assigned)}")
                    else:
                        drivers_unassigned = response_data.get('driversUnassigned', [])
                        drivers_assigned = response_data.get('driversAssigned', [])
                        logger.debug(f"Drivers Unassigned (flat): {len(drivers_unassigned)}")
                        logger.debug(f"Drivers Assigned (flat): {len(drivers_assigned)}")
                
                return {
                    "request": {
                        "url": API_URL,
                        "method": "GET",
                        "headers": headers,
                        "timestamp": datetime.now().isoformat(),
                        "source_id": source_id,
                        "parameter": parameter,
                        "string_param": string_param
                    },
                    "response": {
                        "status_code": response.status_code,
                        "response_time_seconds": response.elapsed.total_seconds(),
                        "content_type": response.headers.get('content-type', 'unknown'),
                        "content_length": len(response.text),
                        "data": response_data
                    }
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON Structure: Invalid - {e}")
                return {
                    "request": {
                        "url": API_URL,
                        "method": "GET",
                        "headers": headers,
                        "timestamp": datetime.now().isoformat()
                    },
                    "response": {
                        "status_code": response.status_code,
                        "response_time_seconds": response.elapsed.total_seconds(),
                        "content_type": response.headers.get('content-type', 'unknown'),
                        "content_length": len(response.text),
                        "raw_text": response.text,
                        "json_error": str(e)
                    }
                }
        else:
            logger.error(f"Error Response: {response.text[:500]}")
            return {
                "request": {
                    "url": API_URL,
                    "method": "GET", 
                    "headers": headers,
                    "timestamp": datetime.now().isoformat()
                },
                "response": {
                    "status_code": response.status_code,
                    "response_time_seconds": response.elapsed.total_seconds(),
                    "content_type": response.headers.get('content-type', 'unknown'),
                    "error_text": response.text
                }
            }
            
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {
            "request": {
                "url": API_URL,
                "method": "GET",
                "headers": headers,
                "timestamp": datetime.now().isoformat()
            },
            "response": {
                "error": str(e),
                "error_type": type(e).__name__
            }
        }

def main():
    """
    Test script to capture API request/response and assignment results
    """
    logger.info("API REQUEST/RESPONSE CAPTURE TEST")
    logger.info("="*60)
    
    # Configuration
    source_id = "UC_logisticllp"  # Update this to match your API format
    parameter = 1
    string_param = "Evening%20shift"  # Use plain text, let requests handle URL encoding
    
    logger.info(f"Test Configuration:")
    logger.info(f"Source ID: {source_id}")
    logger.info(f"Parameter: {parameter}")
    logger.info(f"String Parameter: {string_param}")
    logger.info(f"Current Directory: {os.getcwd()}")
    logger.info(f".env File Exists: {os.path.exists('.env')}")
    
    # Step 1: Capture API request/response
    logger.info("="*60)
    logger.info("STEP 1: CAPTURING RAW API DATA")
    logger.info("="*60)
    
    api_capture = capture_api_response(source_id, parameter, string_param)
    
    if not api_capture:
        logger.error("Failed to capture API data")
        return
    
    # Step 2: Run assignment and capture result
    logger.info("="*60)
    logger.info("STEP 2: RUNNING ASSIGNMENT ALGORITHM")
    logger.info("="*60)
    
    logger.info("Starting assignment algorithm...")
    assignment_start_time = datetime.now()
    
    try:
        assignment_result = run_assignment(source_id, parameter, string_param)
        assignment_end_time = datetime.now()
        assignment_duration = (assignment_end_time - assignment_start_time).total_seconds()
        
        logger.info(f"Assignment completed in {assignment_duration:.2f} seconds")
        logger.info(f"Assignment Status: {assignment_result.get('status', 'unknown')}")
        
        # Check if API had any drivers
        api_data = api_capture.get('response', {}).get('data', {})
        total_drivers_in_api = (
            len(api_data.get('driversUnassigned', [])) + 
            len(api_data.get('driversAssigned', []))
        )
        
        if total_drivers_in_api == 0:
            logger.warning(f"⚠️ NO DRIVERS IN API RESPONSE - Assignment will fail")
            logger.warning(f"This explains why assignment returned 0 routes")
        
        if assignment_result.get('status') == 'true':
            routes = assignment_result.get('data', [])
            unassigned_users = assignment_result.get('unassignedUsers', [])
            unassigned_drivers = assignment_result.get('unassignedDrivers', [])
            
            logger.info(f"Routes Created: {len(routes)}")
            logger.info(f"Users Assigned: {sum(len(route.get('assigned_users', [])) for route in routes)}")
            logger.info(f"Users Unassigned: {len(unassigned_users)}")
            logger.info(f"Drivers Used: {len(routes)}")
            logger.info(f"Drivers Unused: {len(unassigned_drivers)}")
            
            if len(routes) == 0 and total_drivers_in_api == 0:
                logger.info("✅ Assignment correctly handled case with no drivers available")
        else:
            logger.error(f"Assignment Error: {assignment_result.get('details', 'Unknown error')}")
            
            # Check if error is related to no drivers
            error_details = assignment_result.get('details', '')
            if 'division by zero' in error_details.lower() or total_drivers_in_api == 0:
                logger.info("💡 This error is likely caused by having no drivers in the API response")
    
    except Exception as e:
        assignment_end_time = datetime.now()
        assignment_duration = (assignment_end_time - assignment_start_time).total_seconds()
        logger.error(f"Assignment failed after {assignment_duration:.2f} seconds: {e}")
        
        assignment_result = {
            "status": "false",
            "error": str(e),
            "error_type": type(e).__name__,
            "data": []
        }
    
    # Step 3: Compile complete test report
    logger.info("="*60)
    logger.info("STEP 3: GENERATING COMPLETE TEST REPORT")
    logger.info("="*60)
    
    complete_report = {
        "test_metadata": {
            "test_name": "API Request/Response Capture Test",
            "timestamp": datetime.now().isoformat(),
            "source_id": source_id,
            "parameter": parameter,
            "string_param": string_param,
            "test_duration_seconds": (datetime.now() - assignment_start_time).total_seconds()
        },
        "api_data_capture": api_capture,
        "assignment_execution": {
            "start_time": assignment_start_time.isoformat(),
            "end_time": assignment_end_time.isoformat(),
            "duration_seconds": assignment_duration,
            "result": assignment_result
        },
        "summary": {
            "api_request_successful": api_capture.get('response', {}).get('status_code') == 200,
            "assignment_successful": assignment_result.get('status') == 'true',
            "total_api_response_size": api_capture.get('response', {}).get('content_length', 0),
            "routes_generated": len(assignment_result.get('data', [])),
            "users_from_api": len(api_capture.get('response', {}).get('data', {}).get('users', [])),
            "drivers_from_api": (
                len(api_capture.get('response', {}).get('data', {}).get('driversUnassigned', [])) +
                len(api_capture.get('response', {}).get('data', {}).get('driversAssigned', []))
            ),
            "api_has_drivers": (
                len(api_capture.get('response', {}).get('data', {}).get('driversUnassigned', [])) +
                len(api_capture.get('response', {}).get('data', {}).get('driversAssigned', []))
            ) > 0
        }
    }
    
    # Save to JSON file
    output_filename = f"api_capture_test_{source_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(output_filename, 'w') as f:
            json.dump(complete_report, f, indent=2, default=str)
        
        logger.info(f"Complete test report saved to: {output_filename}")
        logger.info(f"File size: {os.path.getsize(output_filename)} bytes")
        
        # Display summary
        logger.info(f"TEST SUMMARY:")
        logger.info(f"API Status: {'✅ Success' if complete_report['summary']['api_request_successful'] else '❌ Failed'}")
        logger.info(f"Assignment Status: {'✅ Success' if complete_report['summary']['assignment_successful'] else '❌ Failed'}")
        logger.info(f"API Response Size: {complete_report['summary']['total_api_response_size']} characters")
        logger.info(f"Users from API: {complete_report['summary']['users_from_api']}")
        logger.info(f"Drivers from API: {complete_report['summary']['drivers_from_api']}")
        logger.info(f"Routes Generated: {complete_report['summary']['routes_generated']}")
        
        logger.info(f"REPORT SECTIONS:")
        logger.info(f"• test_metadata: General test information")
        logger.info(f"• api_data_capture: Raw API request/response data")
        logger.info(f"• assignment_execution: Assignment algorithm results")
        logger.info(f"• summary: Key metrics and status")
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        logger.info(f"Report data available in memory but not saved to file")
        
    except Exception as e:
        logger.error(f"Failed to analyze assignment: {e}")
        return None

if __name__ == "__main__":
    main()