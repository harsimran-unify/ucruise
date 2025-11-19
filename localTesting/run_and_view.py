import subprocess
import threading
import time
import webbrowser
import sys
import argparse
import os
import json
import requests
import urllib.parse

# Add parent directory to Python path to import logger and other modules
sys.path.append('..')
from logger import clear_logs

SOURCE_ID = "UC_logisticllp"  # <-- Replace with your real source_id
PARAMETER = 1  # Example numerical parameter
STRING_PARAM = "Evening shift" # Example string parameter (no URL encoding)
CHOICE = " " # Example choice parameter (use "1" to match main.py behavior)

def start_fastapi():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI on 0.0.0.0:5000")
    # Change to parent directory to start FastAPI from root
    parent_dir = os.path.dirname(os.getcwd())
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload", "--log-level", "info"], cwd=parent_dir)

def launch_browser():
    """Launch browser after server starts"""
    time.sleep(5)  # Wait for server to start
    try:
        webbrowser.open("http://localhost:5000/visualize")
        print("🌐 Browser opened at: http://localhost:5000/visualize")
    except Exception as e:
        print(f"⚠️  Could not auto-open browser: {e}")
        print("   Please manually visit: http://localhost:5000/visualize")

def call_assignment_api(source_id, parameter, string_param, choice):
    """Call the FastAPI assignment endpoint"""
    # URL encode the string parameter to handle spaces
    import urllib.parse
    encoded_string_param = urllib.parse.quote(string_param)

    url = f"http://localhost:5000/assign-drivers/{source_id}/{parameter}/{encoded_string_param}/{choice}"

    try:
        print(f"📡 Calling assignment API: {url}")
        print("⏳ Processing assignment (this may take 30-60 seconds)...")
        print("🔄 Server is running assignment algorithm...")

        # Clear any existing cached routes before making the call
        clear_cached_routes()

        response = requests.post(url, timeout=1800)  # Increased timeout
        response.raise_for_status()

        result = response.json()
        print("✅ API response received successfully!")
        return result

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to FastAPI server. Make sure it's running on port 5000.")
        return None
    except requests.exceptions.Timeout:
        print("❌ API request timed out. The assignment is taking longer than expected.")
        print("💡 Try running 'python test_analysis.py' for direct assignment testing")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {e}")
        return None

def clear_cached_routes():
    """Clear any cached route files to ensure fresh data"""
    route_files = [
        "drivers_and_routes.json",
        "drivers_and_routes_capacity.json",
        "drivers_and_routes_balance.json",
        "drivers_and_routes_road_aware.json"
    ]

    for filename in route_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"🗑️ Cleared cached file: {filename}")
            except Exception as e:
                print(f"⚠️ Could not clear {filename}: {e}")

def display_detailed_analytics(result, algorithm_name):
    """Display comprehensive analytics in terminal with enhanced formatting"""
    print("\n" + "🎯" + "="*78 + "🎯")
    print(f"📊 ROUTEFLOW - INTELLIGENT ASSIGNMENT DASHBOARD ({algorithm_name})")
    print("🎯" + "="*78 + "🎯")

    if result["status"] != "true":
        print("❌ Assignment failed - no analytics available")
        return

    # Basic metrics
    routes = result["data"]
    unassigned_users = result.get("unassignedUsers", [])
    unassigned_drivers = result.get("unassignedDrivers", [])

    total_assigned = sum(len(route["assigned_users"]) for route in routes)
    total_users = total_assigned + len(unassigned_users)

    # Handle no users case
    if total_users == 0:
        print("\n📈 SYSTEM OVERVIEW")
        print("─" * 50)
        print(f"   🚗 Active Routes Created: {len(routes)}")
        print(f"   👥 Users Successfully Assigned: 0")
        print(f"   ⚠️  Users Unassigned: 0")
        print(f"   🚙 Drivers Deployed: 0")
        print(f"   💤 Drivers Available: {len(unassigned_drivers)}")
        print(f"   🏆 Total Fleet Capacity: 0 passengers")
        print(f"\nℹ️  No users found for assignment")
        print(f"   📊 System ready for user assignment when users are available")
        return

    total_capacity = sum(route["vehicle_type"] for route in routes)

    # Enhanced Overview Section
    print(f"\n📈 SYSTEM OVERVIEW")
    print("─" * 50)
    print(f"   🤖 Algorithm Used: {result.get('optimization_mode', 'Unknown')}")
    print(f"   🚗 Active Routes Created: {len(routes)}")
    print(f"   👥 Users Successfully Assigned: {total_assigned}")
    print(f"   ⚠️  Users Unassigned: {len(unassigned_users)}")
    print(f"   🚙 Drivers Deployed: {len(routes)}")
    print(f"   💤 Drivers Available: {len(unassigned_drivers)}")
    print(f"   🏆 Total Fleet Capacity: {total_capacity} passengers")
    if total_capacity > 0:
        print(f"   📊 Overall Capacity Utilization: {(total_assigned/total_capacity*100):.1f}%")
    else:
        print(f"   📊 Overall Capacity Utilization: N/A (no capacity available)")

    # Route Performance Analysis
    print(f"\n🚗 DETAILED ROUTE PERFORMANCE ANALYSIS")
    print("─" * 50)

    for i, route in enumerate(routes):
        assigned = len(route["assigned_users"])
        capacity = route["vehicle_type"]
        utilization = assigned / capacity if capacity > 0 else 0

        # Efficiency categorization
        if utilization >= 0.8:
            efficiency_icon = "🟢"
            efficiency_label = "EXCELLENT"
        elif utilization >= 0.6:
            efficiency_icon = "🟡"
            efficiency_label = "GOOD"
        elif utilization >= 0.4:
            efficiency_icon = "🟠"
            efficiency_label = "FAIR"
        else:
            efficiency_icon = "🔴"
            efficiency_label = "NEEDS OPTIMIZATION"

        print(f"   Route {i+1:2d}: {efficiency_icon} {efficiency_label:15} | "
              f"{assigned}/{capacity} users ({utilization*100:5.1f}%) | "
              f"Driver: {route['driver_id']} | Vehicle: {route.get('vehicle_id', 'N/A')}")

    if unassigned_users:
        print(f"\n⚠️  UNASSIGNED USERS REQUIRING ATTENTION")
        print("─" * 50)
        for i, user in enumerate(unassigned_users[:5]):  # Show first 5
            print(f"   {i+1}. User {user['user_id']}: Location ({user.get('lat', 'N/A')}, {user.get('lng', 'N/A')})")
        if len(unassigned_users) > 5:
            print(f"   ... and {len(unassigned_users) - 5} more users need manual assignment")

    print("\n" + "🎯" + "="*78 + "🎯")
    print("🌐 ACCESS FULL INTERACTIVE DASHBOARD: http://localhost:5000/visualize")
    print("📊 Real-time analytics, route optimization, and performance monitoring available")
    print("🎯" + "="*78 + "🎯\n")

def wait_for_server():
    """Wait for FastAPI server to be ready"""
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            if response.status_code == 200:
                print("✅ FastAPI server is ready!")
                print("🌐 Server URL: http://localhost:5000")
                print("📊 Health check: PASSED")
                return True
        except requests.exceptions.RequestException:
            pass

        if attempt == 0:
            print("⏳ Waiting for FastAPI server to start...")
        if attempt % 5 == 0 and attempt > 0:
            print(f"⏳ Still waiting... ({attempt + 1}/{max_attempts}) - Server starting...")
        time.sleep(2)

    print("❌ Server failed to start within timeout period")
    print("💡 Try manually running: uvicorn main:app --host 0.0.0.0 --port 5000")
    return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Driver Assignment Dashboard')
    parser.add_argument('--source-id', type=str, default=SOURCE_ID,
                       help=f'Source ID for API (default: {SOURCE_ID})')
    parser.add_argument('--parameter', type=int, default=PARAMETER,
                       help=f'Parameter value (default: {PARAMETER})')
    parser.add_argument('--string-param', type=str, default=STRING_PARAM,
                       help=f'String parameter (default: {STRING_PARAM})')
    parser.add_argument('--choice', type=str, default=CHOICE,
                       help=f'Choice parameter (default: {CHOICE})')
    args = parser.parse_args()

    # Clear logs at the start
    clear_logs()

    print("🚀 Starting Driver Assignment Dashboard...")
    print(f"📍 Source ID: {args.source_id}")
    print(f"📊 Parameter: {args.parameter}")
    print(f"📝 String Parameter: {args.string_param}")
    print(f"🎯 Choice: {args.choice}")

    try:
        print("\n🔧 Starting FastAPI Server...")

        # Start server in background
        server_thread = threading.Thread(target=start_fastapi, daemon=True)
        server_thread.start()

        # Wait for server to be ready
        if not wait_for_server():
            print("❌ Failed to start server")
            exit(1)

        print("\n🤖 Running Assignment with Automatic Algorithm Detection...")
        print("=" * 60)
        print(f"🔄 Using parameters: source_id={args.source_id}, parameter={args.parameter}, string_param='{args.string_param}', choice={args.choice}")

        # Call the assignment API
        result = call_assignment_api(args.source_id, args.parameter, args.string_param, args.choice)

        if not result:
            print("❌ Assignment API call failed")
            print("💡 The server may still be processing. Check http://localhost:5000/routes for results")
            print("💡 Or try running 'python test_analysis.py' for direct testing")
            exit(1)

        print(f"📋 Result status: {result.get('status', 'unknown')}")
        print(f"🔍 Result keys: {list(result.keys())}")

        if result["status"] == "true":
            # Get the algorithm name from the result
            algorithm_name = result.get("optimization_mode", "AUTO-DETECTED ALGORITHM")
            algorithm_name = algorithm_name.replace("_", " ").upper()

            print(f"✅ {algorithm_name} assignment completed successfully!")
            print(f"📊 Routes Created: {len(result.get('data', []))}")

            # Display detailed analytics
            display_detailed_analytics(result, algorithm_name)

        else:
            print("❌ Assignment failed:")
            print(f"   Error: {result.get('details', 'Unknown error')}")
            print(f"   Please check your configuration and API credentials")
            exit(1)

        print("\n🌐 Launching Dashboard...")

        # Launch browser
        browser_thread = threading.Thread(target=launch_browser, daemon=True)
        browser_thread.start()

        print("\n✅ Dashboard is running!")
        print("📱 Dashboard URL: http://localhost:5000/visualize")
        print("📊 API Endpoint: http://localhost:5000/routes")
        print("🔍 Health Check: http://localhost:5000/health")
        print("\n⌨️  Press Ctrl+C to stop the server")

        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down dashboard...")
            print("👋 Goodbye!")

    except KeyboardInterrupt:
        print("\n🛑 Dashboard startup interrupted by user")
        exit(0)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print(f"📋 Error type: {type(e).__name__}")
        exit(1)