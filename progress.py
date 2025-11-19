import sys
import time
from datetime import datetime

class ProgressTracker:
    def __init__(self):
        self.start_time = None
        self.current_stage = ""
        self.stages = [
            "Data Loading & Validation",
            "Geographic Clustering", 
            "Capacity Sub-clustering",
            "Driver Assignment",
            "Local Optimization",
            "Global Optimization",
            "Final Merge & Validation"
        ]
        self.current_stage_index = 0
        self.stage_details = {}
        self.assignment_started = False # Added to track if assignment has started

    def start_assignment(self, source_id, mode):
        self.start_time = datetime.now()
        self.assignment_started = True # Set to True when assignment starts
        print(f"\n🚀 Starting {mode} Assignment")
        print(f"📋 Source ID: {source_id}")
        print(f"⏰ Started at: {self.start_time.strftime('%H:%M:%S')}")
        print("="*60)

    def start_stage(self, stage_name, details=""):
        if not self.assignment_started: # Check if assignment has started
            return

        self.current_stage = stage_name
        if stage_name in self.stages:
            self.current_stage_index = self.stages.index(stage_name) + 1

        # Progress bar
        progress = (self.current_stage_index / len(self.stages)) * 100
        bar_length = 30
        filled_length = int(bar_length * self.current_stage_index // len(self.stages))
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(f"\n📊 Stage {self.current_stage_index}/{len(self.stages)}: {stage_name}")
        print(f"[{bar}] {progress:.1f}%")
        if details:
            print(f"   {details}")

        self.stage_details[stage_name] = {
            'start_time': datetime.now(),
            'details': details
        }

    def update_stage_progress(self, message):
        if not self.assignment_started: # Check if assignment has started
            return

        elapsed = datetime.now() - self.stage_details.get(self.current_stage, {}).get('start_time', datetime.now())
        print(f"   ⏳ {message} (Elapsed: {elapsed.total_seconds():.1f}s)")

    def complete_stage(self, summary):
        if not self.assignment_started: # Check if assignment has started
            return

        if self.current_stage in self.stage_details:
            elapsed = datetime.now() - self.stage_details[self.current_stage]['start_time']
            print(f"   ✅ {summary} (Completed in {elapsed.total_seconds():.1f}s)")

    def fail_stage(self, reason: str):
        """Mark current stage as failed"""
        if self.current_stage:
            # This part assumes self.stages is a list of dictionaries, which it is not.
            # The original code had a list of strings for stages.
            # For now, we'll just print the error and rely on fail_assignment for overall failure.
            print(f"❌ Stage '{self.current_stage}' failed: {reason}")
        else:
            print(f"❌ Assignment failed: {reason}")

    def show_final_summary(self, result):
        """Show final summary of assignment"""
        if not self.assignment_started:
            return

        routes = result.get("data", [])
        unassigned_users = result.get("unassignedUsers", [])
        total_assigned = sum(len(route.get("assigned_users", [])) for route in routes)

        print(f"\n{'='*60}")
        print(f"🎯 ASSIGNMENT COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"📊 Routes Created: {len(routes)}")
        print(f"👥 Users Assigned: {total_assigned}")
        print(f"⚠️  Users Unassigned: {len(unassigned_users)}")
        print(f"⏰ Total Time: {result.get('execution_time', 0):.1f}s")
        print(f"🎪 Algorithm: {result.get('optimization_mode', 'Unknown').upper()}")
        print(f"{'='*60}")

        self.assignment_started = False

    def fail_assignment(self, error_message):
        """Handle assignment failure"""
        if not self.assignment_started:
            return

        print(f"\n{'='*60}")
        print(f"❌ ASSIGNMENT FAILED")
        print(f"{'='*60}")
        print(f"🚫 Error: {error_message}")
        # Assuming start_time is correctly set in start_assignment
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        print(f"⏰ Failed after: {elapsed_time.total_seconds():.1f}s")
        print(f"{'='*60}")

        self.assignment_started = False

# Global progress tracker
progress_tracker = ProgressTracker()

def get_progress_tracker():
    return progress_tracker