
import threading
import time
import os
from logger import get_logger
import queue

logger = get_logger()


def _haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in km. Local copy so the mock fallback never needs
    the algorithm package (which pulls in pandas/requests/etc.)."""
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371


class RoadNetworkManager:
    """Manages road network loading with persistent background loading and retry logic"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.road_network = None
            self._loading = False
            self._loaded = False
            self._load_error = None
            self._retry_count = 0
            self._max_retries = 3
            self._retry_delay = 2  # seconds
            self._load_thread = None
            self._use_mock = False
            self._initialized = True
            
            # Start loading immediately on initialization
            self.start_loading()
    
    def start_loading(self, graphml_path=None):
        """Start loading road network in background with retry logic"""
        # If no path provided, use absolute path to GraphML file
        if graphml_path is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up from roadNetwork/ to project root
            graphml_path = os.path.join(script_dir, 'roadNetwork', 'tricity_main_roads.graphml')

        if self._loading or self._loaded:
            return
        
        self._loading = True

        def load_worker():
            try:
                while self._retry_count < self._max_retries and not self._loaded:
                    try:
                        logger.info(f"🔄 Attempt {self._retry_count + 1}/{self._max_retries}: Loading road network...")

                        import roadNetwork.roadNetwork as road_network_module
                        self.road_network = road_network_module.RoadNetwork(graphml_path)
                        self._loaded = True
                        self._use_mock = False
                        logger.info("✅ Road network loaded successfully!")
                        break

                    except Exception as e:
                        self._retry_count += 1
                        logger.warning(f"⚠️ Attempt {self._retry_count} failed: {e}")

                        if self._retry_count < self._max_retries:
                            logger.info(f"Retrying in {self._retry_delay} seconds...")
                            time.sleep(self._retry_delay)
                        else:
                            logger.error(f"❌ Failed to load road network after {self._max_retries} attempts")
                            self._load_error = e
                            self._create_mock_network()
                            self._use_mock = True
                            self._loaded = True
            finally:
                # Always reset the flag so is_loading()/get_status() report
                # correctly once loading finished (success or failure).
                self._loading = False
        
        self._load_thread = threading.Thread(target=load_worker, daemon=True, name="RoadNetworkLoader")
        self._load_thread.start()
        logger.info("🚀 Road network loading started in background")
    
    def _create_mock_network(self):
        """Create mock road network for fallback"""
        class MockRoadNetwork:
            def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
                if not user_positions:
                    return 1.0

                avg_dist_from_driver = sum(
                    _haversine_distance(driver_pos[0], driver_pos[1], u[0], u[1])
                    for u in user_positions
                ) / len(user_positions)

                avg_dist_from_office = sum(
                    _haversine_distance(office_pos[0], office_pos[1], u[0], u[1])
                    for u in user_positions
                ) / len(user_positions)

                score = max(0, 1.0 - (avg_dist_from_driver / 50.0) - (avg_dist_from_office / 100.0))
                return min(1.0, score)

            def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos, max_detour_ratio=1.5, route_type="optimization"):
                return True

            def get_road_distance(self, lat1, lon1, lat2, lon2):
                return _haversine_distance(lat1, lon1, lat2, lon2)

            def find_nearest_road_node(self, lat, lon):
                return None, None

            def simplify_path_nodes(self, path, max_nodes=10):
                return path

        self.road_network = MockRoadNetwork()
        logger.warning("⚠️ Using mock road network (limited functionality)")
    
    def get_road_network(self, wait_timeout=10, force_wait=False):
        """
        Get road network, waiting for actual network to load

        Args:
            wait_timeout: Maximum seconds to wait for actual network
            force_wait: If True, never fall back to the temporary mock — wait
                for the real network up to wait_timeout and return None if it
                is still not ready
        """
        # If already loaded, return immediately — unless a temporary mock is in
        # place while the real network is still loading in the background (in
        # that case fall through to the wait logic below)
        if self._loaded and not (self._use_mock and self._loading):
            if self._use_mock:
                if force_wait:
                    # Caller demanded the real network; retries are exhausted
                    # so it is not coming — report that honestly.
                    return None
                if wait_timeout > 0:
                    logger.warning("⚠️ Using mock road network")
            return self.road_network

        # Wait for loading to complete (with optimized polling)
        if wait_timeout > 0:
            logger.info(f"⏳ Waiting up to {wait_timeout}s for road network to load...")
            start_time = time.time()

            # Adaptive sleep - start with shorter intervals
            sleep_interval = 0.05  # 50ms initially

            while (self._loading or force_wait) and (time.time() - start_time) < wait_timeout:
                if self._loaded:
                    break
                time.sleep(sleep_interval)
                # Gradually increase sleep interval to reduce CPU usage
                sleep_interval = min(0.2, sleep_interval * 1.5)

            # Check if loaded successfully
            if self._loaded and not self._use_mock:
                logger.info("✅ Road network ready!")
                return self.road_network

        # If timeout is 0, return immediately (non-blocking)
        if wait_timeout == 0:
            if self._loaded:
                return self.road_network
            else:
                logger.debug("Road network not ready yet, returning None")
                return None

        # force_wait callers never get the temporary mock
        if force_wait:
            logger.warning(f"Road network not ready after {wait_timeout}s force wait")
            return None

        # If not loaded and not using mock yet, create mock as last resort
        if not self._loaded:
            logger.warning("Road network not ready, using temporary mock")
            self._create_mock_network()
            self._loaded = True
            self._use_mock = True

        return self.road_network
    
    def is_loaded(self):
        """Check if road network is loaded (actual or mock)"""
        return self._loaded
    
    def is_loading(self):
        """Check if road network is currently loading"""
        return self._loading
    
    def is_using_mock(self):
        """Check if using mock network"""
        return self._use_mock
    
    def get_status(self):
        """Get detailed status"""
        return {
            'loaded': self._loaded,
            'loading': self._loading,
            'using_mock': self._use_mock,
            'retry_count': self._retry_count,
            'has_error': self._load_error is not None
        }

# Global instance - starts loading immediately
road_network_manager = RoadNetworkManager()
