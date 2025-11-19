
import threading
import time
import os
from logger import get_logger
import queue

logger = get_logger()

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
            while self._retry_count < self._max_retries and not self._loaded:
                try:
                    logger.info(f"🔄 Attempt {self._retry_count + 1}/{self._max_retries}: Loading road network...")
                    
                    import roadNetwork.roadNetwork as road_network_module
                    self.road_network = road_network_module.RoadNetwork(graphml_path)
                    self._loaded = True
                    self._use_mock = False
                    logger.info("✅ Road network loaded successfully!")
                    return
                    
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
            
            self._loading = False
        
        self._load_thread = threading.Thread(target=load_worker, daemon=True, name="RoadNetworkLoader")
        self._load_thread.start()
        logger.info("🚀 Road network loading started in background")
    
    def _create_mock_network(self):
        """Create mock road network for fallback"""
        from algorithm.base.base import haversine_distance
        
        class MockRoadNetwork:
            def get_route_coherence_score(self, driver_pos, user_positions, office_pos):
                if not user_positions:
                    return 1.0
                
                avg_dist_from_driver = sum(
                    haversine_distance(driver_pos[0], driver_pos[1], u[0], u[1])
                    for u in user_positions
                ) / len(user_positions)
                
                avg_dist_from_office = sum(
                    haversine_distance(office_pos[0], office_pos[1], u[0], u[1])
                    for u in user_positions
                ) / len(user_positions)
                
                score = max(0, 1.0 - (avg_dist_from_driver / 50.0) - (avg_dist_from_office / 100.0))
                return min(1.0, score)
            
            def is_user_on_route_path(self, driver_pos, current_user_positions, user_pos, office_pos, max_detour_ratio=1.5, route_type="optimization"):
                return True
            
            def get_road_distance(self, lat1, lon1, lat2, lon2):
                return haversine_distance(lat1, lon1, lat2, lon2)
            
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
            force_wait: If True, always wait for actual network (no mock fallback during wait)
        """
        # If already loaded, return immediately
        if self._loaded:
            if self._use_mock and wait_timeout > 0:
                logger.warning("⚠️ Using mock road network")
            return self.road_network
        
        # Wait for loading to complete (with optimized polling)
        if self._loading and wait_timeout > 0:
            logger.info(f"⏳ Waiting up to {wait_timeout}s for road network to load...")
            start_time = time.time()
            
            # Adaptive sleep - start with shorter intervals
            sleep_interval = 0.05  # 50ms initially
            
            while self._loading and (time.time() - start_time) < wait_timeout:
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
            elif not self._use_mock:
                logger.debug("Road network not ready yet, returning None")
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
