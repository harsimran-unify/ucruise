"""
Centralized configuration for road-aware routing thresholds and parameters
"""

class RoadAwareConfig:
    """Centralized configuration for all road-aware routing parameters"""

    # Detour ratio thresholds (consistency across all modules)
    MAX_DETOUR_RATIO_DEFAULT = 2.5  # Base threshold for most operations
    MAX_DETOUR_RATIO_STRICT = 1.5   # For tight route optimization
    MAX_DETOUR_RATIO_RELAXED = 3.0  # For initial clustering/coverage

    # Distance and proximity thresholds
    MAX_REASONABLE_RATIO = 3.0       # For coherence validation
    NEAR_PATH_THRESHOLD_KM = 0.5     # Path proximity check
    CANDIDATE_CUTOFF_KM = 1.0        # Candidate filtering
    MAX_SEARCH_RADIUS_KM = 5.0       # Node search radius

    # Speed and weight calculation
    BASE_SPEED_KMH = 50.0           # Reference speed for factor calculation
    MIN_SPEED_FACTOR = 0.5          # Minimum speed multiplier
    MAX_SPEED_FACTOR = 2.0          # Maximum speed multiplier

    # Cache and precision settings
    COORDINATE_PRECISION = 6         # Decimal places for coordinate rounding
    CACHE_SIZE_LIMIT = 10000        # Maximum cache entries

    # Fallback behavior
    USE_BEARING_FALLBACK = True     # Use bearing when nodes missing
    FALLBACK_DETOUR_RATIO = 1.25    # Conservative fallback threshold
    FALLBACK_SEARCH_RADIUS_KM = 10.0 # Fallback search radius

    # Cache configuration
    CACHE_CLEANUP_THRESHOLD = 8000   # Cache cleanup threshold

    # Coherence scoring weights
    DIRECTIONAL_PENALTY_WEIGHT = 0.3
    BACKTRACK_PENALTY_WEIGHT = 0.4
    CORRIDOR_BONUS_WEIGHT = 0.2
    SEQUENCE_PENALTY_WEIGHT = 0.1

    # Road efficiency thresholds
    MIN_ROAD_EFFICIENCY = 0.5

    # Road type weights for routing
    ROAD_TYPE_WEIGHTS = {
        'motorway': 0.8,
        'trunk': 0.9,
        'primary': 1.0,
        'secondary': 1.1,
        'tertiary': 1.2,
        'residential': 1.3,
        'service': 1.5,
        'track': 2.0
    }

    # Coverage mode settings
    COVERAGE_MAX_DIST_TO_OFFICE_KM = 15.0
    DEFAULT_NEAR_PATH_THRESHOLD_KM = 1.0
    COHERENCE_TOLERANCE_DEFAULT = 0.1
    COHERENCE_TOLERANCE_COVERAGE = 0.2

    # Backtracking penalty settings
    MAX_BACKTRACK_PENALTY = 0.5
    BACKTRACK_PENALTY_SCALE = 5.0

    @classmethod
    def get_detour_ratio_for_mode(cls, mode="default"):
        """Get appropriate detour ratio based on operation mode"""
        mode_mapping = {
            "strict": cls.MAX_DETOUR_RATIO_STRICT,
            "default": cls.MAX_DETOUR_RATIO_DEFAULT,
            "relaxed": cls.MAX_DETOUR_RATIO_RELAXED,
            "coverage": cls.MAX_DETOUR_RATIO_RELAXED,
            "optimization": cls.MAX_DETOUR_RATIO_STRICT
        }
        return mode_mapping.get(mode, cls.MAX_DETOUR_RATIO_DEFAULT)

    @classmethod
    def get_all_config(cls):
        """Return all configuration as a dictionary"""
        return {
            'max_detour_ratio_default': cls.MAX_DETOUR_RATIO_DEFAULT,
            'max_detour_ratio_strict': cls.MAX_DETOUR_RATIO_STRICT,
            'max_detour_ratio_relaxed': cls.MAX_DETOUR_RATIO_RELAXED,
            'max_reasonable_ratio': cls.MAX_REASONABLE_RATIO,
            'near_path_threshold_km': cls.NEAR_PATH_THRESHOLD_KM,
            'candidate_cutoff_km': cls.CANDIDATE_CUTOFF_KM,
            'max_search_radius_km': cls.MAX_SEARCH_RADIUS_KM,
            'base_speed_kmh': cls.BASE_SPEED_KMH,
            'min_speed_factor': cls.MIN_SPEED_FACTOR,
            'max_speed_factor': cls.MAX_SPEED_FACTOR,
            'coordinate_precision': cls.COORDINATE_PRECISION,
            'cache_size_limit': cls.CACHE_SIZE_LIMIT,
            'use_bearing_fallback': cls.USE_BEARING_FALLBACK,
            'fallback_detour_ratio': cls.FALLBACK_DETOUR_RATIO
        }