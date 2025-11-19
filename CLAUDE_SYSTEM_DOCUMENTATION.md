# Cab Clustering API System Documentation

## Overview
This is a sophisticated cab routing and clustering system built with FastAPI that optimizes driver-passenger assignments based on various algorithms and priorities.

## System Architecture

### Main Components

#### 1. **FastAPI Application (`main.py`)**
- **Purpose**: Entry point for all API requests
- **Key Functions**:
  - Routes requests to appropriate algorithms based on priority settings
  - Handles different optimization modes (route efficiency, capacity, balanced, road-aware)
  - Processes incoming requests and formats responses

#### 2. **Algorithm Routing System**
Located in `main.py` lines 60-103, the system uses a priority-based routing mechanism:

```python
# Current Priority Levels:
# Priority 1: route_efficiency
# Priority 2: capacity_optimization
# Priority 3: balanced_optimization
# Priority 4: road_aware_routing
# Priority 5: female_safety (TO BE ADDED)
```

#### 3. **Algorithm Modules**
Each algorithm follows a consistent structure:
- `algorithm/route_efficiency/route_efficiency.py`
- `algorithm/capacity/capacity.py`
- `algorithm/balanced/balanced.py`
- `algorithm/road_aware/road_aware.py`

**Common Pattern**:
```python
from fastapi import Blueprint
from algorithm.base.base import load_env_and_fetch_data
# Algorithm-specific logic
# Clustering and assignment
# Result formatting
```

#### 4. **Base System (`algorithm/base/base.py`)**
- **Purpose**: Core data processing and API communication
- **Key Function**: `load_env_and_fetch_data()`
  - Fetches data from external API
  - Processes user and driver information
  - **Handles safety configuration**: `"safety": {"female_safety": 1}`

#### 5. **Configuration (`config.json`)**
- Stores algorithm parameters and settings
- Contains vehicle types, capacity limits, and optimization parameters

## Data Flow

### 1. **API Request Reception**
```
Client → FastAPI (main.py) → Algorithm Router
```

### 2. **Data Processing**
```
main.py → load_env_and_fetch_data() → External API → User/Driver Data
```

### 3. **Algorithm Selection**
Based on `ride_settings.pic_priority` from API data:
```json
"ride_settings": {
  "pic_priority": 2  // Determines which algorithm to use
}
```

### 4. **Clustering & Assignment**
```
Selected Algorithm → Clustering Logic → Driver-Passenger Assignment
```

### 5. **Response Formatting**
```
Algorithm Result → JSON Response → Client
```

## Current Algorithm Capabilities

### 1. **Route Efficiency (Priority 1)**
- Focuses on minimizing total travel distance
- Optimizes for shortest routes

### 2. **Capacity Optimization (Priority 2)**
- Maximizes vehicle utilization
- Ensures optimal passenger-to-vehicle ratios

### 3. **Balanced Optimization (Priority 3)**
- Balances between efficiency and capacity
- Provides middle-ground solution

### 4. **Road-Aware Routing (Priority 4)**
- Considers road conditions and traffic
- Advanced routing with real-time factors

## API Data Structure

### Input Data Format
```json
{
  "data": {
    "users": [
      {
        "id": 335242,
        "first_name": "Kritika",
        "gender": "Female",  // IMPORTANT FOR FEMALE SAFETY
        "latitude": "30.7087476",
        "longitude": "76.8424893",
        // ... other user data
      }
    ],
    "drivers": {
      "driversAssigned": [],
      "driversUnassigned": [
        // Driver data with capacity and location
      ]
    },
    "safety": {
      "female_safety": 1  // NEW: Safety configuration
    },
    "ride_settings": {
      "pic_priority": 2  // Algorithm selection
    }
  }
}
```

### Output Data Format
```json
{
  "data": [
    {
      "driver_id": "335471",
      "vehicle_id": "59",
      "assigned_users": [
        {
          "user_id": "335337",
          "first_name": "Gourav",
          // User assignment details
        }
      ]
    }
  ],
  "unassignedUsers": [],
  "unassignedDrivers": []
}
```

## Key Integration Points for Female Safety Algorithm

### 1. **Safety Configuration Detection**
- Location: `algorithm/base/base.py` in `load_env_and_fetch_data()`
- The system already parses: `"safety": {"female_safety": 1}`

### 2. **Algorithm Routing**
- Location: `main.py` lines 60-103
- Add new priority level for female safety

### 3. **User Gender Processing**
- Available in user data: `"gender": "Female"`
- Can be used for female-specific clustering

### 4. **Implementation Template**
- Use `algorithm/capacity/capacity.py` as a template
- Follows established patterns for imports, processing, and output

## File Structure
```
clustering_cab_api replit routes/
├── main.py                          # FastAPI application and routing
├── config.json                      # System configuration
├── algorithm/
│   ├── base/
│   │   └── base.py                  # Core data processing
│   ├── route_efficiency/
│   ├── capacity/
│   ├── balanced/
│   └── road_aware/
├── localTesting/
│   └── api_capture_test_*.json      # Test data samples
└── CLAUDE_SYSTEM_DOCUMENTATION.md   # This file
```

## Current State with Female Safety

### What's Already Available:
1. ✅ Safety configuration parsing in API data
2. ✅ User gender information in data
3. ✅ Priority-based algorithm routing system
4. ✅ Template patterns for implementing new algorithms

### What Needs to Be Added:
1. 🔧 Female safety algorithm module
2. 🔧 Algorithm routing update for new priority
3. 🔧 Configuration parameters for female safety
4. 🔧 Female-specific clustering logic

## Next Steps for Female Safety Implementation

1. **Create `algorithm/female_safety/female_safety.py`**
2. **Update `main.py` algorithm routing**
3. **Add configuration to `config.json`**
4. **Implement female-specific clustering logic**
5. **Test with existing API data**

This documentation provides a complete understanding of the system architecture and where the female safety algorithm should be integrated.