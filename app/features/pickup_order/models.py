"""
Request/response models for the pickup-order API (pydantic v1 API — the
project pins pydantic<2; do NOT modernize @validator / class Config).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class UserWithPickupOrder(BaseModel):
    user_id: str
    latitude: float = Field(..., description="User latitude - accepts string or float")
    longitude: float = Field(..., description="User longitude - accepts string or float")
    office_distance: Optional[float] = None
    first_name: Optional[str] = None
    last_name: Optional[Any] = None  # Can be null
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    employee_shift: Optional[str] = None
    shift_type: Optional[str] = None
    route_no: Optional[int] = None
    pickup_order: Optional[int] = None

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class DriverWithRoutes(BaseModel):
    driver_id: str
    vehicle_id: Optional[str] = None
    latitude: float = Field(..., description="Driver latitude - accepts string or float")
    longitude: float = Field(..., description="Driver longitude - accepts string or float")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    vehicle_name: Optional[str] = None
    vehicle_no: Optional[str] = None
    capacity: Optional[int] = None
    chasis_no: Optional[str] = None
    color: Optional[str] = None
    registration_no: Optional[str] = None
    shift_type_id: Optional[int] = None
    route_no: Optional[int] = None
    assigned_users: List[UserWithPickupOrder] = []

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class CompanyInfo(BaseModel):
    id: int
    address: Optional[str] = None
    latitude: float = Field(..., description="Company latitude - accepts string or float")
    longitude: float = Field(..., description="Company longitude - accepts string or float")

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class DataPayload(BaseModel):
    company: CompanyInfo
    routes: List[DriverWithRoutes]


class RecalculatePickupOrderRequest(BaseModel):
    data: DataPayload


class RecalculatePickupOrderResponse(BaseModel):
    status: str
    message: str
    data: DataPayload
    optimization_summary: Optional[Dict[str, Any]] = None


# ============== V2 API Models (Simplified) ==============

class DriverLocationV2(BaseModel):
    latitude: float
    longitude: float


class AssignedUserV2(BaseModel):
    user_id: int
    latitude: float
    longitude: float
    pickup_order: Optional[int] = None
    route_no: int

    @validator('user_id', 'route_no', pre=True)
    def parse_integers(cls, v):
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"Invalid integer value: {v}")
        return v

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class RouteV2(BaseModel):
    driver_id: int
    route_no: int
    driver_location: DriverLocationV2
    assigned_users: List[AssignedUserV2] = []

    @validator('driver_id', 'route_no', pre=True)
    def parse_integers(cls, v):
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"Invalid integer value: {v}")
        return v

    class Config:
        extra = "allow"


class CompanyInfoV2(BaseModel):
    latitude: float
    longitude: float

    @validator('latitude', 'longitude', pre=True)
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid coordinate value: {v}")
        return v

    class Config:
        extra = "allow"


class RecalculatePickupOrderRequestV2(BaseModel):
    company: CompanyInfoV2
    routes: List[RouteV2]  # No nesting - flat structure


class RecalculatePickupOrderResponseV2(BaseModel):
    status: str
    message: str
    company: CompanyInfoV2
    routes: List[RouteV2]
