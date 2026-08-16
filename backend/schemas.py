"""
FastAPI Backend - Pydantic Request & Response Schemas Module
Validates authentication, profile management, prediction inputs, and dashboard metrics.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, EmailStr, Field


# -----------------------------------------------------------------------------
# 1. AUTHENTICATION SCHEMAS
# -----------------------------------------------------------------------------
class UserRegisterRequest(BaseModel):
    first_name: str = Field(..., example="Alex")
    last_name: str = Field(..., example="Morgan")
    email: EmailStr = Field(..., example="alex.morgan@company.com")
    password: str = Field(..., min_length=6, example="Password123")
    organization_type: Optional[str] = "Startup"
    education_category: Optional[str] = "College / University Student"
    school_name: Optional[str] = ""
    standard: Optional[str] = ""
    university_name: Optional[str] = ""
    degree: Optional[str] = ""
    academic_year: Optional[str] = ""
    designation: Optional[str] = ""
    experience_level: Optional[str] = ""


class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., example="alex.morgan@company.com")
    password: str = Field(..., example="Password123")


class AuthResponse(BaseModel):
    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# 2. USER PROFILE SCHEMAS
# -----------------------------------------------------------------------------
class UserProfileUpdateRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization_type: Optional[str] = None
    education_category: Optional[str] = None
    school_name: Optional[str] = None
    standard: Optional[str] = None
    university_name: Optional[str] = None
    degree: Optional[str] = None
    academic_year: Optional[str] = None
    designation: Optional[str] = None
    experience_level: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    old_password: str = Field(..., example="OldPassword123")
    new_password: str = Field(..., min_length=6, example="NewPassword123")


class UserProfileResponse(BaseModel):
    id: int
    _id: str
    first_name: str
    last_name: str
    email: str
    organization_type: Optional[str] = ""
    education_category: Optional[str] = ""
    school_name: Optional[str] = ""
    standard: Optional[str] = ""
    university_name: Optional[str] = ""
    degree: Optional[str] = ""
    academic_year: Optional[str] = ""
    designation: Optional[str] = ""
    experience_level: Optional[str] = ""
    profile_image: Optional[str] = None
    created_at: Optional[str] = ""


# -----------------------------------------------------------------------------
# 3. PREDICTION SCHEMAS (EXACT 20 FRONTEND ML FEATURES)
# -----------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    user_id: str = Field(..., example="1")
    email: Optional[str] = "user@company.com"
    project_name: str = Field(..., example="Global AI Migration")
    project_type: str = Field(..., example="Software Development")
    industry_sector: str = Field(..., example="IT")
    methodology: str = Field(..., example="Agile")
    region: str = Field(..., example="Asia Pacific")
    priority: str = Field(..., example="High")
    planned_duration_days: float = Field(..., example=180.0)
    budget_usd: float = Field(..., example=250000.0)
    requirement_changes_count: float = Field(..., example=4.0)
    vendor_dependency_count: float = Field(..., example=2.0)
    milestones_missed: float = Field(..., example=1.0)
    team_size: float = Field(..., example=12.0)
    team_avg_experience_years: float = Field(..., example=5.5)
    team_turnover_pct: float = Field(..., example=10.0)
    resource_availability_pct: float = Field(..., example=85.0)
    communication_score: float = Field(..., example=75.0)
    sponsor_engagement_score: float = Field(..., example=80.0)
    tech_complexity_score: float = Field(..., example=65.0)
    scope_clarity_score: float = Field(..., example=70.0)
    external_dependency_score: float = Field(..., example=35.0)
    defect_count: float = Field(..., example=5.0)


class PredictionRecordResponse(BaseModel):
    id: int
    _id: str
    user_id: str
    email: str
    project_name: str
    risk_level: str
    risk_score: float
    model_predicted_category: Optional[str] = None
    risk_category: Optional[str] = None
    overall_risk_score: Optional[float] = None
    prediction_confidence: Optional[float] = None
    input_features: Dict[str, Any]
    analyzed_at: str


class PredictionResponse(BaseModel):
    success: bool
    message: str
    model_predicted_category: Optional[str] = None
    risk_category: Optional[str] = None
    overall_risk_score: Optional[float] = None
    prediction_confidence: Optional[float] = None
    risk_score: Optional[float] = None
    weighted_risk_score: Optional[float] = None
    class_probabilities: Optional[Dict[str, float]] = None
    prediction_record: Optional[PredictionRecordResponse] = None


# -----------------------------------------------------------------------------
# 4. DASHBOARD SCHEMAS
# -----------------------------------------------------------------------------
class DashboardMetricsResponse(BaseModel):
    total_projects: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    critical_risk_count: Optional[int] = 0
    avg_risk_score_pct: str
    avg_risk_score_num: float
    predictions: List[Dict[str, Any]]
