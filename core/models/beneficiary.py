# core/models/beneficiary.py
"""Pydantic models for beneficiary data with validation."""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum

class RiskLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class SchemeType(str, Enum):
    PDS = "PDS"
    PAHAL = "PAHAL"
    PM_KISAN = "PM_KISAN"
    PENSION = "PENSION"
    SCHOLARSHIP = "SCHOLARSHIP"

class Beneficiary(BaseModel):
    """Core beneficiary data model."""
    
    beneficiary_id: str
    aadhaar_masked: str
    name: str
    address: str
    phone_masked: Optional[str] = None
    bank_masked: str
    ifsc_code: Optional[str] = None
    annual_income: float = Field(..., ge=0)
    occupation: Optional[str] = None
    family_size: Optional[int] = Field(None, ge=1, le=20)
    district: str
    state: str
    pincode: Optional[str] = None
    registration_date: date
    status: str = "active"
    
    # Fraud analysis fields
    fraud_score: Optional[float] = Field(None, ge=0, le=100)
    risk_level: Optional[RiskLevel] = None
    
    class Config:
        use_enum_values = True

class FraudIndicator(BaseModel):
    """Individual detection engine output."""
    engine: str  # 'rule', 'velocity', 'ml', 'anomaly', 'graph'
    score: float = Field(..., ge=0, le=100)
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    details: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, float]] = None

class FraudReport(BaseModel):
    """Complete fraud analysis for one beneficiary."""
    beneficiary_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Composite scores
    overall_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    
    # Individual engine scores
    rule_score: float = 0
    velocity_score: float = 0
    ml_score: float = 0
    anomaly_score: float = 0
    graph_score: float = 0
    
    # Explanations
    indicators: List[FraudIndicator] = []
    primary_reasons: List[str] = []
    recommended_action: str
    
    # Related entities (for graph view)
    linked_beneficiaries: Optional[List[str]] = None
    suspicious_agents: Optional[List[str]] = None
    
    class Config:
        use_enum_values = True