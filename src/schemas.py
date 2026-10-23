from pydantic import BaseModel
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str
    service: str
    companies_loaded: int

class CompanySummary(BaseModel):
    company: str
    latest_quarter: str
    latest_revenue_usd: float
    latest_operating_profit_usd: float
    latest_operating_margin_pct: float
    debt_to_cash_ratio: float

class ForecastResponse(BaseModel):
    company: str
    historical_dates: List[str]
    actual_operating_profit: List[Optional[float]]
    predicted_operating_profit: List[float]
    next_quarter_forecast_usd: float

class RiskResponse(BaseModel):
    company: str
    distress_propensity_score: float
    audit_triage_tier: str
    historical_distress_count: int
    operating_threshold_score: float
    validation_precision: float
    validation_recall: float
    calibration_note: str
    score_interpretation: str

class ShapFeatureAttribution(BaseModel):
    feature: str
    attribution_weight: float

class ExplainResponse(BaseModel):
    company: str
    dominant_driver: str
    attributions: List[ShapFeatureAttribution]
    base_value: float
    model_raw_prediction: float

class AnomalyRecord(BaseModel):
    company: str
    date: str
    revenue_usd: float
    operating_profit_usd: float
    reason: str

class AnomaliesResponse(BaseModel):
    total_anomalies_flagged: int
    anomalies: List[AnomalyRecord]