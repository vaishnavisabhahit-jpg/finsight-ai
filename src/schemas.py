from pydantic import BaseModel, Field
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
    distress_risk_probability_pct: float
    audit_triage_tier: str
    historical_distress_count: int

class ShapFeatureAttribution(BaseModel):
    feature: str
    attribution_weight: float

class ExplainResponse(BaseModel):
    company: str
    dominant_driver: str
    attributions: List[ShapFeatureAttribution]

class AnomalyRecord(BaseModel):
    company: str
    date: str
    revenue_usd: float
    operating_profit_usd: float
    reason: str

class AnomaliesResponse(BaseModel):
    total_anomalies_flagged: int
    anomalies: List[AnomalyRecord]