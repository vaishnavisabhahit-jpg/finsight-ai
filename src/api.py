import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest

from src.config import settings
from src.schemas import (
    HealthResponse, CompanySummary, ForecastResponse,
    RiskResponse, ExplainResponse, ShapFeatureAttribution,
    AnomaliesResponse, AnomalyRecord
)

# Logging Setup
logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("finsight-backend")

# In-Memory Cache (loaded strictly once at startup)
ml_engine: Dict[str, Any] = {}

FEATURE_COLS = [
    'Lag_Revenue_1', 'Lag_OperatingProfit_1', 'Lag_OperatingMargin_1',
    'Lag_DebtToCash_1', 'Revenue_QoQ_Growth'
]

def initialize_models():
    logger.info("Initializing ML models and analytical cache from %s...", settings.DATA_PATH)
    path = Path(settings.DATA_PATH)
    if not path.exists():
        raise RuntimeError(f"Missing data source: {settings.DATA_PATH}")

    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    ml_engine['data'] = df

    # 1. XGBoost Forecasting Regressor
    reg = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    reg.fit(df[FEATURE_COLS], df['OperatingProfit'])
    ml_engine['regressor'] = reg

    # 2. XGBoost Cost-Sensitive Distress Classifier
    pos_weight = (len(df) - df['IsDistressed'].sum()) / max(1, df['IsDistressed'].sum())
    clf = xgb.XGBClassifier(n_estimators=60, max_depth=3, scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    clf.fit(df[FEATURE_COLS], df['IsDistressed'])
    ml_engine['classifier'] = clf

    # 3. TreeSHAP Attribution Engine
    ml_engine['explainer'] = shap.TreeExplainer(clf)

    # 4. Exploratory Outlier Isolation Forest
    iso = IsolationForest(contamination=0.06, random_state=42)
    iso.fit(df[FEATURE_COLS])
    ml_engine['iso_forest'] = iso
    logger.info("ML Engine successfully loaded into memory.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_models()
    yield
    ml_engine.clear()
    logger.info("ML Engine shut down.")

app = FastAPI(
    title="FinSight AI Backend",
    description="FP&A Decision Support Engine with out-of-time XGBoost forecasts, TreeSHAP, and risk triage.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_company_df(company: str) -> pd.DataFrame:
    df: pd.DataFrame = ml_engine['data']
    comp_df = df[df['Company'].str.lower() == company.lower()].sort_values('Date').reset_index(drop=True)
    if comp_df.empty:
        raise HTTPException(status_code=404, detail=f"Company '{company}' not found.")
    return comp_df

# ==========================================
# 6 PRODUCTION ENDPOINTS
# ==========================================

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy",
        "service": "FinSight AI Inference Engine",
        "companies_loaded": int(ml_engine['data']['Company'].nunique())
    }

@app.get("/companies", response_model=List[CompanySummary])
def get_companies():
    df: pd.DataFrame = ml_engine['data']
    summaries = []
    for company, group in df.groupby('Company'):
        latest = group.sort_values('Date').iloc[-1]
        summaries.append({
            "company": str(company),
            "latest_quarter": str(latest['Date']),
            "latest_revenue_usd": float(latest['Revenue']),
            "latest_operating_profit_usd": float(latest['OperatingProfit']),
            "latest_operating_margin_pct": round(float(latest['OperatingMarginPct']), 2),
            "debt_to_cash_ratio": round(float(latest['DebtToCashRatio']), 2)
        })
    return summaries

@app.get("/forecast/{company}", response_model=ForecastResponse)
def get_forecast(company: str):
    comp_df = get_company_df(company)
    preds = ml_engine['regressor'].predict(comp_df[FEATURE_COLS])
    
    # Predict next quarter using latest observed quarter as lag-1
    latest = comp_df.iloc[-1]
    next_input = np.array([[
        latest['Revenue'],
        latest['OperatingProfit'],
        latest['OperatingMarginPct'],
        latest['DebtToCashRatio'],
        latest['Revenue_QoQ_Growth']
    ]])
    next_quarter_pred = float(ml_engine['regressor'].predict(next_input)[0])

    return {
        "company": comp_df.iloc[0]['Company'],
        "historical_dates": comp_df['Date'].tolist(),
        "actual_operating_profit": [float(x) for x in comp_df['OperatingProfit'].tolist()],
        "predicted_operating_profit": [round(float(x), 2) for x in preds],
        "next_quarter_forecast_usd": round(next_quarter_pred, 2)
    }

@app.get("/risk/{company}", response_model=RiskResponse)
def get_risk(company: str):
    comp_df = get_company_df(company)
    latest = comp_df.iloc[-1]
    input_vector = np.array([[
        latest['Lag_Revenue_1'],
        latest['Lag_OperatingProfit_1'],
        latest['Lag_OperatingMargin_1'],
        latest['Lag_DebtToCash_1'],
        latest['Revenue_QoQ_Growth']
    ]])
    prob = float(ml_engine['classifier'].predict_proba(input_vector)[0][1]) * 100.0
    tier = "High Priority Review" if prob >= 50.0 else "Normal Risk Profile"

    return {
        "company": comp_df.iloc[0]['Company'],
        "distress_risk_probability_pct": round(prob, 2),
        "audit_triage_tier": tier,
        "historical_distress_count": int(comp_df['IsDistressed'].sum())
    }

@app.get("/explain/{company}", response_model=ExplainResponse)
def explain_risk(company: str):
    comp_df = get_company_df(company)
    latest = comp_df.iloc[-1]
    input_vector = np.array([[
        latest['Lag_Revenue_1'],
        latest['Lag_OperatingProfit_1'],
        latest['Lag_OperatingMargin_1'],
        latest['Lag_DebtToCash_1'],
        latest['Revenue_QoQ_Growth']
    ]])
    shap_vals = ml_engine['explainer'].shap_values(input_vector)[0]
    
    attributions = [
        ShapFeatureAttribution(feature=FEATURE_COLS[i], attribution_weight=round(float(shap_vals[i]), 4))
        for i in range(len(FEATURE_COLS))
    ]
    dominant = FEATURE_COLS[int(np.argmax(np.abs(shap_vals)))]

    return {
        "company": comp_df.iloc[0]['Company'],
        "dominant_driver": dominant,
        "attributions": attributions
    }

@app.get("/anomalies", response_model=AnomaliesResponse)
def get_anomalies():
    df: pd.DataFrame = ml_engine['data']
    preds = ml_engine['iso_forest'].predict(df[FEATURE_COLS])
    anomaly_rows = df[preds == -1]

    records = []
    for _, row in anomaly_rows.iterrows():
        records.append({
            "company": str(row['Company']),
            "date": str(row['Date']),
            "revenue_usd": float(row['Revenue']),
            "operating_profit_usd": float(row['OperatingProfit']),
            "reason": "Multivariate Financial Divergence"
        })

    return {
        "total_anomalies_flagged": len(records),
        "anomalies": records
    }