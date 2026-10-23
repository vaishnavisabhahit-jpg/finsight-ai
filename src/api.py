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
from sklearn.metrics import precision_recall_curve

from src.config import settings
from src.schemas import (
    HealthResponse, CompanySummary, ForecastResponse,
    RiskResponse, ExplainResponse, ShapFeatureAttribution,
    AnomaliesResponse, AnomalyRecord
)

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("finsight-backend")

ml_engine: Dict[str, Any] = {}

FEATURE_COLS = [
    'Lag_Revenue_1', 'Lag_OperatingProfit_1', 'Lag_OperatingMargin_1',
    'Lag_DebtToCash_1', 'Revenue_QoQ_Growth'
]

def resolve_data_path() -> Path:
    # Check configured path first
    target = Path(settings.DATA_PATH)
    if target.exists():
        return target
    
    # Fallback to local root searches
    fallbacks = [
        Path("data/synthetic_fpa_features.csv"),
        Path("data/sec_features.csv"),
        Path(__file__).resolve().parent.parent / "data" / "synthetic_fpa_features.csv",
        Path(__file__).resolve().parent.parent / "data" / "sec_features.csv"
    ]
    for p in fallbacks:
        if p.exists():
            return p

    raise RuntimeError(f"Missing data source. Tried {settings.DATA_PATH} and standard fallback paths.")

def initialize_models():
    data_path = resolve_data_path()
    logger.info("Initializing ML models from %s...", data_path)

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    ml_engine['data'] = df

    # 1. XGBoost Forecasting Regressor
    reg = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    reg.fit(df[FEATURE_COLS], df['OperatingProfit'])
    ml_engine['regressor'] = reg

    # 2. Strict Out-of-Time Temporal Split for Distress Model & Calibration
    sorted_dates = sorted(df['Date'].unique())
    split_idx = int(len(sorted_dates) * 0.75)
    split_date = sorted_dates[split_idx]
    
    train_df = df[df['Date'] < split_date].copy()
    val_df = df[df['Date'] >= split_date].copy()
    ml_engine['split_date'] = split_date

    pos_weight = (len(train_df) - train_df['IsDistressed'].sum()) / max(1, train_df['IsDistressed'].sum())
    clf = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=3,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    clf.fit(train_df[FEATURE_COLS], train_df['IsDistressed'])
    ml_engine['classifier'] = clf

    # Precision-Recall Optimization on Validation Fold
    val_scores = clf.predict_proba(val_df[FEATURE_COLS])[:, 1]
    y_val = val_df['IsDistressed'].values
    
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores)
    f1_scores = np.divide(
        2 * (precisions[:-1] * recalls[:-1]),
        (precisions[:-1] + recalls[:-1]),
        out=np.zeros_like(thresholds),
        where=(precisions[:-1] + recalls[:-1]) > 0
    )
    best_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    val_prec = float(precisions[best_idx]) if len(precisions) > 0 else 0.0
    val_rec = float(recalls[best_idx]) if len(recalls) > 0 else 0.0

    ml_engine['threshold'] = best_threshold
    ml_engine['val_precision'] = val_prec
    ml_engine['val_recall'] = val_rec

    # 3. TreeSHAP Attribution Engine
    ml_engine['explainer'] = shap.TreeExplainer(clf)

    # 4. Outlier Isolation Forest
    iso = IsolationForest(contamination=0.06, random_state=42)
    iso.fit(df[FEATURE_COLS])
    ml_engine['iso_forest'] = iso
    logger.info("Engine initialized. Data: %s | Threshold: %.4f | Val Precision: %.2f | Val Recall: %.2f",
                data_path, best_threshold, val_prec, val_rec)

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_models()
    yield
    ml_engine.clear()
    logger.info("ML Engine shut down.")

app = FastAPI(
    title="FinSight AI Backend",
    description="FP&A Decision Support Engine with out-of-time XGBoost forecasts, TreeSHAP attributions, and PR-calibrated triage.",
    version="2.2.0",
    lifespan=lifespan
)

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
    raw_score = float(ml_engine['classifier'].predict_proba(input_vector)[0][1])
    threshold = ml_engine['threshold']
    
    tier = "Elevated Distress-Propensity Tier" if raw_score >= threshold else "Baseline Propensity Tier"

    return {
        "company": comp_df.iloc[0]['Company'],
        "distress_propensity_score": round(raw_score, 4),
        "audit_triage_tier": tier,
        "historical_distress_count": int(comp_df['IsDistressed'].sum()),
        "operating_threshold_score": round(threshold, 4),
        "validation_precision": round(ml_engine['val_precision'], 2),
        "validation_recall": round(ml_engine['val_recall'], 2),
        "calibration_note": (
            f"Threshold ({round(threshold, 4)}) calibrated on out-of-time validation fold to optimize F1 score. "
            f"Validation performance: Precision={round(ml_engine['val_precision'], 2)}, Recall={round(ml_engine['val_recall'], 2)}."
        ),
        "score_interpretation": "Class-weighted propensity score from cost-sensitive XGBoost, not an absolute default probability."
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
    
    explainer = ml_engine['explainer']
    shap_vals = explainer.shap_values(input_vector)[0]
    base_val = float(explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0])
    margin_pred = float(ml_engine['classifier'].predict(xgb.DMatrix(input_vector), output_margin=True)[0])
    
    attributions = [
        ShapFeatureAttribution(feature=FEATURE_COLS[i], attribution_weight=round(float(shap_vals[i]), 4))
        for i in range(len(FEATURE_COLS))
    ]
    dominant = FEATURE_COLS[int(np.argmax(np.abs(shap_vals)))]

    return {
        "company": comp_df.iloc[0]['Company'],
        "dominant_driver": dominant,
        "attributions": attributions,
        "base_value": round(base_val, 4),
        "model_raw_prediction": round(margin_pred, 4)
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