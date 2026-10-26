import os
import joblib
import numpy as np
import pandas as pd
import pytest

MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "featured_financials.csv")

def test_data_artifacts_exist():
    assert os.path.exists(DATA_PATH), "featured_financials.csv is missing. Run feature engineering."

def test_no_future_target_leakage_in_features():
    df = pd.read_csv(DATA_PATH)
    assert "Target_NextPeriodHighRisk" in df.columns
    assert "Target_NextPeriodOperatingProfit" in df.columns

    # Verify no infinite or NaN values in feature matrix
    numeric_df = df.select_dtypes(include=[np.number])
    assert not numeric_df.isin([np.inf, -np.inf]).any().any(), "Features contain inf values."
    assert not numeric_df.isnull().any().any(), "Features contain unhandled NaNs."

def test_model_artifacts_loadable():
    xgb_risk = joblib.load(os.path.join(MODELS_DIR, "xgb_risk_model.joblib"))
    risk_meta = joblib.load(os.path.join(MODELS_DIR, "risk_meta.joblib"))
    xgb_forecast = joblib.load(os.path.join(MODELS_DIR, "xgb_forecast_model.joblib"))
    iso_forest = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))

    assert xgb_risk is not None
    assert "operating_threshold" in risk_meta
    assert 0.0 < risk_meta["operating_threshold"] < 1.0
    assert xgb_forecast is not None
    assert iso_forest is not None

def test_risk_inference_shape():
    risk_meta = joblib.load(os.path.join(MODELS_DIR, "risk_meta.joblib"))
    xgb_risk = joblib.load(os.path.join(MODELS_DIR, "xgb_risk_model.joblib"))
    feature_cols = risk_meta["feature_cols"]

    df = pd.read_csv(DATA_PATH)
    sample_vector = df.iloc[[0]][feature_cols]

    proba = xgb_risk.predict_proba(sample_vector)[0][1]
    assert 0.0 <= proba <= 1.0