import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.api import app

def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_sec_features_data_integrity():
    df = pd.read_csv("data/sec_features.csv")
    expected_cols = [
        'Revenue', 'OperatingProfit', 'Cash', 'Debt', 
        'Lag_Revenue_1', 'Lag_OperatingProfit_1', 'Lag_DebtToCash_1', 
        'IsDistressed', 'Split'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing critical column: {col}"
    
    # Verify temporal split partitions exist and no nulls
    assert "Train" in df["Split"].values
    assert "Test" in df["Split"].values
    assert df.isnull().sum().sum() == 0, "Null values found in processed feature set"

def test_api_inference_endpoint():
    with TestClient(app) as client:
        payload = {
            "lag_revenue": 90000000.0,
            "lag_operating_profit": 15000000.0,
            "lag_operating_margin": 16.67,
            "lag_debt_to_cash": 1.2,
            "revenue_qoq_growth": 0.05
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        res_data = response.json()
        assert "predicted_operating_profit_usd" in res_data
        assert "distress_risk_probability_pct" in res_data
        assert 0.0 <= res_data["distress_risk_probability_pct"] <= 100.0
        assert res_data["audit_triage_tier"] in ["High Priority Review", "Normal Risk Profile"]