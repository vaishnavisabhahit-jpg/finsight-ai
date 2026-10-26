import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api import app, initialize_models

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def run_startup():
    initialize_models()

def test_health_check_returns_loaded_state():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["companies_loaded"] > 0

def test_companies_returns_synthetic_cohorts():
    response = client.get("/companies")
    assert response.status_code == 200
    companies = response.json()
    assert isinstance(companies, list)
    assert len(companies) > 0
    assert "(Synthetic)" in companies[0]["company"]

def test_forecast_horizon_and_shape_match():
    target = client.get("/companies").json()[0]["company"]
    response = client.get(f"/forecast/{target}")
    assert response.status_code == 200
    body = response.json()
    
    assert len(body["historical_dates"]) == len(body["actual_operating_profit"])
    assert len(body["actual_operating_profit"]) == len(body["predicted_operating_profit"])
    assert isinstance(body["next_quarter_forecast_usd"], float)

def test_risk_triage_returns_pr_calibrated_fields():
    target = client.get("/companies").json()[0]["company"]
    response = client.get(f"/risk/{target}")
    assert response.status_code == 200
    body = response.json()
    
    assert "distress_propensity_score" in body
    assert "operating_threshold_score" in body
    assert "validation_precision" in body
    assert "validation_recall" in body
    assert body["audit_triage_tier"] in ["Elevated Distress-Propensity Tier", "Baseline Propensity Tier"]
    assert "calibration_note" in body
    assert "score_interpretation" in body

def test_shap_additivity_identity():
    target = client.get("/companies").json()[0]["company"]
    response = client.get(f"/explain/{target}")
    assert response.status_code == 200
    body = response.json()
    
    attribution_sum = sum(item["attribution_weight"] for item in body["attributions"])
    reconstructed = attribution_sum + body["base_value"]
    
    assert np.isclose(reconstructed, body["model_raw_prediction"], atol=1e-2)