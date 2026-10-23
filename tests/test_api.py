import pytest
from fastapi.testclient import TestClient
from src.api import app, ml_engine, initialize_models

# Initialize TestClient
client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def run_startup():
    # Explicitly ensure in-memory models and thresholds initialize for testing
    initialize_models()

def test_health_check():
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
    # Verify synthetic naming convention exists
    first_name = companies[0]["company"]
    assert "(Synthetic)" in first_name

def test_fundamentals_retrieval():
    response = client.get("/companies")
    assert response.status_code == 200
    target_company = response.json()[0]["company"]

    forecast_res = client.get(f"/forecast/{target_company}")
    assert forecast_res.status_code == 200
    body = forecast_res.json()
    assert body["company"] == target_company
    assert len(body["historical_dates"]) > 0

def test_forecast_returns_expected_series_shape():
    response = client.get("/companies")
    target_company = response.json()[0]["company"]

    response = client.get(f"/forecast/{target_company}")
    assert response.status_code == 200
    body = response.json()
    assert "actual_operating_profit" in body
    assert "predicted_operating_profit" in body
    assert "next_quarter_forecast_usd" in body
    assert len(body["actual_operating_profit"]) == len(body["predicted_operating_profit"])

def test_shap_attribution_schema():
    response = client.get("/companies")
    target_company = response.json()[0]["company"]

    response = client.get(f"/explain/{target_company}")
    assert response.status_code == 200
    body = response.json()
    assert "dominant_driver" in body
    assert "attributions" in body
    assert len(body["attributions"]) == 5

def test_risk_triage_returns_calibrated_percentile_fields():
    response = client.get("/companies")
    target_company = response.json()[0]["company"]

    response = client.get(f"/risk/{target_company}")
    assert response.status_code == 200
    body = response.json()
    
    # Assert calibration honesty fields
    assert "distress_risk_probability_pct" in body
    assert "p75_baseline_threshold_pct" in body
    assert "audit_triage_tier" in body
    assert body["audit_triage_tier"] in ["Elevated Anomaly Tier", "Baseline Distribution Tier"]
    assert "calibration_note" in body
    assert "score_interpretation" in body