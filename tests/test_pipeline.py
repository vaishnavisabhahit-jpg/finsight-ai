import os
import pandas as pd
from fastapi.testclient import TestClient
from src.api import app, initialize_models

def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["companies_loaded"] > 0

def test_sec_features_data_integrity():
    csv_path = "data/sec_features.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist"
    df = pd.read_csv(csv_path)
    required_cols = [
        "Company", "Date", "Revenue", "OperatingProfit",
        "Lag_Revenue_1", "Lag_OperatingProfit_1", "Lag_OperatingMargin_1",
        "Lag_DebtToCash_1", "Revenue_QoQ_Growth", "IsDistressed"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    assert len(df) > 0

def test_api_inference_endpoint():
    # Production inference route is GET /forecast/{company}
    with TestClient(app) as client:
        # Retrieve active companies
        comp_res = client.get("/companies")
        assert comp_res.status_code == 200
        first_company = comp_res.json()[0]["company"]

        # Run model inference on first cohort
        response = client.get(f"/forecast/{first_company}")
        assert response.status_code == 200
        data = response.json()
        assert data["company"] == first_company
        assert "next_quarter_forecast_usd" in data
        assert isinstance(data["next_quarter_forecast_usd"], float)