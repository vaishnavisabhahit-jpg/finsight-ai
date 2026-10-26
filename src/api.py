import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure models are loaded
MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "featured_financials.csv")

try:
  forecast_model = joblib.load(
      os.path.join(MODELS_DIR, "xgb_forecast_model.joblib")
  )
  forecast_features = joblib.load(
      os.path.join(MODELS_DIR, "forecast_features.joblib")
  )
except Exception as e:
  forecast_model = None
  forecast_features = None


@app.get("/surface/{company}")
def get_model_driven_surface(company: str):
  """Generates a dynamic 3D sensitivity surface driven by the trained XGBoost model.

  Axes: Timeline Horizons (Q+1 to Q+5) x Macro Revenue/Margin Shocks (-30% to
  +30%)
  """
  if forecast_model is None or not os.path.exists(DATA_PATH):
    raise HTTPException(
        status_code=500, detail="Forecasting model or dataset not loaded."
    )

  df = pd.read_csv(DATA_PATH)
  company_records = df[df["BusinessUnit"] == company].sort_values("Date")

  if company_records.empty:
    # Fallback to NovaTech Retail or first entity if synthetic matching
    company_records = df[
        df["BusinessUnit"] == "NovaTech Retail (Synthetic)"
    ].sort_values("Date")
    if company_records.empty:
      company_records = df.sort_values("Date")

  latest_vector = company_records.iloc[-1][forecast_features].copy()

  quarters = ["Q+1 Horizon", "Q+2 Horizon", "Q+3 Horizon", "Q+4 Horizon", "Q+5 Horizon"]
  scenarios = [
      "Severe Stress (-30%)",
      "Moderate Downside (-15%)",
      "Baseline Trajectory (0%)",
      "Moderate Upside (+15%)",
      "Strong Expansion (+30%)",
  ]
  shock_multipliers = [-0.30, -0.15, 0.0, 0.15, 0.30]

  z_matrix = []

  for shock in shock_multipliers:
    row_predictions = []
    shocked_vector = latest_vector.copy()

    # Apply macro stress shock to revenue, margins, and momentum
    if "GrossMarginPct" in shocked_vector:
      shocked_vector["GrossMarginPct"] += shock * 15.0
    if "OperatingMarginPct" in shocked_vector:
      shocked_vector["OperatingMarginPct"] += shock * 12.0
    if "Revenue_Lag1" in shocked_vector:
      shocked_vector["Revenue_Lag1"] *= 1.0 + shock
    if "Rolling3M_AvgRev" in shocked_vector:
      shocked_vector["Rolling3M_AvgRev"] *= 1.0 + shock

    for step in range(len(quarters)):
      # Compound time-decay or horizon drift
      step_vector = shocked_vector.copy()
      if "OperatingMargin_Lag1" in step_vector:
        step_vector["OperatingMargin_Lag1"] += (shock * 2.0) * (step + 1)

      input_array = pd.DataFrame([step_vector])[forecast_features]
      pred_val = float(forecast_model.predict(input_array)[0])

      # Scale to thousands ($k) for UI presentation
      row_predictions.append(round(pred_val / 1000.0, 2))

    z_matrix.append(row_predictions)

  return {"x": quarters, "y": scenarios, "z": z_matrix}