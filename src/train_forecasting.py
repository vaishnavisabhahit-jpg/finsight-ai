import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = os.path.join("data", "featured_financials.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "GrossMarginPct",
    "OperatingMarginPct",
    "ReceivablesToRevenueRatio",
    "CashBufferRatio",
    "DebtToCashRatio",
    "BudgetVarianceRevenue",
    "BudgetVarianceExpenses",
    "Revenue_Lag1",
    "Revenue_Lag2",
    "Expenses_Lag1",
    "OperatingProfit_Lag1",
    "OperatingMargin_Lag1",
    "RevenueGrowthMoM",
    "ExpenseGrowthMoM",
    "OperatingProfitGrowthMoM",
    "Rolling3M_AvgRev",
    "Rolling3M_AvgExp",
    "Rolling3M_AvgMargin",
]
TARGET_COL = "Target_NextPeriodOperatingProfit"


def train_and_benchmark_forecaster():
  if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Missing {DATA_PATH}. Run src/feature_engineering.py first."
    )

  df = pd.read_csv(DATA_PATH)
  df["Date"] = pd.to_datetime(df["Date"])
  df = df.sort_values("Date").reset_index(drop=True)

  # Strict chronological split: 70% Train, 15% Validation, 15% Test
  n = len(df)
  train_end = int(n * 0.70)
  val_end = int(n * 0.85)

  train_df = df.iloc[:train_end]
  val_df = df.iloc[train_end:val_end]
  test_df = df.iloc[val_end:]

  X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
  X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]
  X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

  # 1. Baseline 1: Naive Persistence (Last Quarter's Operating Profit)
  # Uses the historical lag that was already known at time t
  naive_preds_test = test_df["OperatingProfit_Lag1"].values

  # 2. Baseline 2: Ridge Linear Regression
  ridge = Ridge(alpha=1.0)
  ridge.fit(X_train, y_train)
  ridge_preds_test = ridge.predict(X_test)

  # 3. Candidate: XGBoost Regressor
  xgb_reg = XGBRegressor(
      n_estimators=120,
      max_depth=3,
      learning_rate=0.04,
      subsample=0.85,
      colsample_bytree=0.85,
      random_state=42,
  )
  xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
  xgb_preds_test = xgb_reg.predict(X_test)

  def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R² Score": round(r2, 4),
    }

  naive_metrics = calc_metrics(y_test, naive_preds_test)
  ridge_metrics = calc_metrics(y_test, ridge_preds_test)
  xgb_metrics = calc_metrics(y_test, xgb_preds_test)

  # Save trained forecaster model
  joblib.dump(xgb_reg, os.path.join(MODELS_DIR, "xgb_forecast_model.joblib"))
  joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "forecast_features.joblib"))

  print("=" * 65)
  print("OUT-OF-TIME FINANCIAL FORECASTING BENCHMARK (HELD-OUT TEST SET):")
  results_df = pd.DataFrame([
      {"Model": "Naive Baseline (Lag-1 Persistence)", **naive_metrics},
      {"Model": "Ridge Regression Baseline", **ridge_metrics},
      {"Model": "XGBoost Regressor (Tuned)", **xgb_metrics},
  ])
  print(results_df.to_string(index=False))
  print("=" * 65)
  print(f"Model saved to {MODELS_DIR}/xgb_forecast_model.joblib")


if __name__ == "__main__":
  train_and_benchmark_forecaster()