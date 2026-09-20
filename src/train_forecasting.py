import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

def evaluate_forecast(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}

def run_forecasting_pipeline():
    input_path = os.path.join("data", "featured_financials.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}. Run src/feature_engineering.py first.")
        
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "BusinessUnit"]).reset_index(drop=True)
    
    # Candidate features available at prediction time (t-1 & past ratios only)
    feature_cols = [
        "GrossMarginPct", "OperatingMarginPct", "ReceivablesToRevenueRatio",
        "CashBufferRatio", "DebtToCashRatio", "Revenue_Lag1", "Revenue_Lag2",
        "Expenses_Lag1", "OperatingProfit_Lag1", "OperatingMargin_Lag1",
        "RevenueGrowthMoM", "ExpenseGrowthMoM", "OperatingProfitGrowthMoM",
        "Rolling3M_AvgRev", "Rolling3M_AvgExp"
    ]
    
    targets = ["Revenue", "TotalExpenses", "OperatingProfit"]
    
    # Enforce Time-Aware Split (Last 12 months per Business Unit reserved for Test)
    split_date = df["Date"].max() - pd.DateOffset(months=12)
    train_df = df[df["Date"] <= split_date].copy()
    test_df = df[df["Date"] > split_date].copy()
    
    print(f"Training split: {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(train_df)} rows)")
    print(f"Testing split:  {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(test_df)} rows)\n")
    
    results = []
    forecast_output = test_df[["Date", "BusinessUnit"]].copy()
    
    for target in targets:
        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test = test_df[feature_cols]
        y_test = test_df[target]
        
        # --- 1. Baseline Model: Lag 1 Persistence (Naïve historical forecast) ---
        lag_col = "Revenue_Lag1" if target == "Revenue" else ("Expenses_Lag1" if target == "TotalExpenses" else "OperatingProfit_Lag1")
        baseline_pred = test_df[lag_col]
        base_metrics = evaluate_forecast(y_test, baseline_pred, f"Baseline (Lag-1) [{target}]")
        results.append(base_metrics)
        
        # --- 2. Advanced Candidate: XGBoost Regressor ---
        xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_metrics = evaluate_forecast(y_test, xgb_pred, f"XGBoost [{target}]")
        results.append(xgb_metrics)
        
        # Save predictions and actuals to export dataframe
        forecast_output[f"Actual_{target}"] = y_test.values
        forecast_output[f"Forecast_{target}"] = np.round(xgb_pred, 2)
        forecast_output[f"Baseline_{target}"] = baseline_pred.values
        forecast_output[f"Variance_{target}"] = np.round(xgb_pred - y_test.values, 2)
        forecast_output[f"VariancePct_{target}"] = np.round(((xgb_pred - y_test.values) / y_test.values) * 100, 2)
    
    # Display summary metrics
    metrics_df = pd.DataFrame(results)
    print("=== Forecasting Model Evaluation Summary ===")
    print(metrics_df.to_string(index=False))
    print("\n" + "=" * 44)
    
    # Export for Power BI
    os.makedirs("outputs", exist_ok=True)
    export_path = os.path.join("outputs", "forecast_predictions.csv")
    forecast_output.to_csv(export_path, index=False)
    print(f"Predictions successfully exported to: {export_path}")

if __name__ == "__main__":
    run_forecasting_pipeline()