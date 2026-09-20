import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def run_anomaly_and_consolidation():
    data_path = os.path.join("data", "featured_financials.csv")
    forecast_path = os.path.join("outputs", "forecast_predictions.csv")
    risk_path = os.path.join("outputs", "risk_predictions.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run previous steps first.")
        
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "BusinessUnit"]).reset_index(drop=True)
    
    # --- 1. Module C: Financial Anomaly Detection (Isolation Forest) ---
    anomaly_features = [
        "RevenueGrowthMoM", "ExpenseGrowthMoM", "OperatingMarginPct",
        "ReceivablesToRevenueRatio", "CashBufferRatio"
    ]
    
    # Set expected contamination to 5% of monthly observations
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df["AnomalyRaw"] = iso_forest.fit_predict(df[anomaly_features])
    
    # -1 indicates anomaly in scikit-learn, 1 is normal
    df["IsAnomaly"] = np.where(df["AnomalyRaw"] == -1, 1, 0)
    # Decision function: lower values mean more anomalous
    df["AnomalyScore"] = np.round(-iso_forest.decision_function(df[anomaly_features]), 4)
    
    # Determine the primary metric driver for each anomaly
    anomaly_reasons = []
    for _, row in df.iterrows():
        if row["IsAnomaly"] == 1:
            if row["ExpenseGrowthMoM"] > 25:
                anomaly_reasons.append("Unusual Expense Spike")
            elif row["ReceivablesToRevenueRatio"] > 1.1:
                anomaly_reasons.append("Abnormal Receivables Surge")
            elif row["RevenueGrowthMoM"] < -15:
                anomaly_reasons.append("Severe Revenue Drop")
            elif row["CashBufferRatio"] < 1.0:
                anomaly_reasons.append("Critically Low Cash Buffer")
            else:
                anomaly_reasons.append("Multivariate Outlier")
        else:
            anomaly_reasons.append("Normal")
            
    df["AnomalyReason"] = anomaly_reasons
    
    print("=== Financial Anomaly Detection Summary ===")
    print(f"Total Anomalies Flagged: {df['IsAnomaly'].sum()} out of {len(df)} records")
    print(df[df["IsAnomaly"] == 1][["Date", "BusinessUnit", "AnomalyReason", "AnomalyScore"]])
    
    # --- 2. Master Consolidation for Power BI Decision Layer ---
    # Merge actuals, forecast predictions, risk metrics, and anomaly flags
    forecast_df = pd.read_csv(forecast_path)
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    
    risk_df = pd.read_csv(risk_path)
    risk_df["Date"] = pd.to_datetime(risk_df["Date"])
    
    # Left join forecast and risk onto the master featured dataset
    master = pd.merge(df, forecast_df, on=["Date", "BusinessUnit"], how="left")
    master = pd.merge(master, risk_df[["Date", "BusinessUnit", "Predicted_RiskProbPct", "Predicted_RiskCategory", "TopRiskDriver"]], on=["Date", "BusinessUnit"], how="left")
    
    # Fill non-test historical rows for Power BI visual continuity
    master["Predicted_RiskCategory"] = master["Predicted_RiskCategory"].fillna("Historical Record")
    master["TopRiskDriver"] = master["TopRiskDriver"].fillna("N/A")
    
    # Format Date cleanly for BI ingestion
    master["Date"] = master["Date"].dt.strftime("%Y-%m-%d")
    
    output_master = os.path.join("outputs", "finsight_master_analytics.csv")
    master.to_csv(output_master, index=False)
    print(f"\nConsolidated master analytics dataset created at: {output_master}")
    print(f"Total columns ready for Power BI: {len(master.columns)}")

if __name__ == "__main__":
    run_anomaly_and_consolidation()