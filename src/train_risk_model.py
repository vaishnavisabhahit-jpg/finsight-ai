import os
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from xgboost import XGBClassifier

def run_risk_pipeline():
    input_path = os.path.join("data", "featured_financials.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}. Run src/feature_engineering.py first.")
        
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "BusinessUnit"]).reset_index(drop=True)
    
    feature_cols = [
        "GrossMarginPct", "OperatingMarginPct", "ReceivablesToRevenueRatio",
        "CashBufferRatio", "DebtToCashRatio", "RevenueGrowthMoM", 
        "ExpenseGrowthMoM", "OperatingProfitGrowthMoM"
    ]
    target_col = "Target_NextPeriodHighRisk"
    
    # Enforce Time-Aware Split (Last 12 months reserved for testing)
    split_date = df["Date"].max() - pd.DateOffset(months=12)
    train_df = df[df["Date"] <= split_date].copy()
    test_df = df[df["Date"] > split_date].copy()
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    # --- 1. Baseline Classifier: Logistic Regression ---
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_base = log_reg.predict(X_test)
    y_prob_base = log_reg.predict_proba(X_test)[:, 1]
    
    print("=== Baseline Model (Logistic Regression) ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_base):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_base):.3f}\n")
    
    # --- 2. Advanced Classifier: XGBoost ---
    xgb_clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42
    )
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    
    print("=== Champion Model (XGBoost Classifier) ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.3f}")
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, y_pred_xgb))
    
    # --- 3. SHAP Explainability Engine ---
    print("Computing TreeSHAP values for model explainability...")
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_test)
    
    # Calculate mean absolute SHAP value per feature
    global_importance = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_SHAP_Impact": np.round(global_importance, 4)
    }).sort_values("Mean_SHAP_Impact", ascending=False)
    
    print("\n=== Global Risk Drivers (SHAP Top Features) ===")
    print(feature_importance_df.to_string(index=False))
    
    # --- 4. Export Table for Power BI Risk Intelligence Page ---
    risk_output = test_df[["Date", "BusinessUnit"]].copy()
    risk_output["Actual_NextPeriodRisk"] = y_test.values
    risk_output["Predicted_RiskProbPct"] = np.round(y_prob_xgb * 100, 1)
    risk_output["Predicted_RiskCategory"] = np.where(
        y_prob_xgb >= 0.70, "High Risk",
        np.where(y_prob_xgb >= 0.40, "Moderate Risk", "Low Risk")
    )
    
    # Extract top individual risk contributor per observation using SHAP
    top_drivers = []
    for i in range(len(test_df)):
        obs_shap = shap_values[i]
        top_feature_idx = np.argmax(np.abs(obs_shap))
        top_drivers.append(feature_cols[top_feature_idx])
        
    risk_output["TopRiskDriver"] = top_drivers
    
    os.makedirs("outputs", exist_ok=True)
    export_path = os.path.join("outputs", "risk_predictions.csv")
    risk_output.to_csv(export_path, index=False)
    print(f"\nRisk intelligence table exported to: {export_path}")

if __name__ == "__main__":
    run_risk_pipeline()