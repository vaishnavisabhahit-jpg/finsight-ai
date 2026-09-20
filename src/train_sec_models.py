import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import shap

def run_sec_pipeline(input_path='data/sec_features.csv', output_path='outputs/sec_master_analytics.csv'):
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    feature_cols = ['Lag_Revenue_1', 'Lag_OperatingProfit_1', 'Lag_OperatingMargin_1', 'Lag_DebtToCash_1', 'Revenue_QoQ_Growth']
    
    train_df = df[df['Split'] == 'Train'].copy()
    test_df = df[df['Split'] == 'Test'].copy()
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # -------------------------------------------------------------
    # 1. Multi-Target Forecasting Models
    # -------------------------------------------------------------
    # Revenue Forecasting
    reg_rev = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    reg_rev.fit(X_train, train_df['Revenue'])
    df['Pred_Revenue'] = reg_rev.predict(df[feature_cols])
    
    # Operating Profit Forecasting
    reg_profit = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    reg_profit.fit(X_train, train_df['OperatingProfit'])
    df['Pred_OperatingProfit'] = reg_profit.predict(df[feature_cols])
    
    # Evaluate Holdout MAPE vs Persistence Baseline
    test_profit_actual = test_df['OperatingProfit'].values
    test_profit_pred = reg_profit.predict(X_test)
    test_profit_baseline = test_df['Lag_OperatingProfit_1'].values
    
    # Safe MAPE calculation avoiding division by zero
    mask = np.abs(test_profit_actual) > 1e-4
    model_mape = np.mean(np.abs((test_profit_actual[mask] - test_profit_pred[mask]) / test_profit_actual[mask])) * 100
    baseline_mape = np.mean(np.abs((test_profit_actual[mask] - test_profit_baseline[mask]) / test_profit_actual[mask])) * 100
    
    print("\n================== FORECASTING BENCHMARK ==================")
    print(f"Operating Profit Holdout Baseline MAPE : {baseline_mape:.2f}%")
    print(f"Operating Profit Holdout Model MAPE    : {model_mape:.2f}%")
    if baseline_mape > 0:
        improvement = ((baseline_mape - model_mape) / baseline_mape) * 100
        print(f"Relative Error Improvement             : {improvement:.2f}%")
    print("===========================================================\n")
    
    # -------------------------------------------------------------
    # 2. Cost-Sensitive Distress Risk Ranking (<10% Prevalance)
    # -------------------------------------------------------------
    # Scale positive class weight to counter rare-event distribution
    pos_weight = (len(train_df) - train_df['IsDistressed'].sum()) / max(1, train_df['IsDistressed'].sum())
    clf = xgb.XGBClassifier(n_estimators=60, max_depth=3, scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    clf.fit(X_train, train_df['IsDistressed'])
    
    df['Predicted_RiskProbPct'] = np.round(clf.predict_proba(df[feature_cols])[:, 1] * 100, 2)
    
    # -------------------------------------------------------------
    # 3. Model Explainability via TreeSHAP (Dominant Drivers)
    # -------------------------------------------------------------
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df[feature_cols])
    
    # Identify row-level top predictive driver
    top_drivers = []
    for row in shap_values:
        idx = np.argmax(np.abs(row))
        top_drivers.append(feature_cols[idx])
    df['TopRiskDriver'] = top_drivers
    
    # -------------------------------------------------------------
    # 4. Exploratory Outlier Screening (Isolation Forest)
    # -------------------------------------------------------------
    iso = IsolationForest(contamination=0.06, random_state=42)
    iso.fit(df[feature_cols])
    df['IsAnomaly'] = np.where(iso.predict(df[feature_cols]) == -1, 1, 0)
    df['AnomalyReason'] = np.where(df['IsAnomaly'] == 1, 'Multivariate Ratio Divergence', 'Normal Profile')
    
    Path('outputs').mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Master real analytical export written to {output_path}")

if __name__ == '__main__':
    run_sec_pipeline()