import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_PATH = os.path.join("data", "featured_financials.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

ANOMALY_FEATURE_COLS = [
    "GrossMarginPct",
    "OperatingMarginPct",
    "ReceivablesToRevenueRatio",
    "CashBufferRatio",
    "DebtToCashRatio",
    "RevenueGrowthMoM",
    "ExpenseGrowthMoM",
    "OperatingProfitGrowthMoM",
]

def inject_synthetic_accounting_anomalies(df: pd.DataFrame, injection_rate: float = 0.05, seed: int = 42):
    """
    Creates an evaluation dataset with known, injected financial statement distortions:
    1. Receivables Inflation (Revenue-Receivables Decoupling)
    2. Severe Margin Collapse with Expense Inflation
    3. Extreme Debt-to-Cash Divergence
    """
    np.random.seed(seed)
    eval_df = df.copy()
    n_injections = int(len(eval_df) * injection_rate)
    injection_indices = np.random.choice(eval_df.index, size=n_injections, replace=False)
    
    eval_df["is_injected_anomaly"] = 0
    
    for idx in injection_indices:
        anomaly_type = np.random.choice(["channel_stuffing", "margin_collapse", "liquidity_drain"])
        if anomaly_type == "channel_stuffing":
            eval_df.loc[idx, "ReceivablesToRevenueRatio"] *= np.random.uniform(3.5, 6.0)
            eval_df.loc[idx, "RevenueGrowthMoM"] -= np.random.uniform(25.0, 50.0)
        elif anomaly_type == "margin_collapse":
            eval_df.loc[idx, "GrossMarginPct"] -= np.random.uniform(30.0, 60.0)
            eval_df.loc[idx, "ExpenseGrowthMoM"] += np.random.uniform(40.0, 80.0)
        elif anomaly_type == "liquidity_drain":
            eval_df.loc[idx, "CashBufferRatio"] *= np.random.uniform(0.05, 0.20)
            eval_df.loc[idx, "DebtToCashRatio"] *= np.random.uniform(4.0, 8.0)
            
        eval_df.loc[idx, "is_injected_anomaly"] = 1
        
    return eval_df

def train_and_validate_anomaly_detector():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run src/feature_engineering.py first.")
        
    df = pd.read_csv(DATA_PATH)
    
    # 1. Train Unsupervised Isolation Forest on Clean/Historical Data
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.06,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(df[ANOMALY_FEATURE_COLS])
    
    # 2. Validation via Injection Test
    eval_df = inject_synthetic_accounting_anomalies(df, injection_rate=0.06, seed=42)
    
    # Isolation Forest scores: lower/negative = more anomalous. We invert so higher = more anomalous.
    raw_scores = iso_forest.score_samples(eval_df[ANOMALY_FEATURE_COLS])
    anomaly_scores = -raw_scores
    
    y_true = eval_df["is_injected_anomaly"].values
    
    # 3. Compute Recall@K (e.g. Top 6%, 10%, 15% scored records)
    def compute_recall_at_k(scores, ground_truth, k_pct):
        top_k_threshold = np.percentile(scores, 100 - (k_pct * 100))
        predicted_anomalies = (scores >= top_k_threshold).astype(int)
        tp = np.sum((predicted_anomalies == 1) & (ground_truth == 1))
        total_anomalies = np.sum(ground_truth == 1)
        return (tp / total_anomalies) if total_anomalies > 0 else 0.0

    recall_at_6 = compute_recall_at_k(anomaly_scores, y_true, 0.06)
    recall_at_10 = compute_recall_at_k(anomaly_scores, y_true, 0.10)
    recall_at_15 = compute_recall_at_k(anomaly_scores, y_true, 0.15)
    
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)
    
    # 4. Save Model Artifacts
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    joblib.dump(ANOMALY_FEATURE_COLS, os.path.join(MODELS_DIR, "anomaly_features.joblib"))
    
    print("=" * 65)
    print("ISOLATION FOREST ANOMALY VALIDATION REPORT (FALSIFIABILITY TEST):")
    print(f"Total Evaluated Records: {len(eval_df)} | Injected Red Flags: {np.sum(y_true)}")
    print("-" * 65)
    print(f"Recall@6% Budget  : {recall_at_6:.2%}")
    print(f"Recall@10% Budget : {recall_at_10:.2%}")
    print(f"Recall@15% Budget : {recall_at_15:.2%}")
    print(f"Anomaly ROC-AUC   : {roc_auc:.4f}")
    print(f"Anomaly PR-AUC    : {pr_auc:.4f}")
    print("=" * 65)
    print(f"Saved calibrated anomaly model to {MODELS_DIR}/isolation_forest.joblib")

if __name__ == "__main__":
    train_and_validate_anomaly_detector()