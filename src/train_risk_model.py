import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
TARGET_COL = "Target_NextPeriodHighRisk"


def train_and_benchmark():
  if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Missing {DATA_PATH}. Run src/feature_engineering.py first."
    )

  df = pd.read_csv(DATA_PATH)
  df["Date"] = pd.to_datetime(df["Date"])
  df = df.sort_values("Date").reset_index(drop=True)

  n = len(df)
  train_end = int(n * 0.70)
  val_end = int(n * 0.85)

  train_df = df.iloc[:train_end]
  val_df = df.iloc[train_end:val_end]
  test_df = df.iloc[val_end:]

  print("=" * 65)
  print("OUT-OF-TIME TEMPORAL SPLIT BREAKDOWN:")
  print(
      f"Train Fold: {len(train_df)} samples | Positives:"
      f" {int(train_df[TARGET_COL].sum())}"
      f" ({train_df[TARGET_COL].mean():.2%})"
  )
  print(
      f"Val Fold  : {len(val_df)} samples | Positives:"
      f" {int(val_df[TARGET_COL].sum())} ({val_df[TARGET_COL].mean():.2%})"
  )
  print(
      f"Test Fold : {len(test_df)} samples | Positives:"
      f" {int(test_df[TARGET_COL].sum())} ({test_df[TARGET_COL].mean():.2%})"
  )
  print("=" * 65)

  X_train = train_df[FEATURE_COLS]
  y_train = train_df[TARGET_COL].astype(int)

  X_val = val_df[FEATURE_COLS]
  y_val = val_df[TARGET_COL].astype(int)

  X_test = test_df[FEATURE_COLS]
  y_test = test_df[TARGET_COL].astype(int)

  neg_count = (y_train == 0).sum()
  pos_count = max((y_train == 1).sum(), 1)
  scale_pos = float(neg_count / pos_count)

  # Baseline Model: L2-regularized Logistic Regression
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.transform(X_val)
  X_test_scaled = scaler.transform(X_test)

  lr_baseline = LogisticRegression(
      max_iter=1000, class_weight="balanced", random_state=42
  )
  lr_baseline.fit(X_train_scaled, y_train)

  lr_test_proba = lr_baseline.predict_proba(X_test_scaled)[:, 1]

  # Candidate Model: Cost-Sensitive XGBoost
  xgb_model = XGBClassifier(
      n_estimators=100,
      max_depth=4,
      learning_rate=0.05,
      scale_pos_weight=scale_pos,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      eval_metric="logloss",
  )
  xgb_model.fit(X_train, y_train)

  xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
  xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]

  # Pick optimal decision threshold on Validation set (cost ratio 5:1 for false negatives)
  precisions, recalls, thresholds = precision_recall_curve(y_val, xgb_val_proba)
  best_thresh = 0.5
  min_cost = float("inf")
  cost_ratio = 5.0

  for t in thresholds:
    preds = (xgb_val_proba >= t).astype(int)
    fn = np.sum((y_val == 1) & (preds == 0))
    fp = np.sum((y_val == 0) & (preds == 1))
    cost = (cost_ratio * fn) + fp
    if cost < min_cost:
      min_cost = cost
      best_thresh = float(t)

  def evaluate_split(y_true, proba, thresh):
    preds = (proba >= thresh).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    pr_auc = average_precision_score(y_true, proba)
    roc_auc = (
        roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else 0.5
    )
    brier = brier_score_loss(y_true, proba)

    return {
        "PR-AUC": round(float(pr_auc), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "Brier Score": round(float(brier), 4),
    }

  lr_test_metrics = evaluate_split(y_test, lr_test_proba, 0.5)
  xgb_test_metrics = evaluate_split(y_test, xgb_test_proba, best_thresh)

  joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_risk_model.joblib"))
  joblib.dump(
      {"operating_threshold": best_thresh, "feature_cols": FEATURE_COLS},
      os.path.join(MODELS_DIR, "risk_meta.joblib"),
  )

  print("=" * 65)
  print(f"VAL-OPTIMIZED OPERATING THRESHOLD: {best_thresh:.4f}")
  print("=" * 65)
  print("HELD-OUT TEST SET BENCHMARKING:")
  results_df = pd.DataFrame([
      {"Model": "Logistic Regression Baseline", **lr_test_metrics},
      {"Model": "Cost-Sensitive XGBoost (Tuned)", **xgb_test_metrics},
  ])
  print(results_df.to_string(index=False))
  print("=" * 65)
  print(f"Model saved to {MODELS_DIR}/xgb_risk_model.joblib")


if __name__ == "__main__":
  train_and_benchmark()