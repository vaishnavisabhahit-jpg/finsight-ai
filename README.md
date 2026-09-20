# FinSight AI — Financial Intelligence & Risk Decision Platform

An end-to-end financial forecasting, distress risk ranking, and decision-support architecture built for corporate financial planning and analysis (FP&A). FinSight AI couples an embedded analytical SQL layer with time-series feature engineering, multi-target predictive modeling, and executive Power BI dashboards while enforcing strict chronological data boundaries.

---

## System Objectives & Problem Formulation

Standard machine learning deployments in corporate finance often degrade in production due to two systemic issues:
1. Lookahead Data Leakage: Conventional randomized train/test splits inadvertently expose models to future data points during time-series feature engineering.
2. Opaque Scoring Under Class Imbalance: Distressed quarters are statistically rare (<12% prevalence), causing naive classifiers to optimize for raw accuracy while failing to triage high-risk operating cycles.

FinSight AI addresses these challenges by enforcing out-of-time evaluation boundaries, benchmarking predictive gains against standard persistence heuristics, and leveraging game-theoretic attribution to audit the dominant drivers of model risk scores.

---

## Methodology & Empirical Evaluation

- Temporal Out-of-Time Validation: Evaluated all models using strict chronological splits (reserving the final 12-month window as unseen holdout data) to eliminate lookahead bias.
- Forecasting Baseline Benchmark: Benchmarked multi-target XGBoost models against a Naive Persistence baseline (y_t = y_{t-1}), achieving a 63% relative reduction in Operating Profit forecasting error (reducing MAPE from 20.04% to 7.41%).
- Distress Risk Triage (<12% Imbalance): Formulated financial distress as a cost-sensitive risk-ranking problem rather than a standard balanced classifier. Optimized decision thresholds to isolate the top-decile riskiest quarters for auditor review.
- Model Explainability (TreeSHAP): Applied game-theoretic Shapley attributions to audit the model dominant predictive drivers, identifying DebtToCashRatio and CashBufferRatio as the dominant features correlated with elevated risk scores.
- Exploratory Outlier Screening: Deployed an Isolation Forest model as an unsupervised heuristic to flag multivariate balance-sheet irregularities for secondary human inspection.

---

## Repository Structure

- sql/: Analytical aggregation SQL views
- src/: Ingestion, chronological feature engineering, XGBoost models, and TreeSHAP scripts
- outputs/: Master analytical feeds
- powerbi/: Interactive 3-page reporting dashboard
