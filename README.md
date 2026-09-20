# FinSight AI — ML-Powered Financial Planning, Forecasting & Risk Intelligence Platform

## Project Overview
FinSight AI is an end-to-end, machine learning–first financial analyst decision-support system designed to automate financial diagnostics, forecast key performance indicators, detect anomalous transactions, and predict forward-looking risk states before they impact operations.

The platform bridges enterprise financial analysis and machine learning by answering four fundamental FP&A questions:
1. **What happened?** — Historical actuals, profitability margins, and working capital ratios.
2. **Why did it happen?** — Financial driver analysis, variance tracking, and TreeSHAP explainability.
3. **What is likely to happen next?** — Multi-target XGBoost regression forecasting for Revenue, Expenses, and Operating Profit.
4. **What deserves attention?** — Multi-dimensional anomaly detection via Isolation Forest and forward-looking financial risk classification.

---

## System Architecture & Data Flow
1. **Data Ingestion & SQL Layer:** 5-year multi-entity financial ledger ingested into SQLite with automated analytical views (`v_financial_ratios`, `v_financial_trends`, `v_budget_variance`).
2. **Feature Engineering Pipeline:** Purely past-oriented features ($t-1$, $t-2$, rolling 3-month averages) eliminating lookahead bias and data leakage.
3. **Forecasting Engine (Module A):** Time-aware XGBoost models evaluated against a Naïve Lag-1 Persistence baseline across 12-month test horizons.
4. **Risk Intelligence (Module B):** Forward-looking ($T+1$) class-weighted XGBoost classifier identifying liquidity and margin distress with TreeSHAP global and local attribution.
5. **Anomaly Detection (Module C):** Unsupervised Isolation Forest isolating working capital spikes and multivariate operational outliers.
6. **Decision Layer:** 3-Page interactive Power BI reporting suite with dynamic DAX calculations.

---

## Validated Model Performance

### 1. Financial Forecasting (Module A)
Evaluated on an out-of-time 12-month holdout set across three business units:

| Target Metric | Baseline (Lag-1) MAPE | Champion XGBoost MAPE | XGBoost MAE | XGBoost RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Operating Profit** | 20.04% | **7.41%** (63% error reduction) | $17,875.15 | $23,169.43 |
| **Revenue** | 5.65% | **4.52%** | $52,825.25 | $77,787.34 |
| **Total Expenses** | 5.47% | **5.56%** | $51,456.26 | $74,589.13 |

### 2. Forward-Looking Risk Prediction (Module B)
Evaluated under natural class imbalance for next-period stress detection:
* **Logistic Regression Baseline:** ROC-AUC: 0.500 | F1-Score: 0.000
* **Champion XGBoost Classifier:** **ROC-AUC: 0.688** | **F1-Score: 0.286** | **Recall on High-Risk Events: 50%**
* **SHAP Risk Attributions:** Top drivers identified as `DebtToCashRatio` (0.508), `CashBufferRatio` (0.380), and `ReceivablesToRevenueRatio` (0.333).

### 3. Anomaly Detection (Module C)
* Isolation Forest flagged **9 high-priority operational anomalies** out of 168 records, pinpointing abnormal receivables surges and cost spikes.

---

## Tech Stack
* **Core Analytics & Modeling:** Python 3.11+, Pandas, NumPy, Scikit-learn, XGBoost, SHAP
* **Database & Query Layer:** SQLite3, SQL DDL/DML analytical views
* **Visualization & BI:** Microsoft Power BI Desktop, DAX (Data Analysis Expressions)