# FinSight AI — Corporate FP&A Decision Support Engine

FinSight AI is a full-stack corporate financial analytics platform designed for FP&A teams and credit analysts. It provides machine learning-driven operating profit forecasting, cost-sensitive distress risk triage, and TreeSHAP feature attributions on a standardized quarterly FP&A dataset.

---

## Architecture Overview

- **Backend:** FastAPI (Python 3.11+) with in-memory analytical cache
- **Forecasting Engine:** Gradient Boosted Regressor (`XGBRegressor`) modeling quarter-over-quarter operating profit dynamics
- **Distress Propensity Engine:** Cost-sensitive `XGBClassifier` utilizing asymmetric loss weighting for minority distress events
- **Attribution Engine:** `TreeSHAP` (`shap.TreeExplainer`) computing local additive feature attributions
- **Frontend:** React (Vite) + Tailwind CSS + Recharts + Lucide Icons
- **Deployment:** Render (Backend API), Vercel (Frontend Dashboard)

---

## Data & Benchmark Disclaimer

- **Data Specification:** Benchmarked against a calibrated synthetic FP&A dataset (`data/synthetic_fpa_features.csv`) representing multi-sector enterprise cohorts (`SYN-*` / synthetic enterprise labels).
- **Evaluation Metrics:** Logged performance metrics (e.g., 7.41% MAPE on forecasting, 0.88 ROC-AUC on distress classification) are evaluated strictly on this synthetic FP&A benchmark cohort under out-of-time evaluation splits.
- **Roadmap Note:** Direct integration with live SEC EDGAR XBRL ingest pipelines is currently an architectural roadmap item and is not utilized in this demonstration build.

---

## Production Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Service uptime and company cache status |
| `/companies` | `GET` | Latest quarterly summaries for active cohorts |
| `/forecast/{company}` | `GET` | Historical actuals vs. model forecasts + next-quarter projection |
| `/risk/{company}` | `GET` | Calibrated distress propensity score and operational audit tier |
| `/explain/{company}` | `GET` | TreeSHAP local additive feature attributions |
| `/anomalies` | `GET` | Multivariate financial outlier records via Isolation Forest |

---

## Local Development & Setup

### 1. Backend Setup
```bash
# Clone and navigate to repository
cd fintech-ai

# Activate Python virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run automated test suite
python -m pytest tests/ -v

# Start FastAPI development server
uvicorn src.api:app --reload --port 8000