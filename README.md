# FinSight AI 📈
*Real-Time SEC Corporate FP&A Intelligence & ML Risk Triage*

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=flat&logo=react&logoColor=black)](https://vitejs.dev/)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://finsight-ai-no4q.onrender.com/docs)
[![Vercel](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=flat&logo=vercel&logoColor=white)](https://finsight-ai-two-tau.vercel.app)

---

## 🌐 Live Deployments

- **Web Dashboard (Vercel):** [https://finsight-ai-two-tau.vercel.app](https://finsight-ai-two-tau.vercel.app)
- **API Swagger Docs (Render):** [https://finsight-ai-no4q.onrender.com/docs](https://finsight-ai-no4q.onrender.com/docs)
- **API Health Check:** [https://finsight-ai-no4q.onrender.com/health](https://finsight-ai-no4q.onrender.com/health)

---

## 🏗️ Architecture Overview

FinSight AI employs a decoupled client-server architecture engineered for sub-100ms analytical responses:

```text
                      +-----------------------------+
                      |       React + Vite UI       |
                      |     (Hosted on Vercel)      |
                      +--------------+--------------+
                                     |
                                     | HTTPS / REST
                                     v
                      +-----------------------------+
                      |       FastAPI Backend       |
                      |     (Hosted on Render)      |
                      +--------------+--------------+
                                     |
           +-------------------------+-------------------------+
           |                         |                         |
           v                         v                         v
+--------------------+    +--------------------+    +--------------------+
|  In-Memory Feature |    |  XGBoost Operating |    |  TreeSHAP Driver   |
|   Analytics Cache  |    |  Profit Forecaster |    |     Explainer      |
+--------------------+    +--------------------+    +--------------------+