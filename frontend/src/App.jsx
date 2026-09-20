import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export default function App() {
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState("");
  const [forecastData, setForecastData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [explainData, setExplainData] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // 1. Initial Load: Companies and Global Anomalies
  useEffect(() => {
    async function init() {
      try {
        const [compRes, anomRes] = await Promise.all([
          axios.get(`${API_BASE}/companies`),
          axios.get(`${API_BASE}/anomalies`)
        ]);
        setCompanies(compRes.data);
        if (compRes.data.length > 0) {
          setSelectedCompany(compRes.data[0].company);
        }
        setAnomalies(anomRes.data.anomalies || []);
      } catch (err) {
        setError("Failed to connect to FinSight backend. Ensure FastAPI is running on port 8000.");
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  // 2. Fetch specific company metrics whenever selectedCompany changes
  useEffect(() => {
    if (!selectedCompany) return;

    async function fetchCompanyDetails() {
      try {
        const [fRes, rRes, eRes] = await Promise.all([
          axios.get(`${API_BASE}/forecast/${selectedCompany}`),
          axios.get(`${API_BASE}/risk/${selectedCompany}`),
          axios.get(`${API_BASE}/explain/${selectedCompany}`)
        ]);
        setForecastData(fRes.data);
        setRiskData(rRes.data);
        setExplainData(eRes.data);
      } catch (err) {
        console.error("Error fetching company details", err);
      }
    }
    fetchCompanyDetails();
  }, [selectedCompany]);

  if (loading) {
    return <div style={{ padding: "40px", color: "#fff", background: "#0f172a", minHeight: "100vh" }}>Loading FinSight AI Platform...</div>;
  }

  if (error) {
    return (
      <div style={{ padding: "40px", color: "#ef4444", background: "#0f172a", minHeight: "100vh" }}>
        <h3>Connection Error</h3>
        <p>{error}</p>
      </div>
    );
  }

  const selectedSummary = companies.find((c) => c.company === selectedCompany) || {};

  // Format forecast series for Recharts
  const chartData = (forecastData?.historical_dates || []).map((date, idx) => ({
    date,
    Actual: forecastData.actual_operating_profit[idx] !== null ? forecastData.actual_operating_profit[idx] / 1e9 : null,
    Predicted: forecastData.predicted_operating_profit[idx] / 1e9
  }));

  return (
    <div style={{ background: "#0b0f19", color: "#f8fafc", minHeight: "100vh", padding: "24px", fontFamily: "sans-serif" }}>
      {/* Header Bar */}
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #1e293b", paddingBottom: "16px" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "24px", fontWeight: "700", color: "#38bdf8" }}>FinSight AI</h1>
          <p style={{ margin: "4px 0 0", color: "#94a3b8", fontSize: "14px" }}>
            Real-Time SEC Corporate FP&A Intelligence & Risk Triage
          </p>
        </div>
        <div>
          <label style={{ fontSize: "14px", color: "#94a3b8", marginRight: "8px" }}>Select Entity:</label>
          <select
            value={selectedCompany}
            onChange={(e) => setSelectedCompany(e.target.value)}
            style={{ background: "#1e293b", color: "#fff", border: "1px solid #334155", padding: "8px 16px", borderRadius: "6px" }}
          >
            {companies.map((c) => (
              <option key={c.company} value={c.company}>{c.company}</option>
            ))}
          </select>
        </div>
      </header>

      {/* KPI Cards */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "16px", marginTop: "24px" }}>
        <div style={{ background: "#1e293b", padding: "16px", borderRadius: "8px" }}>
          <span style={{ color: "#94a3b8", fontSize: "12px" }}>LATEST QUARTER REVENUE</span>
          <h2 style={{ margin: "8px 0 0" }}>${((selectedSummary.latest_revenue_usd || 0) / 1e9).toFixed(2)}B</h2>
        </div>
        <div style={{ background: "#1e293b", padding: "16px", borderRadius: "8px" }}>
          <span style={{ color: "#94a3b8", fontSize: "12px" }}>OPERATING MARGIN</span>
          <h2 style={{ margin: "8px 0 0" }}>{selectedSummary.latest_operating_margin_pct || 0}%</h2>
        </div>
        <div style={{ background: "#1e293b", padding: "16px", borderRadius: "8px" }}>
          <span style={{ color: "#94a3b8", fontSize: "12px" }}>DISTRESS RISK SCORE</span>
          <h2 style={{ margin: "8px 0 0", color: (riskData?.distress_risk_probability_pct || 0) > 50 ? "#ef4444" : "#10b981" }}>
            {riskData?.distress_risk_probability_pct?.toFixed(1) || 0}%
          </h2>
        </div>
        <div style={{ background: "#1e293b", padding: "16px", borderRadius: "8px" }}>
          <span style={{ color: "#94a3b8", fontSize: "12px" }}>TRIAGE STATUS</span>
          <h3 style={{ margin: "8px 0 0", color: riskData?.audit_triage_tier === "High Priority Review" ? "#ef4444" : "#38bdf8" }}>
            {riskData?.audit_triage_tier || "Evaluating..."}
          </h3>
        </div>
      </section>

      {/* Charts Grid */}
      <section style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "24px", marginTop: "24px" }}>
        {/* Forecast Realization Chart */}
        <div style={{ background: "#1e293b", padding: "20px", borderRadius: "8px" }}>
          <h3 style={{ marginTop: 0, fontSize: "16px" }}>Operating Profit: Realized vs. XGBoost Prediction ($ Billions)</h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="date" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155" }} />
              <Legend />
              <Line type="monotone" dataKey="Actual" stroke="#38bdf8" strokeWidth={2} />
              <Line type="monotone" dataKey="Predicted" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* TreeSHAP Feature Attributions */}
        <div style={{ background: "#1e293b", padding: "20px", borderRadius: "8px" }}>
          <h3 style={{ marginTop: 0, fontSize: "16px" }}>TreeSHAP Feature Attributions</h3>
          <p style={{ color: "#94a3b8", fontSize: "12px" }}>Top Driver: <strong style={{ color: "#f8fafc" }}>{explainData?.dominant_driver}</strong></p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart layout="vertical" data={explainData?.attributions || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" stroke="#94a3b8" />
              <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={110} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155" }} />
              <Bar dataKey="attribution_weight" fill="#38bdf8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Flagged Balance Sheet Anomalies */}
      <section style={{ background: "#1e293b", padding: "20px", borderRadius: "8px", marginTop: "24px" }}>
        <h3 style={{ marginTop: 0, fontSize: "16px" }}>Isolation Forest: Flagged Multivariate Irregularities</h3>
        <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "12px", fontSize: "14px" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #334155", textAlign: "left", color: "#94a3b8" }}>
              <th style={{ padding: "8px" }}>Company</th>
              <th style={{ padding: "8px" }}>Quarter</th>
              <th style={{ padding: "8px" }}>Revenue ($B)</th>
              <th style={{ padding: "8px" }}>Operating Profit ($B)</th>
              <th style={{ padding: "8px" }}>Screening Diagnostic</th>
            </tr>
          </thead>
          <tbody>
            {anomalies.map((a, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #1e293b" }}>
                <td style={{ padding: "8px" }}>{a.company}</td>
                <td style={{ padding: "8px" }}>{a.date}</td>
                <td style={{ padding: "8px" }}>${(a.revenue_usd / 1e9).toFixed(2)}B</td>
                <td style={{ padding: "8px" }}>${(a.operating_profit_usd / 1e9).toFixed(2)}B</td>
                <td style={{ padding: "8px", color: "#f59e0b" }}>{a.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}