import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, CartesianGrid, Cell
} from 'recharts';
import { Activity, ShieldAlert, Cpu, BarChart3, AlertTriangle } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function App() {
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState('');
  const [forecast, setForecast] = useState(null);
  const [risk, setRisk] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCompanies();
  }, []);

  useEffect(() => {
    if (selectedCompany) {
      fetchAnalytics(selectedCompany);
    }
  }, [selectedCompany]);

  const fetchCompanies = async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API_BASE_URL}/companies`);
      setCompanies(res.data);
      if (res.data.length > 0) {
        setSelectedCompany(res.data[0].company);
      }
      setLoading(false);
    } catch (err) {
      console.error(err);
      setError('Unable to connect to the analytical inference engine.');
      setLoading(false);
    }
  };

  const fetchAnalytics = async (ticker) => {
    try {
      const [forecastRes, riskRes, explainRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/forecast/${ticker}`),
        axios.get(`${API_BASE_URL}/risk/${ticker}`),
        axios.get(`${API_BASE_URL}/explain/${ticker}`)
      ]);
      setForecast(forecastRes.data);
      setRisk(riskRes.data);
      setExplanation(explainRes.data);
    } catch (err) {
      console.error(err);
    }
  };

  const chartData = forecast
    ? forecast.historical_dates.map((date, idx) => ({
        date,
        Actual: forecast.actual_operating_profit[idx],
        Predicted: forecast.predicted_operating_profit[idx],
      }))
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="text-cyan-400 w-7 h-7" />
            <h1 className="text-2xl font-bold tracking-tight text-white">FinSight AI</h1>
            <span className="text-xs bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded">v2.2 Calibrated</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Corporate FP&A Intelligence & Statistical Feature Attribution (Synthetic Benchmarks)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs text-slate-400 font-medium">Active Cohort:</label>
          <select
            value={selectedCompany}
            onChange={(e) => setSelectedCompany(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-lg px-3 py-2 outline-none focus:border-cyan-500"
          >
            {companies.map((c) => (
              <option key={c.company} value={c.company}>
                {c.company}
              </option>
            ))}
          </select>
        </div>
      </header>

      {error ? (
        <div className="max-w-7xl mx-auto mt-8 p-4 bg-red-950/40 border border-red-800 text-red-200 rounded-lg flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          <span>{error}</span>
        </div>
      ) : (
        <main className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          {/* Risk Triage Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Distress Propensity Triage</span>
              <ShieldAlert className="w-4 h-4 text-amber-400" />
            </div>
            {risk ? (
              <div className="mt-4 space-y-4">
                <div>
                  <div className="text-3xl font-extrabold text-white">
                    {risk.distress_propensity_score}
                  </div>
                  <span className={`inline-block mt-2 text-xs px-2.5 py-1 rounded-full font-medium ${
                    risk.audit_triage_tier === 'Elevated Distress-Propensity Tier'
                      ? 'bg-amber-950 text-amber-300 border border-amber-800'
                      : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                  }`}>
                    {risk.audit_triage_tier}
                  </span>
                </div>
                <div className="text-xs text-slate-400 leading-relaxed border-t border-slate-800/80 pt-3">
                  <p><strong className="text-slate-300">Calibrated Threshold:</strong> {risk.operating_threshold_score}</p>
                  <p className="mt-0.5"><strong className="text-slate-300">Val Precision / Recall:</strong> {risk.validation_precision} / {risk.validation_recall}</p>
                  <p className="mt-1.5 text-[11px] text-slate-400">{risk.calibration_note}</p>
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-500 mt-4">Loading metrics...</div>
            )}
          </div>

          {/* Forecasting Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 md:col-span-2 shadow-sm">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Operating Profit Forecast (XGBoost)</span>
              <Activity className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="h-64 mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} tickFormatter={(val) => `$${(val / 1e6).toFixed(0)}M`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }}
                    formatter={(val) => [`$${Number(val).toLocaleString()}`, '']}
                  />
                  <Line type="monotone" dataKey="Actual" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="Predicted" stroke="#a855f7" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* SHAP Attribution Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 md:col-span-3 shadow-sm">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">TreeSHAP Feature Attributions</span>
                <span className="block text-[11px] text-slate-400 mt-0.5">
                  Additive statistical associations relative to expected value (not causal proof)
                </span>
              </div>
              <BarChart3 className="w-4 h-4 text-purple-400" />
            </div>
            {explanation ? (
              <div className="mt-4">
                <div className="text-xs text-slate-300 mb-3">
                  Dominant Predictive Contributor: <strong className="text-purple-300">{explanation.dominant_driver}</strong>
                </div>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={explanation.attributions} layout="vertical" margin={{ left: 40, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                      <XAxis type="number" stroke="#64748b" fontSize={11} />
                      <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={11} width={130} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }} />
                      <Bar dataKey="attribution_weight">
                        {explanation.attributions.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.attribution_weight >= 0 ? '#06b6d4' : '#f43f5e'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-500 mt-4">Loading attributions...</div>
            )}
          </div>
        </main>
      )}
    </div>
  );
}