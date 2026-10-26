Set-Content -Path "frontend\src\App.jsx" -Encoding utf8 -Value @'
import React, { useState, useEffect } from 'react';
import { Cpu } from 'lucide-react';
import GlassCard3D from './components/GlassCard3D';
import GlowingSurface3D from './components/GlowingSurface3D';
import AnomalyGalaxy3D from './components/AnomalyGalaxy3D';
import FinTech3DCore from './components/FinTech3DCore';

const API_BASE = "https://finsight-ai-no4q.onrender.com";

export default function App() {
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState("NovaTech Retail (Synthetic)");
  const [forecastData, setForecastData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [explainData, setExplainData] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/companies`)
      .then(res => res.json())
      .then(data => {
        setCompanies(data);
        if (data.length > 0 && !selectedCompany) {
          setSelectedCompany(data[0].company);
        }
      })
      .catch(err => console.error("Error fetching companies:", err));
  }, []);

  useEffect(() => {
    if (!selectedCompany) return;
    const encoded = encodeURIComponent(selectedCompany);

    Promise.all([
      fetch(`${API_BASE}/forecast/${encoded}`).then(r => r.json()),
      fetch(`${API_BASE}/risk/${encoded}`).then(r => r.json()),
      fetch(`${API_BASE}/explain/${encoded}`).then(r => r.json())
    ])
      .then(([forecast, risk, explain]) => {
        setForecastData(forecast);
        setRiskData(risk);
        setExplainData(explain);
      })
      .catch(err => console.error("Error fetching analytics:", err));
  }, [selectedCompany]);

  const isElevated = riskData?.audit_triage_tier?.toLowerCase().includes("elevated");

  return (
    <div className="relative min-h-screen bg-[#0B0F1A] text-[#E6EAF2] font-sans overflow-x-hidden selection:bg-[#00F5D4] selection:text-black">
      <div 
        className="fixed inset-0 pointer-events-none opacity-[0.09] z-0"
        style={{
          backgroundImage: `linear-gradient(to right, #7B61FF 1px, transparent 1px),
                            linear-gradient(to bottom, #7B61FF 1px, transparent 1px)`,
          backgroundSize: '45px 45px'
        }}
      />
      <div className="fixed top-[-10%] left-[-10%] w-[650px] h-[650px] rounded-full bg-[#7B61FF]/12 blur-[160px] pointer-events-none" />
      <div className="fixed top-[30%] right-[-10%] w-[600px] h-[600px] rounded-full bg-[#00F5D4]/10 blur-[160px] pointer-events-none" />

      <header className="relative z-10 border-b border-[#7B61FF]/25 bg-[#0B0F1A]/80 backdrop-blur-xl px-6 py-4">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#7B61FF]/40 bg-[#7B61FF]/10 shadow-[0_0_25px_rgba(123,97,255,0.35)]">
              <Cpu className="h-6 w-6 text-[#00F5D4]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-black tracking-wider text-[#E6EAF2]">FinSight</span>
                <span className="rounded border border-[#7B61FF]/40 bg-[#7B61FF]/20 px-2 py-0.5 text-xs font-mono font-bold text-[#00F5D4]">
                  AI v2.2 Calibrated
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#E6EAF2]/60">FP&A DECISION SUPPORT & ML INFERENCE</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs font-mono text-[#E6EAF2]/60">COHORT:</label>
            <select
              value={selectedCompany}
              onChange={(e) => setSelectedCompany(e.target.value)}
              className="rounded-xl border border-[#7B61FF]/40 bg-[#121829] px-4 py-2 font-mono text-sm text-[#00F5D4] shadow-inner focus:border-[#00F5D4] focus:outline-none"
            >
              {companies.map((c) => (
                <option key={c.company} value={c.company} className="bg-[#0B0F1A] text-[#E6EAF2]">
                  {c.company}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <GlassCard3D
            title="Distress Propensity Score"
            value={riskData ? riskData.distress_propensity_score * 100 : 0.33}
            suffix="%"
            subtitle="Class-weighted cost-sensitive XGBoost"
            status={isElevated ? "danger" : "positive"}
            tag={riskData?.audit_triage_tier || "Baseline Propensity"}
            footer={`Operating Threshold: ${riskData?.operating_threshold_score || 0.2317} | Val Recall: 1.0`}
          />

          <GlassCard3D
            title="Operating Profit Forecast"
            value={forecastData?.next_quarter_forecast_usd ? forecastData.next_quarter_forecast_usd / 1000 : 24.5}
            prefix="$"
            suffix="k"
            subtitle="Walk-forward temporal XGBoost projection"
            status="gold"
            tag="Q+1 Projection"
            footer="Zero-leakage out-of-time evaluation fold"
          />

          <GlassCard3D
            title="TreeSHAP Base Vector"
            value={explainData?.base_value || -1.41}
            subtitle={`Dominant: ${explainData?.dominant_driver || 'Lag_OperatingProfit_1'}`}
            status="neutral"
            tag="Game Theoretic"
            footer={`Raw Logit Margin: ${explainData?.model_raw_prediction || -6.44}`}
          />
        </div>

        <GlowingSurface3D selectedCompany={selectedCompany} />

        <AnomalyGalaxy3D />

        <FinTech3DCore 
          distressScore={riskData?.distress_propensity_score ?? 0.0033}
          tier={riskData?.audit_triage_tier ?? "Baseline Propensity Tier"}
        />
      </main>
    </div>
  );
}
'@