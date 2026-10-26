import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

export default function AnomalyGalaxy3D() {
  const { normalPoints, anomalyPoints } = useMemo(() => {
    const normal = { x: [], y: [], z: [], text: [] };
    const anomalies = { x: [], y: [], z: [], text: [] };

    for (let i = 0; i < 420; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const r = Math.cbrt(Math.random()) * 2.2;
      const sinPhi = Math.sin(phi);

      normal.x.push(r * sinPhi * Math.cos(theta));
      normal.y.push(r * sinPhi * Math.sin(theta));
      normal.z.push(r * Math.cos(phi) * 0.7);
      normal.text.push(`Ledger #${1000 + i}<br>Z-Score: ${(Math.random() * 1.2).toFixed(2)}<br>Status: In-Bounds Baseline`);
    }

    for (let j = 0; j < 38; j++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = 3.2 + Math.random() * 2.6;
      const height = (Math.random() - 0.5) * 3.8;

      anomalies.x.push(Math.cos(angle) * dist);
      anomalies.y.push(Math.sin(angle) * dist);
      anomalies.z.push(height);
      anomalies.text.push(`<b>ANOMALY ALERT #${j + 1}</b><br>Isolation Margin: -${(0.45 + Math.random() * 0.4).toFixed(3)}<br>Divergence: Out-of-Distribution`);
    }

    return { normalPoints: normal, anomalyPoints: anomalies };
  }, []);

  return (
    <div className="relative w-full rounded-2xl border border-[#FF3D71]/30 bg-[#121829]/70 backdrop-blur-2xl p-5 shadow-[0_0_50px_rgba(255,61,113,0.12)] overflow-hidden">
      <div className="flex items-center justify-between pb-3 border-b border-[#7B61FF]/20 font-mono text-xs text-[#E6EAF2]">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-[#FF3D71] animate-ping" />
          <span className="font-bold tracking-wider text-[#FF3D71]">ANOMALY GALAXY CLUSTER (ISOLATION SPACE)</span>
        </div>
        <span className="text-[#E6EAF2]/60">ISOLATION FOREST LATENT PROJECTION</span>
      </div>

      <div className="h-[460px] w-full">
        <Plot
          data={[
            {
              type: 'scatter3d',
              mode: 'markers',
              name: 'Normal Ledger Flow',
              x: normalPoints.x,
              y: normalPoints.y,
              z: normalPoints.z,
              text: normalPoints.text,
              hoverinfo: 'text',
              marker: {
                size: 3.5,
                color: normalPoints.z,
                colorscale: [
                  [0.0, '#7B61FF'],
                  [0.5, '#00F5D4'],
                  [1.0, '#38bdf8']
                ],
                opacity: 0.65,
              }
            },
            {
              type: 'scatter3d',
              mode: 'markers',
              name: 'Detected Distress Anomalies',
              x: anomalyPoints.x,
              y: anomalyPoints.y,
              z: anomalyPoints.z,
              text: anomalyPoints.text,
              hoverinfo: 'text',
              marker: {
                size: 6.5,
                color: '#FF3D71',
                symbol: 'diamond',
                opacity: 0.95,
                line: { color: '#FFFFFF', width: 1.5 }
              }
            }
          ]}
          layout={{
            autosize: true,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            showlegend: true,
            legend: {
              x: 0,
              y: 1,
              font: { color: '#E6EAF2', family: 'monospace', size: 10 },
              bgcolor: 'rgba(18, 24, 41, 0.6)'
            },
            margin: { l: 0, r: 0, b: 10, t: 10 },
            scene: {
              xaxis: { visible: false },
              yaxis: { visible: false },
              zaxis: { visible: false },
              camera: {
                eye: { x: 1.5, y: 1.5, z: 1.2 }
              }
            }
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      <div className="pt-2 flex justify-between font-mono text-[10px] text-[#E6EAF2]/40 border-t border-[#7B61FF]/15">
        <span>NEBULA: 420 NORMAL LEDGER NODES</span>
        <span className="text-[#FF3D71]">OUTLIERS: 38 DIVERGENT TRANSACTIONS CAUGHT</span>
      </div>
    </div>
  );
}
