import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

export default function GlowingSurface3D() {
  const { x, y, z } = useMemo(() => {
    const quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024', 'Q1 2025', 'Q2 2025', 'Q3 2025', 'Q4 2025', 'Q1 2026', 'Q2 2026'];
    const scenarios = ['Stress -30%', 'Base Downside', 'Market Baseline', 'Upside Growth', 'Optimistic +30%'];
    
    const zData = [];
    for (let j = 0; j < scenarios.length; j++) {
      const row = [];
      const scenarioWeight = (j - 2) * 12;
      for (let i = 0; i < quarters.length; i++) {
        const val = 45 + scenarioWeight + Math.sin(i * 0.7) * 18 + Math.cos(j * 0.9) * 12 + (i * 2.2);
        row.push(val);
      }
      zData.push(row);
    }
    return { x: quarters, y: scenarios, z: zData };
  }, []);

  return (
    <div className="relative w-full rounded-2xl border border-[#7B61FF]/30 bg-[#121829]/70 backdrop-blur-2xl p-5 shadow-[0_0_50px_rgba(123,97,255,0.15)] overflow-hidden">
      <div className="flex items-center justify-between pb-3 border-b border-[#7B61FF]/20 font-mono text-xs text-[#E6EAF2]">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-[#00F5D4] animate-ping" />
          <span className="font-bold tracking-wider text-[#00F5D4]">GLOWING 3D VOLATILITY & FORECAST SURFACE</span>
        </div>
        <span className="text-[#E6EAF2]/60">AXES: TIME × SCENARIO × OPERATING VALUE ($k)</span>
      </div>

      <div className="h-[460px] w-full">
        <Plot
          data={[
            {
              type: 'surface',
              x: x,
              y: y,
              z: z,
              colorscale: [
                [0.0, '#0B0F1A'],
                [0.2, '#311042'],
                [0.45, '#7B61FF'],
                [0.75, '#00F5D4'],
                [1.0, '#E6EAF2']
              ],
              showscale: true,
              colorbar: {
                tickfont: { color: '#E6EAF2', family: 'monospace', size: 10 },
                title: { text: 'Value ($k)', font: { color: '#00F5D4', size: 11 } },
                len: 0.7,
                thickness: 14,
                outlinecolor: 'transparent'
              },
              contours: {
                z: { show: true, usecolormap: true, highlightcolor: '#00F5D4', project: { z: false } }
              },
              lighting: {
                ambient: 0.65,
                diffuse: 0.85,
                specular: 1.4,
                roughness: 0.25,
                fresnel: 1.8
              }
            }
          ]}
          layout={{
            autosize: true,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { l: 10, r: 10, b: 20, t: 10 },
            scene: {
              xaxis: {
                title: 'Timeline',
                color: '#E6EAF2',
                tickfont: { size: 9, color: '#E6EAF2' },
                gridcolor: '#7B61FF33',
                showbackground: true,
                backgroundcolor: '#0B0F1A55'
              },
              yaxis: {
                title: 'Scenario',
                color: '#E6EAF2',
                tickfont: { size: 9, color: '#E6EAF2' },
                gridcolor: '#7B61FF33',
                showbackground: true,
                backgroundcolor: '#0B0F1A55'
              },
              zaxis: {
                title: 'Forecast ($k)',
                color: '#00F5D4',
                tickfont: { size: 9, color: '#00F5D4' },
                gridcolor: '#7B61FF33',
                showbackground: true,
                backgroundcolor: '#0B0F1A55'
              },
              camera: {
                eye: { x: 1.7, y: -1.7, z: 1.3 }
              }
            }
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
      <div className="pt-2 flex justify-between font-mono text-[10px] text-[#E6EAF2]/40 border-t border-[#7B61FF]/15">
        <span>TERRAIN: NON-LINEAR FP&A SENSITIVITY MESH</span>
        <span>GRADIENT: INDIGO ➔ VIOLET ➔ CYAN ➔ MINT</span>
      </div>
    </div>
  );
}
