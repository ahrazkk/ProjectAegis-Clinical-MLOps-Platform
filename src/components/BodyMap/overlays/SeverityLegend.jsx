// SeverityLegend.jsx — Severity scale + data source badge
import React from 'react';
import { getSeverityColor } from '../organRegistry';

const SEVERITY_LEVELS = [
  { value: 0.8, label: 'Severe' },
  { value: 0.5, label: 'Moderate' },
  { value: 0.2, label: 'Mild' },
  { value: 0, label: 'Normal' },
];

const DATA_QUALITY_LABELS = {
  none: { label: 'No Data', color: 'text-slate-600', bg: 'bg-slate-800/50' },
  prediction: { label: 'Predicted', color: 'text-amber-500', bg: 'bg-amber-500/10' },
  enriched: { label: 'Drug Data', color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
  full: { label: 'Full Evidence', color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
};

export default function SeverityLegend({ dataQuality = 'none', isMobile = false }) {
  const quality = DATA_QUALITY_LABELS[dataQuality] || DATA_QUALITY_LABELS.none;

  return (
    <div className={`
      absolute z-10 flex flex-col gap-2
      ${isMobile ? 'bottom-2 left-2' : 'bottom-4 left-4'}
    `}>
      {/* Data source badge */}
      <div className={`
        inline-flex items-center gap-1.5 px-2 py-1 rounded border border-white/5
        ${quality.bg} backdrop-blur-sm
      `}>
        <span className={`w-1.5 h-1.5 rounded-full ${quality.color.replace('text-', 'bg-')}`} />
        <span className={`text-[9px] font-mono uppercase tracking-wider ${quality.color}`}>
          {quality.label}
        </span>
      </div>

      {/* Severity scale */}
      <div className="flex items-center gap-1.5 px-2 py-1.5 rounded bg-[#0a0f1a]/80 backdrop-blur-sm border border-white/5">
        {SEVERITY_LEVELS.map(({ value, label }) => {
          const info = getSeverityColor(value);
          return (
            <div key={label} className="flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-sm"
                style={{ background: info.fill }}
              />
              {!isMobile && (
                <span className="text-[8px] text-slate-500 font-mono">{label}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
