// ComparisonToggle.jsx — Drug A vs Drug B vs Combined view toggle
import React from 'react';

export default function ComparisonToggle({ drugs = [], activeView = 'combined', onViewChange, isMobile = false }) {
  if (drugs.length < 2) return null;

  const views = [
    { key: 'combined', label: 'Combined', color: '#a855f7' },
    { key: 'drugA', label: drugs[0] || 'Drug A', color: '#22d3ee' },
    { key: 'drugB', label: drugs[1] || 'Drug B', color: '#f97316' },
  ];

  return (
    <div className={`
      absolute z-10 flex items-center gap-1
      ${isMobile ? 'bottom-10 left-2 right-2 justify-center' : 'bottom-4 right-4'}
    `}>
      {views.map(({ key, label, color }) => {
        const active = activeView === key;
        return (
          <button
            key={key}
            onClick={() => onViewChange?.(key)}
            className={`
              px-2 py-1 rounded border transition-all text-[9px] font-mono uppercase tracking-wider
              ${active
                ? 'border-white/20 bg-white/5'
                : 'border-white/5 bg-transparent opacity-40 hover:opacity-70'
              }
            `}
            style={{ color: active ? color : '#64748b' }}
          >
            {isMobile ? label.slice(0, 8) : label}
          </button>
        );
      })}
    </div>
  );
}
