// MechanismLegend.jsx — Legend explaining node/edge types
import React from 'react';
import { DRUG_COLORS, ENZYME_ROLE_COLORS, TARGET_ACTION_COLORS, EDGE_STYLES } from '../mechanismGraphEngine';

export default function MechanismLegend({ isOffline, isLightTheme = false }) {
  const panelClass = isLightTheme
    ? 'bg-white/86 border-slate-500/35 shadow-slate-800/10'
    : 'bg-black/75 border-white/8';
  const titleClass = isLightTheme ? 'text-slate-700/80' : 'text-white/40';
  const dividerClass = isLightTheme ? 'border-slate-400/25' : 'border-white/5';
  const labelClass = isLightTheme ? 'text-slate-700/75' : 'text-white/40';

  return (
    <div className="absolute bottom-3 left-3 z-20 pointer-events-none">
      <div className={`backdrop-blur-md border rounded-lg shadow-xl px-3 py-2 pointer-events-auto min-w-[140px] ${panelClass}`}>
        <div className={`text-[8px] uppercase tracking-wider mb-1.5 font-mono ${titleClass}`}>Legend</div>

        {/* Node types */}
        <div className="space-y-1 mb-2">
          <LegendItem shape="hexagon" color={DRUG_COLORS.A} label="Drug A" labelClass={labelClass} />
          <LegendItem shape="hexagon" color={DRUG_COLORS.B} label="Drug B" labelClass={labelClass} />
          <LegendItem shape="hexagon" color={ENZYME_ROLE_COLORS.substrate} label="CYP Enzyme" labelClass={labelClass} />
          <LegendItem shape="hexagon" color={TARGET_ACTION_COLORS.inhibitor} label="Protein Target" labelClass={labelClass} />
          <LegendItem shape="hexagon" color="#6b7280" label="Side Effect" labelClass={labelClass} />
        </div>

        {/* Edge types */}
        <div className={`border-t pt-1.5 space-y-1 ${dividerClass}`}>
          <EdgeLegend color={EDGE_STYLES.substrate.stroke} dash={false} label="Substrate" labelClass={labelClass} />
          <EdgeLegend color={EDGE_STYLES.inhibitor.stroke} dash={true} label="Inhibitor" labelClass={labelClass} />
          <EdgeLegend color={EDGE_STYLES.inducer.stroke} dash={true} label="Inducer" labelClass={labelClass} />
          <EdgeLegend color={EDGE_STYLES.targets.stroke} dash={false} label="Targets" labelClass={labelClass} />
          <EdgeLegend color={EDGE_STYLES.conflict.stroke} dash={false} label="Conflict" thick labelClass={labelClass} />
        </div>

        {/* Offline badge */}
        {isOffline && (
          <div className={`mt-2 border-t pt-1.5 ${dividerClass}`}>
            <span className={`text-[7px] font-mono ${isLightTheme ? 'text-amber-700/85' : 'text-yellow-400/60'}`}>
              CYP data only (API offline)
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function LegendItem({ shape, color, label, labelClass }) {
  return (
    <div className="flex items-center gap-1.5">
      <svg width={12} height={12} viewBox="0 0 12 12">
        {shape === 'roundedRect' && <rect x={1} y={3} width={10} height={6} rx={2} fill="none" stroke={color} strokeWidth={1} />}
        {shape === 'hexagon' && <polygon points="6,1 10.5,3.5 10.5,8.5 6,11 1.5,8.5 1.5,3.5" fill="none" stroke={color} strokeWidth={1} />}
        {shape === 'circle' && <circle cx={6} cy={6} r={4.5} fill="none" stroke={color} strokeWidth={1} />}
        {shape === 'diamond' && <polygon points="6,1 11,6 6,11 1,6" fill="none" stroke={color} strokeWidth={1} />}
      </svg>
      <span className={`text-[7px] font-mono ${labelClass}`}>{label}</span>
    </div>
  );
}

function EdgeLegend({ color, dash, label, thick, labelClass }) {
  return (
    <div className="flex items-center gap-1.5">
      <svg width={16} height={6} viewBox="0 0 16 6">
        <line
          x1={0} y1={3} x2={16} y2={3}
          stroke={color}
          strokeWidth={thick ? 2.5 : 1.5}
          strokeDasharray={dash ? '4,2' : 'none'}
        />
      </svg>
      <span className={`text-[7px] font-mono ${labelClass}`}>{label}</span>
    </div>
  );
}
