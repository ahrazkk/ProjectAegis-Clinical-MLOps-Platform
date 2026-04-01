// MechanismLegend.jsx — Legend explaining node/edge types
import React from 'react';
import { DRUG_COLORS, ENZYME_ROLE_COLORS, TARGET_ACTION_COLORS, EDGE_STYLES } from '../mechanismGraphEngine';

export default function MechanismLegend({ isOffline }) {
  return (
    <div className="absolute bottom-3 left-3 z-20 pointer-events-none">
      <div className="bg-black/75 backdrop-blur-md border border-white/8 rounded-lg shadow-xl px-3 py-2 pointer-events-auto min-w-[140px]">
        <div className="text-[8px] text-white/40 uppercase tracking-wider mb-1.5 font-mono">Legend</div>

        {/* Node types */}
        <div className="space-y-1 mb-2">
          <LegendItem shape="hexagon" color={DRUG_COLORS.A} label="Drug A" />
          <LegendItem shape="hexagon" color={DRUG_COLORS.B} label="Drug B" />
          <LegendItem shape="hexagon" color={ENZYME_ROLE_COLORS.substrate} label="CYP Enzyme" />
          <LegendItem shape="hexagon" color={TARGET_ACTION_COLORS.inhibitor} label="Protein Target" />
          <LegendItem shape="hexagon" color="#6b7280" label="Side Effect" />
        </div>

        {/* Edge types */}
        <div className="border-t border-white/5 pt-1.5 space-y-1">
          <EdgeLegend color={EDGE_STYLES.substrate.stroke} dash={false} label="Substrate" />
          <EdgeLegend color={EDGE_STYLES.inhibitor.stroke} dash={true} label="Inhibitor" />
          <EdgeLegend color={EDGE_STYLES.inducer.stroke} dash={true} label="Inducer" />
          <EdgeLegend color={EDGE_STYLES.targets.stroke} dash={false} label="Targets" />
          <EdgeLegend color={EDGE_STYLES.conflict.stroke} dash={false} label="Conflict" thick />
        </div>

        {/* Offline badge */}
        {isOffline && (
          <div className="mt-2 border-t border-white/5 pt-1.5">
            <span className="text-[7px] font-mono text-yellow-400/60">CYP data only (API offline)</span>
          </div>
        )}
      </div>
    </div>
  );
}

function LegendItem({ shape, color, label }) {
  return (
    <div className="flex items-center gap-1.5">
      <svg width={12} height={12} viewBox="0 0 12 12">
        {shape === 'roundedRect' && <rect x={1} y={3} width={10} height={6} rx={2} fill="none" stroke={color} strokeWidth={1} />}
        {shape === 'hexagon' && <polygon points="6,1 10.5,3.5 10.5,8.5 6,11 1.5,8.5 1.5,3.5" fill="none" stroke={color} strokeWidth={1} />}
        {shape === 'circle' && <circle cx={6} cy={6} r={4.5} fill="none" stroke={color} strokeWidth={1} />}
        {shape === 'diamond' && <polygon points="6,1 11,6 6,11 1,6" fill="none" stroke={color} strokeWidth={1} />}
      </svg>
      <span className="text-[7px] text-white/40 font-mono">{label}</span>
    </div>
  );
}

function EdgeLegend({ color, dash, label, thick }) {
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
      <span className="text-[7px] text-white/40 font-mono">{label}</span>
    </div>
  );
}
