// NodeTooltip.jsx — Hover tooltip with biological details
import React from 'react';
import { ENZYME_ROLE_COLORS, TARGET_ACTION_COLORS, DRUG_COLORS } from '../mechanismGraphEngine';

export default function NodeTooltip({ node, x, y, containerRect }) {
  if (!node) return null;

  // Clamp tooltip position within bounds
  const tooltipW = 180;
  const tooltipH = 80;
  let tx = x + 20;
  let ty = y - 10;
  if (containerRect) {
    if (tx + tooltipW > containerRect.width) tx = x - tooltipW - 10;
    if (ty + tooltipH > containerRect.height) ty = containerRect.height - tooltipH - 10;
    if (ty < 0) ty = 10;
  }

  return (
    <div
      className="absolute z-30 pointer-events-none"
      style={{ left: tx, top: ty }}
    >
      <div className="bg-black/90 backdrop-blur-md border border-white/15 rounded-md shadow-xl px-2.5 py-2 min-w-[160px] max-w-[200px]">
        {/* Name */}
        <div className="text-[10px] font-bold text-white/90 font-mono">{node.label}</div>

        {/* Type badge */}
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="text-[7px] uppercase text-white/30 tracking-wider">{node.type.replace('_', ' ')}</span>
          {node.isConflict && (
            <span className="text-[6px] px-1 py-0.5 rounded bg-red-500/15 text-red-400 font-mono uppercase">conflict</span>
          )}
        </div>

        {/* Type-specific info */}
        {node.type === 'drug' && (
          <div className="mt-1 space-y-0.5">
            {node.therapeutic_class && (
              <div className="text-[8px] text-white/40">{node.therapeutic_class}</div>
            )}
            <div className="text-[7px] text-white/25">Slot: {node.slot || '—'}</div>
          </div>
        )}

        {node.type === 'enzyme' && (
          <div className="mt-1 space-y-0.5">
            {Object.entries(node.enzymeRoles || {}).map(([slot, role]) => (
              <div key={slot} className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: slot === 'A' ? DRUG_COLORS.A : DRUG_COLORS.B }} />
                <span className="text-[7px] text-white/50 font-mono">{role}</span>
              </div>
            ))}
          </div>
        )}

        {node.type === 'target' && (
          <div className="mt-1 space-y-0.5">
            {node.gene && <div className="text-[8px] text-white/50 font-mono">Gene: {node.gene}</div>}
            <div className="text-[8px]" style={{ color: TARGET_ACTION_COLORS[node.action] || '#6b7280' }}>
              Action: {node.action || 'unknown'}
            </div>
          </div>
        )}

        {node.type === 'side_effect' && (
          <div className="mt-1 space-y-0.5">
            {node.organ_system && (
              <div className="text-[8px] text-white/40">System: {node.organ_system}</div>
            )}
            <div className="text-[8px] text-white/30">
              Severity: {typeof node.severity === 'number' ? (node.severity * 100).toFixed(0) + '%' : node.severity || 'unknown'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
