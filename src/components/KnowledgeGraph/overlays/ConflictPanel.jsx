// ConflictPanel.jsx — Shows detected CYP/target/side-effect conflicts
import React from 'react';
import { AlertTriangle, Beaker, Target, Heart } from 'lucide-react';

const RISK_COLORS = {
  high: '#ef4444',
  moderate: '#eab308',
  low: '#22c55e',
};

export default function ConflictPanel({ conflicts, conflictSummary, interaction }) {
  if (!conflicts || conflicts.length === 0) return null;

  const cypConflicts = conflicts.filter(c => c.type === 'cyp');
  const targetConflicts = conflicts.filter(c => c.type === 'target');
  const seConflicts = conflicts.filter(c => c.type === 'side_effect');
  const overallRisk = conflictSummary?.overall_risk || 'low';

  return (
    <div className="absolute top-12 right-3 z-20 pointer-events-none">
      <div className="w-64 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg shadow-2xl overflow-hidden pointer-events-auto max-h-[60vh] overflow-y-auto scrollbar-thin">
        {/* Header */}
        <div className="px-3 py-2 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5" style={{ color: RISK_COLORS[overallRisk] }} />
            <span className="text-[10px] font-bold text-white/80 uppercase tracking-wider">Conflicts</span>
          </div>
          <span
            className="text-[8px] font-mono px-1.5 py-0.5 rounded uppercase"
            style={{
              color: RISK_COLORS[overallRisk],
              background: `${RISK_COLORS[overallRisk]}15`,
            }}
          >
            {overallRisk} risk
          </span>
        </div>

        <div className="px-3 py-2 space-y-2">
          {/* CYP Conflicts */}
          {cypConflicts.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <Beaker className="w-3 h-3 text-cyan-400" />
                <span className="text-[8px] text-white/40 uppercase tracking-wider">CYP Enzyme Conflicts</span>
              </div>
              {cypConflicts.map((c, i) => (
                <div key={i} className="ml-4 mb-1.5 border-l-2 pl-2 py-0.5" style={{ borderColor: RISK_COLORS[c.risk_level] || '#eab308' }}>
                  <div className="text-[9px] text-white/70 font-mono font-bold">{c.enzyme}</div>
                  <div className="text-[7px] text-white/40">{c.drug1_role} / {c.drug2_role}</div>
                  {c.risk && <div className="text-[7px] text-white/30 mt-0.5">{c.risk}</div>}
                </div>
              ))}
            </div>
          )}

          {/* Shared Targets */}
          {targetConflicts.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <Target className="w-3 h-3 text-purple-400" />
                <span className="text-[8px] text-white/40 uppercase tracking-wider">Shared Targets</span>
              </div>
              {targetConflicts.map((c, i) => (
                <div key={i} className="ml-4 mb-1.5 border-l-2 border-purple-500/30 pl-2 py-0.5">
                  <div className="text-[9px] text-white/70 font-mono font-bold">{c.gene || c.target}</div>
                  <div className="text-[7px] text-white/40">{c.drug1_action} / {c.drug2_action}</div>
                  {c.risk && <div className="text-[7px] text-white/30 mt-0.5">{c.risk}</div>}
                </div>
              ))}
            </div>
          )}

          {/* Shared Side Effects */}
          {seConflicts.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <Heart className="w-3 h-3 text-pink-400" />
                <span className="text-[8px] text-white/40 uppercase tracking-wider">Shared Side Effects</span>
              </div>
              {seConflicts.map((c, i) => (
                <div key={i} className="ml-4 mb-1 text-[8px] text-white/50 font-mono">
                  <span style={{ color: RISK_COLORS[c.combined_risk] || '#6b7280' }}>●</span>{' '}
                  {c.name} {c.organ_system && <span className="text-white/20">({c.organ_system})</span>}
                </div>
              ))}
            </div>
          )}

          {/* Direct interaction info */}
          {interaction?.severity && (
            <div className="border-t border-white/5 pt-2">
              <div className="text-[8px] text-white/40 uppercase tracking-wider mb-1">Known Interaction</div>
              <div className="text-[9px] text-white/60">{interaction.mechanism?.slice(0, 120) || interaction.description?.slice(0, 120)}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
