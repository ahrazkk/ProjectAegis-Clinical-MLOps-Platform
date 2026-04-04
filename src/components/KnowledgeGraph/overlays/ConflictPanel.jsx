// ConflictPanel.jsx — Shows detected CYP/target/side-effect conflicts
import React from 'react';
import { AlertTriangle, Beaker, Target, Heart } from 'lucide-react';

const RISK_COLORS = {
  high: '#ef4444',
  moderate: '#eab308',
  low: '#22c55e',
  unknown: '#94a3b8',
};

function buildCausalitySteps(conflict) {
  if (!conflict) return [];

  if (conflict.type === 'cyp') {
    const riskLevel = String(conflict.risk_level || 'unknown').toLowerCase();
    return [
      `${conflict.enzyme || 'Shared CYP'} appears in both drug metabolism pathways.`,
      `Role overlap detected: Drug A (${conflict.drug1_role || 'unknown'}) vs Drug B (${conflict.drug2_role || 'unknown'}).`,
      conflict.risk || 'This role combination can alter exposure for one or both drugs.',
      riskLevel === 'high'
        ? 'Clinical implication: monitor for toxicity or therapeutic failure and consider regimen adjustment.'
        : 'Clinical implication: monitor response and consider dose/monitoring adjustments if symptoms emerge.',
    ];
  }

  if (conflict.type === 'target') {
    return [
      `${conflict.gene || conflict.target || 'Target'} is affected by both drugs.`,
      `Action profile: Drug A (${conflict.drug1_action || 'unknown'}) and Drug B (${conflict.drug2_action || 'unknown'}).`,
      conflict.risk || 'Dual modulation of this target may amplify or counter expected pharmacodynamic response.',
      'Clinical implication: watch for exaggerated or blunted therapeutic effects tied to this pathway.',
    ];
  }

  if (conflict.type === 'side_effect') {
    return [
      `${conflict.name || 'Side effect'} is linked to both drugs.`,
      `${conflict.organ_system ? `${conflict.organ_system} system involvement detected.` : 'Multi-source adverse event overlap detected.'}`,
      `Combined risk level is ${conflict.combined_risk || 'unknown'}.`,
      'Clinical implication: monitor symptom burden and evaluate whether one component can be substituted.',
    ];
  }

  return [
    'Conflict signal detected for this node.',
    conflict.risk || 'Multiple risk indicators overlap for this mechanism.',
  ];
}

export default function ConflictPanel({ conflicts, conflictSummary, interaction }) {
  if (!conflicts || conflicts.length === 0) return null;

  const cypConflicts = conflicts.filter(c => c.type === 'cyp');
  const targetConflicts = conflicts.filter(c => c.type === 'target');
  const seConflicts = conflicts.filter(c => c.type === 'side_effect');
  const overallRisk = conflictSummary?.overall_risk || 'low';

  const ConflictBlock = ({ conflict, accentColor, title }) => {
    const steps = buildCausalitySteps(conflict);
    return (
      <div className="ml-4 mb-2 border-l-2 pl-2 py-1" style={{ borderColor: accentColor }}>
        <div className="text-[9px] text-white/75 font-mono font-bold">{title}</div>
        <div className="text-[7px] text-white/45 mt-0.5">
          {conflict.drug1_role && conflict.drug2_role
            ? `${conflict.drug1_role} / ${conflict.drug2_role}`
            : conflict.drug1_action && conflict.drug2_action
              ? `${conflict.drug1_action} / ${conflict.drug2_action}`
              : conflict.combined_risk
                ? `combined ${conflict.combined_risk}`
                : 'mechanism overlap'}
        </div>

        <div className="mt-1.5 space-y-1">
          {steps.slice(0, 4).map((step, index) => (
            <div key={`${conflict.id || title}-${index}`} className="text-[7px] text-white/35 leading-relaxed">
              {index + 1}. {step}
            </div>
          ))}
        </div>
      </div>
    );
  };

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
                <ConflictBlock
                  key={c.id || `cyp-${i}`}
                  conflict={c}
                  accentColor={RISK_COLORS[c.risk_level] || '#eab308'}
                  title={c.enzyme || 'CYP conflict'}
                />
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
                <ConflictBlock
                  key={c.id || `target-${i}`}
                  conflict={c}
                  accentColor="rgba(168,85,247,0.55)"
                  title={c.gene || c.target || 'Target overlap'}
                />
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
                <ConflictBlock
                  key={c.id || `se-${i}`}
                  conflict={c}
                  accentColor={RISK_COLORS[c.combined_risk] || '#6b7280'}
                  title={`${c.name || 'Side effect'}${c.organ_system ? ` (${c.organ_system})` : ''}`}
                />
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
