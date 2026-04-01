import React from 'react';
import { Activity, AlertTriangle, Layers, ShieldAlert, Sparkles } from 'lucide-react';

const FACTOR_ORDER = [
  'pairwise_baseline',
  'enzyme_competition',
  'target_overlap',
  'organ_burden',
  'network_stress',
];

const DEFAULT_WEIGHTS = {
  pairwise_baseline: 0.4,
  enzyme_competition: 0.25,
  target_overlap: 0.15,
  organ_burden: 0.1,
  network_stress: 0.1,
};

const FACTOR_META = {
  pairwise_baseline: {
    label: 'Pairwise Baseline',
    description: 'Combines max and average pairwise interaction risk from all drug pairs.',
    formula: 'min(1, 0.70 x max_pair_risk + 0.30 x average_pair_risk)',
    gradient: 'linear-gradient(90deg, #22d3ee 0%, #3b82f6 100%)',
  },
  enzyme_competition: {
    label: 'Enzyme Competition',
    description: 'Captures CYP substrate, inhibitor, and inducer conflicts as metabolic pressure.',
    formula: 'mean(conflict_scores across detected CYP enzymes)',
    gradient: 'linear-gradient(90deg, #10b981 0%, #f59e0b 100%)',
  },
  target_overlap: {
    label: 'Target Overlap',
    description: 'Measures how often drug pairs share biological targets in the graph.',
    formula: 'min(1, 0.70 x overlap_ratio + 0.30 x normalized_avg_shared_targets)',
    gradient: 'linear-gradient(90deg, #64748b 0%, #38bdf8 100%)',
  },
  organ_burden: {
    label: 'Organ Burden',
    description: 'Aggregates strongest physiological system stress (e.g., CYP450) from edges.',
    formula: 'mean(top 3 impacted system severities)',
    gradient: 'linear-gradient(90deg, #0ea5e9 0%, #f97316 100%)',
  },
  network_stress: {
    label: 'Network Stress',
    description: 'Uses edge density, high-risk density, and hub pressure to gauge systemic load.',
    formula: 'min(1, 0.40 x edge_density + 0.35 x high_risk_density + 0.25 x hub_pressure)',
    gradient: 'linear-gradient(90deg, #06b6d4 0%, #ef4444 100%)',
  },
};

const LEVEL_COLORS = {
  low: 'text-risk-low border-risk-low/40 bg-risk-low/10',
  medium: 'text-risk-medium border-risk-medium/40 bg-risk-medium/10',
  high: 'text-risk-high border-risk-high/40 bg-risk-high/10',
  critical: 'text-risk-critical border-risk-critical/40 bg-risk-critical/10',
};

const clampScore = (value) => Math.max(0, Math.min(1, Number(value) || 0));
const asPct = (value, digits = 0) => `${(clampScore(value) * 100).toFixed(digits)}%`;

function getFactorDetail(key, factors, drugCount) {
  if (key === 'pairwise_baseline') {
    const maxRisk = clampScore(factors.pairwise_baseline?.max_pair_risk);
    const avgRisk = clampScore(factors.pairwise_baseline?.average_pair_risk);
    const pairwiseScore = Math.min(1, (0.7 * maxRisk) + (0.3 * avgRisk));
    return `0.70 x ${asPct(maxRisk)} + 0.30 x ${asPct(avgRisk)} = ${asPct(pairwiseScore)}`;
  }

  if (key === 'enzyme_competition') {
    const conflictCount = Number(factors.enzyme_competition?.conflict_count || 0);
    return `${conflictCount} CYP conflict${conflictCount === 1 ? '' : 's'} detected`;
  }

  if (key === 'target_overlap') {
    const overlapPairs = Number(factors.target_overlap?.pair_count_with_overlap || 0);
    const totalPairs = Number(factors.pairwise_baseline?.total_pairs || ((drugCount * (drugCount - 1)) / 2));
    return `${overlapPairs}/${Math.max(totalPairs, 1)} pair${totalPairs === 1 ? '' : 's'} with shared targets`;
  }

  if (key === 'organ_burden') {
    const topSystem = factors.organ_burden?.top_systems?.[0] || 'N/A';
    return `Top impacted system: ${topSystem}`;
  }

  if (key === 'network_stress') {
    const hubDegree = Number(factors.network_stress?.hub_degree || 0);
    const hubName = factors.network_stress?.hub_drug || 'N/A';
    const hubPressure = clampScore(hubDegree / Math.max(1, drugCount - 1));
    return `Hub pressure ${asPct(hubPressure)} (${hubName})`;
  }

  return '';
}

function FactorBar({ label, description, formula, score = 0, weight = 0, contribution = 0, detail = '', gradient }) {
  const pct = Math.round(clampScore(score) * 100);
  const contributionPct = Math.round(clampScore(contribution) * 100);
  const fillWidth = pct > 0 ? Math.max(pct, 2) : 0;

  return (
    <div className="space-y-2.5 border border-theme bg-theme-primary/60 p-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-theme-secondary">{label}</span>
        <span className="text-[10px] font-mono text-theme-primary">{pct}%</span>
      </div>

      <p className="text-[10px] text-theme-muted leading-relaxed">{description}</p>

      <div className="h-3 w-full bg-theme-secondary border border-theme overflow-hidden">
        <div
          className="h-full transition-all duration-700 ease-out"
          style={{
            width: `${fillWidth}%`,
            minWidth: fillWidth > 0 ? '8px' : '0px',
            background: gradient,
            boxShadow: '0 0 14px rgba(56, 189, 248, 0.35)',
          }}
        />
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] uppercase tracking-wider text-theme-muted">
        <span>Weight {Math.round(clampScore(weight) * 100)}%</span>
        <span>Contribution {contributionPct}%</span>
        {detail && <span className="text-theme-secondary">{detail}</span>}
      </div>

      <p className="text-[10px] text-theme-dim font-mono">{formula}</p>
    </div>
  );
}

export default function PolypharmacyDigitalTwin({ drugs = [], twinResult, isMobile = false }) {
  if (!drugs || drugs.length === 0) {
    return (
      <div className="h-full w-full flex items-center justify-center p-6">
        <div className="text-center border border-theme bg-theme-panel p-6 max-w-lg">
          <Layers className="w-8 h-8 mx-auto mb-3 text-theme-dim" />
          <p className="text-sm text-theme-secondary uppercase tracking-widest">Digital Twin Ready</p>
          <p className="text-xs text-theme-muted mt-2">Select multiple drugs and run analysis to build an N-order toxicity profile.</p>
        </div>
      </div>
    );
  }

  if (!twinResult) {
    return (
      <div className="h-full w-full flex items-center justify-center p-6">
        <div className="text-center border border-theme-accent/30 bg-theme-accent/5 p-6 max-w-lg">
          <Sparkles className="w-8 h-8 mx-auto mb-3 text-theme-accent" />
          <p className="text-sm text-theme-secondary uppercase tracking-widest">Run Polypharmacy Analysis</p>
          <p className="text-xs text-theme-muted mt-2">
            This tab shows N-order digital twin factors after analysis for 3 or more drugs.
          </p>
        </div>
      </div>
    );
  }

  const summary = twinResult.summary || {};
  const factors = twinResult.factors || {};
  const recommendations = twinResult.recommendations || [];
  const toxicityPct = Math.round(clampScore(summary.toxicity_score) * 100);
  const riskLevel = summary.risk_level || 'low';
  const networkHub = factors.network_stress?.hub_drug || 'N/A';
  const topSystem = factors.organ_burden?.top_systems?.[0] || 'N/A';

  const weights = twinResult.explainability?.weights || DEFAULT_WEIGHTS;
  const weightedContributions = twinResult.explainability?.weighted_contributions || {};

  const factorRows = FACTOR_ORDER.map((key) => {
    const score = clampScore(factors[key]?.score || 0);
    const weight = clampScore(weights[key] ?? DEFAULT_WEIGHTS[key] ?? 0);
    const contribution = clampScore(weightedContributions[key] ?? (score * weight));
    const meta = FACTOR_META[key];

    return {
      key,
      score,
      weight,
      contribution,
      label: meta.label,
      description: meta.description,
      formula: meta.formula,
      gradient: meta.gradient,
      detail: getFactorDetail(key, factors, drugs.length),
    };
  });

  const computedScore = factorRows.reduce((acc, row) => acc + row.contribution, 0);
  const equationLine = factorRows
    .map((row) => `${row.weight.toFixed(2)} x ${row.score.toFixed(2)}`)
    .join(' + ');
  const equationDeltaPct = Math.abs(computedScore - clampScore(summary.toxicity_score || 0)) * 100;

  return (
    <div className="h-full overflow-y-auto p-4 md:p-6 bg-theme-primary">
      <div className={`grid gap-4 ${isMobile ? 'grid-cols-1' : 'grid-cols-3'}`}>
        <div className="border border-theme bg-theme-panel p-4">
          <div className="flex items-center gap-2 mb-2">
            <ShieldAlert className="w-4 h-4 text-theme-accent" />
            <p className="text-[10px] uppercase tracking-widest text-theme-muted">Composite Toxicity</p>
          </div>
          <p className="text-2xl font-mono text-theme-primary">{toxicityPct}%</p>
          <span className={`inline-block mt-2 px-2 py-1 text-[10px] uppercase tracking-widest border ${LEVEL_COLORS[riskLevel] || LEVEL_COLORS.low}`}>
            {riskLevel}
          </span>
          <p className="text-[10px] text-theme-muted mt-2">Confidence: {(summary.confidence_tier || 'heuristic').replace('-', ' ')}</p>
        </div>

        <div className="border border-theme bg-theme-panel p-4">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-theme-accent" />
            <p className="text-[10px] uppercase tracking-widest text-theme-muted">Network Hub</p>
          </div>
          <p className="text-sm font-mono text-theme-secondary truncate">{networkHub}</p>
          <p className="text-[10px] text-theme-muted mt-2">Highest connected medication in current tension graph.</p>
        </div>

        <div className="border border-theme bg-theme-panel p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-theme-accent" />
            <p className="text-[10px] uppercase tracking-widest text-theme-muted">Top Organ Burden</p>
          </div>
          <p className="text-sm font-mono text-theme-secondary truncate">{topSystem}</p>
          <p className="text-[10px] text-theme-muted mt-2">Most impacted physiological bucket from current combination.</p>
        </div>
      </div>

      <div className="mt-4 border border-theme bg-theme-panel p-4">
        <p className="text-[10px] uppercase tracking-widest text-theme-muted mb-2">Transparent Scoring Math</p>
        <p className="text-xs text-theme-secondary leading-relaxed">
          Composite toxicity is a weighted sum of factor scores, so the model is inspectable instead of a black box.
        </p>
        <div className="mt-3 border border-theme bg-theme-primary/70 p-3">
          <p className="text-[11px] font-mono text-theme-primary break-all">
            {equationLine} = {computedScore.toFixed(3)} ({Math.round(clampScore(computedScore) * 100)}%)
          </p>
          <p className="text-[10px] text-theme-muted mt-2">
            Reported score: {toxicityPct}% | Recomputed delta: {equationDeltaPct.toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="mt-4 border border-theme bg-theme-panel p-4">
        <p className="text-[10px] uppercase tracking-widest text-theme-muted mb-3">Factor Breakdown</p>
        <div className="space-y-3">
          {factorRows.map((row) => (
            <FactorBar
              key={row.key}
              label={row.label}
              description={row.description}
              formula={row.formula}
              score={row.score}
              weight={row.weight}
              contribution={row.contribution}
              detail={row.detail}
              gradient={row.gradient}
            />
          ))}
        </div>
      </div>

      <div className="mt-4 border border-theme bg-theme-panel p-4">
        <p className="text-[10px] uppercase tracking-widest text-theme-muted mb-3">Clinical Recommendations</p>
        {recommendations.length === 0 ? (
          <p className="text-xs text-theme-muted">No recommendations generated.</p>
        ) : (
          <ul className="space-y-2">
            {recommendations.map((item, idx) => (
              <li key={`rec-${idx}`} className="text-xs text-theme-secondary leading-relaxed border-l-2 border-theme-accent/50 pl-3">
                {item}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
