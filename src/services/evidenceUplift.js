const PRIORITY_WEIGHT = {
  high: 3,
  medium: 2,
  low: 1,
};

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function parseRatio(value) {
  if (value === null || value === undefined) return null;

  if (typeof value === 'string') {
    const normalized = value.trim();
    if (!normalized) return null;

    const hasPercent = normalized.includes('%');
    const parsed = Number(normalized.replace('%', ''));
    if (!Number.isFinite(parsed)) return null;

    if (hasPercent || parsed > 1) return clamp01(parsed / 100);
    return clamp01(parsed);
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  if (parsed > 1) return clamp01(parsed / 100);
  return clamp01(parsed);
}

export function normalizeDisagreementLevel(level) {
  const normalized = String(level || 'none').toLowerCase();
  if (normalized === 'moderate') return 'medium';
  if (normalized === 'severe') return 'high';
  if (normalized === 'minimal') return 'low';
  if (['none', 'low', 'medium', 'high'].includes(normalized)) return normalized;
  return 'none';
}

export function normalizeConfidenceBand(confidenceBand) {
  const normalized = String(confidenceBand || 'unknown').toLowerCase();
  if (normalized === 'moderate') return 'medium';
  if (normalized === 'strong') return 'high';
  if (normalized === 'weak') return 'low';
  if (['high', 'medium', 'low'].includes(normalized)) return normalized;
  return 'unknown';
}

function parseDateCandidate(value) {
  if (!value || typeof value !== 'string') return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed;
}

export function deriveEvidenceFreshness(interactionEvidence) {
  const candidates = [
    interactionEvidence?.faers_data?.last_updated,
    interactionEvidence?.faers_data?.freshness?.last_updated,
    interactionEvidence?.faers_data?.metadata?.last_updated,
  ];

  const parsed = candidates.map(parseDateCandidate).find(Boolean);
  if (!parsed) {
    return { daysOld: null, label: 'unknown' };
  }

  const daysOld = Math.max(0, Math.round((Date.now() - parsed.getTime()) / 86400000));
  if (daysOld <= 30) return { daysOld, label: 'fresh' };
  if (daysOld <= 180) return { daysOld, label: 'recent' };
  return { daysOld, label: 'stale' };
}

export function getCoverageRatio(summary) {
  const sourceCoverageRatio = parseRatio(summary?.source_coverage?.ratio);
  if (Number.isFinite(sourceCoverageRatio)) return sourceCoverageRatio;

  const primaryCoverageRatio = parseRatio(summary?.primary_source_coverage?.ratio);
  if (Number.isFinite(primaryCoverageRatio)) return primaryCoverageRatio;

  return null;
}

function mean(values) {
  if (!Array.isArray(values) || values.length === 0) return null;
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (!finiteValues.length) return null;
  const total = finiteValues.reduce((sum, value) => sum + value, 0);
  return total / finiteValues.length;
}

function extractSupportSignal(entry) {
  if (!entry || typeof entry !== 'object') return null;

  const candidates = [
    entry.support,
    entry.strength,
    entry.confidence,
    entry.score,
    entry.weighted_support,
  ];

  for (const candidate of candidates) {
    const parsed = parseRatio(candidate);
    if (Number.isFinite(parsed)) return parsed;
  }

  return null;
}

export function classifyEvidenceSource(raw) {
  const sourceRaw = String(raw || '').toLowerCase();
  if (!sourceRaw) return 'modelSignals';

  if (sourceRaw.includes('kg') || sourceRaw.includes('graph') || sourceRaw.includes('twosides')) {
    return 'knowledgeGraph';
  }

  if (sourceRaw.includes('pubmed') || sourceRaw.includes('literature') || sourceRaw.includes('paper') || sourceRaw.includes('corpus')) {
    return 'literature';
  }

  if (sourceRaw.includes('faers') || sourceRaw.includes('openfda') || sourceRaw.includes('real-world')) {
    return 'realWorld';
  }

  if (sourceRaw.includes('cyp') || sourceRaw.includes('mechan')) {
    return 'mechanistic';
  }

  return 'modelSignals';
}

export function countIndependentEvidenceSources(interactionEvidence) {
  const evidenceChain = Array.isArray(interactionEvidence?.evidence_chain)
    ? interactionEvidence.evidence_chain
    : [];

  const categories = new Set();

  evidenceChain.forEach((entry) => {
    categories.add(classifyEvidenceSource(entry?.source || entry?.source_name));
  });

  const faersReports = Number(interactionEvidence?.faers_data?.total_reports || 0);
  const hasFaersReactions = Array.isArray(interactionEvidence?.faers_data?.top_reactions)
    && interactionEvidence.faers_data.top_reactions.length > 0;
  if (faersReports > 0 || hasFaersReactions) {
    categories.add('realWorld');
  }

  if (Array.isArray(interactionEvidence?.common_side_effects) && interactionEvidence.common_side_effects.length > 0) {
    categories.add('clinicalSignals');
  }

  if (Array.isArray(interactionEvidence?.affected_systems) && interactionEvidence.affected_systems.length > 0) {
    categories.add('modelSignals');
  }

  return categories.size;
}

function priorityForCoverage(coverageRatio, independentSourceCount) {
  if (Number.isFinite(coverageRatio) && coverageRatio < 0.4) return 'high';
  if (Number.isFinite(independentSourceCount) && independentSourceCount < 2) return 'high';
  return 'medium';
}

function deriveFallbackSupport({
  evidenceNodeCount,
  sideEffectSignals,
  independentSourceCount,
  affectedSystemCount,
  hasRealWorldEvidence,
}) {
  const nodeTerm = Math.min(0.34, Math.max(0, evidenceNodeCount) * 0.08);
  const signalTerm = Math.min(0.24, Math.max(0, sideEffectSignals) * 0.035);
  const sourceTerm = Math.min(0.2, Math.max(0, independentSourceCount) * 0.05);
  const affectedTerm = Math.min(0.1, Math.max(0, affectedSystemCount) * 0.025);
  const realWorldTerm = hasRealWorldEvidence ? 0.1 : 0;

  return clamp01(0.14 + nodeTerm + signalTerm + sourceTerm + affectedTerm + realWorldTerm);
}

function deriveFallbackCoverage({
  independentSourceCount,
  evidenceNodeCount,
  hasRealWorldEvidence,
  hasMechanisticDetail,
}) {
  const sourceTerm = Math.min(0.56, Math.max(0, independentSourceCount) * 0.16);
  const depthTerm = Math.min(0.2, Math.max(0, evidenceNodeCount) * 0.045);
  const realWorldTerm = hasRealWorldEvidence ? 0.12 : 0;
  const mechanisticTerm = hasMechanisticDetail ? 0.08 : 0;

  return clamp01(0.06 + sourceTerm + depthTerm + realWorldTerm + mechanisticTerm);
}

function deriveFallbackUncertainty({
  disagreementLevel,
  coverageRatio,
  independentSourceCount,
  evidenceNodeCount,
  freshnessLabel,
  sideEffectSignals,
}) {
  const normalizedDisagreement = normalizeDisagreementLevel(disagreementLevel);
  const normalizedFreshness = String(freshnessLabel || 'unknown').toLowerCase();

  const disagreementTerm = normalizedDisagreement === 'high'
    ? 0.34
    : normalizedDisagreement === 'medium'
      ? 0.22
      : normalizedDisagreement === 'low'
        ? 0.11
        : 0.06;

  const sourcePenalty = independentSourceCount <= 1
    ? 0.24
    : independentSourceCount === 2
      ? 0.16
      : independentSourceCount === 3
        ? 0.1
        : 0.06;

  const depthPenalty = evidenceNodeCount === 0
    ? 0.18
    : evidenceNodeCount < 3
      ? 0.1
      : evidenceNodeCount < 6
        ? 0.06
        : 0.03;

  const freshnessPenalty = normalizedFreshness === 'stale'
    ? 0.17
    : normalizedFreshness === 'recent'
      ? 0.1
      : normalizedFreshness === 'fresh'
        ? 0.05
        : 0.12;

  const coveragePenalty = Number.isFinite(coverageRatio)
    ? (1 - clamp01(coverageRatio)) * 0.28
    : 0.16;

  const signalPenalty = sideEffectSignals === 0
    ? 0.08
    : sideEffectSignals < 2
      ? 0.05
      : 0.02;

  return clamp01(
    0.04
    + disagreementTerm
    + sourcePenalty
    + depthPenalty
    + freshnessPenalty
    + coveragePenalty
    + signalPenalty
  );
}

export function buildConfidenceUpliftPlan({
  scopeLabel = 'this system',
  coverageRatio,
  hasRealWorldEvidence,
  disagreementLevel,
  freshnessLabel,
  hasMechanisticDetail,
  sideEffectSignalCount,
  independentSourceCount,
  evidenceNodeCount,
  maxSteps = 5,
}) {
  const plan = [];
  const normalizedCoverage = Number.isFinite(coverageRatio) ? clamp01(coverageRatio) : null;
  const normalizedDisagreement = normalizeDisagreementLevel(disagreementLevel);
  const normalizedFreshness = String(freshnessLabel || 'unknown').toLowerCase();
  const normalizedSideEffectSignals = Number.isFinite(sideEffectSignalCount)
    ? Math.max(0, sideEffectSignalCount)
    : 0;
  const normalizedSourceCount = Number.isFinite(independentSourceCount)
    ? Math.max(0, independentSourceCount)
    : 0;
  const normalizedEvidenceNodeCount = Number.isFinite(evidenceNodeCount)
    ? Math.max(0, evidenceNodeCount)
    : 0;

  if (!hasRealWorldEvidence) {
    plan.push({
      key: 'real-world-enrichment',
      action: `Add real-world safety evidence (FAERS/OpenFDA) for ${scopeLabel}`,
      expectedGain: 0.14,
      priority: 'high',
      currentState: 'No direct FAERS/OpenFDA signal is currently linked.',
    });
  }

  if (
    normalizedCoverage === null
    || normalizedCoverage < 0.6
    || normalizedSourceCount < 3
  ) {
    plan.push({
      key: 'source-coverage',
      action: 'Increase source coverage with at least one additional independent source',
      expectedGain: 0.11,
      priority: priorityForCoverage(normalizedCoverage, normalizedSourceCount),
      currentState: normalizedCoverage === null
        ? `Independent source count: ${normalizedSourceCount}`
        : `Coverage ${(normalizedCoverage * 100).toFixed(0)}%, independent sources ${normalizedSourceCount}`,
    });
  }

  if (normalizedDisagreement === 'high' || normalizedDisagreement === 'medium') {
    plan.push({
      key: 'resolve-disagreement',
      action: 'Resolve cross-source disagreement with targeted literature verification',
      expectedGain: normalizedDisagreement === 'high' ? 0.13 : 0.08,
      priority: 'high',
      currentState: `Current disagreement level is ${normalizedDisagreement}.`,
    });
  }

  if (normalizedFreshness === 'stale' || normalizedFreshness === 'unknown') {
    plan.push({
      key: 'refresh-recency',
      action: 'Refresh evidence recency to reduce temporal uncertainty',
      expectedGain: 0.05,
      priority: 'medium',
      currentState: `Evidence freshness is ${normalizedFreshness}.`,
    });
  }

  if (!hasMechanisticDetail) {
    plan.push({
      key: 'mechanistic-coverage',
      action: `Add mechanistic pathway rationale for ${scopeLabel}`,
      expectedGain: 0.07,
      priority: 'medium',
      currentState: 'Mechanistic pathway detail is missing or sparse.',
    });
  }

  if (normalizedSideEffectSignals < 2) {
    plan.push({
      key: 'clinical-signal-density',
      action: 'Increase side-effect signal density from validated clinical sources',
      expectedGain: 0.06,
      priority: 'low',
      currentState: `Only ${normalizedSideEffectSignals} validated side-effect signal(s) detected.`,
    });
  }

  if (normalizedEvidenceNodeCount < 3) {
    plan.push({
      key: 'evidence-depth',
      action: 'Increase evidence-chain depth with additional corroborating nodes',
      expectedGain: 0.05,
      priority: 'medium',
      currentState: `Evidence chain currently has ${normalizedEvidenceNodeCount} node(s).`,
    });
  }

  const upliftPlan = plan
    .sort((a, b) => {
      if (b.expectedGain !== a.expectedGain) {
        return b.expectedGain - a.expectedGain;
      }
      return (PRIORITY_WEIGHT[b.priority] || 0) - (PRIORITY_WEIGHT[a.priority] || 0);
    })
    .slice(0, Math.max(1, maxSteps));

  const potentialGain = clamp01(
    upliftPlan.reduce((sum, step) => sum + (Number(step.expectedGain) || 0), 0)
  );

  return {
    upliftPlan,
    potentialGain,
    readinessScore: clamp01(1 - potentialGain),
  };
}

export function buildInteractionEvidenceUplift(interactionEvidence) {
  const summary = interactionEvidence?.evidence_summary && typeof interactionEvidence.evidence_summary === 'object'
    ? interactionEvidence.evidence_summary
    : {};

  const weightedSupportSummary = parseRatio(summary?.weighted_support_score);
  const weightedUncertaintySummary = parseRatio(summary?.weighted_uncertainty_score);
  const confidenceBand = normalizeConfidenceBand(summary?.confidence_band);
  const disagreementLevel = normalizeDisagreementLevel(summary?.disagreement?.level);
  const coverageRatioSummary = getCoverageRatio(summary);

  const freshness = deriveEvidenceFreshness(interactionEvidence);
  const faersReports = Number(interactionEvidence?.faers_data?.total_reports || 0);
  const hasFaersReactions = Array.isArray(interactionEvidence?.faers_data?.top_reactions)
    && interactionEvidence.faers_data.top_reactions.length > 0;
  const hasRealWorldEvidence = faersReports > 0 || hasFaersReactions;

  const evidenceChain = Array.isArray(interactionEvidence?.evidence_chain)
    ? interactionEvidence.evidence_chain
    : [];

  const sideEffectSignals = Array.isArray(interactionEvidence?.common_side_effects)
    ? interactionEvidence.common_side_effects.length
    : 0;
  const evidenceNodeCount = evidenceChain.length;
  const affectedSystemCount = Array.isArray(interactionEvidence?.affected_systems)
    ? interactionEvidence.affected_systems.length
    : 0;
  const independentSourceCount = countIndependentEvidenceSources(interactionEvidence);

  const hasMechanisticDetail = evidenceChain.some((entry) => {
      const sourceType = classifyEvidenceSource(entry?.source || entry?.source_name);
      return sourceType === 'mechanistic';
    });

  const supportSignals = evidenceChain
    .map(extractSupportSignal)
    .filter((value) => Number.isFinite(value));

  const fallbackSupport = supportSignals.length > 0
    ? mean(supportSignals)
    : deriveFallbackSupport({
      evidenceNodeCount,
      sideEffectSignals,
      independentSourceCount,
      affectedSystemCount,
      hasRealWorldEvidence,
    });

  const resolvedCoverageRatio = Number.isFinite(coverageRatioSummary)
    ? coverageRatioSummary
    : deriveFallbackCoverage({
      independentSourceCount,
      evidenceNodeCount,
      hasRealWorldEvidence,
      hasMechanisticDetail,
    });

  const resolvedSupport = Number.isFinite(weightedSupportSummary)
    ? weightedSupportSummary
    : fallbackSupport;

  const resolvedUncertainty = Number.isFinite(weightedUncertaintySummary)
    ? weightedUncertaintySummary
    : deriveFallbackUncertainty({
      disagreementLevel,
      coverageRatio: resolvedCoverageRatio,
      independentSourceCount,
      evidenceNodeCount,
      freshnessLabel: freshness.label,
      sideEffectSignals,
    });

  const { upliftPlan, potentialGain, readinessScore } = buildConfidenceUpliftPlan({
    scopeLabel: 'this interaction',
    coverageRatio: resolvedCoverageRatio,
    hasRealWorldEvidence,
    disagreementLevel,
    freshnessLabel: freshness.label,
    hasMechanisticDetail,
    sideEffectSignalCount: sideEffectSignals,
    independentSourceCount,
    evidenceNodeCount,
    maxSteps: 5,
  });

  return {
    metrics: {
      weightedSupport: clamp01(resolvedSupport),
      weightedUncertainty: clamp01(resolvedUncertainty),
      confidenceBand,
      disagreementLevel,
      coverageRatio: resolvedCoverageRatio,
      freshnessLabel: freshness.label,
      freshnessDaysOld: freshness.daysOld,
      hasRealWorldEvidence,
      faersReports,
      independentSourceCount,
      sideEffectSignals,
      evidenceNodeCount,
    },
    upliftPlan,
    potentialGain,
    readinessScore,
  };
}
