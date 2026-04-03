// bodyMapDataService.js — Data enrichment pipeline for Body Map V2
// Merges multiple data sources into a unified per-organ severity + detail model.
// Sources: affectedSystems (prediction), drugInfoCache (side effects), interactionEvidence, CYP data.

import { normalizeOrganSystem, ORGAN_SYSTEMS } from './organRegistry';
import { getOfflineCYPProfile } from '../KnowledgeGraph/cypFallbackData';
import { buildConfidenceUpliftPlan as buildSharedConfidenceUpliftPlan } from '../../services/evidenceUplift';

// ─── Main enrichment function ──────────────────────────────────────────────
// Returns: { organs, drugPathway, cypLiverLoad, dataQuality, isEmpty }
export function enrichBodyMapData({ affectedSystems, drugs, drugInfoCache, interactionEvidence, polypharmacyResult, result }) {
  const organs = buildBaseOrganMap();
  let dataQuality = 'none'; // none | prediction | enriched | full

  // Layer 1: Prediction-based affected_systems (lowest fidelity)
  if (affectedSystems && typeof affectedSystems === 'object') {
    applyAffectedSystems(organs, affectedSystems);
    dataQuality = 'prediction';
  }

  // Layer 2: Polypharmacy body_map (if available)
  if (polypharmacyResult?.body_map) {
    applyAffectedSystems(organs, polypharmacyResult.body_map);
    dataQuality = 'prediction';
  }

  // Layer 2.5: Pairwise prediction fallback (handles generic systems like "Systemic/Categorical")
  if (result) {
    const enriched = enrichFromPairwiseResult(organs, result);
    if (enriched && dataQuality === 'none') dataQuality = 'prediction';
  }

  // Layer 3: Drug info cache — real side effects with organ_system
  if (drugInfoCache && drugs?.length) {
    const enriched = enrichFromDrugInfo(organs, drugs, drugInfoCache);
    if (enriched) dataQuality = 'enriched';
  }

  // Layer 4: Interaction evidence — common side effects, FAERS data
  if (interactionEvidence) {
    const enriched = enrichFromEvidence(organs, interactionEvidence);
    if (enriched) dataQuality = 'full';
  }

  // Layer 5: CYP liver load computation
  const cypLiverLoad = computeCYPLiverLoad(drugs || []);
  if (cypLiverLoad > 0) {
    const liverOrgan = organs.liver;
    liverOrgan.severity = Math.max(liverOrgan.severity, cypLiverLoad * 0.8);
    liverOrgan.cypLoad = cypLiverLoad;
    liverOrgan.details.push({
      type: 'cyp',
      label: 'CYP450 Hepatic Load',
      description: cypLiverLoad > 0.7
        ? 'High metabolic burden — multiple CYP substrates/inhibitors'
        : cypLiverLoad > 0.4
          ? 'Moderate metabolic burden — shared CYP pathways'
          : 'Low metabolic burden',
      severity: cypLiverLoad,
    });
  }

  // Compute drug pathway activity
  const drugPathway = computeDrugPathway(organs, drugs);

  // Build unified per-system evidence profiles for BodyMap UI panels
  const systemEvidence = buildSystemEvidence(organs, interactionEvidence, drugs || []);

  // Is the whole thing empty?
  const isEmpty = !drugs?.length || Object.values(organs).every(o => o.severity === 0);

  return { organs, drugPathway, cypLiverLoad, dataQuality, isEmpty, systemEvidence };
}

// ─── Build base organ map ──────────────────────────────────────────────────
function buildBaseOrganMap() {
  const map = {};
  for (const key of Object.keys(ORGAN_SYSTEMS)) {
    map[key] = {
      id: key,
      severity: 0,
      sideEffects: [],     // [{ name, severity, frequency, drug, source }]
      drugContributions: [], // [{ drug, contribution }]
      details: [],          // [{ type, label, description, severity }]
      faersCount: 0,
      cypLoad: 0,
    };
  }
  return map;
}

// ─── Layer 1: Apply severity map ───────────────────────────────────────────
function applyAffectedSystems(organs, systemMap) {
  for (const [key, severity] of Object.entries(systemMap)) {
    const normalizedSeverity = clamp01(
      typeof severity === 'number' ? severity : parseSeverity(severity)
    );
    const organId = normalizeOrganSystem(key);
    if (organId && organs[organId]) {
      organs[organId].severity = Math.max(organs[organId].severity, normalizedSeverity);
      continue;
    }

    // Fallback for generic or non-standard system labels from pairwise predictions.
    const fallbackWeights = mapGenericSystemToOrgans(key);
    if (!fallbackWeights) continue;

    for (const [fallbackOrgan, weight] of Object.entries(fallbackWeights)) {
      if (!organs[fallbackOrgan]) continue;
      organs[fallbackOrgan].severity = Math.max(
        organs[fallbackOrgan].severity,
        clamp01(normalizedSeverity * weight)
      );
    }
  }
}

function mapGenericSystemToOrgans(rawSystem) {
  const text = String(rawSystem || '').toLowerCase();
  if (!text) return null;

  if (text.includes('cyp') || text.includes('metabolic') || text.includes('hepatic') || text.includes('liver')) {
    return { liver: 1.0, gi: 0.55 };
  }

  if (text.includes('cardio') || text.includes('vascular') || text.includes('qt') || text.includes('heart')) {
    return { heart: 1.0, blood: 0.72 };
  }

  if (text.includes('neuro') || text.includes('cns') || text.includes('seroton')) {
    return { brain: 1.0 };
  }

  if (text.includes('resp') || text.includes('pulmo') || text.includes('lung')) {
    return { lungs: 1.0 };
  }

  if (text.includes('renal') || text.includes('kidney')) {
    return { kidney: 1.0 };
  }

  if (text.includes('gastro') || text.includes('gi')) {
    return { gi: 1.0, liver: 0.5 };
  }

  if (text.includes('heme') || text.includes('blood') || text.includes('platelet') || text.includes('coag')) {
    return { blood: 1.0 };
  }

  if (text.includes('systemic') || text.includes('categorical') || text.includes('interaction') || text.includes('unknown')) {
    return {
      liver: 0.82,
      kidney: 0.74,
      heart: 0.72,
      blood: 0.7,
      gi: 0.66,
      brain: 0.6,
    };
  }

  return null;
}

function enrichFromPairwiseResult(organs, result) {
  if (!result || typeof result !== 'object') return false;

  const riskScore = clamp01(Number(result?.risk_score));
  const affectedSystems = Array.isArray(result?.affected_systems)
    ? result.affected_systems
    : [];

  const before = Object.values(organs).some((organ) => (organ?.severity || 0) > 0);

  if (affectedSystems.length > 0) {
    const mapped = {};
    affectedSystems.forEach((entry) => {
      const systemName = typeof entry === 'string' ? entry : (entry?.system || entry?.target_system);
      if (!systemName) return;

      const entrySeverity = clamp01(
        Number.isFinite(Number(entry?.severity))
          ? Number(entry.severity)
          : riskScore > 0
            ? Math.max(0.3, riskScore)
            : 0.45
      );

      mapped[systemName] = Math.max(mapped[systemName] || 0, entrySeverity);
    });

    if (Object.keys(mapped).length > 0) {
      applyAffectedSystems(organs, mapped);
    }
  }

  const afterAffectedSystems = Object.values(organs).some((organ) => (organ?.severity || 0) > 0);

  // Final fallback: if model gave a risk but no usable organ/system detail, render a systemic body burden.
  if (!afterAffectedSystems && riskScore > 0) {
    applyAffectedSystems(organs, {
      systemic: Math.max(0.35, riskScore * 0.9),
    });
  }

  const after = Object.values(organs).some((organ) => (organ?.severity || 0) > 0);
  return !before && after;
}

// ─── Layer 3: Enrich from drugInfoCache side effects ───────────────────────
function enrichFromDrugInfo(organs, drugs, drugInfoCache) {
  let enriched = false;

  for (const drug of drugs) {
    const drugName = typeof drug === 'string' ? drug : drug?.name;
    if (!drugName) continue;

    const info = drugInfoCache[drugName] || drugInfoCache[drugName.toLowerCase()];
    if (!info?.side_effects?.length) continue;

    for (const se of info.side_effects) {
      const organId = normalizeOrganSystem(se.organ_system);
      if (!organId || !organs[organId]) continue;

      const severityWeight = parseSeverity(se.severity || se.severity_weight);

      organs[organId].sideEffects.push({
        name: se.name || se.effect,
        severity: severityWeight,
        frequency: se.frequency || null,
        drug: drugName,
        source: 'drug-info',
      });

      // Boost organ severity based on side effect count and weight
      const currentMax = organs[organId].severity;
      const seBump = Math.min(0.9, severityWeight * 0.7);
      organs[organId].severity = Math.max(currentMax, seBump);

      enriched = true;
    }

    // Track drug contribution per organ
    for (const organId of Object.keys(organs)) {
      const drugSEs = organs[organId].sideEffects.filter(s => s.drug === drugName);
      if (drugSEs.length > 0) {
        const avgSev = drugSEs.reduce((sum, s) => sum + s.severity, 0) / drugSEs.length;
        organs[organId].drugContributions.push({ drug: drugName, contribution: avgSev });
      }
    }
  }

  return enriched;
}

// ─── Layer 4: Enrich from interaction evidence ─────────────────────────────
function enrichFromEvidence(organs, evidence) {
  let enriched = false;

  // Common side effects from interaction
  if (evidence.common_side_effects?.length) {
    for (const se of evidence.common_side_effects) {
      const organId = normalizeOrganSystem(se.organ_system);
      if (!organId || !organs[organId]) continue;

      organs[organId].sideEffects.push({
        name: se.name || se.effect,
        severity: parseSeverity(se.severity),
        frequency: se.frequency || null,
        drug: 'interaction',
        source: 'interaction-evidence',
      });

      organs[organId].severity = Math.max(organs[organId].severity, parseSeverity(se.severity) * 0.8);
      enriched = true;
    }
  }

  // Affected systems from interaction
  if (evidence.affected_systems?.length) {
    for (const sys of evidence.affected_systems) {
      const organId = normalizeOrganSystem(sys.system || sys);
      if (!organId || !organs[organId]) continue;
      const sev = sys.severity || 0.5;
      organs[organId].severity = Math.max(organs[organId].severity, sev);
    }
  }

  // FAERS real-world evidence
  if (evidence.faers_data) {
    const faers = evidence.faers_data;
    if (faers.adverse_events?.length) {
      for (const ae of faers.adverse_events) {
        const organId = normalizeOrganSystem(ae.organ_system || ae.system_organ_class);
        if (!organId || !organs[organId]) continue;
        organs[organId].faersCount += ae.count || ae.case_count || 1;
        enriched = true;
      }
    }
  }

  return enriched;
}

// ─── CYP liver load computation ────────────────────────────────────────────
function computeCYPLiverLoad(drugs) {
  const enzymeBurden = {};
  const CYP_ENZYMES = ['CYP1A2', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A4', 'CYP2B6', 'CYP2C8'];

  for (const drug of drugs) {
    const drugName = typeof drug === 'string' ? drug : drug?.name;
    if (!drugName) continue;

    const profile = getOfflineCYPProfile(drugName);
    if (!profile) continue;

    for (const enzyme of CYP_ENZYMES) {
      const roles = profile[enzyme];
      if (!roles?.length) continue;

      if (!enzymeBurden[enzyme]) enzymeBurden[enzyme] = { substrates: 0, inhibitors: 0, inducers: 0 };

      for (const role of roles) {
        if (role === 'substrate') enzymeBurden[enzyme].substrates++;
        else if (role.startsWith('inhibitor')) enzymeBurden[enzyme].inhibitors++;
        else if (role === 'inducer') enzymeBurden[enzyme].inducers++;
      }
    }
  }

  // Score: competition on same enzyme is dangerous
  let load = 0;
  for (const enzyme of Object.keys(enzymeBurden)) {
    const b = enzymeBurden[enzyme];
    // Multiple substrates competing for same enzyme
    if (b.substrates >= 2) load += 0.3;
    // Substrate + inhibitor = metabolism blocked
    if (b.substrates >= 1 && b.inhibitors >= 1) load += 0.4;
    // Substrate + inducer = metabolism accelerated
    if (b.substrates >= 1 && b.inducers >= 1) load += 0.2;
    // Inhibitor + inducer = unpredictable
    if (b.inhibitors >= 1 && b.inducers >= 1) load += 0.3;
  }

  return Math.min(1, load);
}

// ─── Drug pathway computation ──────────────────────────────────────────────
function computeDrugPathway(organs, drugs) {
  if (!drugs?.length) return null;

  // Find which organs are actually affected
  const activeOrgans = Object.entries(organs)
    .filter(([_, o]) => o.severity > 0)
    .map(([id, o]) => ({ id, severity: o.severity }))
    .sort((a, b) => b.severity - a.severity);

  return {
    route: 'oral', // Default oral route for now
    activeOrgans,
    drugCount: drugs.length,
  };
}

const DISAGREEMENT_PENALTY = {
  none: 0,
  low: 0.05,
  medium: 0.1,
  high: 0.16,
};

const SOURCE_RELIABILITY_WEIGHTS = {
  knowledgeGraph: 0.78,
  literature: 0.84,
  realWorld: 0.8,
  mechanistic: 0.73,
  modelSignals: 0.68,
};

const SOURCE_RECENCY_HALFLIFE_DAYS = {
  knowledgeGraph: 540,
  literature: 720,
  realWorld: 180,
  mechanistic: 900,
  modelSignals: 365,
};

const ORGAN_MONITORING_FOCUS = {
  brain: ['Track sedation and cognitive status', 'Watch for neurologic dose stacking', 'Review CNS depressant combinations'],
  heart: ['Monitor BP and heart rate trends', 'Check ECG/QT risk if applicable', 'Review electrolyte-sensitive interactions'],
  lungs: ['Track respiratory rate and oxygenation', 'Monitor sedative-opioid combinations', 'Assess bronchospasm symptom progression'],
  liver: ['Review LFT trend and hepatic burden', 'Audit CYP substrate/inhibitor overlap', 'Reassess dosing window and timing'],
  kidney: ['Monitor creatinine/eGFR trend', 'Evaluate nephrotoxic stacking', 'Check renal-dose adjustment opportunities'],
  gi: ['Track GI intolerance and bleeding signals', 'Review absorption and timing conflicts', 'Check GI-protective mitigation options'],
  blood: ['Monitor bleeding and hematologic signals', 'Check anticoagulant/antiplatelet overlap', 'Review CBC-related safety markers'],
  skin: ['Track rash progression and severity', 'Watch hypersensitivity escalation signs', 'Review culprit-agent chronology'],
  musculoskeletal: ['Monitor myalgia or weakness progression', 'Review statin-interaction risk factors', 'Track mobility-limiting symptom burden'],
  endocrine: ['Track glucose and hormonal shifts', 'Review endocrine-active co-medications', 'Reassess metabolic side-effect trend'],
};

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function parseDateCandidate(value) {
  if (!value || typeof value !== 'string') return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed;
}

function deriveFreshness(interactionEvidence) {
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

function buildEvidenceHighlights(organ) {
  const sideEffectSignals = getUniqueSideEffects(organ)
    .slice(0, 3)
    .map((signal) => ({
      title: signal?.name || 'Clinical finding',
      score: parseSeverity(signal?.severity),
      source: signal?.source || 'clinical-signal',
      detail: signal?.frequency || null,
    }));

  const detailSignals = (organ?.details || [])
    .slice(0, 3)
    .map((detail) => ({
      title: detail?.label || detail?.type || 'Mechanistic note',
      score: parseSeverity(detail?.severity),
      source: detail?.type || 'mechanistic',
      detail: detail?.description || null,
    }));

  return [...sideEffectSignals, ...detailSignals]
    .sort((a, b) => (b.score || 0) - (a.score || 0))
    .slice(0, 4);
}

function classifyEvidenceSource(raw) {
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

function buildGlobalSourcePriors(interactionEvidence) {
  const priors = {
    knowledgeGraph: 0,
    literature: 0,
    realWorld: 0,
    mechanistic: 0,
  };

  const evidenceChain = Array.isArray(interactionEvidence?.evidence_chain)
    ? interactionEvidence.evidence_chain
    : [];

  evidenceChain.forEach((entry) => {
    const category = classifyEvidenceSource(entry?.source || entry?.source_name);
    if (category === 'modelSignals') return;
    priors[category] += 1;
  });

  return priors;
}

function parseEvidenceTimestamp(value) {
  if (value == null) return null;

  if (typeof value === 'number' && Number.isFinite(value) && value > 1900 && value < 2200) {
    return new Date(Date.UTC(Math.floor(value), 0, 1));
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return null;

    if (/^\d{4}$/.test(trimmed)) {
      return new Date(Date.UTC(Number(trimmed), 0, 1));
    }

    const parsed = new Date(trimmed);
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }

  return null;
}

function decayScore(daysOld, halfLife) {
  if (!Number.isFinite(daysOld) || daysOld < 0) return 0.6;
  const safeHalfLife = Number.isFinite(halfLife) && halfLife > 0 ? halfLife : 365;
  return clamp01(Math.exp((-Math.log(2) * daysOld) / safeHalfLife));
}

function weightedBreakdownAverage(breakdown = {}, weights = {}) {
  let weightedSum = 0;
  let totalWeight = 0;

  for (const [key, value] of Object.entries(breakdown || {})) {
    const componentWeight = Number(value) || 0;
    if (componentWeight <= 0) continue;
    const metric = Number(weights[key]);
    if (!Number.isFinite(metric)) continue;
    weightedSum += componentWeight * metric;
    totalWeight += componentWeight;
  }

  if (totalWeight <= 0) return 0;
  return clamp01(weightedSum / totalWeight);
}

function buildSourceRecencyMetrics(interactionEvidence, freshnessLabel) {
  const evidenceChain = Array.isArray(interactionEvidence?.evidence_chain)
    ? interactionEvidence.evidence_chain
    : [];

  const latestByCategory = {
    knowledgeGraph: null,
    literature: null,
    realWorld: null,
    mechanistic: null,
    modelSignals: null,
  };

  evidenceChain.forEach((entry) => {
    const category = classifyEvidenceSource(entry?.source || entry?.source_name);
    const timestamp = [
      entry?.published_at,
      entry?.last_updated,
      entry?.updated_at,
      entry?.timestamp,
      entry?.date,
      entry?.year,
    ]
      .map(parseEvidenceTimestamp)
      .find(Boolean);

    if (!timestamp) return;

    if (!latestByCategory[category] || timestamp > latestByCategory[category]) {
      latestByCategory[category] = timestamp;
    }
  });

  const fallbackRealWorldScore = freshnessLabel === 'fresh'
    ? 0.92
    : freshnessLabel === 'recent'
      ? 0.76
      : freshnessLabel === 'stale'
        ? 0.5
        : 0.62;

  const fallbackScores = {
    knowledgeGraph: 0.66,
    literature: 0.64,
    realWorld: fallbackRealWorldScore,
    mechanistic: 0.61,
    modelSignals: 0.68,
  };

  const scoresByCategory = {};
  const daysByCategory = {};

  Object.keys(latestByCategory).forEach((category) => {
    const timestamp = latestByCategory[category];
    if (!timestamp) {
      scoresByCategory[category] = fallbackScores[category] ?? 0.6;
      daysByCategory[category] = null;
      return;
    }

    const daysOld = Math.max(0, Math.round((Date.now() - timestamp.getTime()) / 86400000));
    const score = decayScore(daysOld, SOURCE_RECENCY_HALFLIFE_DAYS[category]);
    scoresByCategory[category] = score;
    daysByCategory[category] = daysOld;
  });

  return { scoresByCategory, daysByCategory };
}

function buildSourceReliabilityProfile({
  sourceBreakdown,
  disagreementLevel,
  support,
  severity,
  sideEffectCount,
  detailCount,
  faersCount,
}) {
  const qualityScore = weightedBreakdownAverage(sourceBreakdown, SOURCE_RELIABILITY_WEIGHTS);
  const disagreementFactor = clamp01(1 - ((DISAGREEMENT_PENALTY[disagreementLevel] ?? 0.08) / 0.16));
  const alignmentFactor = clamp01(1 - Math.abs((Number(support) || 0) - (Number(severity) || 0)));
  const densityFactor = clamp01((Math.min((sideEffectCount + detailCount + (faersCount > 0 ? 1 : 0)), 6)) / 6);

  const consistencyTrend = clamp01(
    (alignmentFactor * 0.45)
    + (disagreementFactor * 0.4)
    + (densityFactor * 0.15)
  );

  const sourceReliabilityScore = clamp01(
    (qualityScore * 0.74)
    + (consistencyTrend * 0.26)
  );

  return {
    qualityScore,
    consistencyTrend,
    sourceReliabilityScore,
  };
}

function normalizeBreakdown(rawBreakdown) {
  const total = Object.values(rawBreakdown).reduce((sum, value) => sum + (Number(value) || 0), 0);
  if (total <= 0) {
    return Object.fromEntries(Object.keys(rawBreakdown).map((key) => [key, 0]));
  }

  return Object.fromEntries(
    Object.entries(rawBreakdown).map(([key, value]) => [key, clamp01((Number(value) || 0) / total)])
  );
}

function getRecencyRisk(freshnessLabel) {
  if (freshnessLabel === 'fresh') return 0.08;
  if (freshnessLabel === 'recent') return 0.24;
  if (freshnessLabel === 'stale') return 0.68;
  return 0.5;
}

function buildUncertaintyDecomposition({
  evidenceCount,
  disagreementLevel,
  support,
  severity,
  recencyScore,
  faersCount,
}) {
  const sparsity = clamp01((4 - Math.min(evidenceCount, 4)) / 4);
  const disagreement = clamp01((DISAGREEMENT_PENALTY[disagreementLevel] ?? 0.08) / 0.16);
  const recencyRisk = clamp01(1 - (Number(recencyScore) || 0));
  const crossSourceVariance = clamp01(Math.abs((Number(support) || 0) - (Number(severity) || 0)) * 1.35);
  const realWorldGap = clamp01((Number(severity) > 0 && Number(faersCount) === 0) ? 0.82 : 0.18);

  const decomposition = {
    dataSparsity: sparsity,
    sourceDisagreement: disagreement,
    recencyRisk,
    crossSourceVariance,
    realWorldGap,
  };

  const labels = {
    dataSparsity: 'Data sparsity',
    sourceDisagreement: 'Source disagreement',
    recencyRisk: 'Recency risk',
    crossSourceVariance: 'Cross-source variance',
    realWorldGap: 'Real-world evidence gap',
  };

  const topDrivers = Object.entries(decomposition)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([key, value]) => ({
      key,
      label: labels[key] || key,
      score: clamp01(value),
    }));

  return { decomposition, topDrivers };
}

function buildConfidenceUpliftPlan({
  severity,
  coverage,
  faersCount,
  disagreementLevel,
  freshnessLabel,
  detailCount,
  sideEffectCount,
  evidenceCount,
}) {
  const independentSourceCount = [
    detailCount > 0 ? 1 : 0,
    sideEffectCount > 0 ? 1 : 0,
    faersCount > 0 ? 1 : 0,
    severity > 0.2 ? 1 : 0,
  ].reduce((sum, value) => sum + value, 0);

  return buildSharedConfidenceUpliftPlan({
    scopeLabel: 'this system',
    coverageRatio: coverage,
    hasRealWorldEvidence: faersCount > 0,
    disagreementLevel,
    freshnessLabel,
    hasMechanisticDetail: detailCount > 0,
    sideEffectSignalCount: sideEffectCount,
    independentSourceCount,
    evidenceNodeCount: evidenceCount,
    maxSteps: 5,
  });
}

function buildSystemEvidence(organs, interactionEvidence, drugs) {
  const summary = interactionEvidence?.evidence_summary && typeof interactionEvidence.evidence_summary === 'object'
    ? interactionEvidence.evidence_summary
    : {};

  const weightedSupport = Number(summary?.weighted_support_score);
  const weightedUncertainty = Number(summary?.weighted_uncertainty_score);
  const globalCoverage = Number(summary?.source_coverage?.ratio);
  const disagreementLevel = String(summary?.disagreement?.level || 'none').toLowerCase();
  const disagreementPenalty = DISAGREEMENT_PENALTY[disagreementLevel] ?? 0.08;
  const summaryReasons = Array.isArray(summary?.uncertainty_reasons)
    ? summary.uncertainty_reasons.filter((reason) => typeof reason === 'string' && reason.trim().length > 0)
    : [];
  const freshness = deriveFreshness(interactionEvidence);
  const sourcePriors = buildGlobalSourcePriors(interactionEvidence);
  const sourceRecencyMetrics = buildSourceRecencyMetrics(interactionEvidence, freshness.label);

  const profiles = {};

  for (const [organKey, organ] of Object.entries(organs || {})) {
    const severity = clamp01(Number(organ?.severity || 0));
    const detailCount = Array.isArray(organ?.details) ? organ.details.length : 0;
    const sideEffectCount = Array.isArray(organ?.sideEffects) ? getUniqueSideEffects(organ).length : 0;
    const faersCount = Number(organ?.faersCount || 0);
    const evidenceCount = sideEffectCount + detailCount + (faersCount > 0 ? 1 : 0) + (severity > 0 ? 1 : 0);

    const fallbackSupport = clamp01((severity * 0.62) + (Math.min(sideEffectCount + detailCount, 8) / 8) * 0.38);
    const support = Number.isFinite(weightedSupport) ? clamp01(weightedSupport) : fallbackSupport;

    const uncertaintyBase = Number.isFinite(weightedUncertainty) ? clamp01(weightedUncertainty) : 0.34;
    const sparsityPenalty = evidenceCount < 2 ? 0.22 : evidenceCount < 4 ? 0.08 : 0;
    const faersPenalty = severity > 0 && faersCount === 0 ? 0.08 : 0;
    const uncertainty = clamp01(uncertaintyBase + sparsityPenalty + faersPenalty);

    const coverage = Number.isFinite(globalCoverage)
      ? clamp01(globalCoverage)
      : clamp01(evidenceCount === 0 ? 0 : 0.35 + (Math.min(evidenceCount, 8) / 8) * 0.65);

    const uncertaintyReasons = [...summaryReasons];
    if (evidenceCount < 2) uncertaintyReasons.push('Limited organ-specific evidence signals');
    if (severity > 0 && faersCount === 0) uncertaintyReasons.push('No real-world FAERS support for this system yet');
    if (disagreementLevel !== 'none') uncertaintyReasons.push('Cross-source disagreement detected');

    const uniqueReasons = Array.from(new Set(uncertaintyReasons)).slice(0, 5);
    const evidenceHighlights = buildEvidenceHighlights(organ);

    const sourceSet = new Set();
    (organ?.sideEffects || []).forEach((signal) => sourceSet.add(signal?.source || 'signal'));
    (organ?.details || []).forEach((detail) => sourceSet.add(detail?.type || 'mechanistic'));
    if (faersCount > 0) sourceSet.add('faers');
    if ((organ?.cypLoad || 0) > 0) sourceSet.add('cyp');
    if ((interactionEvidence?.evidence_chain || []).length > 0) sourceSet.add('evidence-chain');

    const sourceBreakdownRaw = {
      knowledgeGraph: (sourcePriors.knowledgeGraph * 0.45) + (severity * 0.16),
      literature: (sourcePriors.literature * 0.45) + (sideEffectCount * 0.22),
      realWorld: (sourcePriors.realWorld * 0.45) + Math.min(1, faersCount / 30),
      mechanistic: (sourcePriors.mechanistic * 0.4) + (detailCount * 0.32) + ((organ?.cypLoad || 0) * 0.38),
      modelSignals: (severity * 0.75) + (Math.min(sideEffectCount + detailCount, 6) / 6) * 0.25,
    };
    const sourceBreakdown = normalizeBreakdown(sourceBreakdownRaw);

    const recencyScore = weightedBreakdownAverage(sourceBreakdown, sourceRecencyMetrics.scoresByCategory);
    const {
      qualityScore,
      consistencyTrend,
      sourceReliabilityScore,
    } = buildSourceReliabilityProfile({
      sourceBreakdown,
      disagreementLevel,
      support,
      severity,
      sideEffectCount,
      detailCount,
      faersCount,
    });

    const confidenceScore = clamp01(
      (support * 0.45)
      + (coverage * 0.18)
      + (severity * 0.12)
      + ((Math.min(evidenceCount, 6) / 6) * 0.09)
      + (sourceReliabilityScore * 0.1)
      + (recencyScore * 0.06)
      - (uncertainty * 0.35)
      - disagreementPenalty
    );
    const certaintyScore = clamp01(
      (1 - uncertainty) * 0.8
      + (sourceReliabilityScore * 0.12)
      + (recencyScore * 0.08)
    );
    const confidenceBand = confidenceScore >= 0.75 ? 'high' : confidenceScore >= 0.45 ? 'medium' : 'low';

    const { decomposition: uncertaintyDecomposition, topDrivers: uncertaintyTopDrivers } = buildUncertaintyDecomposition({
      evidenceCount,
      disagreementLevel,
      support,
      severity,
      recencyScore,
      faersCount,
    });

    const { upliftPlan, potentialGain } = buildConfidenceUpliftPlan({
      severity,
      coverage,
      faersCount,
      disagreementLevel,
      freshnessLabel: freshness.label,
      detailCount,
      sideEffectCount,
      evidenceCount,
    });

    const monitoringFocus = ORGAN_MONITORING_FOCUS[organKey] || [
      'Track symptom progression and burden',
      'Review interaction mechanism alignment',
      'Reassess dosing and monitoring interval',
    ];

    const recommendation = severity > 0.7 || uncertainty > 0.65
      ? 'Escalate monitoring'
      : severity > 0.4
        ? 'Focused review'
        : 'Routine surveillance';

    profiles[organKey] = {
      support,
      uncertainty,
      certaintyScore,
      coverage,
      confidenceScore,
      confidencePotentialScore: clamp01(confidenceScore + potentialGain),
      sourceReliabilityScore,
      sourceConsistencyTrend: consistencyTrend,
      sourceQualityScore: qualityScore,
      recencyScore,
      sourceRecencyByCategory: sourceRecencyMetrics.scoresByCategory,
      sourceRecencyDaysByCategory: sourceRecencyMetrics.daysByCategory,
      confidenceBand,
      disagreementLevel,
      evidenceCount,
      signalCount: sideEffectCount + detailCount,
      faersCount,
      sourceBreakdown,
      uncertaintyReasons: uniqueReasons,
      uncertaintyDecomposition,
      uncertaintyTopDrivers,
      confidenceUpliftPlan: upliftPlan,
      evidenceHighlights,
      sources: Array.from(sourceSet).sort(),
      recommendation,
      monitoringFocus,
      freshnessLabel: freshness.label,
      freshnessDaysOld: freshness.daysOld,
      regimenDrugCount: Array.isArray(drugs) ? drugs.length : 0,
    };
  }

  return profiles;
}

// ─── Severity parsing helper ───────────────────────────────────────────────
function parseSeverity(raw) {
  if (typeof raw === 'number') return Math.min(1, Math.max(0, raw));
  if (typeof raw === 'string') {
    const lower = raw.toLowerCase();
    if (lower === 'severe' || lower === 'high' || lower === 'serious') return 0.8;
    if (lower === 'moderate' || lower === 'medium') return 0.5;
    if (lower === 'mild' || lower === 'low' || lower === 'minor') return 0.3;
    // Try parsing as number
    const num = parseFloat(raw);
    if (!isNaN(num)) return Math.min(1, Math.max(0, num));
  }
  return 0.4; // Default moderate
}

// ─── Deduplicate side effects per organ ────────────────────────────────────
export function getUniqueSideEffects(organ) {
  if (!organ?.sideEffects?.length) return [];

  const seen = new Map();
  for (const se of organ.sideEffects) {
    const key = (se.name || '').toLowerCase();
    if (!seen.has(key) || se.severity > seen.get(key).severity) {
      seen.set(key, se);
    }
  }

  return Array.from(seen.values()).sort((a, b) => b.severity - a.severity);
}
