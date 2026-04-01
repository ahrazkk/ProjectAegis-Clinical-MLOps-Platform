// bodyMapDataService.js — Data enrichment pipeline for Body Map V2
// Merges multiple data sources into a unified per-organ severity + detail model.
// Sources: affectedSystems (prediction), drugInfoCache (side effects), interactionEvidence, CYP data.

import { normalizeOrganSystem, ORGAN_SYSTEMS } from './organRegistry';
import { getOfflineCYPProfile } from '../KnowledgeGraph/cypFallbackData';

// ─── Main enrichment function ──────────────────────────────────────────────
// Returns: { organs, drugPathway, cypLiverLoad, dataQuality, isEmpty }
export function enrichBodyMapData({ affectedSystems, drugs, drugInfoCache, interactionEvidence, polypharmacyResult }) {
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

  // Is the whole thing empty?
  const isEmpty = !drugs?.length || Object.values(organs).every(o => o.severity === 0);

  return { organs, drugPathway, cypLiverLoad, dataQuality, isEmpty };
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
    const organId = normalizeOrganSystem(key) || key;
    if (organs[organId]) {
      organs[organId].severity = Math.max(organs[organId].severity, typeof severity === 'number' ? severity : 0.5);
    }
  }
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
