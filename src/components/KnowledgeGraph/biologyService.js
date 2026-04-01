// biologyService.js — API client for drug biology data (Knowledge Graph V2)
// Uses EXISTING deployed endpoints (drug-info, interaction-info) as primary,
// with new biology endpoints as enhancement when backend is updated,
// and CYP fallback data as last resort.

import { getDrugInfo, getInteractionInfo } from '../../services/api';
import { buildOfflineDrugBiology, buildOfflineMechanismMap, getOfflineCYPProfile } from './cypFallbackData';

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';
const TIMEOUT_MS = 15000;

// ─── In-memory cache ────────────────────────────────────────────────────────
const cache = new Map();
const CACHE_TTL = 5 * 60 * 1000;

function getCached(key) {
  const entry = cache.get(key);
  if (entry && (Date.now() - entry.time) < CACHE_TTL) return entry.data;
  return null;
}

function setCache(key, data) {
  cache.set(key, { data, time: Date.now() });
}

// ─── Generic fetch with timeout ─────────────────────────────────────────────
async function fetchJSON(url, timeoutMs = TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
    });
    clearTimeout(timeoutId);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') throw new Error('Request timed out');
    throw err;
  }
}

// ─── Fetch drug biology (layered strategy) ──────────────────────────────────
// 1. Try new /graph/drug-biology/ endpoint (if backend updated)
// 2. Fall back to existing /drug-info/ endpoint + CYP data
// 3. Fall back to offline CYP-only data
export async function fetchDrugBiology(drugName) {
  const key = `bio:${drugName.toLowerCase()}`;
  const cached = getCached(key);
  if (cached) return { ...cached, _cached: true };

  // Strategy 1: Try new biology endpoint
  try {
    const data = await fetchJSON(
      `${API_BASE}/graph/drug-biology/?drug=${encodeURIComponent(drugName)}&include_side_effects=true`
    );
    if (data && !data.error) {
      const result = { ...data, source: 'biology-api' };
      setCache(key, result);
      return result;
    }
  } catch (e) {
    // New endpoint not deployed yet, try existing
  }

  // Strategy 2: Use existing drug-info endpoint + CYP fallback
  try {
    const drugInfo = await getDrugInfo(drugName, true);
    if (drugInfo && !drugInfo.error) {
      const cypProfile = getOfflineCYPProfile(drugName) || {};
      const cyp_metabolism = buildCYPMetabolismFromProfile(cypProfile);

      const result = {
        drug: {
          id: drugInfo.drugbank_id || drugInfo.id || '',
          name: drugInfo.name || drugName,
          therapeutic_class: drugInfo.therapeutic_class || drugInfo.category || '',
          description: drugInfo.description || '',
        },
        cyp_metabolism,
        targets: (drugInfo.targets || []).map(t => ({
          id: t.uniprot_id || t.id || '',
          name: t.name || '',
          gene: t.gene_name || t.gene || '',
          action: t.action_type || t.action || 'unknown',
        })),
        side_effects: (drugInfo.side_effects || []).slice(0, 12).map(se => ({
          name: typeof se === 'string' ? se : (se.name || se.effect || ''),
          organ_system: se.organ_system || '',
          severity: se.severity_weight || se.severity || 0,
          frequency: se.frequency || '',
        })),
        interaction_count: drugInfo.interaction_count || drugInfo.total_interactions || 0,
        faers_data: drugInfo.faers_data || null,
        source: 'drug-info-api',
      };
      setCache(key, result);
      return result;
    }
  } catch (e) {
    // API offline entirely
  }

  // Strategy 3: Offline CYP data only
  const offline = buildOfflineDrugBiology(drugName);
  if (offline) {
    setCache(key, offline);
    return offline;
  }

  // Nothing at all — return minimal stub
  return {
    drug: { id: '', name: drugName, therapeutic_class: '' },
    cyp_metabolism: { substrates: [], inhibitors: [], inducers: [] },
    targets: [],
    side_effects: [],
    interaction_count: 0,
    source: 'none',
  };
}

// ─── Fetch mechanism map (layered strategy) ─────────────────────────────────
export async function fetchMechanismMap(drug1Name, drug2Name) {
  const sorted = [drug1Name.toLowerCase(), drug2Name.toLowerCase()].sort();
  const key = `map:${sorted[0]}:${sorted[1]}`;
  const cached = getCached(key);
  if (cached) return { ...cached, _cached: true };

  // Strategy 1: Try new mechanism-map endpoint
  try {
    const data = await fetchJSON(
      `${API_BASE}/graph/mechanism-map/?drug1=${encodeURIComponent(drug1Name)}&drug2=${encodeURIComponent(drug2Name)}`
    );
    if (data && !data.error) {
      const result = { ...data, source: 'mechanism-api' };
      setCache(key, result);
      return result;
    }
  } catch (e) {
    // Not deployed yet
  }

  // Strategy 2: Use existing interaction-info endpoint + CYP analysis
  try {
    const intxInfo = await getInteractionInfo(drug1Name, drug2Name, true);
    if (intxInfo && !intxInfo.error) {
      const offlineMap = buildOfflineMechanismMap(drug1Name, drug2Name) || {};

      const result = {
        drugs: [
          { id: '', name: drug1Name, therapeutic_class: intxInfo.drug_a_info?.therapeutic_class || '' },
          { id: '', name: drug2Name, therapeutic_class: intxInfo.drug_b_info?.therapeutic_class || '' },
        ],
        shared_enzymes: offlineMap.shared_enzymes || [],
        cyp_interactions: offlineMap.cyp_interactions || [],
        shared_targets: (intxInfo.common_targets || []).map(t => ({
          target: t.name || t.target || '',
          gene: t.gene_name || t.gene || '',
          drug1_action: t.drug1_action || 'unknown',
          drug2_action: t.drug2_action || 'unknown',
          risk: t.drug1_action === t.drug2_action ? `Additive ${t.drug1_action} effect` : '',
        })),
        shared_side_effects: (intxInfo.common_side_effects || []).map(se => ({
          name: typeof se === 'string' ? se : (se.name || ''),
          organ_system: se.organ_system || '',
          combined_risk: 'moderate',
        })),
        interaction: {
          severity: intxInfo.severity || intxInfo.risk_level || '',
          mechanism: intxInfo.mechanism || intxInfo.mechanism_hypothesis || '',
          description: intxInfo.description || '',
          evidence_level: intxInfo.evidence_level || '',
          risk_score: intxInfo.risk_score || 0,
        },
        affected_systems: intxInfo.affected_systems || [],
        conflict_summary: {
          ...(offlineMap.conflict_summary || {}),
          target_overlaps: (intxInfo.common_targets || []).length,
          shared_side_effects: (intxInfo.common_side_effects || []).length,
        },
        source: 'interaction-info-api',
      };

      // Recalculate overall risk
      const cs = result.conflict_summary;
      cs.overall_risk = (cs.cyp_conflicts >= 2 || result.interaction.severity === 'major' || result.interaction.severity === 'severe')
        ? 'high'
        : (cs.cyp_conflicts >= 1 || cs.target_overlaps >= 1 || cs.shared_side_effects >= 3)
          ? 'moderate'
          : 'low';

      setCache(key, result);
      return result;
    }
  } catch (e) {
    // API offline
  }

  // Strategy 3: Offline CYP-only
  const offline = buildOfflineMechanismMap(drug1Name, drug2Name);
  if (offline) {
    setCache(key, offline);
    return offline;
  }

  return null;
}

// ─── Helper: build CYP metabolism structure from profile ────────────────────
function buildCYPMetabolismFromProfile(profile) {
  const cyp_metabolism = { substrates: [], inhibitors: [], inducers: [] };
  for (const [enzyme, roles] of Object.entries(profile)) {
    for (const role of roles) {
      if (role === 'substrate') {
        cyp_metabolism.substrates.push(enzyme);
      } else if (role.includes('inhibitor')) {
        const strength = role.includes('strong') ? 'strong' : role.includes('moderate') ? 'moderate' : 'weak';
        cyp_metabolism.inhibitors.push({ enzyme, strength });
      } else if (role === 'inducer') {
        cyp_metabolism.inducers.push(enzyme);
      }
    }
  }
  return cyp_metabolism;
}

// ─── Cache management ───────────────────────────────────────────────────────
export function clearBiologyCache() {
  cache.clear();
}
