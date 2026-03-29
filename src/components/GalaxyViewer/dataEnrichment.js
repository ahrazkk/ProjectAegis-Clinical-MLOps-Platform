// dataEnrichment.js — Thin caching wrapper around api.js for Galaxy Viewer
// Enriches drug info and interaction data on-demand when user interacts

import {
  getDrugInfo,
  getInteractionInfo,
  getRealWorldEvidence,
  getTherapeuticAlternatives,
  predictDDI,
} from '../../services/api';

// ─── Cache ───────────────────────────────────────────────────────────────────
const cache = new Map();

/**
 * Enrich a single drug with info from backend
 * Returns: { name, smiles, therapeutic_class, side_effects[], cyp_metabolism, faers_data, ... }
 */
export async function enrichDrug(drugName) {
  if (!drugName) return null;
  const key = `drug:${drugName.toLowerCase()}`;
  if (cache.has(key)) return cache.get(key);

  try {
    const data = await getDrugInfo(drugName, true);
    cache.set(key, data);
    return data;
  } catch (err) {
    console.warn(`[enrichDrug] Failed for "${drugName}":`, err.message);
    return null;
  }
}

/**
 * Enrich a drug pair interaction
 * Returns: { risk_score, severity, confidence, mechanism_hypothesis, affected_systems, ... }
 */
export async function enrichInteraction(drugA, drugB) {
  if (!drugA || !drugB) return null;
  const key = `interaction:${[drugA, drugB].sort().join(':').toLowerCase()}`;
  if (cache.has(key)) return cache.get(key);

  try {
    // Fetch both prediction and interaction info in parallel
    const [prediction, interactionInfo] = await Promise.allSettled([
      predictDDI({ name: drugA }, { name: drugB }, { includeExplanation: true }),
      getInteractionInfo(drugA, drugB, true),
    ]);

    const data = {
      prediction: prediction.status === 'fulfilled' ? prediction.value : null,
      interactionInfo: interactionInfo.status === 'fulfilled' ? interactionInfo.value : null,
    };

    cache.set(key, data);
    return data;
  } catch (err) {
    console.warn(`[enrichInteraction] Failed for "${drugA}" + "${drugB}":`, err.message);
    return null;
  }
}

/**
 * Fetch real-world evidence (FDA FAERS) for a drug pair
 */
export async function enrichEvidence(drugA, drugB = null) {
  if (!drugA) return null;
  const key = `evidence:${[drugA, drugB].filter(Boolean).sort().join(':').toLowerCase()}`;
  if (cache.has(key)) return cache.get(key);

  try {
    const data = await getRealWorldEvidence(drugA, drugB);
    cache.set(key, data);
    return data;
  } catch (err) {
    console.warn(`[enrichEvidence] Failed:`, err.message);
    return null;
  }
}

/**
 * Fetch therapeutic alternatives for a drug
 */
export async function fetchAlternatives(drugName, interactingWith = null) {
  if (!drugName) return null;
  const key = `alternatives:${drugName.toLowerCase()}:${(interactingWith || '').toLowerCase()}`;
  if (cache.has(key)) return cache.get(key);

  try {
    const data = await getTherapeuticAlternatives(drugName, interactingWith);
    cache.set(key, data);
    return data;
  } catch (err) {
    console.warn(`[fetchAlternatives] Failed:`, err.message);
    return null;
  }
}

/**
 * Clear the enrichment cache
 */
export function clearCache() {
  cache.clear();
}

/**
 * Check if data for a drug is already cached
 */
export function isCached(drugName) {
  return cache.has(`drug:${drugName?.toLowerCase()}`);
}
