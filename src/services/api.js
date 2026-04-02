/**
 * Project Aegis - API Service
 * 
 * This module handles all communication with the Django backend API.
 * Endpoints:
 * - /api/v1/predict/ - DDI prediction for 2 drugs
 * - /api/v1/polypharmacy/ - N-way drug interaction analysis
 * - /api/v1/chat/ - GraphRAG research assistant
 * - /api/v1/search/ - Drug search
 * - /api/v1/health/ - System health check
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

const API_TIMEOUT_MS = 30000;
const API_MAX_RETRIES = 2;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function summarizeErrorBody(text, status) {
    if (!text) {
        return `HTTP error! status: ${status}`;
    }

    const normalized = String(text).trim();
    const looksLikeHtml = /^<!doctype html>|^<html/i.test(normalized);

    if (!looksLikeHtml) {
        return normalized;
    }

    const routeMatch = normalized.match(/Page not found at\s*([^<\n]+)/i);
    if (routeMatch) {
        return `Route not found: ${routeMatch[1].trim()} (HTTP ${status})`;
    }

    const titleMatch = normalized.match(/<title>([^<]+)<\/title>/i);
    if (titleMatch) {
        return `${titleMatch[1].trim()} (HTTP ${status})`;
    }

    return `Received HTML error response (HTTP ${status})`;
}

/**
 * Generic fetch wrapper with error handling
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;

    const defaultHeaders = {
        'Content-Type': 'application/json',
    };

    const config = {
        ...options,
        headers: {
            ...defaultHeaders,
            ...options.headers,
        },
    };

    let lastError;

    for (let attempt = 0; attempt <= API_MAX_RETRIES; attempt++) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

        try {
            const response = await fetch(url, {
                ...config,
                signal: controller.signal,
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorMessage = `HTTP error! status: ${response.status}`;

                try {
                    const contentType = response.headers.get('content-type') || '';
                    if (contentType.includes('application/json')) {
                        const errorData = await response.json();
                        errorMessage = errorData.detail || errorData.error || errorMessage;
                    } else {
                        const text = await response.text();
                        if (text) {
                            errorMessage = summarizeErrorBody(text, response.status);
                        }
                    }
                } catch {
                    // Fall back to status-based message.
                }

                const httpError = new Error(errorMessage);
                httpError.status = response.status;

                // Retry only transient server-side failures.
                if (response.status >= 500 && attempt < API_MAX_RETRIES) {
                    await delay(400 * (attempt + 1));
                    continue;
                }

                throw httpError;
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            lastError = error;

            const isTransientNetworkError =
                error.name === 'AbortError' ||
                error.name === 'TypeError' ||
                /Failed to fetch|NetworkError|Load failed/i.test(error.message || '');

            if (isTransientNetworkError && attempt < API_MAX_RETRIES) {
                await delay(400 * (attempt + 1));
                continue;
            }

            break;
        }
    }

    console.error(`API Error [${endpoint}]:`, lastError);
    throw lastError;
}

/**
 * Health check - verify backend is running
 */
export async function checkHealth() {
    return apiRequest('/health/');
}

/**
 * Search for drugs by name
 * @param {string} query - Search query (min 2 characters)
 * @returns {Promise<{results: Array}>}
 */
export async function searchDrugs(query) {
    if (!query || query.length < 2) {
        return { results: [] };
    }
    return apiRequest(`/search/?q=${encodeURIComponent(query)}`);
}

/**
 * Predict DDI between two drugs
 * @param {Object} drugA - Drug A info {name, smiles?, drugbank_id?}
 * @param {Object} drugB - Drug B info {name, smiles?, drugbank_id?}
 * @param {Object} options - Additional options
 * @returns {Promise<DDIPredictionResponse>}
 */
export async function predictDDI(drugA, drugB, options = {}) {
    return apiRequest('/predict/', {
        method: 'POST',
        body: JSON.stringify({
            drug_a: drugA,
            drug_b: drugB,
            include_explanation: options.includeExplanation ?? true,
            include_alternatives: options.includeAlternatives ?? false,
        }),
    });
}

/**
 * Analyze polypharmacy (N-way drug interactions)
 * @param {Array} drugs - Array of drug objects [{name, smiles?}, ...]
 * @returns {Promise<PolypharmacyResponse>}
 */
export async function analyzePolypharmacy(drugs) {
    return apiRequest('/polypharmacy/', {
        method: 'POST',
        body: JSON.stringify({ drugs }),
    });
}

/**
 * Analyze polypharmacy with N-order Digital Twin scoring.
 * @param {Array} drugs - Array of drug objects [{name, smiles?}, ...]
 * @returns {Promise<Object>}
 */
export async function analyzePolypharmacyDigitalTwin(drugs) {
    return apiRequest('/polypharmacy-digital-twin/', {
        method: 'POST',
        body: JSON.stringify({ drugs }),
    });
}

/**
 * Send message to GraphRAG chatbot
 * @param {string} message - User message
 * @param {Array} contextDrugs - Current drugs in context
 * @param {string} sessionId - Chat session ID
 * @returns {Promise<ChatResponse>}
 */
export async function sendChatMessage(message, contextDrugs = [], sessionId = null) {
    return apiRequest('/chat/', {
        method: 'POST',
        body: JSON.stringify({
            message,
            context_drugs: contextDrugs,
            session_id: sessionId,
        }),
    });
}

/**
 * Get prediction history
 * @param {number} limit - Max records to fetch
 * @returns {Promise<Array>}
 */
export async function getPredictionHistory(limit = 20) {
    return apiRequest(`/history/?limit=${limit}`);
}

/**
 * Get enhanced drug information including side effects and FAERS data
 * @param {string} drugName - Drug name
 * @param {boolean} includeFaers - Whether to include OpenFDA data (slower)
 * @returns {Promise<EnhancedDrugInfo>}
 */
export async function getDrugInfo(drugName, includeFaers = true) {
    const params = new URLSearchParams({
        name: drugName,
        faers: includeFaers.toString()
    });
    return apiRequest(`/drug-info/?${params}`);
}

/**
 * Get enhanced interaction information including real-world evidence
 * @param {string} drug1 - First drug name
 * @param {string} drug2 - Second drug name
 * @param {boolean} includeFaers - Whether to include OpenFDA data
 * @returns {Promise<EnhancedInteractionInfo>}
 */
export async function getInteractionInfo(drug1, drug2, includeFaers = true) {
    const params = new URLSearchParams({
        drug1,
        drug2,
        faers: includeFaers.toString()
    });
    return apiRequest(`/interaction-info/?${params}`);
}

/**
 * Get real-world evidence from FDA FAERS
 * @param {string} drug1 - First drug name
 * @param {string} drug2 - Optional second drug name for co-occurrence
 * @returns {Promise<FAERSData>}
 */
export async function getRealWorldEvidence(drug1, drug2 = null) {
    const params = new URLSearchParams({ drug1 });
    if (drug2) params.append('drug2', drug2);
    return apiRequest(`/real-world-evidence/?${params}`);
}

/**
 * Get database statistics for dashboard
 * @returns {Promise<DatabaseStats>}
 */
export async function getDatabaseStats() {
    return apiRequest('/stats/');
}

/**
 * Compute calibration QA metrics from labeled score arrays.
 * @param {Object} payload
 * @param {Array<number>} payload.labels - Binary labels (0/1)
 * @param {Array<number>} payload.rawScores - Raw model probabilities
 * @param {Array<number>} payload.calibratedScores - Calibrated probabilities
 * @param {number} [payload.nBins=10] - Bin count for ECE
 * @param {number} [payload.nBootstrap=1000] - Bootstrap sample count for CIs
 * @param {number} [payload.seed=42] - Bootstrap RNG seed
 * @returns {Promise<Object>}
 */
export async function computeCalibrationMetrics({
    labels,
    rawScores,
    calibratedScores,
    nBins = 10,
    nBootstrap = 1000,
    seed = 42,
}) {
    return apiRequest('/calibration/metrics/', {
        method: 'POST',
        body: JSON.stringify({
            labels,
            raw_scores: rawScores,
            calibrated_scores: calibratedScores,
            n_bins: nBins,
            n_bootstrap: nBootstrap,
            seed,
        }),
    });
}

/**
 * Get therapeutic alternatives for a drug
 * @param {string} drugName - Drug to find alternatives for
 * @param {string} interactingWith - Optional drug that has problematic interaction
 * @returns {Promise<AlternativesResponse>}
 */
export async function getTherapeuticAlternatives(drugName, interactingWith = null) {
    const params = new URLSearchParams({ drug: drugName });
    if (interactingWith) params.append('interacting_with', interactingWith);
    return apiRequest(`/alternatives/?${params}`);
}

/**
 * Compare multiple drugs side-by-side
 * @param {Array<string>} drugNames - Array of drug names (2-5)
 * @returns {Promise<ComparisonResponse>}
 */
export async function compareDrugs(drugNames) {
    return apiRequest('/compare/', {
        method: 'POST',
        body: JSON.stringify({ drugs: drugNames }),
    });
}

// ============== Type Definitions (for reference) ==============

/**
 * @typedef {Object} DDIPredictionResponse
 * @property {string} drug_a - Name of drug A
 * @property {string} drug_b - Name of drug B
 * @property {number} risk_score - Risk score (0-1)
 * @property {number} raw_score - Uncalibrated model score (0-1)
 * @property {number} calibrated_score - Calibrated model score (0-1)
 * @property {string} risk_level - 'low' | 'medium' | 'high' | 'critical'
 * @property {string} severity - 'no_interaction' | 'minor' | 'moderate' | 'major' | 'severe' | 'high' | 'critical'
 * @property {number} confidence - Model confidence (0-1)
 * @property {string} mechanism_hypothesis - Explanation of interaction mechanism
 * @property {Array} affected_systems - Organ systems affected
 * @property {PredictionProvenance} provenance - Prediction lineage and fallback metadata
 * @property {Object} explanation - XAI explanation data
 */

/**
 * @typedef {Object} PredictionProvenance
 * @property {string} model_version - Backend model version tag
 * @property {string} prediction_path - Path used to produce prediction
 * @property {string} model_used - Concrete model implementation used
 * @property {string} calibration_method - Calibration method applied
 * @property {string} calibration_version - Calibration artifact/version
 * @property {string|null} fallback_reason - Why fallback logic was used (if applicable)
 */

/**
 * @typedef {Object} PolypharmacyResponse
 * @property {Array<string>} drugs - List of drug names
 * @property {Array<InteractionEdge>} interactions - Network edges
 * @property {number} total_interactions - Count of interactions
 * @property {number} max_risk_score - Highest risk score
 * @property {string} overall_risk_level - Overall risk level
 * @property {string} hub_drug - Drug with most interactions
 * @property {number} hub_interaction_count - Hub interaction count
 * @property {Object} body_map - Organ system risk mapping
 */

/**
 * @typedef {Object} InteractionEdge
 * @property {string} source - Source drug name
 * @property {string} target - Target drug name
 * @property {number} risk_score - Edge risk score
 * @property {string} severity - Severity level
 * @property {Array<string>} affected_systems - Affected organs
 */

/**
 * @typedef {Object} ChatResponse
 * @property {string} response - AI response text
 * @property {Array} sources - Citation sources
 * @property {Array<string>} related_drugs - Related drug names
 * @property {string} session_id - Session identifier
 */

export default {
    checkHealth,
    searchDrugs,
    predictDDI,
    analyzePolypharmacy,
    analyzePolypharmacyDigitalTwin,
    sendChatMessage,
    getPredictionHistory,
    getDrugInfo,
    getInteractionInfo,
    getRealWorldEvidence,
    getDatabaseStats,
    computeCalibrationMetrics,
    getTherapeuticAlternatives,
    compareDrugs,
};
