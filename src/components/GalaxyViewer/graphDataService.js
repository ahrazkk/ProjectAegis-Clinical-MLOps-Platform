// graphDataService.js — Dynamic data layer for the Galaxy Viewer
// Fetches graph data from the backend API (Neo4j AuraDB) with caching and fallback.
// Replaces the static gnn_real_data.json import for live data.

import { searchDrugs } from '../../services/api';

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';
const TIMEOUT_MS = 20000;

// ─── In-memory cache ────────────────────────────────────────────────────────
const cache = {
  nodes: null,
  nodesTime: 0,
  edges: null,
  edgesTime: 0,
  neighborhoods: new Map(), // key: drugId → { data, time }
};

const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function isCacheValid(time) {
  return time > 0 && (Date.now() - time) < CACHE_TTL;
}

// ─── Performance config (user-adjustable) ───────────────────────────────────
const DEFAULT_LIMITS = {
  maxNodes: 2000,
  maxEdges: 50000,
  maxNeighborhood: 500,
  maxHops: 3,
};

let performanceLimits = { ...DEFAULT_LIMITS };

export function setPerformanceLimits(limits) {
  performanceLimits = { ...performanceLimits, ...limits };
}

export function getPerformanceLimits() {
  return { ...performanceLimits };
}

// ─── Generic fetch with timeout ─────────────────────────────────────────────
async function fetchWithTimeout(url, timeoutMs = TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
    });
    clearTimeout(timeoutId);
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
    }
    return await response.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') throw new Error('Request timed out');
    throw err;
  }
}

// ─── Fetch all graph nodes ──────────────────────────────────────────────────
// Returns { nodes: [...], total, returned }
export async function fetchGraphNodes(limit) {
  const max = limit || performanceLimits.maxNodes;

  if (cache.nodes && isCacheValid(cache.nodesTime)) {
    return { nodes: cache.nodes.slice(0, max), total: cache.nodes.length, cached: true };
  }

  const data = await fetchWithTimeout(
    `${API_BASE}/graph/nodes/?limit=${max}`
  );

  // Cache the full result
  cache.nodes = data.nodes || [];
  cache.nodesTime = Date.now();

  return { nodes: cache.nodes.slice(0, max), total: data.total || cache.nodes.length, cached: false };
}

// ─── Fetch all graph edges ──────────────────────────────────────────────────
// Returns { edges: [...], total }
// Fetches in batches if > limit per page
export async function fetchGraphEdges(limit) {
  const max = limit || performanceLimits.maxEdges;

  if (cache.edges && isCacheValid(cache.edgesTime)) {
    return { edges: cache.edges.slice(0, max), total: cache.edges.length, cached: true };
  }

  // Fetch in pages of 5000
  const pageSize = 5000;
  let allEdges = [];
  let offset = 0;
  let total = Infinity;

  while (allEdges.length < max && offset < total) {
    const data = await fetchWithTimeout(
      `${API_BASE}/graph/edges/?limit=${pageSize}&offset=${offset}`
    );
    const batch = data.edges || [];
    total = data.total || 0;
    allEdges = allEdges.concat(batch);
    offset += pageSize;
    if (batch.length < pageSize) break;
  }

  cache.edges = allEdges;
  cache.edgesTime = Date.now();

  return { edges: allEdges.slice(0, max), total, cached: false };
}

// ─── Fetch drug neighborhood ────────────────────────────────────────────────
// Returns { center, nodes, edges, total_nodes, total_edges }
export async function fetchNeighborhood(drugName, hops) {
  const maxHops = hops || performanceLimits.maxHops;
  const cacheKey = `${drugName.toLowerCase()}:${maxHops}`;

  const cached = cache.neighborhoods.get(cacheKey);
  if (cached && isCacheValid(cached.time)) {
    return { ...cached.data, cached: true };
  }

  const data = await fetchWithTimeout(
    `${API_BASE}/graph/neighborhood/?drug=${encodeURIComponent(drugName)}&hops=${maxHops}&limit=${performanceLimits.maxNeighborhood}`
  );

  cache.neighborhoods.set(cacheKey, { data, time: Date.now() });

  return { ...data, cached: false };
}

// ─── Build full graph data (nodes + adjacency) from API ─────────────────────
// This is the main function that replaces the static JSON import.
// Returns { nodes: [...], adj: {...} } in the same format as gnn_real_data.json
export async function fetchFullGraphData(limits) {
  const nodeLimit = limits?.maxNodes || performanceLimits.maxNodes;
  const edgeLimit = limits?.maxEdges || performanceLimits.maxEdges;

  // Fetch nodes and edges in parallel
  const [nodeResult, edgeResult] = await Promise.all([
    fetchGraphNodes(nodeLimit),
    fetchGraphEdges(edgeLimit),
  ]);

  const nodes = nodeResult.nodes;
  const edges = edgeResult.edges;

  // Build adjacency list from edges
  const adj = {};
  for (const edge of edges) {
    const s = edge.source;
    const t = edge.target;
    if (!adj[s]) adj[s] = [];
    if (!adj[t]) adj[t] = [];
    // Avoid duplicates in adjacency
    if (!adj[s].includes(t)) adj[s].push(t);
    if (!adj[t].includes(s)) adj[t].push(s);
  }

  // Generate pseudo-positions using a deterministic layout
  // (Fibonacci sphere + jitter based on category for visual separation)
  const positionedNodes = nodes.map((n, i) => ({
    id: n.id,
    name: n.name,
    type: n.therapeutic_class || n.category || 'Unknown',
    category: n.category || '',
    degree: n.degree || 0,
    smiles: n.smiles || '',
    // Generate 3D positions using golden-ratio sphere distribution
    pos: fibonacciSpherePosition(i, nodes.length, 30),
  }));

  return {
    nodes: positionedNodes,
    adj,
    edgeMeta: edges, // Keep severity info for edge coloring
    stats: {
      totalNodes: nodeResult.total,
      totalEdges: edgeResult.total,
      fetchedNodes: nodes.length,
      fetchedEdges: edges.length,
    },
  };
}

// ─── Fibonacci sphere for initial node placement ────────────────────────────
// Distributes N points evenly on a sphere of given radius
function fibonacciSpherePosition(index, total, radius) {
  const goldenRatio = (1 + Math.sqrt(5)) / 2;
  const theta = 2 * Math.PI * index / goldenRatio;
  const phi = Math.acos(1 - (2 * (index + 0.5)) / total);

  // Add deterministic jitter for visual interest
  const jitter = ((index * 7919) % 1000) / 1000 * 0.15; // pseudo-random 0-0.15

  const x = (radius + jitter * radius) * Math.sin(phi) * Math.cos(theta);
  const y = (radius + jitter * radius) * Math.sin(phi) * Math.sin(theta);
  const z = (radius + jitter * radius) * Math.cos(phi);

  return [x, y, z];
}

// ─── Fallback: load static data if API is unavailable ───────────────────────
let staticDataPromise = null;

export async function fetchStaticFallback() {
  if (staticDataPromise) return staticDataPromise;

  staticDataPromise = import('../../assets/gnn_real_data.json')
    .then(module => {
      const raw = module.default || module;
      return {
        nodes: raw.nodes || [],
        adj: raw.adj || {},
        edgeMeta: [],
        stats: {
          totalNodes: raw.nodes?.length || 0,
          totalEdges: Object.values(raw.adj || {}).reduce((s, a) => s + a.length, 0) / 2,
          fetchedNodes: raw.nodes?.length || 0,
          fetchedEdges: Object.values(raw.adj || {}).reduce((s, a) => s + a.length, 0) / 2,
          isStatic: true,
        },
      };
    })
    .catch(() => ({
      nodes: [],
      adj: {},
      edgeMeta: [],
      stats: { totalNodes: 0, totalEdges: 0, fetchedNodes: 0, fetchedEdges: 0, isStatic: true },
    }));

  return staticDataPromise;
}

// ─── Main entry: fetch with fallback ────────────────────────────────────────
// Tries API first, falls back to static JSON if API fails
export async function loadGraphData(limits) {
  try {
    const data = await fetchFullGraphData(limits);
    if (data.nodes.length > 0) {
      return { ...data, source: 'api' };
    }
    throw new Error('API returned 0 nodes');
  } catch (apiErr) {
    console.warn('[GalaxyViewer] API unavailable, falling back to static data:', apiErr.message);
    const fallback = await fetchStaticFallback();
    return { ...fallback, source: 'static', apiError: apiErr.message };
  }
}

// ─── Cache management ───────────────────────────────────────────────────────
export function clearGraphCache() {
  cache.nodes = null;
  cache.nodesTime = 0;
  cache.edges = null;
  cache.edgesTime = 0;
  cache.neighborhoods.clear();
}

export function getCacheStats() {
  return {
    hasNodes: !!cache.nodes,
    nodeCount: cache.nodes?.length || 0,
    hasEdges: !!cache.edges,
    edgeCount: cache.edges?.length || 0,
    neighborhoodsCached: cache.neighborhoods.size,
    nodesAge: cache.nodesTime ? Math.round((Date.now() - cache.nodesTime) / 1000) : null,
  };
}
