// graphDataService.js — Dynamic data layer for the Galaxy Viewer
// Fetches graph data from the backend API (Neo4j AuraDB) with caching and fallback.
// Replaces the static gnn_real_data.json import for live data.

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

const NAME_ALIASES = {
  tylenol: 'acetaminophen',
  advil: 'ibuprofen',
  motrin: 'ibuprofen',
  aleve: 'naproxen',
  coumadin: 'warfarin',
  lipitor: 'atorvastatin',
  zocor: 'simvastatin',
  crestor: 'rosuvastatin',
  norvasc: 'amlodipine',
  prilosec: 'omeprazole',
  nexium: 'esomeprazole',
  zantac: 'ranitidine',
  pepcid: 'famotidine',
};

function normalizeDrugName(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function resolveNameCandidates(name) {
  const normalized = normalizeDrugName(name);
  const alias = NAME_ALIASES[normalized] || null;
  return Array.from(new Set([normalized, alias].filter(Boolean)));
}

function asFiniteNumber(value) {
  if (value === null || value === undefined) return null;

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const lowered = trimmed.toLowerCase();
    if (lowered === 'null' || lowered === 'undefined' || lowered === 'nan') {
      return null;
    }
  }

  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function parseCoordinateTriplet(a, b, c) {
  const x = asFiniteNumber(a);
  const y = asFiniteNumber(b);
  const z = asFiniteNumber(c);
  if (x === null || y === null || z === null) return null;
  return [x, y, z];
}

function parseNodePosition(node) {
  if (!node || typeof node !== 'object') return null;

  if (Array.isArray(node.pos) && node.pos.length >= 3) {
    return parseCoordinateTriplet(node.pos[0], node.pos[1], node.pos[2]);
  }

  if (node.pos && typeof node.pos === 'object') {
    const posObj = parseCoordinateTriplet(node.pos.x, node.pos.y, node.pos.z);
    if (posObj) return posObj;
  }

  const direct = parseCoordinateTriplet(node.x, node.y, node.z);
  if (direct) return direct;

  const tsne = parseCoordinateTriplet(node.tsne_x, node.tsne_y, node.tsne_z);
  if (tsne) return tsne;

  const embedding = parseCoordinateTriplet(node.embedding_x, node.embedding_y, node.embedding_z);
  if (embedding) return embedding;

  return null;
}

function averagePositions(entries) {
  if (!entries || entries.length === 0) return null;

  let wx = 0;
  let wy = 0;
  let wz = 0;
  let totalWeight = 0;

  entries.forEach(({ pos, weight }) => {
    const w = Number.isFinite(weight) && weight > 0 ? weight : 1;
    wx += pos[0] * w;
    wy += pos[1] * w;
    wz += pos[2] * w;
    totalWeight += w;
  });

  if (totalWeight <= 0) return null;
  return [wx / totalWeight, wy / totalWeight, wz / totalWeight];
}

function hashString(value) {
  let hash = 2166136261;
  const str = String(value || '');
  for (let i = 0; i < str.length; i += 1) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function getOrCreateAdj(adj, id) {
  if (!adj[id]) adj[id] = [];
  return adj[id];
}

function buildTopologyLayoutForComponent(componentIds, adj, knownPositions, componentIndex, anchorCenter = null) {
  if (!componentIds || componentIds.length === 0) return;

  const compSet = new Set(componentIds);
  const degrees = new Map(componentIds.map((id) => [id, (adj[id] || []).filter(n => compSet.has(n)).length]));

  let hubId = componentIds[0];
  componentIds.forEach((id) => {
    if ((degrees.get(id) || 0) > (degrees.get(hubId) || 0)) {
      hubId = id;
    }
  });

  const distance = new Map();
  const queue = [hubId];
  distance.set(hubId, 0);

  while (queue.length > 0) {
    const curr = queue.shift();
    const currDist = distance.get(curr) || 0;
    (adj[curr] || []).forEach((next) => {
      if (!compSet.has(next) || distance.has(next)) return;
      distance.set(next, currDist + 1);
      queue.push(next);
    });
  }

  const groups = new Map();
  componentIds.forEach((id) => {
    const d = distance.get(id) || 0;
    if (!groups.has(d)) groups.set(d, []);
    groups.get(d).push(id);
  });

  const sortedDists = Array.from(groups.keys()).sort((a, b) => a - b);
  sortedDists.forEach((d) => groups.get(d).sort());

  const center = anchorCenter || (() => {
    const angle = componentIndex * 1.61803398875;
    const radius = 46 + componentIndex * 4;
    return [Math.cos(angle) * radius, ((componentIndex % 5) - 2) * 6, Math.sin(angle) * radius];
  })();

  const basePhase = (hashString(hubId) % 360) * (Math.PI / 180);

  sortedDists.forEach((distValue) => {
    const group = groups.get(distValue) || [];
    const ringRadius = distValue === 0 ? 0 : 3.2 * distValue;
    const count = group.length;
    group.forEach((id, idx) => {
      let x = center[0];
      let y = center[1];
      let z = center[2];

      if (distValue > 0) {
        const localAngle = basePhase + (2 * Math.PI * idx) / Math.max(count, 1);
        x += Math.cos(localAngle) * ringRadius;
        z += Math.sin(localAngle) * ringRadius;
        y += (((idx % 3) - 1) * 0.9) + distValue * 0.28;
      }

      knownPositions.set(id, [x, y, z]);
    });
  });
}

function connectedComponents(nodeIds, adj) {
  const remaining = new Set(nodeIds);
  const components = [];

  while (remaining.size > 0) {
    const seed = remaining.values().next().value;
    const queue = [seed];
    const comp = [];
    remaining.delete(seed);

    while (queue.length > 0) {
      const curr = queue.shift();
      comp.push(curr);
      (adj[curr] || []).forEach((next) => {
        if (!remaining.has(next)) return;
        remaining.delete(next);
        queue.push(next);
      });
    }

    components.push(comp);
  }

  return components;
}

let staticDataPromise = null;
let realCoordIndexPromise = null;

async function getRealCoordinateIndex() {
  if (realCoordIndexPromise) return realCoordIndexPromise;

  realCoordIndexPromise = import('../../assets/gnn_real_data.json')
    .then((module) => {
      const raw = module.default || module;
      const byName = new Map();

      (raw.nodes || []).forEach((node) => {
        const pos = parseNodePosition(node);
        if (!pos) return;

        resolveNameCandidates(node.name).forEach((candidate) => {
          if (!byName.has(candidate)) {
            byName.set(candidate, pos);
          }
        });
      });

      return { byName };
    })
    .catch(() => ({ byName: new Map() }));

  return realCoordIndexPromise;
}

function getAtlasPosition(name, index) {
  const candidates = resolveNameCandidates(name);
  for (const candidate of candidates) {
    const pos = index.byName.get(candidate);
    if (pos) return pos;
  }
  return null;
}

function buildMeaningfulPositions(nodes, adj, atlasIndex) {
  const knownPositions = new Map();
  const sourceById = new Map();
  const unresolvedIds = [];

  nodes.forEach((node) => {
    const id = String(node.id || '');
    if (!id) return;

    const apiPos = parseNodePosition(node);
    if (apiPos) {
      knownPositions.set(id, apiPos);
      sourceById.set(id, 'api-coordinate');
      return;
    }

    const atlasPos = getAtlasPosition(node.name, atlasIndex);
    if (atlasPos) {
      knownPositions.set(id, atlasPos);
      sourceById.set(id, 'model-tsne');
      return;
    }

    unresolvedIds.push(id);
  });

  // Interpolate unknown nodes from neighbors with known positions.
  let frontier = unresolvedIds.slice();
  for (let iter = 0; iter < 10 && frontier.length > 0; iter += 1) {
    const next = [];

    frontier.forEach((id) => {
      const neighborEntries = (adj[id] || [])
        .filter((nId) => knownPositions.has(nId))
        .map((nId) => {
          const degree = (adj[nId] || []).length;
          return {
            pos: knownPositions.get(nId),
            weight: 1 / Math.max(1, degree),
          };
        });

      const blended = averagePositions(neighborEntries);
      if (blended) {
        knownPositions.set(id, blended);
        sourceById.set(id, 'topology-interpolated');
      } else {
        next.push(id);
      }
    });

    frontier = next;
  }

  // For isolated unresolved components, place deterministically from topology only.
  if (frontier.length > 0) {
    const components = connectedComponents(frontier, adj);

    components.forEach((component, compIdx) => {
      const componentSet = new Set(component);
      const anchorEntries = [];

      component.forEach((id) => {
        (adj[id] || []).forEach((neighborId) => {
          if (componentSet.has(neighborId)) return;
          const pos = knownPositions.get(neighborId);
          if (!pos) return;
          anchorEntries.push({ pos, weight: 1 });
        });
      });

      const anchorCenter = averagePositions(anchorEntries);
      buildTopologyLayoutForComponent(component, adj, knownPositions, compIdx, anchorCenter);
      component.forEach((id) => sourceById.set(id, 'topology-derived'));
    });
  }

  const sourceCounts = {};
  sourceById.forEach((source) => {
    sourceCounts[source] = (sourceCounts[source] || 0) + 1;
  });

  return { knownPositions, sourceById, sourceCounts };
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
  const atlasIndex = await getRealCoordinateIndex();

  // Build adjacency list from edges
  const adj = {};
  for (const edge of edges) {
    const s = String(edge.source || '');
    const t = String(edge.target || '');
    if (!s || !t || s === t) continue;
    getOrCreateAdj(adj, s);
    getOrCreateAdj(adj, t);
    // Avoid duplicates in adjacency
    if (!adj[s].includes(t)) adj[s].push(t);
    if (!adj[t].includes(s)) adj[t].push(s);
  }

  // Ensure every node appears in adjacency, even if isolated.
  nodes.forEach((n) => {
    const id = String(n.id || '');
    if (!id) return;
    getOrCreateAdj(adj, id);
  });

  const { knownPositions, sourceById, sourceCounts } = buildMeaningfulPositions(nodes, adj, atlasIndex);

  const positionedNodes = nodes.map((n) => {
    const id = String(n.id || '');
    const pos = knownPositions.get(id) || [0, 0, 0];
    return {
      id,
    name: n.name,
    type: n.therapeutic_class || n.category || 'Unknown',
    category: n.category || '',
    degree: n.degree || 0,
    smiles: n.smiles || '',
      pos,
      position_source: sourceById.get(id) || 'unknown',
    };
  });

  return {
    nodes: positionedNodes,
    adj,
    edgeMeta: edges, // Keep severity info for edge coloring
    stats: {
      totalNodes: nodeResult.total,
      totalEdges: edgeResult.total,
      fetchedNodes: nodes.length,
      fetchedEdges: edges.length,
      coordinateSources: sourceCounts,
    },
  };
}

// ─── Fallback: load static data if API is unavailable ───────────────────────
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
