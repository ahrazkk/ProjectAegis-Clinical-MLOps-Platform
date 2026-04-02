// graphEngine.js — Pure computation module for GNN Galaxy Viewer
// No React dependencies — all functions are pure JS
// Now accepts dynamic data from graphDataService instead of static JSON

// ─── Module-level graph data (set dynamically) ─────────────────────────────
let _graphData = { nodes: [], adj: {}, edgeMeta: [] };

export function setGraphData(data) {
  _graphData = {
    nodes: data.nodes || [],
    adj: data.adj || {},
    edgeMeta: data.edgeMeta || [],
  };
}

export function getGraphData() {
  return _graphData;
}

// ─── Therapeutic class meta-categories ─────────────────────────────────────
const CLASS_MAP = {
  // Cardiovascular
  'Antihypertensive': 'Cardiovascular', 'ACE Inhibitor': 'Cardiovascular', 'ARB': 'Cardiovascular',
  'Beta Blocker': 'Cardiovascular', 'Calcium Channel Blocker': 'Cardiovascular', 'Diuretic': 'Cardiovascular',
  'Antiarrhythmic': 'Cardiovascular', 'Vasodilator': 'Cardiovascular', 'Statin': 'Cardiovascular',
  'Anticoagulant': 'Cardiovascular', 'Antiplatelet': 'Cardiovascular', 'Fibrate': 'Cardiovascular',
  // CNS
  'SSRI': 'CNS Agents', 'SNRI': 'CNS Agents', 'TCA': 'CNS Agents', 'Antipsychotic': 'CNS Agents',
  'Benzodiazepine': 'CNS Agents', 'Anticonvulsant': 'CNS Agents', 'Anxiolytic': 'CNS Agents',
  'Sedative': 'CNS Agents', 'Hypnotic': 'CNS Agents', 'Antidepressant': 'CNS Agents',
  'Mood Stabilizer': 'CNS Agents', 'Stimulant': 'CNS Agents', 'Opioid': 'CNS Agents',
  // Anti-infective
  'Antibiotic': 'Anti-infectives', 'Antifungal': 'Anti-infectives', 'Antiviral': 'Anti-infectives',
  'Antiparasitic': 'Anti-infectives', 'Antimalarial': 'Anti-infectives', 'Antiretroviral': 'Anti-infectives',
  'Fluoroquinolone': 'Anti-infectives', 'Macrolide': 'Anti-infectives', 'Cephalosporin': 'Anti-infectives',
  // Anti-inflammatory
  'NSAID': 'Anti-inflammatory', 'Corticosteroid': 'Anti-inflammatory', 'DMARD': 'Anti-inflammatory',
  'Immunosuppressant': 'Immunology', 'Immunomodulator': 'Immunology', 'Biologic': 'Immunology',
  // Endocrine
  'Antidiabetic': 'Endocrine', 'Thyroid': 'Endocrine', 'Insulin': 'Endocrine', 'Sulfonylurea': 'Endocrine',
  'Hormone': 'Endocrine', 'Contraceptive': 'Endocrine',
  // Oncology
  'Antineoplastic': 'Oncology', 'Chemotherapy': 'Oncology', 'Kinase Inhibitor': 'Oncology',
  // GI
  'PPI': 'Gastrointestinal', 'Antacid': 'Gastrointestinal', 'Antiemetic': 'Gastrointestinal',
  'Laxative': 'Gastrointestinal', 'H2 Blocker': 'Gastrointestinal',
  // Respiratory
  'Bronchodilator': 'Respiratory', 'Antihistamine': 'Respiratory', 'Decongestant': 'Respiratory',
  // Analgesic
  'Analgesic': 'Analgesics', 'Anesthetic': 'Analgesics',
  // Musculoskeletal
  'Muscle Relaxant': 'Musculoskeletal', 'Bisphosphonate': 'Musculoskeletal',
};

// Category → color mapping (editorial palette)
export const CATEGORY_COLORS = {
  'Cardiovascular':    '#ef4444',
  'CNS Agents':        '#8B5CF6',
  'Anti-infectives':   '#10b981',
  'Anti-inflammatory': '#f97316',
  'Immunology':        '#EC4899',
  'Endocrine':         '#fbbf24',
  'Oncology':          '#dc2626',
  'Gastrointestinal':  '#06b6d4',
  'Respiratory':       '#3b82f6',
  'Analgesics':        '#f59e0b',
  'Musculoskeletal':   '#14b8a6',
  'Other':             '#475569',
};

export const CATEGORY_LIST = Object.keys(CATEGORY_COLORS);

function classifyDrug(type) {
  if (!type || type === 'Unknown') return 'Other';
  // Direct match
  if (CLASS_MAP[type]) return CLASS_MAP[type];
  // Partial match
  const lowerType = type.toLowerCase();
  for (const [key, cat] of Object.entries(CLASS_MAP)) {
    if (lowerType.includes(key.toLowerCase())) return cat;
  }
  return 'Other';
}

const DRUG_NAME_ALIASES = {
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
  erbitux: 'cetuximab',
};

function normalizeLookupName(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// ─── Build node dictionary from current graph data ──────────────────────────
export function buildNodeDict(scale = 0.35) {
  const nodes = _graphData.nodes;
  if (!nodes || nodes.length === 0) return {};
  const dict = {};
  nodes.forEach((n, index) => {
    const pos = n.pos || [0, 0, 0];
    dict[n.id] = {
      ...n,
      index,
      pos: [pos[0] * scale, pos[1] * scale, pos[2] * scale],
      category: classifyDrug(n.type || n.therapeutic_class || n.category),
      isA: false,
      isB: false,
      isSelected: false,
      hopA: Infinity,
      hopB: Infinity,
      hopAny: Infinity,
    };
  });
  return dict;
}

// ─── Find drug by name ─────────────────────────────────────────────────────
export function findDrugByName(nodeDict, name) {
  if (!name) return null;

  const normalized = normalizeLookupName(name);
  const alias = DRUG_NAME_ALIASES[normalized] || null;
  const candidates = Array.from(new Set([normalized, alias].filter(Boolean)));

  for (const candidate of candidates) {
    // Exact normalized match
    for (const id of Object.keys(nodeDict)) {
      const nodeName = normalizeLookupName(nodeDict[id].name);
      if (nodeName === candidate) return id;
    }

    // Partial normalized match
    for (const id of Object.keys(nodeDict)) {
      const nodeName = normalizeLookupName(nodeDict[id].name);
      if (nodeName.includes(candidate) || candidate.includes(nodeName)) return id;
    }
  }

  return null;
}

// ─── BFS hop distances ─────────────────────────────────────────────────────
export function bfsHops(adj, startId, maxHops = 3) {
  if (startId === null || !adj) return new Map();
  const distances = new Map();
  distances.set(startId, 0);
  const queue = [{ id: startId, d: 0 }];
  while (queue.length > 0) {
    const curr = queue.shift();
    if (curr.d >= maxHops) continue;
    const neighbors = adj[curr.id] || [];
    for (const nxt of neighbors) {
      if (!distances.has(nxt)) {
        distances.set(nxt, curr.d + 1);
        queue.push({ id: nxt, d: curr.d + 1 });
      }
    }
  }
  return distances;
}

export function bfsHopsMulti(adj, startIds = [], maxHops = 3) {
  if (!adj || !Array.isArray(startIds) || startIds.length === 0) return new Map();

  const distances = new Map();
  const queue = [];

  startIds.forEach((startId) => {
    if (startId !== null && startId !== undefined && !distances.has(startId)) {
      distances.set(startId, 0);
      queue.push({ id: startId, d: 0 });
    }
  });

  while (queue.length > 0) {
    const curr = queue.shift();
    if (curr.d >= maxHops) continue;

    const neighbors = adj[curr.id] || [];
    for (const nxt of neighbors) {
      if (!distances.has(nxt)) {
        distances.set(nxt, curr.d + 1);
        queue.push({ id: nxt, d: curr.d + 1 });
      }
    }
  }

  return distances;
}

// ─── Shortest path (BFS) ──────────────────────────────────────────────────
export function shortestPath(adj, fromId, toId) {
  if (!adj || fromId === null || toId === null || fromId === toId) return [];
  const visited = new Set([fromId]);
  const queue = [{ id: fromId, path: [fromId] }];
  while (queue.length > 0) {
    const curr = queue.shift();
    const neighbors = adj[curr.id] || [];
    for (const nxt of neighbors) {
      if (nxt === toId) return [...curr.path, toId];
      if (!visited.has(nxt)) {
        visited.add(nxt);
        queue.push({ id: nxt, path: [...curr.path, nxt] });
      }
    }
  }
  return []; // no path found
}

function edgeKey(a, b) {
  const u = String(a);
  const v = String(b);
  return u < v ? `${u}-${v}` : `${v}-${u}`;
}

function shouldKeepEmbeddingBackgroundEdge(key) {
  // Deterministic sparse sampling to reduce visual clutter in global embedding view.
  let hash = 2166136261;
  for (let i = 0; i < key.length; i += 1) {
    hash ^= key.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % 18 === 0; // ~5.5% of background edges
}

function squaredDistance3(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

function buildEmbeddingKnnEdgeMap(nodeDict, k = 8) {
  const ids = Object.keys(nodeDict || {});
  const edgeMap = new Map();
  if (ids.length < 2) return edgeMap;

  const clampedK = Math.max(1, Math.min(Math.round(k) || 8, ids.length - 1));

  for (let i = 0; i < ids.length; i += 1) {
    const sourceId = ids[i];
    const sourcePos = nodeDict[sourceId]?.pos;
    if (!sourcePos) continue;

    const nearest = [];

    for (let j = 0; j < ids.length; j += 1) {
      if (i === j) continue;

      const targetId = ids[j];
      const targetPos = nodeDict[targetId]?.pos;
      if (!targetPos) continue;

      const distanceSq = squaredDistance3(sourcePos, targetPos);
      const candidate = { id: targetId, distanceSq };

      let insertAt = -1;
      for (let idx = 0; idx < nearest.length; idx += 1) {
        if (distanceSq < nearest[idx].distanceSq) {
          insertAt = idx;
          break;
        }
      }

      if (insertAt === -1) {
        if (nearest.length < clampedK) {
          nearest.push(candidate);
        }
      } else {
        nearest.splice(insertAt, 0, candidate);
        if (nearest.length > clampedK) nearest.pop();
      }
    }

    nearest.forEach((neighbor) => {
      const key = edgeKey(sourceId, neighbor.id);
      const existing = edgeMap.get(key);
      if (existing && existing.distanceSq <= neighbor.distanceSq) return;

      edgeMap.set(key, {
        startId: sourceId,
        endId: neighbor.id,
        start: nodeDict[sourceId].pos,
        end: nodeDict[neighbor.id].pos,
        color: '#334155',
        opacity: 0.03,
        lineWidth: 1,
        role: 'embedding-knn',
        severity: 'unknown',
        distance: Math.sqrt(neighbor.distanceSq),
        distanceSq: neighbor.distanceSq,
        priority: 0,
      });
    });
  }

  return edgeMap;
}

function upsertEmbeddingEdge(edgeMap, nodeDict, nodeA, nodeB, style) {
  if (!nodeA || !nodeB || nodeA === nodeB) return;
  if (!nodeDict[nodeA] || !nodeDict[nodeB]) return;

  const key = edgeKey(nodeA, nodeB);
  const existing = edgeMap.get(key);
  const nextPriority = style.priority || 1;
  const currentPriority = existing?.priority || 0;
  if (existing && currentPriority > nextPriority) return;

  const base = existing || {
    startId: nodeA,
    endId: nodeB,
    start: nodeDict[nodeA].pos,
    end: nodeDict[nodeB].pos,
    distance: null,
    distanceSq: null,
    severity: 'unknown',
  };

  edgeMap.set(key, {
    ...base,
    ...style,
    priority: nextPriority,
  });
}

function normalizeSeverityLevel(severity) {
  const raw = String(severity || 'unknown').toLowerCase();
  if (raw === 'severe') return 'critical';
  if (raw === 'high') return 'major';
  if (raw === 'none' || raw === 'no_interaction') return 'minor';
  if (raw === 'minor' || raw === 'moderate' || raw === 'major' || raw === 'critical') return raw;
  return 'unknown';
}

function shortestPathAvoidingNodes(adj, fromId, toId, blockedNodes = new Set()) {
  if (!adj || fromId === null || toId === null || fromId === toId) return [];

  const visited = new Set([fromId]);
  const queue = [{ id: fromId, path: [fromId] }];

  while (queue.length > 0) {
    const curr = queue.shift();
    const neighbors = adj[curr.id] || [];

    for (const nxt of neighbors) {
      if (nxt !== toId && blockedNodes.has(nxt)) continue;
      if (visited.has(nxt)) continue;

      if (nxt === toId) {
        return [...curr.path, toId];
      }

      visited.add(nxt);
      queue.push({ id: nxt, path: [...curr.path, nxt] });
    }
  }

  return [];
}

function scorePathReuse(path, usedNodes, usedEdges, selectedSet) {
  if (!Array.isArray(path) || path.length < 2) return Number.POSITIVE_INFINITY;

  let reusedNodeCount = 0;
  let reusedEdgeCount = 0;

  for (let i = 0; i < path.length; i += 1) {
    const curr = path[i];
    const isIntermediate = i > 0 && i < path.length - 1;
    if (isIntermediate && !selectedSet.has(curr) && usedNodes.has(curr)) {
      reusedNodeCount += 1;
    }

    if (i > 0) {
      const prev = path[i - 1];
      if (usedEdges.has(edgeKey(prev, curr))) {
        reusedEdgeCount += 1;
      }
    }
  }

  // Penalize reused bridge nodes more than reused edges to diversify connector routes.
  return reusedNodeCount * 3 + reusedEdgeCount;
}

function choosePreferredConnectorPath(adj, fromId, toId, usedNodes, usedEdges, selectedSet) {
  const basePath = shortestPath(adj, fromId, toId);
  if (!basePath || basePath.length < 2) {
    return { path: [], hops: Number.POSITIVE_INFINITY, reuseScore: Number.POSITIVE_INFINITY };
  }

  const baseHops = basePath.length - 1;
  const blockedNodes = new Set(
    Array.from(usedNodes).filter(id => id !== fromId && id !== toId && !selectedSet.has(id))
  );

  let preferredPath = basePath;
  if (blockedNodes.size > 0) {
    const altPath = shortestPathAvoidingNodes(adj, fromId, toId, blockedNodes);
    const altHops = altPath.length > 0 ? altPath.length - 1 : Number.POSITIVE_INFINITY;

    // Allow a small hop increase if it avoids repeatedly routing through the same connector hub.
    if (altPath.length > 0 && altHops <= baseHops + 1) {
      preferredPath = altPath;
    }
  }

  return {
    path: preferredPath,
    hops: preferredPath.length - 1,
    reuseScore: scorePathReuse(preferredPath, usedNodes, usedEdges, selectedSet),
  };
}

function computeMinimalConnector(adj, selectedIds) {
  const uniqueIds = Array.from(new Set((selectedIds || []).filter(Boolean)));
  const nodeSet = new Set(uniqueIds);
  const edgeSet = new Set();
  const connectorPaths = [];

  if (!adj || uniqueIds.length < 2) {
    return { nodeSet, edgeSet, connectorPaths };
  }

  const selectedSet = new Set(uniqueIds);
  const idToIndex = new Map(uniqueIds.map((id, idx) => [id, idx]));
  const parent = uniqueIds.map((_, idx) => idx);

  const findRoot = (idx) => {
    let root = idx;
    while (parent[root] !== root) {
      root = parent[root];
    }
    while (parent[idx] !== idx) {
      const next = parent[idx];
      parent[idx] = root;
      idx = next;
    }
    return root;
  };

  const unionIds = (idA, idB) => {
    const idxA = idToIndex.get(idA);
    const idxB = idToIndex.get(idB);
    if (idxA === undefined || idxB === undefined) return;
    const rootA = findRoot(idxA);
    const rootB = findRoot(idxB);
    if (rootA !== rootB) {
      parent[rootB] = rootA;
    }
  };

  const buildSelectedComponents = () => {
    const byRoot = new Map();
    uniqueIds.forEach((id) => {
      const root = findRoot(idToIndex.get(id));
      if (!byRoot.has(root)) byRoot.set(root, new Set());
      byRoot.get(root).add(id);
    });
    return Array.from(byRoot.values());
  };

  // Keep all direct selected-selected edges in Focus mode so visible direct links are never dropped.
  uniqueIds.forEach((u) => {
    (adj[u] || []).forEach((v) => {
      if (!selectedSet.has(v)) return;
      const key = edgeKey(u, v);
      if (edgeSet.has(key)) return;
      edgeSet.add(key);
      unionIds(u, v);
    });
  });

  const usedConnectorNodes = new Set();
  const usedConnectorEdges = new Set(edgeSet);
  let components = buildSelectedComponents();

  while (components.length > 1) {
    let best = null;

    for (let i = 0; i < components.length; i += 1) {
      for (let j = i + 1; j < components.length; j += 1) {
        for (const fromId of components[i]) {
          for (const toId of components[j]) {
            const candidate = choosePreferredConnectorPath(
              adj,
              fromId,
              toId,
              usedConnectorNodes,
              usedConnectorEdges,
              selectedSet,
            );

            if (!candidate.path || candidate.path.length < 2) continue;

            if (
              !best ||
              candidate.hops < best.hops ||
              (candidate.hops === best.hops && candidate.reuseScore < best.reuseScore)
            ) {
              best = {
                ...candidate,
                componentA: i,
                componentB: j,
              };
            }
          }
        }
      }
    }

    if (!best) {
      break;
    }

    const path = best.path;
    connectorPaths.push(path);

    for (let i = 0; i < path.length; i += 1) {
      const nodeId = path[i];
      nodeSet.add(nodeId);

      if (i > 0) {
        const prev = path[i - 1];
        const key = edgeKey(prev, nodeId);
        edgeSet.add(key);
        usedConnectorEdges.add(key);
      }

      if (i > 0 && i < path.length - 1 && !selectedSet.has(nodeId)) {
        usedConnectorNodes.add(nodeId);
      }
    }

    const selectedOnPath = path.filter(nodeId => selectedSet.has(nodeId));
    if (selectedOnPath.length >= 2) {
      const rootId = selectedOnPath[0];
      for (let i = 1; i < selectedOnPath.length; i += 1) {
        unionIds(rootId, selectedOnPath[i]);
      }
    }

    components = buildSelectedComponents();
  }

  return { nodeSet, edgeSet, connectorPaths };
}

// ─── Get adjacency list from current data ──────────────────────────────────
export function getAdj() {
  return _graphData.adj || {};
}

// ─── Get edge metadata (severity) for a pair ───────────────────────────────
const _edgeMetaIndex = new Map();

export function buildEdgeMetaIndex() {
  _edgeMetaIndex.clear();
  for (const e of (_graphData.edgeMeta || [])) {
    const key = e.source < e.target ? `${e.source}-${e.target}` : `${e.target}-${e.source}`;
    _edgeMetaIndex.set(key, e);
  }
}

export function getEdgeMeta(nodeA, nodeB) {
  const key = nodeA < nodeB ? `${nodeA}-${nodeB}` : `${nodeB}-${nodeA}`;
  return _edgeMetaIndex.get(key) || null;
}

// ─── Compute full subgraph state ──────────────────────────────────────────
export function computeSubgraph(
  nodeDict,
  drugAId,
  drugBId,
  maxHops = 3,
  selectedDrugIds = [],
  interactionPairs = [],
  viewMode = 'galaxy',
  filters = null,
) {
  const adj = _graphData.adj || {};
  const isFocusMode = viewMode === 'focus';
  const isEmbeddingMode = viewMode === 'embedding';
  const embeddingEdgeMode = isEmbeddingMode ? String(filters?.embeddingEdgeMode || 'knn').toLowerCase() : 'graph';
  const embeddingK = isEmbeddingMode ? Math.max(2, Math.min(20, Math.round(filters?.embeddingK || 8))) : 8;

  const selectedSet = new Set((selectedDrugIds || []).filter(Boolean));
  if (drugAId) selectedSet.add(drugAId);
  if (drugBId) selectedSet.add(drugBId);
  const selectedIds = Array.from(selectedSet);

  const interactionPairSet = new Set();
  (interactionPairs || []).forEach((pair) => {
    if (!Array.isArray(pair) || pair.length < 2) return;
    const a = String(pair[0] || '');
    const b = String(pair[1] || '');
    if (!a || !b) return;
    interactionPairSet.add(edgeKey(a, b));
  });

  const minimalConnector = isFocusMode
    ? computeMinimalConnector(adj, selectedIds)
    : { nodeSet: new Set(selectedIds), edgeSet: new Set(), connectorPaths: [] };
  const focusNodeSet = minimalConnector.nodeSet;
  const focusEdgeSet = minimalConnector.edgeSet;

  // Reset hop fields
  Object.values(nodeDict).forEach(n => {
    n.isA = false;
    n.isB = false;
    n.isSelected = false;
    n.isFocusNode = false;
    n.hopA = Infinity;
    n.hopB = Infinity;
    n.hopAny = Infinity;
  });

  if (drugAId && nodeDict[drugAId]) nodeDict[drugAId].isA = true;
  if (drugBId && nodeDict[drugBId]) nodeDict[drugBId].isB = true;
  selectedIds.forEach((id) => {
    if (nodeDict[id]) nodeDict[id].isSelected = true;
  });

  // BFS from both drugs
  const hopsA = bfsHops(adj, drugAId, maxHops);
  const hopsB = bfsHops(adj, drugBId, maxHops);
  const hopsAny = bfsHopsMulti(adj, selectedIds, maxHops);

  hopsA.forEach((d, id) => { if (nodeDict[id]) nodeDict[id].hopA = d; });
  hopsB.forEach((d, id) => { if (nodeDict[id]) nodeDict[id].hopB = d; });
  hopsAny.forEach((d, id) => { if (nodeDict[id]) nodeDict[id].hopAny = d; });

  if (isFocusMode && selectedIds.length > 0) {
    Object.values(nodeDict).forEach((n) => {
      const inFocusNode = focusNodeSet.has(n.id);
      n.isFocusNode = inFocusNode;
      if (!inFocusNode && !n.isA && !n.isB && !n.isSelected) {
        n.hopA = Infinity;
        n.hopB = Infinity;
        n.hopAny = Infinity;
      }
    });
  }

  // Compute shortest path between A and B
  const path = shortestPath(adj, drugAId, drugBId);
  const pathSet = new Set(path);

  // Embedding mode keeps selected drug highlighting but renders the complete latent atlas.
  const hasDrugs = selectedIds.length > 0 && !isEmbeddingMode;

  // Compute edges
  const edges = [];

  if (isEmbeddingMode && embeddingEdgeMode === 'knn') {
    const edgeMap = buildEmbeddingKnnEdgeMap(nodeDict, embeddingK);

    // Keep known selected-selected interaction links visible even if they are not in the KNN manifold.
    selectedIds.forEach((u) => {
      (adj[u] || []).forEach((v) => {
        if (!selectedSet.has(v)) return;
        const isDirectPair = (u === drugAId && v === drugBId) || (u === drugBId && v === drugAId);
        const meta = getEdgeMeta(u, v);
        upsertEmbeddingEdge(edgeMap, nodeDict, u, v, {
          color: isDirectPair ? '#ef4444' : '#22c55e',
          opacity: isDirectPair ? 0.95 : 0.8,
          lineWidth: isDirectPair ? 2.8 : 2.2,
          role: isDirectPair ? 'direct' : 'selected-pair',
          severity: normalizeSeverityLevel(meta?.severity || 'unknown'),
          priority: isDirectPair ? 6 : 4,
        });
      });
    });

    // Highlight known interaction pairs provided from polypharmacy analysis.
    (interactionPairs || []).forEach((pair) => {
      if (!Array.isArray(pair) || pair.length < 2) return;
      const u = String(pair[0] || '');
      const v = String(pair[1] || '');
      if (!u || !v) return;
      const meta = getEdgeMeta(u, v);
      upsertEmbeddingEdge(edgeMap, nodeDict, u, v, {
        color: '#ef4444',
        opacity: 0.92,
        lineWidth: 2.6,
        role: 'interaction-pair',
        severity: normalizeSeverityLevel(meta?.severity || 'unknown'),
        priority: 6,
      });
    });

    // Highlight shortest connector path between selected A/B when available.
    if (path.length >= 2) {
      for (let i = 1; i < path.length; i += 1) {
        const prev = path[i - 1];
        const curr = path[i];
        const meta = getEdgeMeta(prev, curr);
        upsertEmbeddingEdge(edgeMap, nodeDict, prev, curr, {
          color: '#ef4444',
          opacity: 1.0,
          lineWidth: 3,
          role: 'path',
          severity: normalizeSeverityLevel(meta?.severity || 'unknown'),
          priority: 7,
        });
      }
    }

    edgeMap.forEach((edge) => {
      const { priority, distanceSq, ...rest } = edge;
      edges.push(rest);
    });
  } else {
    const processed = new Set();
    Object.keys(adj).forEach(u => {
      const uNode = nodeDict[u];
      if (!uNode) return;
      (adj[u] || []).forEach(v => {
        const key = edgeKey(u, v);
        if (processed.has(key)) return;
        processed.add(key);
        const vNode = nodeDict[v];
        if (!vNode) return;

        const inHopA = uNode.hopA <= maxHops && vNode.hopA <= maxHops;
        const inHopB = uNode.hopB <= maxHops && vNode.hopB <= maxHops;
        const inHopAny = uNode.hopAny <= maxHops && vNode.hopAny <= maxHops;
        const isSelectedPair = uNode.isSelected && vNode.isSelected;
        const isInteractionPair = interactionPairSet.has(key);
        const inFocusConnector = focusEdgeSet.has(key);
        const onPath = pathSet.has(u) && pathSet.has(v) &&
          Math.abs(path.indexOf(u) - path.indexOf(v)) === 1;

        if (isEmbeddingMode) {
          let color = '#334155';
          let opacity = 0.018;
          let lineWidth = 0.9;
          let role = 'background';

          if (onPath || ((uNode.isA && vNode.isB) || (uNode.isB && vNode.isA))) {
            color = '#ef4444';
            opacity = 0.95;
            lineWidth = 2.8;
            role = 'path';
          } else if (isSelectedPair) {
            color = '#22c55e';
            opacity = 0.75;
            lineWidth = 2.2;
            role = 'selected-pair';
          } else if (uNode.isA || vNode.isA) {
            color = '#00d2ff';
            opacity = 0.42;
            lineWidth = 1.8;
            role = 'hopA';
          } else if (uNode.isB || vNode.isB) {
            color = '#ff8c00';
            opacity = 0.42;
            lineWidth = 1.8;
            role = 'hopB';
          }

          if (role !== 'background' || shouldKeepEmbeddingBackgroundEdge(key)) {
            edges.push({
              startId: u,
              endId: v,
              start: uNode.pos,
              end: vNode.pos,
              color,
              opacity,
              lineWidth,
              role,
              severity: 'unknown',
            });
          }
          return;
        }

        let color = '#475569';
        let opacity = 0.04;
        let lineWidth = 1;
        let role = 'background';

        // Get severity from edge metadata
        const meta = getEdgeMeta(u, v);
        const severity = normalizeSeverityLevel(meta?.severity || 'unknown');

        if (onPath) {
          color = '#ef4444'; opacity = 1.0; lineWidth = 3; role = 'path';
        } else if (isInteractionPair) {
          color = '#ef4444'; opacity = 0.92; lineWidth = 2.8; role = 'interaction-pair';
        } else if (isSelectedPair && inFocusConnector) {
          color = '#22c55e'; opacity = 0.95; lineWidth = 2.8; role = 'selected-pair';
        } else if (inFocusConnector) {
          color = '#06b6d4'; opacity = 0.95; lineWidth = 2.8; role = 'focus';
        } else if ((uNode.isA && vNode.isB) || (uNode.isB && vNode.isA)) {
          color = '#ef4444'; opacity = 1.0; lineWidth = 3; role = 'direct';
        } else if (isSelectedPair) {
          color = '#22c55e'; opacity = 0.9; lineWidth = 2.2; role = 'selected-pair';
        } else if (inHopA && inHopB) {
          color = '#a855f7'; opacity = 0.45; lineWidth = 1.5; role = 'bridge';
        } else if (inHopA) {
          color = '#00d2ff';
          opacity = Math.max(0.1, 0.35 - Math.max(uNode.hopA, vNode.hopA) * 0.08);
          role = 'hopA';
        } else if (inHopB) {
          color = '#ff8c00';
          opacity = Math.max(0.1, 0.35 - Math.max(uNode.hopB, vNode.hopB) * 0.08);
          role = 'hopB';
        } else if (inHopAny) {
          color = '#06b6d4';
          opacity = Math.max(0.08, 0.32 - Math.max(uNode.hopAny, vNode.hopAny) * 0.07);
          role = 'multi-hop';
        }

        // Severity-based color tinting for active edges
        if (role !== 'background' && severity === 'critical') {
          color = '#ef4444'; // red for severe
        } else if (role !== 'background' && severity === 'major') {
          color = role === 'path' ? '#ef4444' : '#f97316'; // orange for moderate
        } else if (role !== 'background' && severity === 'moderate') {
          color = role === 'path' ? '#ef4444' : '#f59e0b';
        }

        let shouldInclude = !hasDrugs || inHopA || inHopB || inHopAny || onPath || isSelectedPair || isInteractionPair;
        if (isFocusMode && hasDrugs) {
          shouldInclude = inFocusConnector || isSelectedPair || onPath || (isInteractionPair && focusNodeSet.has(u) && focusNodeSet.has(v));
        }

        if (shouldInclude) {
          edges.push({
            startId: u, endId: v,
            start: uNode.pos, end: vNode.pos,
            color, opacity, lineWidth, role, severity,
          });
        }
      });
    });
  }

  // Stats
  const totalNodes = _graphData.nodes.length;
  const totalEdges = Object.values(adj).reduce((sum, arr) => sum + arr.length, 0) / 2;
  const visibleNodes = Object.values(nodeDict).filter((n) => {
    if (!hasDrugs) return true;
    if (isFocusMode) {
      return n.isFocusNode || n.isA || n.isB || n.isSelected;
    }
    return n.hopAny <= maxHops;
  }).length;

  return {
    nodes: Object.values(nodeDict),
    edges,
    path,
    pathSet,
    stats: {
      totalNodes,
      totalEdges,
      visibleNodes: hasDrugs ? visibleNodes : totalNodes,
      visibleEdges: edges.length,
      pathLength: path.length > 0 ? path.length - 1 : -1,
      sharedNeighbors: Object.values(nodeDict).filter(n => n.hopA <= maxHops && n.hopB <= maxHops && !n.isA && !n.isB).length,
      selectedCount: selectedIds.length,
      interactionPairCount: interactionPairSet.size,
      focusConnectorEdges: focusEdgeSet.size,
      focusPathCount: minimalConnector.connectorPaths.length,
      focusMode: isFocusMode,
    },
  };
}

// ─── Get node visual properties ───────────────────────────────────────────
export function getNodeVisuals(node, hasDrugs, viewMode = 'galaxy') {
  const adj = _graphData.adj || {};
  const degree = (adj[node.id] || []).length;
  const degreeFactor = Math.min(Math.sqrt(degree) / 8, 1); // 0-1 normalized
  const isEmbeddingMode = viewMode === 'embedding';

  let color = '#1e293b';
  let opacity = 0.15;
  let size = 0.1;
  let glow = 0;

  if (node.isA) {
    color = '#00d2ff'; opacity = 1; size = 0.7; glow = 1;
  } else if (node.isB) {
    color = '#ff8c00'; opacity = 1; size = 0.7; glow = 1;
  } else if (node.isSelected) {
    color = '#22c55e'; opacity = 0.95; size = 0.58; glow = 0.9;
  } else if (isEmbeddingMode) {
    // Embedding mode should show global class structure independent of hop shells.
    color = CATEGORY_COLORS[node.category] || '#475569';
    opacity = 0.55 + degreeFactor * 0.18;
    size = 0.24 + degreeFactor * 0.22;
  } else if (node.hopA <= 3 && node.hopB <= 3) {
    color = '#a855f7';
    opacity = 0.75 - Math.max(node.hopA, node.hopB) * 0.12;
    size = 0.35 + degreeFactor * 0.1;
    glow = 0.4;
  } else if (node.hopA <= 3) {
    color = '#00d2ff';
    opacity = 0.6 - node.hopA * 0.15;
    size = 0.3 - node.hopA * 0.04 + degreeFactor * 0.1;
    glow = 0.3 - node.hopA * 0.08;
  } else if (node.hopB <= 3) {
    color = '#ff8c00';
    opacity = 0.6 - node.hopB * 0.15;
    size = 0.3 - node.hopB * 0.04 + degreeFactor * 0.1;
    glow = 0.3 - node.hopB * 0.08;
  } else if (node.hopAny <= 3) {
    color = '#06b6d4';
    opacity = 0.48 - node.hopAny * 0.1;
    size = 0.24 - node.hopAny * 0.03 + degreeFactor * 0.08;
    glow = 0.2;
  } else if (!hasDrugs) {
    // No drugs selected — show by category with degree-based sizing
    color = CATEGORY_COLORS[node.category] || '#475569';
    opacity = 0.25 + degreeFactor * 0.15;
    size = 0.12 + degreeFactor * 0.15;
  }

  return { color, opacity, size: Math.max(size, isEmbeddingMode ? 0.22 : 0.15), glow: Math.max(glow, 0) };
}

// ─── Sample background edges for "no selection" view ─────────────────────
export function sampleBackgroundEdges(maxCount = 3000) {
  const adj = _graphData.adj || {};
  const allEdges = [];
  const seen = new Set();
  Object.keys(adj).forEach(u => {
    (adj[u] || []).forEach(v => {
      const key = u < v ? `${u}-${v}` : `${v}-${u}`;
      if (!seen.has(key)) {
        seen.add(key);
        allEdges.push([u, v]);
      }
    });
  });
  // Shuffle and take maxCount
  for (let i = allEdges.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allEdges[i], allEdges[j]] = [allEdges[j], allEdges[i]];
  }
  return allEdges.slice(0, maxCount);
}

// ─── Apply filters to determine node/edge visibility ─────────────────────
export function applyFilters(nodes, edges, filters) {
  const adj = _graphData.adj || {};
  const {
    visibleClasses = null,
    minDegree = 0,
    edgeDensity = 1.0,
    severityLevels = ['minor', 'moderate', 'major', 'critical', 'unknown'],
  } = filters || {};

  const normalizedSeverityLevels = new Set(
    (severityLevels || []).map(level => normalizeSeverityLevel(level))
  );
  const hasFocusSubset = nodes.some(n => n.isFocusNode);

  // Determine which nodes pass filters
  const nodeVisibility = new Map();
  nodes.forEach(n => {
    let visible = true;

    // Class filter
    if (visibleClasses !== null && !visibleClasses.includes(n.category)) {
      visible = false;
    }

    // Degree filter
    if (minDegree > 0) {
      const degree = (adj[n.id] || []).length;
      if (degree < minDegree && !n.isA && !n.isB && !n.isSelected) {
        visible = false;
      }
    }

    // Selected drugs always visible
    if (n.isA || n.isB || n.isSelected) visible = true;

    // In focus mode, hide non-connector nodes for a clear minimal graph.
    if (hasFocusSubset && !n.isFocusNode && !n.isA && !n.isB && !n.isSelected) {
      visible = false;
    }

    nodeVisibility.set(n.id, visible);
  });

  const edgeDensityValue = Math.max(0, Math.min(1, edgeDensity));
  const importantRoles = new Set(['path', 'direct', 'selected-pair', 'interaction-pair', 'focus']);

  // Filter edges by node visibility, severity, and density.
  const filteredEdges = edges.filter((e, i) => {
    const importantEdge = importantRoles.has(e.role);
    const startVisible = nodeVisibility.get(e.startId) !== false;
    const endVisible = nodeVisibility.get(e.endId) !== false;

    if (!importantEdge && (!startVisible || !endVisible)) {
      return false;
    }

    const edgeSeverity = normalizeSeverityLevel(e.severity);
    if (!importantEdge && normalizedSeverityLevels.size > 0 && !normalizedSeverityLevels.has(edgeSeverity)) {
      return false;
    }

    if (!importantEdge && edgeDensityValue <= 0) {
      return false;
    }

    if (!importantEdge && edgeDensityValue < 1.0) {
      // Deterministic sampling based on index
      return (i % Math.ceil(1 / edgeDensityValue)) === 0;
    }

    return true;
  });

  return { nodeVisibility, filteredEdges };
}
