// graphEngine.js — Pure computation module for GNN Galaxy Viewer
// No React dependencies — all functions are pure JS

import rawGnnData from '../../assets/gnn_real_data.json';

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

// ─── Build node dictionary ─────────────────────────────────────────────────
export function buildNodeDict(scale = 0.35) {
  if (!rawGnnData || !rawGnnData.nodes) return {};
  const dict = {};
  rawGnnData.nodes.forEach((n, index) => {
    dict[n.id] = {
      ...n,
      index,
      pos: [n.pos[0] * scale, n.pos[1] * scale, n.pos[2] * scale],
      category: classifyDrug(n.type),
      isA: false,
      isB: false,
      hopA: Infinity,
      hopB: Infinity,
    };
  });
  return dict;
}

// ─── Find drug by name ─────────────────────────────────────────────────────
export function findDrugByName(nodeDict, name) {
  if (!name) return null;
  const lower = name.toLowerCase();
  // Exact match
  for (const id of Object.keys(nodeDict)) {
    if (nodeDict[id].name.toLowerCase() === lower) return id;
  }
  // Partial match
  for (const id of Object.keys(nodeDict)) {
    if (nodeDict[id].name.toLowerCase().includes(lower)) return id;
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

// ─── Compute full subgraph state ──────────────────────────────────────────
export function computeSubgraph(nodeDict, drugAId, drugBId, maxHops = 3) {
  const adj = rawGnnData.adj || {};

  // Reset hop fields
  Object.values(nodeDict).forEach(n => {
    n.isA = false;
    n.isB = false;
    n.hopA = Infinity;
    n.hopB = Infinity;
  });

  if (drugAId && nodeDict[drugAId]) nodeDict[drugAId].isA = true;
  if (drugBId && nodeDict[drugBId]) nodeDict[drugBId].isB = true;

  // BFS from both drugs
  const hopsA = bfsHops(adj, drugAId, maxHops);
  const hopsB = bfsHops(adj, drugBId, maxHops);

  hopsA.forEach((d, id) => { if (nodeDict[id]) nodeDict[id].hopA = d; });
  hopsB.forEach((d, id) => { if (nodeDict[id]) nodeDict[id].hopB = d; });

  // Compute shortest path between A and B
  const path = shortestPath(adj, drugAId, drugBId);
  const pathSet = new Set(path);

  const hasDrugs = drugAId !== null || drugBId !== null;

  // Compute edges
  const edges = [];
  const processed = new Set();
  Object.keys(adj).forEach(u => {
    const uNode = nodeDict[u];
    if (!uNode) return;
    (adj[u] || []).forEach(v => {
      const key = u < v ? `${u}-${v}` : `${v}-${u}`;
      if (processed.has(key)) return;
      processed.add(key);
      const vNode = nodeDict[v];
      if (!vNode) return;

      const inHopA = uNode.hopA <= maxHops && vNode.hopA <= maxHops;
      const inHopB = uNode.hopB <= maxHops && vNode.hopB <= maxHops;
      const onPath = pathSet.has(u) && pathSet.has(v) &&
        Math.abs(path.indexOf(u) - path.indexOf(v)) === 1;

      let color = '#475569';
      let opacity = 0.04;
      let lineWidth = 1;
      let role = 'background';

      if (onPath) {
        color = '#ef4444'; opacity = 1.0; lineWidth = 3; role = 'path';
      } else if ((uNode.isA && vNode.isB) || (uNode.isB && vNode.isA)) {
        color = '#ef4444'; opacity = 1.0; lineWidth = 3; role = 'direct';
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
      }

      if (!hasDrugs || inHopA || inHopB || onPath) {
        edges.push({
          startId: u, endId: v,
          start: uNode.pos, end: vNode.pos,
          color, opacity, lineWidth, role,
        });
      }
    });
  });

  // Stats
  const visibleNodes = Object.values(nodeDict).filter(n =>
    n.hopA <= maxHops || n.hopB <= maxHops || !hasDrugs
  ).length;

  return {
    nodes: Object.values(nodeDict),
    edges,
    path,
    pathSet,
    stats: {
      totalNodes: rawGnnData.nodes.length,
      totalEdges: Object.values(adj).reduce((sum, arr) => sum + arr.length, 0) / 2,
      visibleNodes: hasDrugs ? visibleNodes : rawGnnData.nodes.length,
      visibleEdges: edges.length,
      pathLength: path.length > 0 ? path.length - 1 : -1,
      sharedNeighbors: Object.values(nodeDict).filter(n => n.hopA <= maxHops && n.hopB <= maxHops && !n.isA && !n.isB).length,
    },
  };
}

// ─── Get node visual properties ───────────────────────────────────────────
export function getNodeVisuals(node, hasDrugs) {
  const adj = rawGnnData.adj || {};
  const degree = (adj[node.id] || []).length;
  const degreeFactor = Math.min(Math.sqrt(degree) / 8, 1); // 0-1 normalized

  let color = '#1e293b';
  let opacity = 0.15;
  let size = 0.1;
  let glow = 0;

  if (node.isA) {
    color = '#00d2ff'; opacity = 1; size = 0.7; glow = 1;
  } else if (node.isB) {
    color = '#ff8c00'; opacity = 1; size = 0.7; glow = 1;
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
  } else if (!hasDrugs) {
    // No drugs selected — show by category with degree-based sizing
    color = CATEGORY_COLORS[node.category] || '#475569';
    opacity = 0.25 + degreeFactor * 0.15;
    size = 0.12 + degreeFactor * 0.15;
  }

  return { color, opacity, size: Math.max(size, 0.15), glow: Math.max(glow, 0) };
}

// ─── Sample background edges for "no selection" view ─────────────────────
export function sampleBackgroundEdges(maxCount = 3000) {
  const adj = rawGnnData.adj || {};
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
  const adj = rawGnnData.adj || {};
  const { visibleClasses, minDegree, edgeDensity } = filters;

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
      if (degree < minDegree && !n.isA && !n.isB) {
        visible = false;
      }
    }

    // Selected drugs always visible
    if (n.isA || n.isB) visible = true;

    nodeVisibility.set(n.id, visible);
  });

  // Edge density sampling for background edges
  const filteredEdges = edges.filter((e, i) => {
    if (e.role === 'background' && edgeDensity < 1.0) {
      // Deterministic sampling based on index
      return (i % Math.ceil(1 / edgeDensity)) === 0;
    }
    return true;
  });

  return { nodeVisibility, filteredEdges };
}

export { rawGnnData };
