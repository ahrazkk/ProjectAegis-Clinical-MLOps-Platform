// mechanismGraphEngine.js — Layout & computation for the biological mechanism map
// Pure functions, no React. Takes biology API data and produces graph layout.

// ─── Node types and their visual config ─────────────────────────────────────
export const NODE_TYPES = {
  drug: { shape: 'roundedRect', baseSize: 60 },
  enzyme: { shape: 'hexagon', baseSize: 40 },
  target: { shape: 'circle', baseSize: 36 },
  side_effect: { shape: 'diamond', baseSize: 28 },
};

// ─── Color palettes ─────────────────────────────────────────────────────────
export const DRUG_COLORS = { A: '#00d2ff', B: '#ff8c00' };

export const ENZYME_ROLE_COLORS = {
  substrate: '#3b82f6',
  inhibitor: '#ef4444',
  inducer: '#22c55e',
  mixed: '#a855f7',
};

export const TARGET_ACTION_COLORS = {
  inhibitor: '#ef4444',
  agonist: '#22c55e',
  antagonist: '#f97316',
  substrate: '#3b82f6',
  binder: '#8b5cf6',
  unknown: '#6b7280',
};

export const SEVERITY_COLORS = {
  high: '#ef4444',
  moderate: '#eab308',
  low: '#22c55e',
};

// ─── Edge types ─────────────────────────────────────────────────────────────
export const EDGE_STYLES = {
  substrate: { stroke: '#3b82f6', dasharray: 'none', width: 1.5 },
  inhibitor: { stroke: '#ef4444', dasharray: '6,3', width: 1.5 },
  inducer: { stroke: '#22c55e', dasharray: '3,3', width: 1.5 },
  targets: { stroke: '#a855f7', dasharray: 'none', width: 1.5 },
  causes: { stroke: '#6b7280', dasharray: 'none', width: 1 },
  conflict: { stroke: '#ef4444', dasharray: 'none', width: 3 },
};

// ─── Build the full mechanism graph ─────────────────────────────────────────
// Takes biology data from API and produces { nodes, edges, conflicts }
export function buildMechanismGraph(drug1Bio, drug2Bio, mechanismMap) {
  const nodes = [];
  const edges = [];
  const conflicts = [];
  const nodeMap = new Map(); // id → node

  const hasTwoDrugs = !!drug2Bio;

  // Helper to add node if not already present
  function addNode(id, type, label, extra = {}) {
    if (nodeMap.has(id)) return nodeMap.get(id);
    const node = { id, type, label, ...extra };
    nodes.push(node);
    nodeMap.set(id, node);
    return node;
  }

  // 1. Add drug nodes
  const d1Name = drug1Bio?.drug?.name || 'Drug A';
  const d1Id = `drug:${d1Name.toLowerCase()}`;
  addNode(d1Id, 'drug', d1Name, {
    slot: 'A',
    color: DRUG_COLORS.A,
    therapeutic_class: drug1Bio?.drug?.therapeutic_class || '',
  });

  let d2Id = null;
  if (hasTwoDrugs) {
    const d2Name = drug2Bio.drug?.name || 'Drug B';
    d2Id = `drug:${d2Name.toLowerCase()}`;
    addNode(d2Id, 'drug', d2Name, {
      slot: 'B',
      color: DRUG_COLORS.B,
      therapeutic_class: drug2Bio.drug?.therapeutic_class || '',
    });
  }

  // 2. Add CYP enzyme nodes + edges
  function addCYPFromBio(bio, drugNodeId, drugSlot) {
    if (!bio?.cyp_metabolism) return;
    const { substrates = [], inhibitors = [], inducers = [] } = bio.cyp_metabolism;

    for (const enzyme of substrates) {
      const eId = `enzyme:${enzyme}`;
      addNode(eId, 'enzyme', enzyme, { enzymeRoles: {} });
      const node = nodeMap.get(eId);
      node.enzymeRoles[drugSlot] = 'substrate';
      edges.push({ source: drugNodeId, target: eId, type: 'substrate', label: 'substrate' });
    }

    for (const inh of inhibitors) {
      const enzyme = typeof inh === 'string' ? inh : inh.enzyme;
      const strength = typeof inh === 'string' ? '' : inh.strength;
      const eId = `enzyme:${enzyme}`;
      addNode(eId, 'enzyme', enzyme, { enzymeRoles: {} });
      const node = nodeMap.get(eId);
      node.enzymeRoles[drugSlot] = 'inhibitor';
      edges.push({ source: drugNodeId, target: eId, type: 'inhibitor', label: `inhibitor${strength ? ` (${strength})` : ''}` });
    }

    for (const enzyme of inducers) {
      const eId = `enzyme:${enzyme}`;
      addNode(eId, 'enzyme', enzyme, { enzymeRoles: {} });
      const node = nodeMap.get(eId);
      node.enzymeRoles[drugSlot] = 'inducer';
      edges.push({ source: drugNodeId, target: eId, type: 'inducer', label: 'inducer' });
    }
  }

  addCYPFromBio(drug1Bio, d1Id, 'A');
  if (hasTwoDrugs) addCYPFromBio(drug2Bio, d2Id, 'B');

  // 3. Add protein target nodes + edges
  function addTargetsFromBio(bio, drugNodeId) {
    if (!bio?.targets) return;
    for (const t of bio.targets) {
      const tId = `target:${t.gene || t.name || t.id}`;
      addNode(tId, 'target', t.name || t.gene || 'Unknown', {
        gene: t.gene || '',
        action: t.action || 'unknown',
      });
      edges.push({ source: drugNodeId, target: tId, type: 'targets', label: t.action || '' });
    }
  }

  addTargetsFromBio(drug1Bio, d1Id);
  if (hasTwoDrugs) addTargetsFromBio(drug2Bio, d2Id);

  // 4. Add side effect nodes + edges
  function addSideEffectsFromBio(bio, drugNodeId) {
    if (!bio?.side_effects) return;
    for (const se of bio.side_effects.slice(0, 8)) {
      const seId = `se:${se.name?.toLowerCase() || 'unknown'}`;
      addNode(seId, 'side_effect', se.name || 'Unknown', {
        organ_system: se.organ_system || '',
        severity: se.severity || 0,
      });
      edges.push({ source: drugNodeId, target: seId, type: 'causes', label: '' });
    }
  }

  addSideEffectsFromBio(drug1Bio, d1Id);
  if (hasTwoDrugs) addSideEffectsFromBio(drug2Bio, d2Id);

  // 5. Detect conflicts from mechanism map
  if (mechanismMap) {
    // CYP enzyme conflicts from API
    for (const se of (mechanismMap.shared_enzymes || [])) {
      if (se.risk_level !== 'low' && se.risk) {
        const eId = `enzyme:${se.enzyme}`;
        conflicts.push({
          id: `conflict:cyp:${se.enzyme}`,
          type: 'cyp',
          nodeId: eId,
          enzyme: se.enzyme,
          drug1_role: se.drug1_role,
          drug2_role: se.drug2_role,
          risk: se.risk,
          risk_level: se.risk_level,
        });
      }
    }

    // Shared target conflicts
    for (const st of (mechanismMap.shared_targets || [])) {
      const tId = `target:${st.gene || st.target}`;
      conflicts.push({
        id: `conflict:target:${st.gene || st.target}`,
        type: 'target',
        nodeId: tId,
        target: st.target,
        gene: st.gene,
        drug1_action: st.drug1_action,
        drug2_action: st.drug2_action,
        risk: st.risk || 'Shared target',
      });
    }

    // Shared side effect conflicts
    for (const sse of (mechanismMap.shared_side_effects || [])) {
      const seId = `se:${sse.name?.toLowerCase() || 'unknown'}`;
      if (sse.combined_risk !== 'low') {
        conflicts.push({
          id: `conflict:se:${sse.name}`,
          type: 'side_effect',
          nodeId: seId,
          name: sse.name,
          organ_system: sse.organ_system,
          combined_risk: sse.combined_risk,
        });
      }
    }
  }

  // 5b. Auto-detect CYP conflicts from local graph data (even without mechanismMap)
  if (hasTwoDrugs) {
    const enzymeNodes = nodes.filter(n => n.type === 'enzyme');
    for (const en of enzymeNodes) {
      const roles = en.enzymeRoles || {};
      if (!roles.A || !roles.B) continue;
      // Already detected by mechanism map?
      if (conflicts.some(c => c.type === 'cyp' && c.enzyme === en.label)) continue;

      let risk = '';
      let risk_level = 'low';
      if (roles.A === 'substrate' && roles.B === 'inhibitor') {
        risk = `${drug2Bio.drug?.name} inhibits ${d1Name} metabolism via ${en.label}`;
        risk_level = 'high';
      } else if (roles.B === 'substrate' && roles.A === 'inhibitor') {
        risk = `${d1Name} inhibits ${drug2Bio.drug?.name} metabolism via ${en.label}`;
        risk_level = 'high';
      } else if (roles.A === 'substrate' && roles.B === 'inducer') {
        risk = `${drug2Bio.drug?.name} induces ${en.label}, reducing ${d1Name} levels`;
        risk_level = 'high';
      } else if (roles.B === 'substrate' && roles.A === 'inducer') {
        risk = `${d1Name} induces ${en.label}, reducing ${drug2Bio.drug?.name} levels`;
        risk_level = 'high';
      } else if (roles.A === 'substrate' && roles.B === 'substrate') {
        risk = `Both compete for ${en.label}`;
        risk_level = 'moderate';
      }

      if (risk_level !== 'low') {
        conflicts.push({
          id: `conflict:cyp:${en.label}`,
          type: 'cyp',
          nodeId: en.id,
          enzyme: en.label,
          drug1_role: roles.A,
          drug2_role: roles.B,
          risk,
          risk_level,
        });
      }
    }

    // 5c. Auto-detect shared targets from local graph
    const targetEdges = edges.filter(e => e.type === 'targets');
    const targetToDrugs = new Map();
    for (const te of targetEdges) {
      if (!targetToDrugs.has(te.target)) targetToDrugs.set(te.target, []);
      targetToDrugs.get(te.target).push(te);
    }
    for (const [tId, tedges] of targetToDrugs) {
      if (tedges.length < 2) continue;
      if (conflicts.some(c => c.type === 'target' && c.nodeId === tId)) continue;
      const tNode = nodeMap.get(tId);
      if (!tNode) continue;
      conflicts.push({
        id: `conflict:target:${tNode.gene || tNode.label}`,
        type: 'target',
        nodeId: tId,
        target: tNode.label,
        gene: tNode.gene,
        drug1_action: tedges[0].label || 'unknown',
        drug2_action: tedges[1].label || 'unknown',
        risk: 'Both drugs act on this target',
      });
    }
  }

  // Mark conflict nodes
  for (const c of conflicts) {
    const node = nodeMap.get(c.nodeId);
    if (node) node.isConflict = true;
  }

  return { nodes, edges, conflicts };
}

// ─── Hex grid helpers ───────────────────────────────────────────────────────
// Pointy-top hex: axial (q, r) → pixel (x, y)
function hexToPixel(q, r, size) {
  const x = size * Math.sqrt(3) * (q + r / 2);
  const y = size * (3 / 2) * r;
  return { x, y };
}

// ─── Hex grid layout ───────────────────────────────────────────────────────
// Snaps all nodes to hex grid positions in concentric rings around center.
// Ring 0: drugs, Ring 1: enzymes, Ring 2: targets, Ring 3: side effects
export function computeRadialLayout(nodes, width, height) {
  const cx = width / 2;
  const cy = height / 2;
  const layout = new Map();

  // Scale hex size to viewport — larger = more spread
  const minDim = Math.min(width, height);
  const hexSize = Math.max(50, minDim * 0.09);

  const drugs = nodes.filter(n => n.type === 'drug');
  const enzymes = nodes.filter(n => n.type === 'enzyme');
  const targets = nodes.filter(n => n.type === 'target');
  const sideEffects = nodes.filter(n => n.type === 'side_effect');

  // Drug positions: center of the grid
  const DRUG_POSITIONS = [
    { q: -1, r: 0 }, // Drug A (left)
    { q: 1, r: 0 },  // Drug B (right)
  ];

  // Ring 1 positions for enzymes (hex ring 2 distance from center)
  const ENZYME_POSITIONS = [
    { q: 0, r: -2 },  // top
    { q: 2, r: -2 },  // top-right
    { q: -2, r: 0 },  // left
    { q: 2, r: 0 },   // right
    { q: 0, r: 2 },   // bottom
    { q: -2, r: 2 },  // bottom-left
    { q: -1, r: -1 }, // inner top-left
    { q: 1, r: -1 },  // inner top-right
    { q: -1, r: 2 },  // inner bottom-left
  ];

  // Ring 2 positions for targets (hex ring 3 distance)
  const TARGET_POSITIONS = [
    { q: 0, r: -3 },  // far top
    { q: 3, r: -3 },  // far top-right
    { q: 3, r: 0 },   // far right
    { q: 0, r: 3 },   // far bottom
    { q: -3, r: 3 },  // far bottom-left
    { q: -3, r: 0 },  // far left
    { q: 1, r: -3 },
    { q: -1, r: -2 },
    { q: 2, r: -3 },
    { q: -2, r: -1 },
    { q: 3, r: -1 },
    { q: -3, r: 1 },
  ];

  // Ring 3 positions for side effects (hex ring 4 distance)
  const SIDE_EFFECT_POSITIONS = [
    { q: 0, r: -4 },
    { q: 4, r: -4 },
    { q: 4, r: 0 },
    { q: 0, r: 4 },
    { q: -4, r: 4 },
    { q: -4, r: 0 },
    { q: 2, r: -4 },
    { q: -2, r: -2 },
    { q: 4, r: -2 },
    { q: -4, r: 2 },
    { q: 2, r: 2 },
    { q: -2, r: 4 },
  ];

  const occupiedCells = new Set();

  function placeGroup(items, positions) {
    items.forEach((node, i) => {
      const pos = positions[i % positions.length];
      const px = hexToPixel(pos.q, pos.r, hexSize);
      layout.set(node.id, { x: cx + px.x, y: cy + px.y });
      occupiedCells.add(`${pos.q},${pos.r}`);
    });
  }

  // Single drug → center it
  if (drugs.length === 1) {
    const px = hexToPixel(0, 0, hexSize);
    layout.set(drugs[0].id, { x: cx + px.x, y: cy + px.y });
    occupiedCells.add('0,0');
  } else {
    placeGroup(drugs, DRUG_POSITIONS);
  }

  placeGroup(enzymes, ENZYME_POSITIONS);
  placeGroup(targets, TARGET_POSITIONS);
  placeGroup(sideEffects, SIDE_EFFECT_POSITIONS);

  return { positions: layout, occupiedCells, hexSize };
}

// ─── Detect conflicts between two drugs (standalone) ────────────────────────
export function detectConflicts(drug1Bio, drug2Bio) {
  const cypConflicts = [];
  const targetOverlaps = [];
  const sharedSideEffects = [];

  if (!drug1Bio || !drug2Bio) return { cypConflicts, targetOverlaps, sharedSideEffects };

  // CYP conflicts
  const subs1 = new Set(drug1Bio.cyp_metabolism?.substrates || []);
  const inhs1 = new Set((drug1Bio.cyp_metabolism?.inhibitors || []).map(i => typeof i === 'string' ? i : i.enzyme));
  const inds1 = new Set(drug1Bio.cyp_metabolism?.inducers || []);
  const subs2 = new Set(drug2Bio.cyp_metabolism?.substrates || []);
  const inhs2 = new Set((drug2Bio.cyp_metabolism?.inhibitors || []).map(i => typeof i === 'string' ? i : i.enzyme));
  const inds2 = new Set(drug2Bio.cyp_metabolism?.inducers || []);

  // Drug2 inhibits Drug1 substrate
  for (const e of subs1) {
    if (inhs2.has(e)) cypConflicts.push({ enzyme: e, perpetrator: drug2Bio.drug?.name, victim: drug1Bio.drug?.name, type: 'inhibition' });
  }
  // Drug1 inhibits Drug2 substrate
  for (const e of subs2) {
    if (inhs1.has(e)) cypConflicts.push({ enzyme: e, perpetrator: drug1Bio.drug?.name, victim: drug2Bio.drug?.name, type: 'inhibition' });
  }
  // Inducer conflicts
  for (const e of subs1) {
    if (inds2.has(e)) cypConflicts.push({ enzyme: e, perpetrator: drug2Bio.drug?.name, victim: drug1Bio.drug?.name, type: 'induction' });
  }
  for (const e of subs2) {
    if (inds1.has(e)) cypConflicts.push({ enzyme: e, perpetrator: drug1Bio.drug?.name, victim: drug2Bio.drug?.name, type: 'induction' });
  }

  // Target overlaps
  const targets1 = new Map((drug1Bio.targets || []).map(t => [t.gene || t.name, t]));
  const targets2 = new Map((drug2Bio.targets || []).map(t => [t.gene || t.name, t]));
  for (const [key, t1] of targets1) {
    if (targets2.has(key)) {
      targetOverlaps.push({ target: key, drug1_action: t1.action, drug2_action: targets2.get(key).action });
    }
  }

  // Shared side effects
  const se1 = new Map((drug1Bio.side_effects || []).map(s => [s.name, s]));
  for (const se of (drug2Bio.side_effects || [])) {
    if (se1.has(se.name)) {
      sharedSideEffects.push({ name: se.name, organ_system: se.organ_system || se1.get(se.name).organ_system });
    }
  }

  return { cypConflicts, targetOverlaps, sharedSideEffects };
}
