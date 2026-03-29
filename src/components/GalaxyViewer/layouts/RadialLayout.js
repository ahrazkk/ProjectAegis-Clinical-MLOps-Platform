// RadialLayout.js — Concentric ring layout with drug at center, hops as rings
// Creates a clean "solar system" view around selected drug(s)

import { CATEGORY_COLORS, CATEGORY_LIST } from '../graphEngine';

/**
 * Compute radial positions for a single-drug focus.
 * Center drug at origin, hop-1 in inner ring, hop-2 in middle, hop-3 in outer.
 * Within each ring, nodes are sorted by therapeutic class for grouped sectors.
 */
export function computeRadialPositions(nodes, drugAId, drugBId, adj, maxHops) {
  const positions = new Map();
  const hasBoth = drugAId && drugBId;

  if (hasBoth) {
    return computeDualRadial(nodes, drugAId, drugBId, adj, maxHops);
  }

  const centerId = drugAId || drugBId;
  if (!centerId) {
    // No drug selected — spiral galaxy fallback
    return computeSpiralFallback(nodes);
  }

  const centerNode = nodes.find(n => n.id === centerId);
  if (!centerNode) return computeSpiralFallback(nodes);

  // Center drug at origin
  positions.set(centerId, [0, 0, 0]);

  // Group nodes by hop distance
  const hopField = drugAId ? 'hopA' : 'hopB';
  const hopBuckets = { 1: [], 2: [], 3: [] };
  const backgroundNodes = [];

  nodes.forEach(n => {
    if (n.id === centerId) return;
    const hop = n[hopField];
    if (hop >= 1 && hop <= maxHops) {
      hopBuckets[hop].push(n);
    } else {
      backgroundNodes.push(n);
    }
  });

  // For each hop ring, sort by category then distribute on circle
  const ringRadii = { 1: 6, 2: 13, 3: 21 };

  for (let hop = 1; hop <= maxHops; hop++) {
    const bucket = hopBuckets[hop];
    if (bucket.length === 0) continue;

    // Sort by category for sector grouping
    bucket.sort((a, b) => {
      const catA = CATEGORY_LIST.indexOf(a.category);
      const catB = CATEGORY_LIST.indexOf(b.category);
      return catA - catB;
    });

    const baseRadius = ringRadii[hop];
    const count = bucket.length;
    const ySpread = hop * 2.5; // Much more vertical spread per hop

    // For large rings, split into two concentric sub-rings
    const useDoubleRing = count > 50;

    bucket.forEach((n, i) => {
      let ringIndex, ringCount, radius;
      if (useDoubleRing) {
        // Alternate between inner and outer sub-ring
        if (i % 2 === 0) {
          ringIndex = Math.floor(i / 2);
          ringCount = Math.ceil(count / 2);
          radius = baseRadius - 1.2;
        } else {
          ringIndex = Math.floor(i / 2);
          ringCount = Math.floor(count / 2);
          radius = baseRadius + 1.2;
        }
      } else {
        ringIndex = i;
        ringCount = count;
        radius = baseRadius;
      }

      const angle = (ringIndex / ringCount) * Math.PI * 2;
      // Add small radial jitter to prevent perfect alignment
      const jitter = (Math.random() - 0.5) * 1.5;
      const x = Math.cos(angle) * (radius + jitter);
      const z = Math.sin(angle) * (radius + jitter * 0.5);
      const y = (Math.random() - 0.5) * ySpread;
      positions.set(n.id, [x, y, z]);
    });
  }

  // Background nodes — pushed to outer shell
  const outerRadius = 28;
  backgroundNodes.forEach((n, i) => {
    const phi = Math.acos(1 - 2 * (i + 0.5) / backgroundNodes.length);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const x = outerRadius * Math.sin(phi) * Math.cos(theta);
    const y = outerRadius * Math.sin(phi) * Math.sin(theta);
    const z = outerRadius * Math.cos(phi);
    positions.set(n.id, [x, y, z]);
  });

  return positions;
}

/**
 * Dual-drug radial: Drug A at left, Drug B at right, shared bridges in center
 */
function computeDualRadial(nodes, drugAId, drugBId, adj, maxHops) {
  const positions = new Map();
  const offsetA = [-14, 0, 0];
  const offsetB = [14, 0, 0];

  positions.set(drugAId, offsetA);
  positions.set(drugBId, offsetB);

  // Categorize nodes
  const bridgeNodes = [];
  const hopANodes = { 1: [], 2: [], 3: [] };
  const hopBNodes = { 1: [], 2: [], 3: [] };
  const backgroundNodes = [];

  nodes.forEach(n => {
    if (n.id === drugAId || n.id === drugBId) return;
    const inA = n.hopA <= maxHops;
    const inB = n.hopB <= maxHops;

    if (inA && inB) {
      bridgeNodes.push(n);
    } else if (inA) {
      const hop = Math.min(n.hopA, 3);
      hopANodes[hop].push(n);
    } else if (inB) {
      const hop = Math.min(n.hopB, 3);
      hopBNodes[hop].push(n);
    } else {
      backgroundNodes.push(n);
    }
  });

  // Bridge nodes arranged in center column
  bridgeNodes.sort((a, b) => (a.hopA + a.hopB) - (b.hopA + b.hopB));
  bridgeNodes.forEach((n, i) => {
    const angle = (i / bridgeNodes.length) * Math.PI * 2;
    const radius = 3 + (Math.min(n.hopA, n.hopB)) * 1.5;
    const x = Math.cos(angle) * radius * 0.5;
    const y = Math.sin(angle) * radius;
    const z = (Math.random() - 0.5) * 3;
    positions.set(n.id, [x, y, z]);
  });

  // Drug A neighborhood — left side
  const ringRadii = { 1: 6, 2: 11, 3: 17 };
  for (let hop = 1; hop <= maxHops; hop++) {
    const bucket = hopANodes[hop];
    bucket.forEach((n, i) => {
      const angle = (i / Math.max(bucket.length, 1)) * Math.PI + Math.PI / 2; // Left hemisphere
      const r = ringRadii[hop];
      positions.set(n.id, [
        offsetA[0] + Math.cos(angle) * r,
        Math.sin(angle) * r * 0.6,
        (Math.random() - 0.5) * hop
      ]);
    });
  }

  // Drug B neighborhood — right side
  for (let hop = 1; hop <= maxHops; hop++) {
    const bucket = hopBNodes[hop];
    bucket.forEach((n, i) => {
      const angle = (i / Math.max(bucket.length, 1)) * Math.PI - Math.PI / 2; // Right hemisphere
      const r = ringRadii[hop];
      positions.set(n.id, [
        offsetB[0] + Math.cos(angle) * r,
        Math.sin(angle) * r * 0.6,
        (Math.random() - 0.5) * hop
      ]);
    });
  }

  // Background — outer shell
  backgroundNodes.forEach((n, i) => {
    const phi = Math.acos(1 - 2 * (i + 0.5) / backgroundNodes.length);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const r = 30;
    positions.set(n.id, [
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi)
    ]);
  });

  return positions;
}

/**
 * Spiral galaxy fallback when no drugs selected
 */
function computeSpiralFallback(nodes) {
  const positions = new Map();
  // Group by category, arrange categories in circle, nodes spiral within
  const groups = {};
  nodes.forEach(n => {
    if (!groups[n.category]) groups[n.category] = [];
    groups[n.category].push(n);
  });

  const categories = Object.keys(groups);
  categories.forEach((cat, catIdx) => {
    const catAngle = (catIdx / categories.length) * Math.PI * 2;
    const catRadius = 12;
    const cx = Math.cos(catAngle) * catRadius;
    const cz = Math.sin(catAngle) * catRadius;

    const nodesInCat = groups[cat];
    nodesInCat.forEach((n, i) => {
      const t = i / nodesInCat.length;
      const spiralAngle = t * Math.PI * 4;
      const spiralRadius = t * 5;
      positions.set(n.id, [
        cx + Math.cos(spiralAngle) * spiralRadius,
        (Math.random() - 0.5) * 3,
        cz + Math.sin(spiralAngle) * spiralRadius
      ]);
    });
  });

  return positions;
}
