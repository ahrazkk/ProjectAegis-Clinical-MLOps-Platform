// PathLayout.js — Linear subway-map style path visualization
// Path nodes in a line, their neighbors branching vertically

import { rawGnnData } from '../graphEngine';

/**
 * Compute path layout: shortest path nodes in a horizontal line,
 * each node's non-path neighbors branch vertically like a subway map.
 */
export function computePathPositions(nodes, path, adj) {
  const positions = new Map();
  if (!path || path.length < 2) return positions;

  const pathSet = new Set(path);
  const spacing = 7;

  // Place path nodes along X axis
  const pathCenter = ((path.length - 1) * spacing) / 2;
  path.forEach((nodeId, index) => {
    positions.set(nodeId, [
      index * spacing - pathCenter,
      0,
      0
    ]);
  });

  // For each path node, fan out its non-path neighbors
  const adjacency = adj || rawGnnData.adj || {};

  path.forEach((nodeId, pathIndex) => {
    const neighbors = (adjacency[nodeId] || []).filter(nId => !pathSet.has(nId));
    const x = pathIndex * spacing - pathCenter;

    // Split neighbors into top and bottom
    const topNeighbors = neighbors.slice(0, Math.ceil(neighbors.length / 2));
    const bottomNeighbors = neighbors.slice(Math.ceil(neighbors.length / 2));

    // Limit displayed neighbors per path node to avoid clutter
    const maxPerSide = 8;

    topNeighbors.slice(0, maxPerSide).forEach((nId, i) => {
      const yOffset = 2.5 + i * 1.8;
      const zOffset = (Math.random() - 0.5) * 2;
      positions.set(nId, [x + (Math.random() - 0.5) * 1.5, yOffset, zOffset]);
    });

    bottomNeighbors.slice(0, maxPerSide).forEach((nId, i) => {
      const yOffset = -(2.5 + i * 1.8);
      const zOffset = (Math.random() - 0.5) * 2;
      positions.set(nId, [x + (Math.random() - 0.5) * 1.5, yOffset, zOffset]);
    });
  });

  // All remaining nodes — push to far background sphere
  nodes.forEach(n => {
    if (!positions.has(n.id)) {
      // Distant background
      const phi = Math.acos(1 - 2 * Math.random());
      const theta = Math.random() * Math.PI * 2;
      const r = 35;
      positions.set(n.id, [
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi)
      ]);
    }
  });

  return positions;
}
