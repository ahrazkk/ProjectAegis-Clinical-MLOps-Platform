// ClusterLayout.js — Force-directed therapeutic class grouping
// Groups drugs by category with each cluster in a distinct region

import { CATEGORY_COLORS, CATEGORY_LIST } from '../graphEngine';

/**
 * Compute cluster positions: each therapeutic class gets a region,
 * nodes within each cluster are arranged in a compact spherical distribution.
 */
export function computeClusterPositions(nodes) {
  const positions = new Map();

  // Group nodes by category
  const groups = {};
  nodes.forEach(n => {
    if (!groups[n.category]) groups[n.category] = [];
    groups[n.category].push(n);
  });

  const categories = Object.keys(groups).sort((a, b) => {
    // Sort by size (largest clusters get prominent positions)
    return groups[b].length - groups[a].length;
  });

  // Place cluster centers on a sphere
  const clusterRadius = 16;
  const clusterCenters = {};

  categories.forEach((cat, i) => {
    // Fibonacci sphere distribution for even spacing
    const phi = Math.acos(1 - 2 * (i + 0.5) / categories.length);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    clusterCenters[cat] = [
      clusterRadius * Math.sin(phi) * Math.cos(theta),
      clusterRadius * Math.sin(phi) * Math.sin(theta) * 0.6, // Flatten Y for better viewing
      clusterRadius * Math.cos(phi),
    ];
  });

  // Within each cluster, distribute nodes in a compact sphere
  categories.forEach(cat => {
    const center = clusterCenters[cat];
    const nodesInCluster = groups[cat];
    const count = nodesInCluster.length;

    // Cluster inner radius scales with node count
    const innerRadius = Math.min(2 + Math.sqrt(count) * 0.5, 6);

    nodesInCluster.forEach((n, i) => {
      if (count === 1) {
        positions.set(n.id, [...center]);
        return;
      }

      // Fibonacci sphere within cluster
      const phi = Math.acos(1 - 2 * (i + 0.5) / count);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const r = innerRadius * Math.cbrt((i + 1) / count); // Pack inner tighter

      positions.set(n.id, [
        center[0] + r * Math.sin(phi) * Math.cos(theta),
        center[1] + r * Math.sin(phi) * Math.sin(theta),
        center[2] + r * Math.cos(phi),
      ]);
    });
  });

  return { positions, clusterCenters, groups };
}

/**
 * Get cluster metadata for rendering cluster bubbles
 */
export function getClusterMeta(nodes) {
  const groups = {};
  nodes.forEach(n => {
    if (!groups[n.category]) groups[n.category] = { count: 0, nodes: [] };
    groups[n.category].count++;
    groups[n.category].nodes.push(n);
  });

  return Object.entries(groups).map(([cat, data]) => ({
    category: cat,
    count: data.count,
    color: CATEGORY_COLORS[cat] || '#475569',
    percentage: ((data.count / nodes.length) * 100).toFixed(1),
  }));
}
