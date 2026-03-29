// ClusterLayout.js — Force-directed therapeutic class grouping
// Groups drugs by category with each cluster in a distinct region

import { CATEGORY_COLORS, CATEGORY_LIST } from '../graphEngine';

/**
 * Compute cluster positions: each therapeutic class gets a region,
 * nodes within each cluster are arranged in a compact spherical distribution
 * with repulsion to prevent overlap.
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

  // Place cluster centers on a sphere — larger radius to prevent overlap
  const clusterRadius = 24;
  const clusterCenters = {};

  categories.forEach((cat, i) => {
    // Fibonacci sphere distribution for even spacing
    const phi = Math.acos(1 - 2 * (i + 0.5) / categories.length);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    clusterCenters[cat] = [
      clusterRadius * Math.sin(phi) * Math.cos(theta),
      clusterRadius * Math.sin(phi) * Math.sin(theta) * 0.85, // Less Y flattening
      clusterRadius * Math.cos(phi),
    ];
  });

  // Within each cluster, distribute nodes with repulsion
  categories.forEach(cat => {
    const center = clusterCenters[cat];
    const nodesInCluster = groups[cat];
    const count = nodesInCluster.length;

    // Cluster inner radius scales more generously with node count
    const innerRadius = Math.min(3 + Math.sqrt(count) * 0.8, 12);

    // Initial placement: fibonacci sphere
    const localPositions = [];
    nodesInCluster.forEach((n, i) => {
      if (count === 1) {
        localPositions.push([0, 0, 0]);
        return;
      }

      const phi = Math.acos(1 - 2 * (i + 0.5) / count);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const r = innerRadius * Math.cbrt((i + 1) / count);

      localPositions.push([
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi),
      ]);
    });

    // Repulsion pass: push overlapping nodes apart (5 iterations)
    const minDist = 0.6;
    for (let iter = 0; iter < 5; iter++) {
      for (let i = 0; i < localPositions.length; i++) {
        for (let j = i + 1; j < localPositions.length; j++) {
          const dx = localPositions[j][0] - localPositions[i][0];
          const dy = localPositions[j][1] - localPositions[i][1];
          const dz = localPositions[j][2] - localPositions[i][2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist < minDist && dist > 0.001) {
            const force = (minDist - dist) * 0.5 / dist;
            const fx = dx * force;
            const fy = dy * force;
            const fz = dz * force;

            localPositions[i][0] -= fx;
            localPositions[i][1] -= fy;
            localPositions[i][2] -= fz;
            localPositions[j][0] += fx;
            localPositions[j][1] += fy;
            localPositions[j][2] += fz;
          }
        }
      }
    }

    // Apply: offset by cluster center
    nodesInCluster.forEach((n, i) => {
      positions.set(n.id, [
        center[0] + localPositions[i][0],
        center[1] + localPositions[i][1],
        center[2] + localPositions[i][2],
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
