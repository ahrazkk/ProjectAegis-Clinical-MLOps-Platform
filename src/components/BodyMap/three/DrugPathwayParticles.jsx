// DrugPathwayParticles.jsx — Particles flowing along drug pathway routes
// Oral → GI → Liver (first-pass) → Heart → Aorta → Target organs
// Drug A = cyan, Drug B = orange. Density scales with target severity.

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { DRUG_PATHWAY } from '../organRegistry';

const DRUG_COLORS = [
  new THREE.Color('#22d3ee'), // Drug A — cyan
  new THREE.Color('#f97316'), // Drug B — orange
];

// Interpolate along a polyline path
function interpolatePath(points, t) {
  if (!points || points.length < 2) return { x: 0, y: 0 };
  const clampedT = Math.max(0, Math.min(1, t));
  const totalSegments = points.length - 1;
  const segmentIndex = Math.min(Math.floor(clampedT * totalSegments), totalSegments - 1);
  const segmentT = (clampedT * totalSegments) - segmentIndex;

  const p0 = points[segmentIndex];
  const p1 = points[segmentIndex + 1];

  return {
    x: p0.x + (p1.x - p0.x) * segmentT,
    y: p0.y + (p1.y - p0.y) * segmentT,
  };
}

// Build full path: oral route + branch to target organ
function buildFullPath(targetOrgan) {
  const oralPath = DRUG_PATHWAY.oral;
  const branch = DRUG_PATHWAY.branches[targetOrgan];
  if (!branch) return oralPath;
  return [...oralPath, ...branch];
}

export default function DrugPathwayParticles({ activeOrgans = [], drugCount = 1 }) {
  const meshRef = useRef();

  const { particleCount, paths, speeds, phases, drugIndices } = useMemo(() => {
    if (!activeOrgans.length) return { particleCount: 0, paths: [], speeds: [], phases: [], drugIndices: [] };

    const allPaths = [];
    const allSpeeds = [];
    const allPhases = [];
    const allDrugIndices = [];

    for (const organ of activeOrgans) {
      const organId = typeof organ === 'string' ? organ : organ.id;
      const severity = typeof organ === 'object' ? organ.severity : 0.5;

      // Number of particles proportional to severity
      const count = Math.max(3, Math.floor(severity * 12));
      const fullPath = buildFullPath(organId);

      for (let i = 0; i < count; i++) {
        allPaths.push(fullPath);
        allSpeeds.push(0.15 + Math.random() * 0.2);
        allPhases.push(Math.random());
        // Alternate drug colors
        allDrugIndices.push(i % Math.max(1, drugCount));
      }
    }

    return {
      particleCount: allPaths.length,
      paths: allPaths,
      speeds: allSpeeds,
      phases: allPhases,
      drugIndices: allDrugIndices,
    };
  }, [activeOrgans, drugCount]);

  // Create geometry
  const { positions, colors, sizes } = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const col = new Float32Array(particleCount * 3);
    const siz = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      const color = DRUG_COLORS[drugIndices[i] % DRUG_COLORS.length];
      col[i * 3] = color.r;
      col[i * 3 + 1] = color.g;
      col[i * 3 + 2] = color.b;
      siz[i] = 1.5 + Math.random() * 1.5;
    }

    return { positions: pos, colors: col, sizes: siz };
  }, [particleCount, drugIndices]);

  // Animate particles
  useFrame((_, delta) => {
    if (!meshRef.current || particleCount === 0) return;

    const posAttr = meshRef.current.geometry.getAttribute('position');
    const posArray = posAttr.array;

    for (let i = 0; i < particleCount; i++) {
      // Advance phase
      phases[i] = (phases[i] + delta * speeds[i]) % 1;

      const path = paths[i];
      const { x, y } = interpolatePath(path, phases[i]);

      // Map SVG coordinates (400×780 viewBox) to Three.js scene coordinates
      // Center at origin, scale to reasonable Three.js units
      const scaleX = 0.05;
      const scaleY = -0.05; // Flip Y for Three.js
      const offsetX = -200 * scaleX;
      const offsetY = 780 * 0.5 * 0.05;

      // Add slight perpendicular wander
      const wander = Math.sin(phases[i] * Math.PI * 6 + i) * 0.15;

      posArray[i * 3] = x * scaleX + offsetX + wander;
      posArray[i * 3 + 1] = y * scaleY + offsetY;
      posArray[i * 3 + 2] = 0;
    }

    posAttr.needsUpdate = true;
  });

  if (particleCount === 0) return null;

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          count={particleCount}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          array={colors}
          count={particleCount}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          array={sizes}
          count={particleCount}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={2}
        vertexColors
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation={false}
      />
    </points>
  );
}
