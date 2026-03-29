// PathParticles.jsx — Animated particles traveling along the shortest path
import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const PARTICLE_COUNT = 20;

export default function PathParticles({ path, nodeDict, visible }) {
  const pointsRef = useRef();

  // Build path polyline from node positions
  const pathPoints = useMemo(() => {
    if (!path || path.length < 2 || !nodeDict) return [];
    return path.map(id => {
      const node = nodeDict[id];
      return node ? new THREE.Vector3(node.pos[0], node.pos[1], node.pos[2]) : null;
    }).filter(Boolean);
  }, [path, nodeDict]);

  // Total path length for parameterization
  const totalLength = useMemo(() => {
    let len = 0;
    for (let i = 1; i < pathPoints.length; i++) {
      len += pathPoints[i].distanceTo(pathPoints[i - 1]);
    }
    return len;
  }, [pathPoints]);

  // Geometry for particles
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const sizes = new Float32Array(PARTICLE_COUNT);
    const opacities = new Float32Array(PARTICLE_COUNT);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      sizes[i] = 0.15 + Math.random() * 0.1;
      opacities[i] = 0.5 + Math.random() * 0.5;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    geo.setAttribute('opacity', new THREE.BufferAttribute(opacities, 1));
    return geo;
  }, []);

  // Particle material
  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      color: '#ffffff',
      size: 0.3,
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
  }, []);

  useFrame(({ clock }) => {
    if (!pointsRef.current || pathPoints.length < 2 || !visible) return;
    const time = clock.getElapsedTime();
    const positions = pointsRef.current.geometry.attributes.position.array;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Each particle has a different phase offset
      const phase = (i / PARTICLE_COUNT);
      const speed = 0.3 + (i % 3) * 0.1;
      const t = ((time * speed + phase) % 1);

      // Interpolate along path
      const targetDist = t * totalLength;
      let accumulated = 0;
      let px = pathPoints[0].x, py = pathPoints[0].y, pz = pathPoints[0].z;

      for (let j = 1; j < pathPoints.length; j++) {
        const segLen = pathPoints[j].distanceTo(pathPoints[j - 1]);
        if (accumulated + segLen >= targetDist) {
          const segT = (targetDist - accumulated) / segLen;
          px = pathPoints[j - 1].x + (pathPoints[j].x - pathPoints[j - 1].x) * segT;
          py = pathPoints[j - 1].y + (pathPoints[j].y - pathPoints[j - 1].y) * segT;
          pz = pathPoints[j - 1].z + (pathPoints[j].z - pathPoints[j - 1].z) * segT;
          break;
        }
        accumulated += segLen;
      }

      // Slight perpendicular offset for visual spread
      const offset = Math.sin(time * 3 + i) * 0.1;
      positions[i * 3] = px + offset;
      positions[i * 3 + 1] = py + Math.cos(time * 2 + i) * 0.1;
      positions[i * 3 + 2] = pz + offset;
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  if (!visible || pathPoints.length < 2) return null;

  return (
    <points ref={pointsRef} geometry={geometry} material={material} frustumCulled={false} />
  );
}
