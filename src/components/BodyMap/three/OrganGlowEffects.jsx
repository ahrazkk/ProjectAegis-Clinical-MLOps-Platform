// OrganGlowEffects.jsx — Bloom sprites at affected organ positions
// Creates soft glowing orbs at each affected organ for ambient luminance.

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { ORGAN_SYSTEMS, getSeverityColor } from '../organRegistry';

// Map SVG viewBox coords to Three.js scene coords
function svgToScene(svgX, svgY) {
  const scaleX = 0.05;
  const scaleY = -0.05;
  return {
    x: svgX * scaleX - 200 * scaleX,
    y: svgY * scaleY + 780 * 0.5 * 0.05,
    z: -0.1, // Slightly behind particles
  };
}

export default function OrganGlowEffects({ organs = {} }) {
  const meshRef = useRef();

  const glowData = useMemo(() => {
    const data = [];
    for (const [key, def] of Object.entries(ORGAN_SYSTEMS)) {
      const severity = organs[key]?.severity || 0;
      if (severity <= 0 || !def.center) continue;

      const colorInfo = getSeverityColor(severity);
      const color = new THREE.Color(colorInfo.fill);
      const pos = svgToScene(def.center.x, def.center.y);

      data.push({
        position: new THREE.Vector3(pos.x, pos.y, pos.z),
        color,
        severity,
        baseSize: def.isOverlay ? 3 : 2,
      });
    }
    return data;
  }, [organs]);

  // Animate pulsing
  const timeRef = useRef(0);
  useFrame((_, delta) => {
    timeRef.current += delta;
    if (!meshRef.current) return;

    const posAttr = meshRef.current.geometry.getAttribute('position');
    const sizeAttr = meshRef.current.geometry.getAttribute('size');
    if (!posAttr || !sizeAttr) return;

    for (let i = 0; i < glowData.length; i++) {
      const d = glowData[i];
      const pulse = 1 + Math.sin(timeRef.current * 2 + i * 1.5) * 0.2 * d.severity;
      sizeAttr.array[i] = d.baseSize * pulse * (0.5 + d.severity);
    }
    sizeAttr.needsUpdate = true;
  });

  if (glowData.length === 0) return null;

  const positions = new Float32Array(glowData.length * 3);
  const colors = new Float32Array(glowData.length * 3);
  const sizes = new Float32Array(glowData.length);

  for (let i = 0; i < glowData.length; i++) {
    const d = glowData[i];
    positions[i * 3] = d.position.x;
    positions[i * 3 + 1] = d.position.y;
    positions[i * 3 + 2] = d.position.z;
    colors[i * 3] = d.color.r;
    colors[i * 3 + 1] = d.color.g;
    colors[i * 3 + 2] = d.color.b;
    sizes[i] = d.baseSize * (0.5 + d.severity);
  }

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" array={positions} count={glowData.length} itemSize={3} />
        <bufferAttribute attach="attributes-color" array={colors} count={glowData.length} itemSize={3} />
        <bufferAttribute attach="attributes-size" array={sizes} count={glowData.length} itemSize={1} />
      </bufferGeometry>
      <pointsMaterial
        size={4}
        vertexColors
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation={false}
      />
    </points>
  );
}
