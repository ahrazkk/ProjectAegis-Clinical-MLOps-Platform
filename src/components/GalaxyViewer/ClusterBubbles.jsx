// ClusterBubbles.jsx — Translucent spheres around therapeutic class clusters
import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { CATEGORY_COLORS } from './graphEngine';

function ClusterBubble({ center, radius, color, label, count }) {
  const meshRef = useRef();

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.getElapsedTime();
    // Gentle breathing
    const scale = radius * (1 + Math.sin(t * 0.8) * 0.02);
    meshRef.current.scale.set(scale, scale, scale);
    meshRef.current.rotation.y = t * 0.03;
  });

  return (
    <group position={center}>
      {/* Translucent sphere */}
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[1, 2]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.035}
          side={THREE.BackSide}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Wireframe shell */}
      <mesh scale={[radius, radius, radius]}>
        <icosahedronGeometry args={[1, 1]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.06}
          wireframe
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Label */}
      <Html
        position={[0, radius + 1, 0]}
        center
        zIndexRange={[50, 0]}
        className="pointer-events-none"
      >
        <div className="text-center whitespace-nowrap">
          <div
            className="text-[10px] font-bold tracking-wider uppercase"
            style={{ color, textShadow: `0 0 10px ${color}40` }}
          >
            {label}
          </div>
          <div className="text-[8px] text-white/25 font-mono">{count} drugs</div>
        </div>
      </Html>
    </group>
  );
}

export default function ClusterBubbles({ clusterCenters, groups, visible }) {
  if (!visible || !clusterCenters || !groups) return null;

  const bubbles = useMemo(() => {
    return Object.entries(clusterCenters).map(([cat, center]) => {
      const nodesInGroup = groups[cat]?.length || 0;
      if (nodesInGroup === 0) return null;
      const radius = Math.min(2 + Math.sqrt(nodesInGroup) * 0.5, 6);
      return {
        center,
        radius,
        color: CATEGORY_COLORS[cat] || '#475569',
        label: cat,
        count: nodesInGroup,
      };
    }).filter(Boolean);
  }, [clusterCenters, groups]);

  return (
    <group>
      {bubbles.map((b, i) => (
        <ClusterBubble key={i} {...b} />
      ))}
    </group>
  );
}
