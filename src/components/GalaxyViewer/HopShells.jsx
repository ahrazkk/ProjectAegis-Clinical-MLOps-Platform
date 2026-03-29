// HopShells.jsx — Translucent concentric spheres showing hop boundaries
import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function HopShell({ center, radius, color, delay = 0 }) {
  const ref = useRef();
  const scaleRef = useRef(0);
  const startTime = useRef(null);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime();
    if (startTime.current === null) startTime.current = t;

    const elapsed = t - startTime.current - delay;
    if (elapsed < 0) {
      ref.current.scale.set(0.001, 0.001, 0.001);
      return;
    }

    // Ease-out scale animation
    const progress = Math.min(1, elapsed * 1.5);
    const eased = 1 - Math.pow(1 - progress, 3);
    scaleRef.current = eased;

    const s = radius * eased;
    ref.current.scale.set(s, s, s);

    // Gentle pulse
    const pulse = 1 + Math.sin(t * 1.2 + delay * 3) * 0.02;
    ref.current.scale.multiplyScalar(pulse);

    // Rotate slowly
    ref.current.rotation.y = t * 0.05;
    ref.current.rotation.x = t * 0.03;
  });

  return (
    <mesh ref={ref} position={center}>
      <icosahedronGeometry args={[1, 2]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.04}
        side={THREE.BackSide}
        depthWrite={false}
        wireframe
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

export default function HopShells({ nodes, drugAId, drugBId, maxHops }) {
  // Compute shell radii from actual node positions at each hop level
  const shells = useMemo(() => {
    const result = [];
    if (!drugAId && !drugBId) return result;

    const computeShells = (drugId, color) => {
      if (!drugId) return;
      const drugNode = nodes.find(n => n.id === drugId);
      if (!drugNode) return;
      const center = drugNode.pos;

      for (let hop = 1; hop <= maxHops; hop++) {
        const hopNodes = nodes.filter(n => {
          const hopDist = drugNode.isA ? n.hopA : n.hopB;
          return hopDist === hop;
        });
        if (hopNodes.length === 0) continue;

        // Compute average distance of nodes at this hop level
        let maxDist = 0;
        hopNodes.forEach(n => {
          const dx = n.pos[0] - center[0];
          const dy = n.pos[1] - center[1];
          const dz = n.pos[2] - center[2];
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (d > maxDist) maxDist = d;
        });

        result.push({
          center,
          radius: maxDist * 0.7, // slightly smaller than max for aesthetic
          color,
          hop,
          delay: hop * 0.3,
        });
      }
    };

    computeShells(drugAId, '#00d2ff');
    computeShells(drugBId, '#ff8c00');

    return result;
  }, [nodes, drugAId, drugBId, maxHops]);

  return (
    <group>
      {shells.map((s, i) => (
        <HopShell
          key={`shell-${i}`}
          center={s.center}
          radius={s.radius}
          color={s.color}
          delay={s.delay}
        />
      ))}
    </group>
  );
}
