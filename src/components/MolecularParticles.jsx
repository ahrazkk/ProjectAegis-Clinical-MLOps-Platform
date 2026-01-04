import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// DNA-like helix structure with molecular nodes
function MolecularHelix({ scrollProgress }) {
  const groupRef = useRef();
  const particleCount = 300;

  const { positions, colors, sizes } = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      // Create helix pattern
      const t = (i / particleCount) * Math.PI * 8;
      const strand = i % 2 === 0 ? 1 : -1;
      const radius = 3 + Math.sin(t * 0.5) * 0.5;

      positions[i * 3] = Math.cos(t) * radius * strand;
      positions[i * 3 + 1] = (i / particleCount - 0.5) * 15;
      positions[i * 3 + 2] = Math.sin(t) * radius;

      // Blue to cyan gradient
      const colorT = i / particleCount;
      colors[i * 3] = 0.2 + colorT * 0.2; // R
      colors[i * 3 + 1] = 0.5 + colorT * 0.3; // G
      colors[i * 3 + 2] = 0.9 + Math.random() * 0.1; // B

      sizes[i] = 0.05 + Math.random() * 0.08;
    }

    return { positions, colors, sizes };
  }, []);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.1;
      groupRef.current.rotation.x = scrollProgress * Math.PI * 0.15;
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.5;
    }
  });

  return (
    <group ref={groupRef}>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={particleCount} array={positions} itemSize={3} />
          <bufferAttribute attach="attributes-color" count={particleCount} array={colors} itemSize={3} />
        </bufferGeometry>
        <pointsMaterial
          size={0.12}
          vertexColors
          transparent
          opacity={0.9}
          sizeAttenuation
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>
    </group>
  );
}

// Floating molecular nodes
function FloatingNodes() {
  const groupRef = useRef();
  const nodeCount = 50;

  const nodes = useMemo(() => {
    return Array.from({ length: nodeCount }, (_, i) => ({
      position: [
        (Math.random() - 0.5) * 25,
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20
      ],
      size: 0.1 + Math.random() * 0.15,
      speed: 0.3 + Math.random() * 0.5,
      offset: Math.random() * Math.PI * 2
    }));
  }, []);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <group ref={groupRef}>
      {nodes.map((node, i) => (
        <mesh key={i} position={node.position}>
          <sphereGeometry args={[node.size, 8, 8]} />
          <meshBasicMaterial
            color={i % 3 === 0 ? '#3B82F6' : i % 3 === 1 ? '#06B6D4' : '#8B5CF6'}
            transparent
            opacity={0.4}
          />
        </mesh>
      ))}
    </group>
  );
}

// Background particle cloud
function ParticleCloud({ scrollProgress }) {
  const pointsRef = useRef();
  const particleCount = 800;

  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      // Spherical distribution
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = 8 + Math.random() * 12;

      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);

      // Subtle color variation
      const brightness = 0.3 + Math.random() * 0.4;
      colors[i * 3] = brightness * 0.3;
      colors[i * 3 + 1] = brightness * 0.5;
      colors[i * 3 + 2] = brightness * 1.0;
    }

    return { positions, colors };
  }, []);

  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.03;
      pointsRef.current.rotation.x = scrollProgress * Math.PI * 0.1;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={particleCount} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={particleCount} array={colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.04}
        vertexColors
        transparent
        opacity={0.6}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

export default function ParticleSystem({ scrollProgress = 0 }) {
  return (
    <group>
      <ParticleCloud scrollProgress={scrollProgress} />
      <MolecularHelix scrollProgress={scrollProgress} />
      <FloatingNodes />
    </group>
  );
}
