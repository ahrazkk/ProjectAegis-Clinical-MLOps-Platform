import React, { useRef, useMemo, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// =============================================================================
// PREMIUM MONOCHROME MOLECULAR VISUALIZATION
// Black & white with subtle cyan/purple accents - pharmaceutical elegance
// =============================================================================

// Glowing atom core with outer ring
function PremiumAtom({ position, radius = 0.12, color = '#ffffff', glowColor = '#00FFFF', isActive = false }) {
  const groupRef = useRef();
  const ringRef = useRef();
  
  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.x = state.clock.elapsedTime * 0.8;
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.5;
    }
  });

  return (
    <group ref={groupRef} position={position}>
      {/* Core sphere */}
      <mesh>
        <sphereGeometry args={[radius, 16, 16]} />
        <meshStandardMaterial 
          color={color}
          emissive={isActive ? glowColor : '#111111'}
          emissiveIntensity={isActive ? 0.5 : 0.1}
          metalness={0.3}
          roughness={0.5}
        />
      </mesh>
      
      {/* Orbital ring for active atoms */}
      {isActive && (
        <mesh ref={ringRef}>
          <torusGeometry args={[radius * 2, 0.008, 8, 32]} />
          <meshBasicMaterial color={glowColor} transparent opacity={0.4} />
        </mesh>
      )}
    </group>
  );
}

// Sleek bond cylinder with optional glow
function BondCylinder({ start, end, radius = 0.03, color = '#888888', glowing = false }) {
  const { position, quaternion, length } = useMemo(() => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const midpoint = new THREE.Vector3().lerpVectors(startVec, endVec, 0.5);
    const direction = new THREE.Vector3().subVectors(endVec, startVec);
    const length = direction.length();
    
    const quaternion = new THREE.Quaternion();
    const up = new THREE.Vector3(0, 1, 0);
    quaternion.setFromUnitVectors(up, direction.clone().normalize());
    
    return { position: midpoint, quaternion, length };
  }, [start, end]);

  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[radius, radius, length, 8]} />
      <meshStandardMaterial 
        color={color} 
        metalness={0.3} 
        roughness={0.5}
        emissive={glowing ? color : '#000000'}
        emissiveIntensity={glowing ? 0.3 : 0}
      />
    </mesh>
  );
}

// =============================================================================
// DNA HELIX - Elegant double helix structure
// =============================================================================
function DNAHelix({ position = [0, 0, 0], scale = 1, color = '#ffffff' }) {
  const groupRef = useRef();
  
  const { strand1, strand2, rungs } = useMemo(() => {
    const strand1Points = [];
    const strand2Points = [];
    const rungs = [];
    const turns = 2;
    const pointsPerTurn = 20;
    const height = 8;
    const radius = 0.6;
    
    for (let i = 0; i < turns * pointsPerTurn; i++) {
      const t = i / pointsPerTurn;
      const y = (i / (turns * pointsPerTurn)) * height - height / 2;
      const angle1 = t * Math.PI * 2;
      const angle2 = angle1 + Math.PI;
      
      strand1Points.push(new THREE.Vector3(
        Math.cos(angle1) * radius,
        y,
        Math.sin(angle1) * radius
      ));
      
      strand2Points.push(new THREE.Vector3(
        Math.cos(angle2) * radius,
        y,
        Math.sin(angle2) * radius
      ));
      
      // Add rungs every few points
      if (i % 4 === 0) {
        rungs.push({
          start: [Math.cos(angle1) * radius, y, Math.sin(angle1) * radius],
          end: [Math.cos(angle2) * radius, y, Math.sin(angle2) * radius]
        });
      }
    }
    
    return {
      strand1: new THREE.BufferGeometry().setFromPoints(strand1Points),
      strand2: new THREE.BufferGeometry().setFromPoints(strand2Points),
      rungs
    };
  }, []);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group ref={groupRef} position={position} scale={scale}>
      {/* Strands */}
      <line geometry={strand1}>
        <lineBasicMaterial color={color} transparent opacity={0.3} />
      </line>
      <line geometry={strand2}>
        <lineBasicMaterial color={color} transparent opacity={0.3} />
      </line>
      
      {/* Rungs */}
      {rungs.map((rung, i) => (
        <BondCylinder 
          key={i} 
          start={rung.start} 
          end={rung.end} 
          radius={0.015} 
          color="#333333" 
        />
      ))}
    </group>
  );
}

// =============================================================================
// MOLECULE STRUCTURE - Warfarin-inspired (anticoagulant) - Premium version
// =============================================================================
function MoleculeStructure({ 
  position = [0, 0, 0], 
  scale = 1, 
  rotation = [0, 0, 0],
  accentColor = '#00FFFF',
  isInteracting = false,
  interactionIntensity = 0
}) {
  const groupRef = useRef();
  const glowRef = useRef();
  
  // Molecule structure - complex polycyclic structure
  const atoms = useMemo(() => [
    // Core benzene ring
    { pos: [0, 0, 0], type: 'C' },
    { pos: [0.6, 0.35, 0], type: 'C' },
    { pos: [0.6, 1.0, 0], type: 'C' },
    { pos: [0, 1.35, 0], type: 'C' },
    { pos: [-0.6, 1.0, 0], type: 'C' },
    { pos: [-0.6, 0.35, 0], type: 'C' },
    // Pyrone ring extension
    { pos: [1.2, 0, 0], type: 'O', accent: true },
    { pos: [1.6, 0.5, 0], type: 'C' },
    { pos: [1.4, 1.15, 0], type: 'C' },
    { pos: [2.1, 0.3, 0], type: 'O', accent: true },
    // Side chain with phenyl
    { pos: [-1.2, 0, 0], type: 'C' },
    { pos: [-1.6, -0.5, 0.3], type: 'C' },
    { pos: [-2.2, -0.3, 0.5], type: 'C' },
    { pos: [-2.6, -0.8, 0.6], type: 'C' },
    { pos: [-2.4, -1.5, 0.4], type: 'C' },
    { pos: [-1.8, -1.7, 0.2], type: 'C' },
    { pos: [-1.4, -1.2, 0.1], type: 'C' },
    // Carbonyl
    { pos: [0, 1.95, 0], type: 'O', accent: true },
  ], []);

  const bonds = useMemo(() => [
    // Benzene ring
    [[0, 0, 0], [0.6, 0.35, 0]],
    [[0.6, 0.35, 0], [0.6, 1.0, 0]],
    [[0.6, 1.0, 0], [0, 1.35, 0]],
    [[0, 1.35, 0], [-0.6, 1.0, 0]],
    [[-0.6, 1.0, 0], [-0.6, 0.35, 0]],
    [[-0.6, 0.35, 0], [0, 0, 0]],
    // Pyrone
    [[0.6, 0.35, 0], [1.2, 0, 0]],
    [[1.2, 0, 0], [1.6, 0.5, 0]],
    [[1.6, 0.5, 0], [1.4, 1.15, 0]],
    [[1.4, 1.15, 0], [0.6, 1.0, 0]],
    [[1.6, 0.5, 0], [2.1, 0.3, 0]],
    // Side chain
    [[-0.6, 0.35, 0], [-1.2, 0, 0]],
    [[-1.2, 0, 0], [-1.6, -0.5, 0.3]],
    [[-1.6, -0.5, 0.3], [-2.2, -0.3, 0.5]],
    [[-2.2, -0.3, 0.5], [-2.6, -0.8, 0.6]],
    [[-2.6, -0.8, 0.6], [-2.4, -1.5, 0.4]],
    [[-2.4, -1.5, 0.4], [-1.8, -1.7, 0.2]],
    [[-1.8, -1.7, 0.2], [-1.4, -1.2, 0.1]],
    [[-1.4, -1.2, 0.1], [-1.6, -0.5, 0.3]],
    // Carbonyl
    [[0, 1.35, 0], [0, 1.95, 0]],
  ], []);

  useFrame((state) => {
    if (groupRef.current) {
      const time = state.clock.elapsedTime;
      groupRef.current.rotation.y = rotation[1] + time * 0.1;
      groupRef.current.rotation.x = rotation[0] + Math.sin(time * 0.2) * 0.08;
      groupRef.current.rotation.z = Math.sin(time * 0.15) * 0.03;
      
      // Floating motion
      groupRef.current.position.y = position[1] + Math.sin(time * 0.4) * 0.2;
      
      // Subtle pulse when interacting
      if (isInteracting && interactionIntensity > 0.3) {
        const pulse = 1 + Math.sin(time * 4) * 0.015 * interactionIntensity;
        groupRef.current.scale.setScalar(scale * pulse);
      }
    }
  });

  return (
    <group ref={groupRef} position={position} scale={scale}>
      {/* Bonds */}
      {bonds.map((bond, i) => (
        <BondCylinder 
          key={`bond-${i}`} 
          start={bond[0]} 
          end={bond[1]} 
          color="#666666" 
          radius={0.028}
          glowing={isInteracting && interactionIntensity > 0.5}
        />
      ))}
      
      {/* Atoms */}
      {atoms.map((atom, i) => (
        <PremiumAtom 
          key={`atom-${i}`} 
          position={atom.pos} 
          color={atom.accent && isInteracting ? accentColor : '#ffffff'}
          glowColor={accentColor}
          radius={atom.accent ? 0.11 : 0.09}
          isActive={atom.accent && isInteracting && interactionIntensity > 0.4}
        />
      ))}
      
      {/* Molecular label - floating text simulation */}
      {isInteracting && interactionIntensity > 0.6 && (
        <mesh position={[0, -1.5, 0]}>
          <planeGeometry args={[1.5, 0.2]} />
          <meshBasicMaterial color={accentColor} transparent opacity={0.1} />
        </mesh>
      )}
    </group>
  );
}

// =============================================================================
// ASPIRIN-LIKE MOLECULE - Premium version
// =============================================================================
function AspirinMolecule({ 
  position = [0, 0, 0], 
  scale = 1, 
  rotation = [0, 0, 0],
  accentColor = '#8B5CF6',
  isInteracting = false,
  interactionIntensity = 0
}) {
  const groupRef = useRef();
  
  const atoms = useMemo(() => [
    // Benzene ring
    { pos: [0, 0, 0], type: 'C' },
    { pos: [0.5, 0.3, 0], type: 'C' },
    { pos: [0.5, 0.9, 0], type: 'C' },
    { pos: [0, 1.2, 0], type: 'C' },
    { pos: [-0.5, 0.9, 0], type: 'C' },
    { pos: [-0.5, 0.3, 0], type: 'C' },
    // Carboxylic acid
    { pos: [1.0, 0, 0], type: 'C' },
    { pos: [1.4, 0.4, 0], type: 'O', accent: true },
    { pos: [1.4, -0.4, 0], type: 'O', accent: true },
    // Acetyl group
    { pos: [-1.0, 0, 0], type: 'O', accent: true },
    { pos: [-1.4, -0.4, 0], type: 'C' },
    { pos: [-1.9, -0.2, 0], type: 'O', accent: true },
    { pos: [-1.4, -1.0, 0], type: 'C' },
    // Extra hydroxyl
    { pos: [0, 1.7, 0], type: 'O', accent: true },
  ], []);

  const bonds = useMemo(() => [
    [[0, 0, 0], [0.5, 0.3, 0]],
    [[0.5, 0.3, 0], [0.5, 0.9, 0]],
    [[0.5, 0.9, 0], [0, 1.2, 0]],
    [[0, 1.2, 0], [-0.5, 0.9, 0]],
    [[-0.5, 0.9, 0], [-0.5, 0.3, 0]],
    [[-0.5, 0.3, 0], [0, 0, 0]],
    [[0.5, 0.3, 0], [1.0, 0, 0]],
    [[1.0, 0, 0], [1.4, 0.4, 0]],
    [[1.0, 0, 0], [1.4, -0.4, 0]],
    [[-0.5, 0.3, 0], [-1.0, 0, 0]],
    [[-1.0, 0, 0], [-1.4, -0.4, 0]],
    [[-1.4, -0.4, 0], [-1.9, -0.2, 0]],
    [[-1.4, -0.4, 0], [-1.4, -1.0, 0]],
    [[0, 1.2, 0], [0, 1.7, 0]],
  ], []);

  useFrame((state) => {
    if (groupRef.current) {
      const time = state.clock.elapsedTime;
      groupRef.current.rotation.y = rotation[1] + time * 0.08;
      groupRef.current.rotation.x = rotation[0] + Math.cos(time * 0.25) * 0.08;
      groupRef.current.rotation.z = Math.cos(time * 0.12) * 0.03;
      groupRef.current.position.y = position[1] + Math.cos(time * 0.45) * 0.2;
      
      if (isInteracting && interactionIntensity > 0.3) {
        const pulse = 1 + Math.sin(time * 4 + Math.PI) * 0.015 * interactionIntensity;
        groupRef.current.scale.setScalar(scale * pulse);
      }
    }
  });

  return (
    <group ref={groupRef} position={position} scale={scale}>
      {bonds.map((bond, i) => (
        <BondCylinder 
          key={`bond-${i}`} 
          start={bond[0]} 
          end={bond[1]} 
          color="#666666" 
          radius={0.028}
          glowing={isInteracting && interactionIntensity > 0.5}
        />
      ))}
      {atoms.map((atom, i) => (
        <PremiumAtom 
          key={`atom-${i}`} 
          position={atom.pos} 
          color={atom.accent && isInteracting ? accentColor : '#ffffff'}
          glowColor={accentColor}
          radius={atom.accent ? 0.11 : 0.09}
          isActive={atom.accent && isInteracting && interactionIntensity > 0.4}
        />
      ))}
    </group>
  );
}

// =============================================================================
// ENERGY PARTICLES - Flow between molecules during interaction
// =============================================================================
function EnergyParticles({ startPos, endPos, intensity = 0, color = '#00FFFF' }) {
  const pointsRef = useRef();
  const count = 80;
  
  const { positions, velocities, lifetimes } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const velocities = [];
    const lifetimes = [];
    
    for (let i = 0; i < count; i++) {
      positions[i * 3] = startPos[0];
      positions[i * 3 + 1] = startPos[1];
      positions[i * 3 + 2] = startPos[2];
      velocities.push(Math.random() * 0.5 + 0.3);
      lifetimes.push(Math.random());
    }
    
    return { positions, velocities, lifetimes };
  }, [startPos]);

  useFrame((state) => {
    if (pointsRef.current && intensity > 0.3) {
      const posArray = pointsRef.current.geometry.attributes.position.array;
      const time = state.clock.elapsedTime;
      
      for (let i = 0; i < count; i++) {
        lifetimes[i] += velocities[i] * 0.02;
        if (lifetimes[i] > 1) {
          lifetimes[i] = 0;
        }
        
        const t = lifetimes[i];
        const noise = Math.sin(time * 3 + i) * 0.3 * (1 - t);
        
        posArray[i * 3] = startPos[0] + (endPos[0] - startPos[0]) * t + noise;
        posArray[i * 3 + 1] = startPos[1] + (endPos[1] - startPos[1]) * t + noise * 0.5;
        posArray[i * 3 + 2] = startPos[2] + (endPos[2] - startPos[2]) * t + noise * 0.3;
      }
      
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
      pointsRef.current.material.opacity = intensity * 0.6;
    }
  });

  if (intensity < 0.3) return null;

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color={color}
        transparent
        opacity={0.4}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// =============================================================================
// INTERACTION EFFECTS - Electric arc / neural pathway style
// =============================================================================
function ElectricArc({ startPos, endPos, intensity = 0 }) {
  const lineRef = useRef();
  const segments = 20;
  
  const geometry = useMemo(() => {
    const points = [];
    for (let i = 0; i <= segments; i++) {
      points.push(new THREE.Vector3(0, 0, 0));
    }
    return new THREE.BufferGeometry().setFromPoints(points);
  }, []);

  useFrame((state) => {
    if (lineRef.current && intensity > 0.4) {
      const time = state.clock.elapsedTime;
      const posArray = lineRef.current.geometry.attributes.position.array;
      
      for (let i = 0; i <= segments; i++) {
        const t = i / segments;
        const noise = Math.sin(time * 10 + t * 15) * 0.15 * (1 - Math.abs(t - 0.5) * 2);
        
        posArray[i * 3] = startPos[0] + (endPos[0] - startPos[0]) * t;
        posArray[i * 3 + 1] = startPos[1] + (endPos[1] - startPos[1]) * t + noise;
        posArray[i * 3 + 2] = startPos[2] + (endPos[2] - startPos[2]) * t + noise * 0.5;
      }
      
      lineRef.current.geometry.attributes.position.needsUpdate = true;
      lineRef.current.material.opacity = (0.2 + Math.sin(time * 8) * 0.1) * intensity;
    }
  });

  if (intensity < 0.4) return null;

  return (
    <line ref={lineRef} geometry={geometry}>
      <lineBasicMaterial color="#ffffff" transparent opacity={0.3} linewidth={2} />
    </line>
  );
}

// =============================================================================
// BACKGROUND ELEMENTS - Deep space pharmaceutical aesthetic
// =============================================================================

// Premium star field with depth layers
function StarField({ count = 2000 }) {
  const pointsRef = useRef();
  
  const { positions, colors, sizes } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = 12 + Math.random() * 40;
      
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
      
      // Mostly white/gray with rare accents
      const brightness = 0.4 + Math.random() * 0.6;
      const accentChance = Math.random();
      
      if (accentChance > 0.985) {
        // Very rare cyan
        colors[i * 3] = 0;
        colors[i * 3 + 1] = brightness;
        colors[i * 3 + 2] = brightness;
        sizes[i] = 0.06 + Math.random() * 0.04;
      } else if (accentChance > 0.97) {
        // Very rare purple
        colors[i * 3] = brightness * 0.5;
        colors[i * 3 + 1] = brightness * 0.3;
        colors[i * 3 + 2] = brightness;
        sizes[i] = 0.06 + Math.random() * 0.04;
      } else {
        // White/gray
        colors[i * 3] = brightness;
        colors[i * 3 + 1] = brightness;
        colors[i * 3 + 2] = brightness;
        sizes[i] = 0.02 + Math.random() * 0.03;
      }
    }
    
    return { positions, colors, sizes };
  }, [count]);

  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.003;
      pointsRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.01) * 0.05;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={count} array={colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.04}
        vertexColors
        transparent
        opacity={0.9}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// Floating wireframe polyhedra - very subtle background geometry
function FloatingPolyhedra({ count = 12 }) {
  const groupRef = useRef();
  
  const shapes = useMemo(() => {
    return Array.from({ length: count }, () => ({
      position: [
        (Math.random() - 0.5) * 35,
        (Math.random() - 0.5) * 18,
        -8 - Math.random() * 15
      ],
      rotation: [Math.random() * Math.PI, Math.random() * Math.PI, 0],
      scale: 0.2 + Math.random() * 0.5,
      rotationSpeed: (Math.random() - 0.5) * 0.2,
      type: Math.floor(Math.random() * 3),
    }));
  }, [count]);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.children.forEach((child, i) => {
        child.rotation.y += shapes[i].rotationSpeed * 0.01;
        child.rotation.x += shapes[i].rotationSpeed * 0.005;
      });
    }
  });

  return (
    <group ref={groupRef}>
      {shapes.map((shape, i) => (
        <mesh key={i} position={shape.position} rotation={shape.rotation} scale={shape.scale}>
          {shape.type === 0 ? (
            <icosahedronGeometry args={[1, 0]} />
          ) : shape.type === 1 ? (
            <octahedronGeometry args={[1, 0]} />
          ) : (
            <dodecahedronGeometry args={[1, 0]} />
          )}
          <meshBasicMaterial color="#1a1a1a" wireframe transparent opacity={0.25} />
        </mesh>
      ))}
    </group>
  );
}

// Orbital ring around interaction zone
function OrbitalRing({ position = [0, 0, 0], radius = 4, intensity = 0 }) {
  const ringRef = useRef();
  
  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.x = Math.PI / 2 + Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.15;
      ringRef.current.material.opacity = 0.05 + intensity * 0.1;
    }
  });

  return (
    <mesh ref={ringRef} position={position}>
      <torusGeometry args={[radius, 0.01, 8, 64]} />
      <meshBasicMaterial color="#ffffff" transparent opacity={0.1} />
    </mesh>
  );
}

// Horizontal scan line - subtle
function ScanLine() {
  const lineRef = useRef();
  
  useFrame((state) => {
    if (lineRef.current) {
      const y = Math.sin(state.clock.elapsedTime * 0.15) * 6;
      lineRef.current.position.y = y;
      lineRef.current.material.opacity = 0.02 + Math.abs(Math.sin(state.clock.elapsedTime * 0.15)) * 0.02;
    }
  });

  return (
    <mesh ref={lineRef} position={[0, 0, -5]}>
      <planeGeometry args={[60, 0.008]} />
      <meshBasicMaterial color="#ffffff" transparent opacity={0.03} side={THREE.DoubleSide} />
    </mesh>
  );
}

// Grid floor - ultra subtle
function GridFloor() {
  return (
    <gridHelper 
      args={[80, 60, '#0a0a0a', '#0a0a0a']} 
      position={[0, -6, 0]}
      rotation={[0, 0, 0]}
    />
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function DrugInteractionBackground({ scrollProgress = 0 }) {
  const [drugAPos, setDrugAPos] = useState([-3.5, 0, 0]);
  const [drugBPos, setDrugBPos] = useState([3.5, 0, 0]);
  const [distance, setDistance] = useState(7);

  // Calculate interaction
  const interactionIntensity = Math.max(0, 1 - (distance / 7));
  const isInteracting = distance < 5.5;
  
  // Center point for effects
  const centerPos = [
    (drugAPos[0] + drugBPos[0]) / 2,
    (drugAPos[1] + drugBPos[1]) / 2,
    (drugAPos[2] + drugBPos[2]) / 2
  ];

  return (
    <group>
      {/* Scene controller */}
      <SceneController 
        scrollProgress={scrollProgress}
        setDrugAPos={setDrugAPos}
        setDrugBPos={setDrugBPos}
        setDistance={setDistance}
      />
      
      {/* Deep background layers */}
      <StarField count={2500} />
      <FloatingPolyhedra count={15} />
      <GridFloor />
      <ScanLine />
      
      {/* DNA helixes in background */}
      <DNAHelix position={[-12, 0, -8]} scale={0.5} color="#333333" />
      <DNAHelix position={[12, -2, -10]} scale={0.4} color="#222222" />
      
      {/* Orbital ring when interacting */}
      <OrbitalRing position={centerPos} radius={distance / 2 + 1} intensity={interactionIntensity} />
      
      {/* Main molecules */}
      <MoleculeStructure 
        position={drugAPos}
        scale={1.5}
        rotation={[0.2, 0, 0]}
        accentColor="#00FFFF"
        isInteracting={isInteracting}
        interactionIntensity={interactionIntensity}
      />
      
      <AspirinMolecule 
        position={drugBPos}
        scale={1.4}
        rotation={[-0.1, Math.PI, 0]}
        accentColor="#8B5CF6"
        isInteracting={isInteracting}
        interactionIntensity={interactionIntensity}
      />
      
      {/* Interaction effects */}
      <ElectricArc 
        startPos={drugAPos}
        endPos={drugBPos}
        intensity={interactionIntensity}
      />
      
      <EnergyParticles 
        startPos={drugAPos}
        endPos={drugBPos}
        intensity={interactionIntensity}
        color="#00FFFF"
      />
      
      <EnergyParticles 
        startPos={drugBPos}
        endPos={drugAPos}
        intensity={interactionIntensity}
        color="#8B5CF6"
      />
      
      {/* Lighting - elegant pharmaceutical */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[8, 8, 8]} intensity={0.5} color="#ffffff" />
      <directionalLight position={[-8, -5, 5]} intensity={0.2} color="#8888ff" />
      <pointLight position={centerPos} intensity={interactionIntensity * 0.5} color="#ffffff" distance={8} />
    </group>
  );
}

// Scene animation controller
function SceneController({ scrollProgress, setDrugAPos, setDrugBPos, setDistance }) {
  useFrame((state) => {
    const time = state.clock.elapsedTime;
    
    // Elegant oscillation - molecules drift toward and away from each other
    const oscillation = (Math.sin(time * 0.12) + 1) / 2; // 0 to 1
    const baseDistance = 1.8 + oscillation * 3.5; // 1.8 to 5.3
    
    const newDrugAPos = [
      -baseDistance - scrollProgress * 2,
      Math.sin(time * 0.3) * 0.4,
      Math.sin(time * 0.2) * 0.25
    ];
    
    const newDrugBPos = [
      baseDistance + scrollProgress * 2,
      Math.cos(time * 0.3) * 0.4,
      Math.cos(time * 0.2) * 0.25
    ];
    
    setDrugAPos(newDrugAPos);
    setDrugBPos(newDrugBPos);
    
    // Calculate distance
    const dx = newDrugAPos[0] - newDrugBPos[0];
    const dy = newDrugAPos[1] - newDrugBPos[1];
    const dz = newDrugAPos[2] - newDrugBPos[2];
    setDistance(Math.sqrt(dx * dx + dy * dy + dz * dz));
  });
  
  return null;
}
