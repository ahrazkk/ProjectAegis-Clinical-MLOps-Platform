import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Stars } from '@react-three/drei';
// Removed post-processing for stability
import * as THREE from 'three';

// CPK Coloring - Standard scientific atom colors (brighter for dark background)
const ATOM_COLORS = {
  C: '#6B7280',   // Carbon - gray
  N: '#60A5FA',   // Nitrogen - bright blue
  O: '#F87171',   // Oxygen - bright red
  S: '#FBBF24',   // Sulfur - yellow
  P: '#FB923C',   // Phosphorus - orange
  F: '#4ADE80',   // Fluorine - green
  Cl: '#34D399',  // Chlorine - emerald
  Br: '#DC2626',  // Bromine - dark red
  I: '#A855F7',   // Iodine - purple
  H: '#E5E7EB',   // Hydrogen - light gray
  default: '#EC4899' // Unknown - pink
};

// Van der Waals radii (scaled for visualization)
const ATOM_RADII = {
  C: 0.38,
  N: 0.34,
  O: 0.32,
  S: 0.42,
  P: 0.44,
  F: 0.30,
  Cl: 0.38,
  Br: 0.40,
  I: 0.44,
  H: 0.22,
  default: 0.35
};

// Improved SMILES parser with better 3D coordinate generation
function generateMoleculeGeometry(smiles, index = 0, totalDrugs = 1) {
  if (!smiles) return { atoms: [], bonds: [] };

  const atoms = [];
  const bonds = [];

  // Spacing between molecules
  const spacing = 8;
  const offsetX = (index - (totalDrugs - 1) / 2) * spacing;

  // Parse SMILES
  let atomIndex = 0;
  let lastAtom = null;
  let bondType = 1;
  const ringAtoms = {};
  const branchStack = [];

  for (let i = 0; i < smiles.length; i++) {
    const char = smiles[i];

    // Handle branching
    if (char === '(') {
      branchStack.push(lastAtom);
      continue;
    }
    if (char === ')') {
      lastAtom = branchStack.pop();
      continue;
    }
    if (char === '[' || char === ']') continue;
    if (char === '=') { bondType = 2; continue; }
    if (char === '#') { bondType = 3; continue; }
    if (char === '-' || char === '+' || char === '@') continue;

    // Parse atoms
    if (/[A-Z]/.test(char)) {
      let element = char;
      if (i + 1 < smiles.length && /[a-z]/.test(smiles[i + 1])) {
        element += smiles[i + 1];
        i++;
      }

      // Create layered spiral structure for aesthetics
      const layer = Math.floor(atomIndex / 6);
      const posInLayer = atomIndex % 6;
      const layerAngle = (posInLayer / 6) * Math.PI * 2 + layer * 0.5;
      const layerRadius = 1.3 + layer * 0.35;
      const layerHeight = (layer - 1) * 0.7 + Math.sin(posInLayer * 0.8) * 0.25;

      const atom = {
        element: element.toUpperCase(),
        position: [
          offsetX + Math.cos(layerAngle) * layerRadius,
          layerHeight,
          Math.sin(layerAngle) * layerRadius
        ],
        index: atomIndex
      };
      atoms.push(atom);

      if (lastAtom !== null) {
        bonds.push({
          start: lastAtom,
          end: atomIndex,
          order: bondType
        });
        bondType = 1;
      }

      lastAtom = atomIndex;
      atomIndex++;
    }

    // Ring closures
    if (/[0-9]/.test(char)) {
      const ringNum = parseInt(char);
      if (ringAtoms[ringNum] !== undefined) {
        bonds.push({
          start: ringAtoms[ringNum],
          end: lastAtom,
          order: 1
        });
        delete ringAtoms[ringNum];
      } else {
        ringAtoms[ringNum] = lastAtom;
      }
    }
  }

  return { atoms, bonds };
}

// Glowing atom with premium materials - Optimized for Stability
function Atom({ position, element, isHighlighted }) {
  const color = ATOM_COLORS[element] || ATOM_COLORS.default;
  const radius = ATOM_RADII[element] || ATOM_RADII.default;

  return (
    <group position={position}>
      <mesh castShadow receiveShadow>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isHighlighted ? 0.4 : 0.15}
          metalness={0.5}
          roughness={0.2}
          envMapIntensity={1}
        />
      </mesh>

      {/* Subtle outer glow for high-risk highlights */}
      {isHighlighted && (
        <mesh>
          <sphereGeometry args={[radius * 1.5, 32, 32]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.15}
            side={THREE.BackSide}
          />
        </mesh>
      )}
    </group>
  );
}

// Stylized bond with metallic finish
function Bond({ start, end, order, isHighlighted }) {
  const offsets = order === 1 ? [[0, 0, 0]] :
    order === 2 ? [[0, 0.05, 0], [0, -0.05, 0]] :
      [[0, 0.07, 0], [0, 0, 0], [0, -0.07, 0]];

  return (
    <group>
      {offsets.map((offset, i) => (
        <Line
          key={i}
          points={[
            [start[0] + offset[0], start[1] + offset[1], start[2] + offset[2]],
            [end[0] + offset[0], end[1] + offset[1], end[2] + offset[2]]
          ]}
          color={isHighlighted ? '#ffffff' : '#94a3b8'}
          lineWidth={isHighlighted ? 3 : 2}
          opacity={isHighlighted ? 0.8 : 0.4}
          transparent
        />
      ))}
    </group>
  );
}

// Complete molecule with floating animation
function Molecule({ smiles, name, index, totalDrugs, isInteracting, riskLevel }) {
  const groupRef = useRef();
  const { atoms, bonds } = useMemo(
    () => generateMoleculeGeometry(smiles, index, totalDrugs),
    [smiles, index, totalDrugs]
  );

  // Floating and rotation animation
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.003;
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.5 + index) * 0.12;
    }
  });

  const spacing = 8;
  const labelY = -3;

  return (
    <group ref={groupRef}>
      {/* Atoms */}
      {atoms.map((atom, i) => (
        <Atom
          key={i}
          position={atom.position}
          element={atom.element}
          isHighlighted={isInteracting}
          pulsePhase={i * 0.3}
        />
      ))}

      {/* Bonds */}
      {bonds.map((bond, i) => (
        <Bond
          key={`bond-${i}`}
          start={atoms[bond.start]?.position || [0, 0, 0]}
          end={atoms[bond.end]?.position || [0, 0, 0]}
          order={bond.order}
          isHighlighted={isInteracting}
        />
      ))}

      {/* Drug name label */}
      <Text
        position={[(index - (totalDrugs - 1) / 2) * spacing, labelY, 0]}
        fontSize={0.45}
        color="#E2E8F0"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.015}
        outlineColor="#0F172A"
      >
        {name}
      </Text>

      {/* Subtle platform ring */}
      <mesh position={[(index - (totalDrugs - 1) / 2) * spacing, labelY - 0.4, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.8, 2.2, 48]} />
        <meshBasicMaterial color="#1E293B" transparent opacity={0.25} />
      </mesh>
    </group>
  );
}

// Interaction beam with particle effect
function InteractionBeam({ riskLevel }) {
  const beamRef = useRef();
  const particlesRef = useRef();

  const riskColors = {
    critical: '#EF4444',
    high: '#F97316',
    medium: '#EAB308',
    low: '#22C55E'
  };

  const color = riskColors[riskLevel] || '#3B82F6';

  // Create particles along the beam
  const particleCount = 25;
  const particles = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 6;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 0.4;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 0.4;
    }
    return pos;
  }, []);

  useFrame((state) => {
    if (beamRef.current) {
      beamRef.current.material.opacity = 0.35 + Math.sin(state.clock.elapsedTime * 3) * 0.15;
    }
    if (particlesRef.current) {
      const positions = particlesRef.current.geometry.attributes.position.array;
      for (let i = 0; i < particleCount; i++) {
        positions[i * 3] += 0.025;
        if (positions[i * 3] > 3.5) positions[i * 3] = -3.5;
      }
      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  return (
    <group>
      {/* Main energy beam */}
      <mesh ref={beamRef} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.06, 0.06, 7, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.45} />
      </mesh>

      {/* Outer glow */}
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.18, 0.18, 7, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.12} />
      </mesh>

      {/* Flowing particles */}
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={particleCount}
            array={particles}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={0.07} color={color} transparent opacity={0.8} sizeAttenuation />
      </points>

      {/* Interaction rings */}
      <mesh rotation={[0, 0, Math.PI / 2]} position={[0, 0, 0]}>
        <torusGeometry args={[0.5, 0.04, 16, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.6} />
      </mesh>
    </group>
  );
}

// Background with cleaner, subtler particles
function BackgroundParticles() {
  const pointsRef = useRef();
  const particleCount = 200; // Reduced count for cleaner look

  const [positions, sizes] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const sz = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 40;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 50;
      sz[i] = Math.random();
    }
    return [pos, sz];
  }, []);

  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.005;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={particleCount} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-size" count={particleCount} array={sizes} itemSize={1} />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color="#38bdf8"
        transparent
        opacity={0.3}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// Main scene composition with professional studio lighting
function Scene({ drugs, result }) {
  const hasResult = result && result.severity !== 'no_interaction';
  const riskLevel = result?.risk_level || 'low';

  return (
    <>
      {/* Studio Lighting Setup */}
      <ambientLight intensity={0.5} />
      <spotLight
        position={[10, 10, 10]}
        angle={0.5}
        penumbra={1}
        intensity={1.2}
        color="#ffffff"
        castShadow
      />
      <pointLight position={[-10, -5, -10]} intensity={0.8} color="#3b82f6" />
      <pointLight position={[5, 0, 5]} intensity={0.5} color="#06b6d4" />

      {/* Background */}
      <BackgroundParticles />
      <Stars radius={100} depth={50} count={1000} factor={4} saturation={0} fade speed={0.5} opacity={0.3} />

      {/* Molecules */}
      {drugs.map((drug, i) => (
        <Molecule
          key={drug.drugbank_id || drug.name || i}
          smiles={drug.smiles}
          name={drug.name}
          index={i}
          totalDrugs={drugs.length}
          isInteracting={hasResult}
          riskLevel={riskLevel}
        />
      ))}

      {/* Interaction visualization */}
      {hasResult && drugs.length === 2 && (
        <InteractionBeam riskLevel={riskLevel} />
      )}

      {/* Camera */}
      <OrbitControls
        enablePan={false}
        minDistance={8}
        maxDistance={25}
        autoRotate={!hasResult}
        autoRotateSpeed={0.5}
        maxPolarAngle={Math.PI * 0.65}
        minPolarAngle={Math.PI * 0.35}
      />
    </>
  );
}

// Empty state component with animation
function EmptyState() {
  return (
    <div className="w-full h-full flex items-center justify-center">
      <div className="text-center">
        <div className="relative w-24 h-24 mx-auto mb-6">
          {/* Animated molecule icon */}
          <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 animate-pulse" />
          <div className="absolute inset-2 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
            <svg className="w-12 h-12 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          {/* Orbiting dots */}
          <div className="absolute inset-0 animate-spin" style={{ animationDuration: '8s' }}>
            <div className="absolute top-0 left-1/2 w-2 h-2 -ml-1 rounded-full bg-blue-500" />
          </div>
          <div className="absolute inset-0 animate-spin" style={{ animationDuration: '12s', animationDirection: 'reverse' }}>
            <div className="absolute bottom-0 left-1/2 w-2 h-2 -ml-1 rounded-full bg-cyan-500" />
          </div>
        </div>
        <p className="text-slate-400 font-medium mb-2">No molecules to display</p>
        <p className="text-sm text-slate-600">Search and add drugs to visualize their structures</p>
      </div>
    </div>
  );
}

// Main export
export default function MoleculeViewer({ drugs = [], result }) {
  if (drugs.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ position: [0, 2, 14], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 1.5]}
      >
        <Scene drugs={drugs} result={result} />
      </Canvas>

      {/* Enhanced Legend */}
      <div className="absolute bottom-4 left-4 flex items-center gap-2.5">
        {[
          { color: '#6B7280', label: 'Carbon' },
          { color: '#60A5FA', label: 'Nitrogen' },
          { color: '#F87171', label: 'Oxygen' },
          { color: '#FBBF24', label: 'Sulfur' },
        ].map(({ color, label }) => (
          <div
            key={label}
            className="flex items-center gap-2 px-3 py-2 bg-slate-900/80 backdrop-blur-sm rounded-xl border border-white/5"
          >
            <div
              className="w-3 h-3 rounded-full shadow-lg"
              style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}50` }}
            />
            <span className="text-xs text-slate-400 font-medium">{label}</span>
          </div>
        ))}
      </div>

      {/* Controls hint */}
      <div className="absolute bottom-4 right-4 px-4 py-2 bg-slate-900/80 backdrop-blur-sm rounded-xl border border-white/5">
        <p className="text-xs text-slate-500">
          <span className="text-slate-400">Drag</span> to rotate • <span className="text-slate-400">Scroll</span> to zoom
        </p>
      </div>

      {/* Data source indicator */}
      <div className="absolute top-4 right-4 px-3 py-1.5 bg-emerald-500/10 backdrop-blur-sm rounded-lg border border-emerald-500/20">
        <p className="text-xs text-emerald-400 font-medium flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          Live Data from Knowledge Graph
        </p>
      </div>
    </div>
  );
}
