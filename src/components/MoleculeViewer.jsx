import React, { useMemo, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere, Box } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

// Atom colors based on element
const ATOM_COLORS = {
  C: '#4a4a4a',
  N: '#3B82F6',
  O: '#EF4444',
  S: '#EAB308',
  P: '#F97316',
  F: '#22C55E',
  Cl: '#22C55E',
  Br: '#A855F7',
  I: '#A855F7',
  H: '#ffffff',
  default: '#94A3B8'
};

const ATOM_RADII = {
  C: 0.4,
  N: 0.35,
  O: 0.35,
  S: 0.5,
  P: 0.45,
  H: 0.25,
  default: 0.35
};

// Parse SMILES into a simple 3D structure (simplified - real implementation would use RDKit)
function generateMoleculeGeometry(smiles, index = 0, totalDrugs = 1) {
  if (!smiles) return { atoms: [], bonds: [] };

  // Simple heuristic parsing for common atoms
  const atoms = [];
  const bonds = [];
  
  // Base position offset for multiple drugs
  const offsetX = (index - (totalDrugs - 1) / 2) * 6;
  
  // Parse SMILES characters
  let atomIndex = 0;
  const atomStack = [];
  let lastAtom = null;
  let bondType = 1;
  let ringAtoms = {};

  for (let i = 0; i < smiles.length; i++) {
    const char = smiles[i];
    
    if (char === '(' || char === ')' || char === '[' || char === ']') continue;
    if (char === '=') { bondType = 2; continue; }
    if (char === '#') { bondType = 3; continue; }
    
    // Check for atoms
    if (/[A-Z]/.test(char)) {
      let element = char;
      if (i + 1 < smiles.length && /[a-z]/.test(smiles[i + 1])) {
        element += smiles[i + 1];
        i++;
      }
      
      // Generate pseudo-random but deterministic position
      const angle = atomIndex * 2.4 + Math.sin(atomIndex * 0.7) * 0.5;
      const radius = 1.5 + Math.sin(atomIndex * 1.3) * 0.5;
      const height = Math.cos(atomIndex * 0.9) * 1.2;
      
      const atom = {
        element: element.toUpperCase(),
        position: [
          offsetX + Math.cos(angle) * radius,
          height,
          Math.sin(angle) * radius
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
    
    // Handle ring closures
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

// Single Atom component
function Atom({ position, element, isHighlighted }) {
  const color = ATOM_COLORS[element] || ATOM_COLORS.default;
  const radius = ATOM_RADII[element] || ATOM_RADII.default;
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current && isHighlighted) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.1);
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[radius, 32, 32]} />
      <meshStandardMaterial
        color={color}
        emissive={isHighlighted ? color : '#000000'}
        emissiveIntensity={isHighlighted ? 0.5 : 0}
        metalness={0.3}
        roughness={0.4}
      />
    </mesh>
  );
}

// Bond component
function Bond({ start, end, order }) {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end)
  ], [start, end]);

  return (
    <Line
      points={points}
      color="#4a5568"
      lineWidth={order === 1 ? 2 : order === 2 ? 3 : 4}
      opacity={0.8}
      transparent
    />
  );
}

// Molecule group component
function Molecule({ smiles, name, index, totalDrugs, isInteracting, riskLevel }) {
  const groupRef = useRef();
  const { atoms, bonds } = useMemo(
    () => generateMoleculeGeometry(smiles, index, totalDrugs),
    [smiles, index, totalDrugs]
  );

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3 + index) * 0.1;
    }
  });

  const highlightColor = riskLevel === 'critical' || riskLevel === 'high' 
    ? '#EF4444' 
    : riskLevel === 'medium' 
    ? '#F59E0B' 
    : '#3B82F6';

  return (
    <group ref={groupRef}>
      {/* Atoms */}
      {atoms.map((atom, i) => (
        <Atom
          key={i}
          position={atom.position}
          element={atom.element}
          isHighlighted={isInteracting}
        />
      ))}

      {/* Bonds */}
      {bonds.map((bond, i) => (
        <Bond
          key={i}
          start={atoms[bond.start]?.position || [0, 0, 0]}
          end={atoms[bond.end]?.position || [0, 0, 0]}
          order={bond.order}
        />
      ))}

      {/* Drug name label */}
      <Text
        position={[(index - (totalDrugs - 1) / 2) * 6, -2.5, 0]}
        fontSize={0.4}
        color="#94A3B8"
        anchorX="center"
        anchorY="middle"
      >
        {name}
      </Text>

      {/* Interaction indicator */}
      {isInteracting && totalDrugs === 2 && index === 0 && (
        <mesh position={[3, 0, 0]}>
          <torusGeometry args={[0.8, 0.1, 16, 32]} />
          <meshStandardMaterial
            color={highlightColor}
            emissive={highlightColor}
            emissiveIntensity={0.5}
            transparent
            opacity={0.6}
          />
        </mesh>
      )}
    </group>
  );
}

// Interaction beam between molecules
function InteractionBeam({ riskLevel }) {
  const beamRef = useRef();
  
  const color = riskLevel === 'critical' || riskLevel === 'high'
    ? '#EF4444'
    : riskLevel === 'medium'
    ? '#F59E0B'
    : '#22C55E';

  useFrame((state) => {
    if (beamRef.current) {
      beamRef.current.material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2) * 0.2;
    }
  });

  return (
    <mesh ref={beamRef} position={[0, 0, 0]}>
      <cylinderGeometry args={[0.05, 0.05, 6, 8]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.8}
        transparent
        opacity={0.5}
      />
    </mesh>
  );
}

// Background particles
function BackgroundParticles() {
  const pointsRef = useRef();
  const particleCount = 200;

  const positions = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 30;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 20;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 30;
    }
    return pos;
  }, []);

  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color="#3B82F6"
        transparent
        opacity={0.4}
        sizeAttenuation
      />
    </points>
  );
}

// Main scene
function Scene({ drugs, result }) {
  const hasResult = result && result.severity !== 'no_interaction';
  const riskLevel = result?.risk_level || 'low';

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} />
      
      <BackgroundParticles />

      {drugs.map((drug, i) => (
        <Molecule
          key={drug.drugbank_id || drug.name}
          smiles={drug.smiles}
          name={drug.name}
          index={i}
          totalDrugs={drugs.length}
          isInteracting={hasResult}
          riskLevel={riskLevel}
        />
      ))}

      {hasResult && drugs.length === 2 && (
        <InteractionBeam riskLevel={riskLevel} />
      )}

      <OrbitControls
        enablePan={false}
        minDistance={5}
        maxDistance={20}
        autoRotate
        autoRotateSpeed={0.5}
      />

      <EffectComposer>
        <Bloom
          intensity={0.5}
          luminanceThreshold={0.6}
          luminanceSmoothing={0.9}
        />
      </EffectComposer>
    </>
  );
}

export default function MoleculeViewer({ drugs = [], result }) {
  if (drugs.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 rounded-3xl bg-white/5 flex items-center justify-center mx-auto mb-4">
            <svg className="w-10 h-10 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <p className="text-sm text-slate-500">Add drugs to view molecular structures</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 0, 12], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
      >
        <Scene drugs={drugs} result={result} />
      </Canvas>
      
      {/* Legend */}
      <div className="absolute bottom-4 left-4 flex items-center gap-4 text-xs">
        <div className="flex items-center gap-2 px-3 py-1.5 bg-black/50 backdrop-blur-sm rounded-lg border border-white/10">
          <div className="w-3 h-3 rounded-full bg-[#4a4a4a]" />
          <span className="text-slate-400">Carbon</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-black/50 backdrop-blur-sm rounded-lg border border-white/10">
          <div className="w-3 h-3 rounded-full bg-[#3B82F6]" />
          <span className="text-slate-400">Nitrogen</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-black/50 backdrop-blur-sm rounded-lg border border-white/10">
          <div className="w-3 h-3 rounded-full bg-[#EF4444]" />
          <span className="text-slate-400">Oxygen</span>
        </div>
      </div>

      {/* Controls hint */}
      <div className="absolute bottom-4 right-4 px-3 py-1.5 bg-black/50 backdrop-blur-sm rounded-lg border border-white/10 text-xs text-slate-500">
        Drag to rotate • Scroll to zoom
      </div>
    </div>
  );
}
