/**
 * ProceduralBody.jsx — Procedural 3D human figure built from Three.js primitives
 * Used as fallback when no GLB model is available.
 * Creates a translucent holographic body with individually addressable organ zones.
 *
 * NOT a flat SVG — this is a real 3D mesh with depth, lighting, and perspective.
 * Uses capsule/sphere/cylinder geometries shaped into an anatomical silhouette.
 */

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import HolographicMaterial from './HolographicMaterial';

// ─── Organ mesh: glowing emissive sphere at anatomical position ────────────
function OrganMesh({ position, scale = [1, 1, 1], severity = 0, organId, onClick, onPointerOver, onPointerOut, color }) {
  const meshRef = useRef();
  const baseColor = color || (severity > 0.7 ? '#ef4444' : severity > 0.4 ? '#f97316' : severity > 0 ? '#eab308' : '#1a3a5c');
  const emissiveIntensity = severity > 0 ? 0.5 + severity * 2 : 0.1;

  useFrame((state) => {
    if (meshRef.current && severity > 0.4) {
      const pulse = 1 + Math.sin(state.clock.elapsedTime * 2 + severity * 5) * 0.08 * severity;
      meshRef.current.scale.set(scale[0] * pulse, scale[1] * pulse, scale[2] * pulse);
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={position}
      scale={scale}
      onClick={(e) => { e.stopPropagation(); onClick?.(organId); }}
      onPointerOver={(e) => { e.stopPropagation(); onPointerOver?.(organId); document.body.style.cursor = 'pointer'; }}
      onPointerOut={(e) => { e.stopPropagation(); onPointerOut?.(null); document.body.style.cursor = 'auto'; }}
    >
      <sphereGeometry args={[1, 24, 24]} />
      <meshStandardMaterial
        color={baseColor}
        emissive={baseColor}
        emissiveIntensity={emissiveIntensity}
        transparent
        opacity={severity > 0 ? 0.4 + severity * 0.4 : 0.15}
        roughness={0.3}
        metalness={0.1}
        depthWrite={false}
      />
    </mesh>
  );
}

// ─── Body part: holographic mesh segment ───────────────────────────────────
function BodyPart({ position = [0, 0, 0], rotation = [0, 0, 0], scale = [1, 1, 1], geometry = 'capsule', args, holoColor = '#51a4de', holoOpacity = 0.12, holoFresnel = 0.45 }) {
  return (
    <mesh position={position} rotation={rotation} scale={scale}>
      {geometry === 'capsule' && <capsuleGeometry args={args || [0.5, 1, 8, 16]} />}
      {geometry === 'sphere' && <sphereGeometry args={args || [0.5, 24, 24]} />}
      {geometry === 'cylinder' && <cylinderGeometry args={args || [0.3, 0.3, 1, 16]} />}
      <HolographicMaterial
        hologramColor={holoColor}
        hologramOpacity={holoOpacity}
        fresnelAmount={holoFresnel}
        fresnelOpacity={0.8}
        scanlineSize={6}
        hologramBrightness={0.8}
        signalSpeed={0.3}
        enableBlinking={false}
        side="DoubleSide"
      />
    </mesh>
  );
}

// ─── Wireframe overlay for body parts ──────────────────────────────────────
function WireframeOverlay({ position = [0, 0, 0], rotation = [0, 0, 0], scale = [1, 1, 1], geometry = 'capsule', args }) {
  return (
    <mesh position={position} rotation={rotation} scale={scale}>
      {geometry === 'capsule' && <capsuleGeometry args={args || [0.5, 1, 8, 16]} />}
      {geometry === 'sphere' && <sphereGeometry args={args || [0.5, 24, 24]} />}
      {geometry === 'cylinder' && <cylinderGeometry args={args || [0.3, 0.3, 1, 16]} />}
      <meshBasicMaterial
        color="#51a4de"
        wireframe
        transparent
        opacity={0.06}
        depthWrite={false}
      />
    </mesh>
  );
}

// ─── Main Procedural Body ──────────────────────────────────────────────────
export default function ProceduralBody({ organs = {}, onOrganClick, onOrganHover, hoveredOrgan }) {
  const groupRef = useRef();

  // Slow auto-rotation
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.15) * 0.15;
    }
  });

  // Body part definitions — shapes an anatomical human figure
  const bodyParts = useMemo(() => [
    // Head
    { pos: [0, 4.2, 0], rot: [0, 0, 0], scale: [0.55, 0.65, 0.55], geo: 'sphere', args: [1, 24, 24] },
    // Neck
    { pos: [0, 3.45, 0], rot: [0, 0, 0], scale: [0.22, 0.2, 0.22], geo: 'cylinder', args: [1, 0.9, 1, 12] },
    // Torso upper (chest)
    { pos: [0, 2.6, 0], rot: [0, 0, 0], scale: [0.75, 0.6, 0.4], geo: 'capsule', args: [1, 0.5, 8, 16] },
    // Torso lower (abdomen)
    { pos: [0, 1.5, 0], rot: [0, 0, 0], scale: [0.65, 0.6, 0.35], geo: 'capsule', args: [1, 0.4, 8, 16] },
    // Pelvis
    { pos: [0, 0.6, 0], rot: [0, 0, 0], scale: [0.6, 0.35, 0.35], geo: 'sphere', args: [1, 16, 16] },
    // Left upper arm
    { pos: [-1.1, 2.8, 0], rot: [0, 0, 0.2], scale: [0.2, 0.55, 0.2], geo: 'capsule', args: [1, 1, 6, 12] },
    // Left forearm
    { pos: [-1.35, 1.7, 0], rot: [0, 0, 0.1], scale: [0.16, 0.55, 0.16], geo: 'capsule', args: [1, 1, 6, 12] },
    // Right upper arm
    { pos: [1.1, 2.8, 0], rot: [0, 0, -0.2], scale: [0.2, 0.55, 0.2], geo: 'capsule', args: [1, 1, 6, 12] },
    // Right forearm
    { pos: [1.35, 1.7, 0], rot: [0, 0, -0.1], scale: [0.16, 0.55, 0.16], geo: 'capsule', args: [1, 1, 6, 12] },
    // Left thigh
    { pos: [-0.3, -0.3, 0], rot: [0, 0, 0.03], scale: [0.25, 0.75, 0.25], geo: 'capsule', args: [1, 1, 6, 12] },
    // Left shin
    { pos: [-0.33, -1.7, 0], rot: [0, 0, 0.01], scale: [0.18, 0.7, 0.18], geo: 'capsule', args: [1, 1, 6, 12] },
    // Right thigh
    { pos: [0.3, -0.3, 0], rot: [0, 0, -0.03], scale: [0.25, 0.75, 0.25], geo: 'capsule', args: [1, 1, 6, 12] },
    // Right shin
    { pos: [0.33, -1.7, 0], rot: [0, 0, -0.01], scale: [0.18, 0.7, 0.18], geo: 'capsule', args: [1, 1, 6, 12] },
    // Left hand
    { pos: [-1.45, 0.95, 0], rot: [0, 0, 0], scale: [0.12, 0.18, 0.08], geo: 'sphere', args: [1, 12, 12] },
    // Right hand
    { pos: [1.45, 0.95, 0], rot: [0, 0, 0], scale: [0.12, 0.18, 0.08], geo: 'sphere', args: [1, 12, 12] },
    // Left foot
    { pos: [-0.35, -2.7, 0.08], rot: [0.3, 0, 0], scale: [0.14, 0.1, 0.22], geo: 'sphere', args: [1, 12, 12] },
    // Right foot
    { pos: [0.35, -2.7, 0.08], rot: [0.3, 0, 0], scale: [0.14, 0.1, 0.22], geo: 'sphere', args: [1, 12, 12] },
  ], []);

  // Organ positions mapped to anatomical locations inside the body
  const organPositions = useMemo(() => ({
    brain:     { pos: [0, 4.2, 0], scale: [0.35, 0.35, 0.35] },
    heart:     { pos: [0.15, 2.7, 0.15], scale: [0.2, 0.22, 0.18] },
    lungs:     { pos: [0, 2.75, 0.05], scale: [0.55, 0.4, 0.3] },
    liver:     { pos: [-0.25, 1.95, 0.12], scale: [0.35, 0.22, 0.2] },
    kidney:    { pos: [0, 1.55, -0.1], scale: [0.45, 0.18, 0.18] },
    gi:        { pos: [0, 1.3, 0.08], scale: [0.4, 0.4, 0.25] },
    endocrine: { pos: [0, 3.4, 0.05], scale: [0.15, 0.1, 0.1] },
    musculoskeletal: { pos: [0, 1.5, 0], scale: [0.6, 1.8, 0.3] },
  }), []);

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      {/* Holographic body shell */}
      {bodyParts.map((part, i) => (
        <React.Fragment key={`body-${i}`}>
          <BodyPart
            position={part.pos}
            rotation={part.rot}
            scale={part.scale}
            geometry={part.geo}
            args={part.args}
          />
          <WireframeOverlay
            position={part.pos}
            rotation={part.rot}
            scale={part.scale.map(s => s * 1.01)}
            geometry={part.geo}
            args={part.args}
          />
        </React.Fragment>
      ))}

      {/* Organ meshes (visible inside the holographic body) */}
      {Object.entries(organPositions).map(([organId, { pos, scale }]) => {
        const organData = organs[organId];
        const severity = organData?.severity || 0;
        const isHovered = hoveredOrgan === organId;

        return (
          <OrganMesh
            key={organId}
            position={pos}
            scale={isHovered ? scale.map(s => s * 1.15) : scale}
            severity={severity}
            organId={organId}
            onClick={onOrganClick}
            onPointerOver={onOrganHover}
            onPointerOut={onOrganHover}
          />
        );
      })}

      {/* Spine line — glowing line down the center */}
      <mesh position={[0, 1.5, -0.15]}>
        <cylinderGeometry args={[0.02, 0.02, 4.5, 6]} />
        <meshBasicMaterial color="#51a4de" transparent opacity={0.15} />
      </mesh>

      {/* Rib hints — subtle horizontal lines */}
      {[-0.3, -0.1, 0.1, 0.3, 0.5].map((yOffset, i) => (
        <mesh key={`rib-${i}`} position={[0, 2.6 + yOffset, 0.15]} rotation={[0, 0, 0]}>
          <torusGeometry args={[0.5 - i * 0.03, 0.008, 4, 24, Math.PI]} />
          <meshBasicMaterial color="#51a4de" transparent opacity={0.08} />
        </mesh>
      ))}
    </group>
  );
}
