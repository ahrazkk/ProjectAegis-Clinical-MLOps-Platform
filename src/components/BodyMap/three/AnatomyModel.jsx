/**
 * AnatomyModel.jsx — Loads a GLB anatomy model with per-organ interaction
 * Automatically traverses the GLTF scene graph and maps named meshes to organ systems.
 * Applies holographic material to body shell, emissive glow to affected organs.
 *
 * To use: place a human anatomy .glb file at /public/models/human-anatomy.glb
 * The model should have meshes named like "Heart", "Lungs", "Liver", "Brain", "Kidney", etc.
 */

import React, { useRef, useMemo, useEffect, useState } from 'react';
import { useGLTF } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import HolographicMaterial from './HolographicMaterial';

// Map mesh names in GLB to our organ system IDs
const MESH_TO_ORGAN = {
  // Common naming conventions in anatomy models
  'heart': 'heart', 'Heart': 'heart', 'HEART': 'heart', 'corazon': 'heart',
  'brain': 'brain', 'Brain': 'brain', 'BRAIN': 'brain', 'cerebro': 'brain', 'cerebrum': 'brain',
  'lung': 'lungs', 'Lung': 'lungs', 'lungs': 'lungs', 'Lungs': 'lungs', 'LUNGS': 'lungs',
  'lung_l': 'lungs', 'lung_r': 'lungs', 'Lung_L': 'lungs', 'Lung_R': 'lungs',
  'LeftLung': 'lungs', 'RightLung': 'lungs',
  'liver': 'liver', 'Liver': 'liver', 'LIVER': 'liver', 'higado': 'liver',
  'kidney': 'kidney', 'Kidney': 'kidney', 'KIDNEY': 'kidney', 'kidneys': 'kidney', 'Kidneys': 'kidney',
  'kidney_l': 'kidney', 'kidney_r': 'kidney', 'Kidney_L': 'kidney', 'Kidney_R': 'kidney',
  'LeftKidney': 'kidney', 'RightKidney': 'kidney',
  'stomach': 'gi', 'Stomach': 'gi', 'STOMACH': 'gi', 'intestine': 'gi', 'Intestine': 'gi',
  'intestines': 'gi', 'Intestines': 'gi', 'colon': 'gi', 'Colon': 'gi',
  'SmallIntestine': 'gi', 'LargeIntestine': 'gi', 'Digestive': 'gi',
  'skin': 'skin', 'Skin': 'skin', 'body': null, 'Body': null, // Body shell = holographic
  'skeleton': 'musculoskeletal', 'Skeleton': 'musculoskeletal', 'bone': 'musculoskeletal', 'Bone': 'musculoskeletal',
  'thyroid': 'endocrine', 'Thyroid': 'endocrine', 'adrenal': 'endocrine', 'Adrenal': 'endocrine',
  'pancreas': 'endocrine', 'Pancreas': 'endocrine',
  'blood': 'blood', 'Blood': 'blood', 'vein': 'blood', 'Vein': 'blood', 'artery': 'blood', 'Artery': 'blood',
};

// Fuzzy match mesh name to organ ID
function meshNameToOrgan(name) {
  if (!name) return null;

  // Direct match
  if (MESH_TO_ORGAN[name] !== undefined) return MESH_TO_ORGAN[name];

  // Partial match (case-insensitive)
  const lower = name.toLowerCase();
  for (const [key, value] of Object.entries(MESH_TO_ORGAN)) {
    if (lower.includes(key.toLowerCase())) return value;
  }

  return null; // Unknown mesh — treat as body shell
}

function getSeverityColor(severity) {
  if (severity > 0.7) return new THREE.Color('#ef4444');
  if (severity > 0.4) return new THREE.Color('#f97316');
  if (severity > 0) return new THREE.Color('#eab308');
  return new THREE.Color('#1a3a5c');
}

export default function AnatomyModel({ modelPath, organs = {}, onOrganClick, onOrganHover, hoveredOrgan }) {
  const { scene } = useGLTF(modelPath);
  const groupRef = useRef();
  const [organMeshes, setOrganMeshes] = useState({});

  // Traverse scene and categorize meshes
  useEffect(() => {
    const meshMap = {};

    scene.traverse((child) => {
      if (child.isMesh) {
        const organId = meshNameToOrgan(child.name);

        if (organId === null) {
          // Body shell — make holographic/transparent
          child.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color('#51a4de'),
            emissive: new THREE.Color('#51a4de'),
            emissiveIntensity: 0.1,
            transparent: true,
            opacity: 0.08,
            wireframe: false,
            depthWrite: false,
            side: THREE.DoubleSide,
          });
        } else if (organId) {
          // Organ mesh — add to map
          if (!meshMap[organId]) meshMap[organId] = [];
          meshMap[organId].push(child);

          // Set initial material
          const severity = organs[organId]?.severity || 0;
          const color = getSeverityColor(severity);
          child.material = new THREE.MeshStandardMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: severity > 0 ? 0.5 + severity * 1.5 : 0.1,
            transparent: true,
            opacity: severity > 0 ? 0.4 + severity * 0.4 : 0.2,
            roughness: 0.3,
            metalness: 0.1,
            depthWrite: false,
          });
        }
      }
    });

    setOrganMeshes(meshMap);
  }, [scene]);

  // Update organ materials when severity changes
  useEffect(() => {
    for (const [organId, meshes] of Object.entries(organMeshes)) {
      const severity = organs[organId]?.severity || 0;
      const color = getSeverityColor(severity);
      const isHovered = hoveredOrgan === organId;

      for (const mesh of meshes) {
        if (mesh.material) {
          mesh.material.color.copy(color);
          mesh.material.emissive.copy(color);
          mesh.material.emissiveIntensity = severity > 0 ? 0.5 + severity * 1.5 : 0.1;
          mesh.material.opacity = severity > 0 ? 0.4 + severity * 0.4 : isHovered ? 0.3 : 0.2;
        }
      }
    }
  }, [organs, hoveredOrgan, organMeshes]);

  // Slow rotation
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.15) * 0.15;
    }
  });

  // Pulse animation for severe organs
  useFrame((state) => {
    for (const [organId, meshes] of Object.entries(organMeshes)) {
      const severity = organs[organId]?.severity || 0;
      if (severity > 0.5) {
        const pulse = 1 + Math.sin(state.clock.elapsedTime * 2 + severity * 5) * 0.05 * severity;
        for (const mesh of meshes) {
          mesh.scale.setScalar(pulse);
        }
      }
    }
  });

  return (
    <group ref={groupRef}>
      <primitive
        object={scene}
        onClick={(e) => {
          e.stopPropagation();
          const organId = meshNameToOrgan(e.object.name);
          if (organId) onOrganClick?.(organId);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          const organId = meshNameToOrgan(e.object.name);
          if (organId) {
            onOrganHover?.(organId);
            document.body.style.cursor = 'pointer';
          }
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          onOrganHover?.(null);
          document.body.style.cursor = 'auto';
        }}
      />
    </group>
  );
}
