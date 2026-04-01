/**
 * ParticleScene.jsx — Full 3D holographic body viewer
 * Renders either a GLB anatomy model or procedural body with:
 * - Holographic translucent material (scanlines, fresnel, blue glow)
 * - Organ glow based on drug severity data
 * - Bloom post-processing
 * - Orbit controls for rotation
 * - Click-to-select organs
 */

import React, { Suspense, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Environment } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import ProceduralBody from './ProceduralBody';

// Check if a GLB model exists at the expected path
const MODEL_PATH = '/models/human-anatomy.glb';

function SceneContent({ organs, onOrganClick, onOrganHover, hoveredOrgan, hasGLB }) {
  // Lazy load AnatomyModel only if GLB exists
  const [AnatomyModel, setAnatomyModel] = useState(null);

  useEffect(() => {
    if (hasGLB) {
      import('./AnatomyModel').then(mod => setAnatomyModel(() => mod.default));
    }
  }, [hasGLB]);

  return (
    <>
      {/* Dark background */}
      <color attach="background" args={['#03050a']} />
      <fog attach="fog" args={['#03050a', 15, 30]} />

      {/* Lighting — soft ambient + accent lights */}
      <ambientLight intensity={0.15} />
      <pointLight position={[5, 8, 5]} intensity={0.4} color="#ffffff" />
      <pointLight position={[-5, -3, -5]} intensity={0.2} color="#3b82f6" />
      <pointLight position={[3, -5, 3]} intensity={0.15} color="#06b6d4" />

      {/* Star field background */}
      <Stars radius={50} depth={30} count={3000} factor={3} saturation={0.2} fade speed={0.5} />

      {/* 3D Body */}
      {hasGLB && AnatomyModel ? (
        <AnatomyModel
          modelPath={MODEL_PATH}
          organs={organs}
          onOrganClick={onOrganClick}
          onOrganHover={onOrganHover}
          hoveredOrgan={hoveredOrgan}
        />
      ) : (
        <ProceduralBody
          organs={organs}
          onOrganClick={onOrganClick}
          onOrganHover={onOrganHover}
          hoveredOrgan={hoveredOrgan}
        />
      )}

      {/* Floor grid */}
      <gridHelper
        args={[20, 40, '#0a2540', '#0a1628']}
        position={[0, -3.2, 0]}
        rotation={[0, 0, 0]}
      />

      {/* Orbit controls — slow auto-rotate, limited vertical */}
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={4}
        maxDistance={12}
        minPolarAngle={Math.PI * 0.2}
        maxPolarAngle={Math.PI * 0.8}
        autoRotate={false}
        target={[0, 1.2, 0]}
      />

      {/* Post-processing */}
      <EffectComposer>
        <Bloom
          luminanceThreshold={0.15}
          luminanceSmoothing={0.9}
          intensity={1.8}
          mipmapBlur
        />
        <Vignette offset={0.3} darkness={0.65} />
      </EffectComposer>
    </>
  );
}

export default function ParticleScene({ organs = {}, drugPathway, drugCount = 1, visible = true, onOrganClick, onOrganHover }) {
  const [hoveredOrgan, setHoveredOrgan] = useState(null);
  const [hasGLB, setHasGLB] = useState(false);

  // Check if GLB model file exists
  useEffect(() => {
    fetch(MODEL_PATH, { method: 'HEAD' })
      .then(res => {
        if (res.ok) setHasGLB(true);
      })
      .catch(() => setHasGLB(false));
  }, []);

  if (!visible) return null;

  const handleOrganClick = (organId) => {
    onOrganClick?.(organId);
  };

  const handleOrganHover = (organId) => {
    setHoveredOrgan(organId);
    onOrganHover?.(organId);
  };

  return (
    <div className="absolute inset-0" style={{ zIndex: 2 }}>
      <Canvas
        camera={{
          fov: 35,
          position: [0, 1.5, 8],
          near: 0.1,
          far: 100,
        }}
        gl={{
          alpha: false,
          antialias: true,
          powerPreference: 'high-performance',
          toneMapping: 3, // ACESFilmic
        }}
        dpr={[1, 2]}
      >
        <Suspense fallback={null}>
          <SceneContent
            organs={organs}
            onOrganClick={handleOrganClick}
            onOrganHover={handleOrganHover}
            hoveredOrgan={hoveredOrgan}
            hasGLB={hasGLB}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
