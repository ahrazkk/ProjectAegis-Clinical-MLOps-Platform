// CameraController.jsx — Smart camera with fly-to, zoom-to-fit, auto-orbit
import React, { useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useGalaxy } from './store';

const tempVec = new THREE.Vector3();
const targetPos = new THREE.Vector3();
const targetLookAt = new THREE.Vector3();

export default function CameraController({ isMobile }) {
  const controlsRef = useRef();
  const { cameraTarget, cameraFit, drugA, drugB } = useGalaxy();
  const { camera } = useThree();

  const flyingTo = useRef(false);
  const flyProgress = useRef(1);
  const startPos = useRef(new THREE.Vector3());
  const startTarget = useRef(new THREE.Vector3());
  const endPos = useRef(new THREE.Vector3());
  const endTarget = useRef(new THREE.Vector3());
  const idleTimer = useRef(0);
  const userInteracted = useRef(false);

  // Trigger fly-to when cameraTarget changes
  useEffect(() => {
    if (!cameraTarget || !controlsRef.current) return;

    const target = new THREE.Vector3(cameraTarget[0], cameraTarget[1], cameraTarget[2]);

    // Compute a good camera position: offset from target
    const offset = new THREE.Vector3(0, 4, 12);
    const camPos = target.clone().add(offset);

    startPos.current.copy(camera.position);
    startTarget.current.copy(controlsRef.current.target);
    endPos.current.copy(camPos);
    endTarget.current.copy(target);

    flyProgress.current = 0;
    flyingTo.current = true;
  }, [cameraTarget]);

  // Zoom to fit when both drugs selected
  useEffect(() => {
    if (!drugA || !drugB || !controlsRef.current) return;

    const a = new THREE.Vector3(drugA.pos[0], drugA.pos[1], drugA.pos[2]);
    const b = new THREE.Vector3(drugB.pos[0], drugB.pos[1], drugB.pos[2]);
    const center = a.clone().add(b).multiplyScalar(0.5);
    const dist = a.distanceTo(b);

    const camDist = Math.max(dist * 1.5, 15);
    const camPos = center.clone().add(new THREE.Vector3(0, camDist * 0.3, camDist));

    startPos.current.copy(camera.position);
    startTarget.current.copy(controlsRef.current.target);
    endPos.current.copy(camPos);
    endTarget.current.copy(center);

    flyProgress.current = 0;
    flyingTo.current = true;
  }, [drugA?.id, drugB?.id]);

  useFrame((_, delta) => {
    if (!controlsRef.current) return;

    // Fly-to animation
    if (flyingTo.current && flyProgress.current < 1) {
      flyProgress.current = Math.min(flyProgress.current + delta * 0.8, 1);
      // Ease in-out cubic
      const t = flyProgress.current;
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      camera.position.lerpVectors(startPos.current, endPos.current, ease);
      controlsRef.current.target.lerpVectors(startTarget.current, endTarget.current, ease);
      controlsRef.current.update();

      if (flyProgress.current >= 1) flyingTo.current = false;
    }

    // Auto-rotate management: pause after interaction, resume after idle
    if (userInteracted.current) {
      idleTimer.current += delta;
      if (idleTimer.current > 6) {
        userInteracted.current = false;
        controlsRef.current.autoRotate = true;
      }
    }
  });

  const handleStart = () => {
    userInteracted.current = true;
    idleTimer.current = 0;
    if (controlsRef.current) controlsRef.current.autoRotate = false;
  };

  return (
    <OrbitControls
      ref={controlsRef}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      autoRotate={true}
      autoRotateSpeed={0.25}
      maxDistance={isMobile ? 80 : 120}
      minDistance={3}
      enableDamping={true}
      dampingFactor={0.05}
      onStart={handleStart}
    />
  );
}
