// InstancedNodes.jsx — Single InstancedMesh for all 1,350 drug nodes
import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { useGalaxy, useGalaxyDispatch } from './store';
import { getNodeVisuals } from './graphEngine';

const tempMatrix = new THREE.Matrix4();
const tempColor = new THREE.Color();
const tempVec = new THREE.Vector3();
const tempQuat = new THREE.Quaternion();
const tempScale = new THREE.Vector3();

// Custom shader material for nodes with fresnel glow
const nodeVertexShader = `
  attribute float instanceOpacity;
  attribute float instanceGlow;
  attribute float instanceScale;

  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying float vOpacity;
  varying float vGlow;
  varying vec3 vColor;

  void main() {
    vColor = instanceColor;
    vOpacity = instanceOpacity;
    vGlow = instanceGlow;
    vNormal = normalize(normalMatrix * normal);

    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    vViewPosition = -mvPosition.xyz;

    gl_Position = projectionMatrix * mvPosition;
  }
`;

const nodeFragmentShader = `
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying float vOpacity;
  varying float vGlow;
  varying vec3 vColor;

  uniform float uTime;

  void main() {
    vec3 viewDir = normalize(vViewPosition);
    float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 2.5);

    // Core color with emissive center
    vec3 coreColor = vColor * (0.6 + vGlow * 0.4);

    // Rim glow (fresnel)
    vec3 rimColor = vColor * fresnel * (0.5 + vGlow * 1.5);

    // Combine
    vec3 finalColor = coreColor + rimColor;

    // Subtle pulse for glowing nodes
    float pulse = 1.0 + sin(uTime * 2.0) * 0.08 * vGlow;
    finalColor *= pulse;

    float alpha = vOpacity * (0.7 + fresnel * 0.3);

    gl_FragColor = vec4(finalColor, alpha);
  }
`;

export default function InstancedNodes({ nodes, hasDrugs, layoutPositions }) {
  const meshRef = useRef();
  const { hoveredNode, selectedNode } = useGalaxy();
  const dispatch = useGalaxyDispatch();
  const { camera } = useThree();

  const count = nodes.length;

  // Buffers for per-instance attributes
  const { opacityArray, glowArray, scaleArray, colorArray, targetOpacity, targetScale, targetGlow, targetColors, currentPos, targetPos } = useMemo(() => {
    return {
      opacityArray: new Float32Array(count),
      glowArray: new Float32Array(count),
      scaleArray: new Float32Array(count),
      colorArray: new Float32Array(count * 3),
      targetOpacity: new Float32Array(count),
      targetScale: new Float32Array(count),
      targetGlow: new Float32Array(count),
      targetColors: new Float32Array(count * 3),
      currentPos: new Float32Array(count * 3), // for layout transitions
      targetPos: new Float32Array(count * 3),
    };
  }, [count]);

  // Entrance animation ref
  const entranceProgress = useRef(0);
  const isInitialized = useRef(false);

  // Create shared geometry and material
  const geometry = useMemo(() => new THREE.SphereGeometry(1, 20, 20), []);
  const material = useMemo(() => {
    const mat = new THREE.ShaderMaterial({
      vertexShader: nodeVertexShader,
      fragmentShader: nodeFragmentShader,
      uniforms: { uTime: { value: 0 } },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    return mat;
  }, []);

  // Set instance matrices (positions) and compute targets
  useEffect(() => {
    if (!meshRef.current || nodes.length === 0) return;
    const mesh = meshRef.current;

    nodes.forEach((node, i) => {
      const visuals = getNodeVisuals(node, hasDrugs);

      // Set target values
      targetScale[i] = visuals.size;
      targetOpacity[i] = visuals.opacity;
      targetGlow[i] = visuals.glow;
      tempColor.set(visuals.color);
      targetColors[i * 3] = tempColor.r;
      targetColors[i * 3 + 1] = tempColor.g;
      targetColors[i * 3 + 2] = tempColor.b;

      // Compute target position (from layout or default T-SNE)
      const layoutPos = layoutPositions?.get?.(node.id);
      const pos = layoutPos || node.pos;
      targetPos[i * 3] = pos[0];
      targetPos[i * 3 + 1] = pos[1];
      targetPos[i * 3 + 2] = pos[2];

      // Initialize current values (for entrance: start at 0)
      if (!isInitialized.current) {
        scaleArray[i] = 0;
        opacityArray[i] = 0;
        glowArray[i] = 0;
        colorArray[i * 3] = tempColor.r;
        colorArray[i * 3 + 1] = tempColor.g;
        colorArray[i * 3 + 2] = tempColor.b;
        currentPos[i * 3] = pos[0];
        currentPos[i * 3 + 1] = pos[1];
        currentPos[i * 3 + 2] = pos[2];
      }

      // Set position matrix
      const cx = isInitialized.current ? currentPos[i * 3] : pos[0];
      const cy = isInitialized.current ? currentPos[i * 3 + 1] : pos[1];
      const cz = isInitialized.current ? currentPos[i * 3 + 2] : pos[2];
      tempVec.set(cx, cy, cz);
      tempQuat.identity();
      tempScale.set(scaleArray[i] || 0.001, scaleArray[i] || 0.001, scaleArray[i] || 0.001);
      tempMatrix.compose(tempVec, tempQuat, tempScale);
      mesh.setMatrixAt(i, tempMatrix);
    });

    mesh.instanceMatrix.needsUpdate = true;
    if (!isInitialized.current) {
      entranceProgress.current = 0;
      isInitialized.current = true;
    }
  }, [nodes, hasDrugs, layoutPositions]);

  // Per-frame animation: lerp current → target + entrance cascade
  useFrame(({ clock }) => {
    if (!meshRef.current || nodes.length === 0) return;
    const mesh = meshRef.current;
    const time = clock.getElapsedTime();
    material.uniforms.uTime.value = time;

    entranceProgress.current = Math.min(entranceProgress.current + 0.016, 3.0);
    const ep = entranceProgress.current;

    let needsUpdate = false;
    const lerpSpeed = 0.08;

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];

      // Entrance cascade: nodes further from center appear later
      const dist = Math.sqrt(node.pos[0] ** 2 + node.pos[1] ** 2 + node.pos[2] ** 2);
      const entranceDelay = dist * 0.08;
      const entranceFactor = Math.min(1, Math.max(0, (ep - entranceDelay) * 1.5));

      // Hovered override
      let tScale = targetScale[i];
      let tOpacity = targetOpacity[i];
      let tGlow = targetGlow[i];

      if (hoveredNode === i) {
        tScale = Math.max(tScale, 0.35);
        tOpacity = 1.0;
        tGlow = 0.8;
      }

      // Apply entrance factor
      tScale *= entranceFactor;
      tOpacity *= entranceFactor;

      // Lerp
      scaleArray[i] += (tScale - scaleArray[i]) * lerpSpeed;
      opacityArray[i] += (tOpacity - opacityArray[i]) * lerpSpeed;
      glowArray[i] += (tGlow - glowArray[i]) * lerpSpeed;

      // Color lerp
      colorArray[i * 3] += (targetColors[i * 3] - colorArray[i * 3]) * lerpSpeed;
      colorArray[i * 3 + 1] += (targetColors[i * 3 + 1] - colorArray[i * 3 + 1]) * lerpSpeed;
      colorArray[i * 3 + 2] += (targetColors[i * 3 + 2] - colorArray[i * 3 + 2]) * lerpSpeed;

      // Lerp position toward target (layout transitions)
      const posLerp = 0.06;
      currentPos[i * 3] += (targetPos[i * 3] - currentPos[i * 3]) * posLerp;
      currentPos[i * 3 + 1] += (targetPos[i * 3 + 1] - currentPos[i * 3 + 1]) * posLerp;
      currentPos[i * 3 + 2] += (targetPos[i * 3 + 2] - currentPos[i * 3 + 2]) * posLerp;

      // Breathing animation for main nodes
      const breathe = node.isA || node.isB ?
        1 + Math.sin(time * 2 + i * 0.5) * 0.06 : 1;

      const s = Math.max(scaleArray[i] * breathe, 0.001);
      tempVec.set(currentPos[i * 3], currentPos[i * 3 + 1], currentPos[i * 3 + 2]);

      // Main drug float
      if (node.isA || node.isB) {
        tempVec.y += Math.sin(time * 1.5 + i) * 0.15;
      }

      tempQuat.identity();
      tempScale.set(s, s, s);
      tempMatrix.compose(tempVec, tempQuat, tempScale);
      mesh.setMatrixAt(i, tempMatrix);
      needsUpdate = true;
    }

    if (needsUpdate) {
      mesh.instanceMatrix.needsUpdate = true;
      // Update attribute buffers
      const geo = mesh.geometry;
      if (geo.attributes.instanceOpacity) {
        geo.attributes.instanceOpacity.array.set(opacityArray);
        geo.attributes.instanceOpacity.needsUpdate = true;
      }
      if (geo.attributes.instanceGlow) {
        geo.attributes.instanceGlow.array.set(glowArray);
        geo.attributes.instanceGlow.needsUpdate = true;
      }
      if (geo.attributes.instanceColor) {
        geo.attributes.instanceColor.array.set(colorArray);
        geo.attributes.instanceColor.needsUpdate = true;
      }
    }
  });

  // Set up instance attributes on mount
  useEffect(() => {
    if (!meshRef.current) return;
    const geo = meshRef.current.geometry;

    geo.setAttribute('instanceOpacity',
      new THREE.InstancedBufferAttribute(opacityArray, 1));
    geo.setAttribute('instanceGlow',
      new THREE.InstancedBufferAttribute(glowArray, 1));
    geo.setAttribute('instanceColor',
      new THREE.InstancedBufferAttribute(colorArray, 3));
  }, [count]);

  // Hover / click handlers via instanceId
  const handlePointerMove = (e) => {
    e.stopPropagation();
    if (e.instanceId !== undefined) {
      dispatch({ type: 'SET_HOVERED', payload: e.instanceId });
      document.body.style.cursor = 'pointer';
    }
  };

  const handlePointerOut = () => {
    dispatch({ type: 'SET_HOVERED', payload: null });
    document.body.style.cursor = 'auto';
  };

  const handleClick = (e) => {
    e.stopPropagation();
    if (e.instanceId !== undefined && nodes[e.instanceId]) {
      dispatch({ type: 'SET_SELECTED_NODE', payload: nodes[e.instanceId] });
    }
  };

  // Hovered node label
  const hoveredNodeData = hoveredNode !== null && hoveredNode !== undefined && nodes[hoveredNode]
    ? nodes[hoveredNode] : null;

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[geometry, material, count]}
        frustumCulled={false}
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
        onClick={handleClick}
      />

      {/* Floating label for hovered node */}
      {hoveredNodeData && (
        <Html
          position={hoveredNodeData.pos}
          center
          zIndexRange={[100, 0]}
          className="pointer-events-none"
          style={{ transform: 'translateY(-20px)' }}
        >
          <div className="bg-black/80 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-md shadow-xl whitespace-nowrap">
            <div className="text-[11px] font-bold text-white tracking-wide">{hoveredNodeData.name}</div>
            <div className="text-[9px] text-white/50 font-mono mt-0.5">{hoveredNodeData.category} · {hoveredNodeData.type}</div>
          </div>
        </Html>
      )}

      {/* Labels for selected drugs */}
      {nodes.filter(n => n.isA || n.isB).map(n => (
        <Html
          key={`label-${n.id}`}
          position={n.pos}
          center
          zIndexRange={[100, 0]}
          className="pointer-events-none"
          style={{ transform: 'translateY(-30px)' }}
        >
          <div
            className="px-3 py-1.5 rounded-sm text-[11px] font-bold tracking-widest uppercase backdrop-blur-md whitespace-nowrap border"
            style={{
              color: n.isA ? '#00d2ff' : '#ff8c00',
              borderColor: n.isA ? '#00d2ff' : '#ff8c00',
              background: 'rgba(5,8,20,0.9)',
              boxShadow: `0 0 20px ${n.isA ? 'rgba(0,210,255,0.3)' : 'rgba(255,140,0,0.3)'}`,
            }}
          >
            {n.isA ? '◆ ' : '◇ '}{n.name}
          </div>
        </Html>
      ))}
    </>
  );
}
