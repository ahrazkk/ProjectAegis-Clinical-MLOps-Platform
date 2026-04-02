// InstancedNodes.jsx — Single InstancedMesh for all 1,350 drug nodes
import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { useGalaxy, useGalaxyDispatch } from './store';
import { getAdj, getNodeVisuals } from './graphEngine';

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
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(vViewPosition);

    vec3 lightDirA = normalize(vec3(0.6, 0.9, 0.4));
    vec3 lightDirB = normalize(vec3(-0.7, -0.2, 0.6));

    float diffuseA = max(dot(normal, lightDirA), 0.0);
    float diffuseB = max(dot(normal, lightDirB), 0.0);
    float fresnel = pow(1.0 - abs(dot(viewDir, normal)), 2.2);

    vec3 ambient = vColor * 0.22;
    vec3 diffuse = vColor * (0.55 * diffuseA + 0.25 * diffuseB + 0.2);
    vec3 rim = vColor * fresnel * (0.18 + vGlow * 0.35);

    float pulse = 1.0 + sin(uTime * 2.0) * 0.05 * vGlow;
    vec3 finalColor = (ambient + diffuse + rim) * pulse;

    float alpha = clamp(vOpacity * (0.72 + fresnel * 0.18), 0.0, 1.0);

    gl_FragColor = vec4(finalColor, alpha);
  }
`;

export default function InstancedNodes({ nodes, hasDrugs, layoutPositions, nodeVisibility }) {
  const meshRef = useRef();
  const { hoveredNode, showLabels, viewMode } = useGalaxy();
  const dispatch = useGalaxyDispatch();

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
  const geometry = useMemo(() => new THREE.SphereGeometry(1, 24, 24), []);
  const material = useMemo(() => {
    const mat = new THREE.ShaderMaterial({
      vertexShader: nodeVertexShader,
      fragmentShader: nodeFragmentShader,
      uniforms: { uTime: { value: 0 } },
      transparent: true,
      depthWrite: true,
      depthTest: true,
      blending: THREE.NormalBlending,
    });
    return mat;
  }, []);

  // Set instance matrices (positions) and compute targets
  useEffect(() => {
    if (!meshRef.current || nodes.length === 0) return;
    const mesh = meshRef.current;

    nodes.forEach((node, i) => {
      const visuals = getNodeVisuals(node, hasDrugs, viewMode);

      // Apply filter visibility — ghost mode for filtered-out nodes
      const isVisible = !nodeVisibility || nodeVisibility.get(node.id) !== false;
      const filterScale = isVisible ? 1.0 : 0.001;
      const filterOpacity = isVisible ? 1.0 : 0;

      // Set target values
      targetScale[i] = visuals.size * filterScale;
      targetOpacity[i] = visuals.opacity * filterOpacity;
      targetGlow[i] = isVisible ? visuals.glow : 0;
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
  }, [nodes, hasDrugs, layoutPositions, nodeVisibility, viewMode]);

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

      // Breathing animation for selected regimen nodes
      const breathe = node.isA || node.isB || node.isSelected ?
        1 + Math.sin(time * 2 + i * 0.5) * 0.06 : 1;

      const s = Math.max(scaleArray[i] * breathe, 0.001);
      tempVec.set(currentPos[i * 3], currentPos[i * 3 + 1], currentPos[i * 3 + 2]);

      // Selected regimen nodes float subtly for emphasis
      if (node.isA || node.isB || node.isSelected) {
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
  const adjacency = useMemo(() => getAdj(), [nodes.length]);

  const labelNodes = useMemo(() => {
    const visibleNodes = nodes.filter(n => !nodeVisibility || nodeVisibility.get(n.id) !== false);
    if (showLabels === 'none') return [];

    const selectedLabels = visibleNodes.filter(n => n.isA || n.isB || n.isSelected);
    const byId = new Map(selectedLabels.map(n => [n.id, n]));

    if (showLabels === 'selected') {
      return Array.from(byId.values());
    }

    // "All" mode intentionally caps labels to high-degree hubs + selected nodes for readability.
    const hubNodes = visibleNodes
      .filter(n => !byId.has(n.id))
      .map(n => ({ node: n, degree: (adjacency[n.id] || []).length }))
      .sort((a, b) => b.degree - a.degree)
      .slice(0, 18)
      .map(item => item.node);

    hubNodes.forEach(n => byId.set(n.id, n));
    return Array.from(byId.values()).slice(0, 24);
  }, [nodes, nodeVisibility, showLabels, adjacency]);

  const getLabelStyle = (node) => {
    if (node.isA) {
      return {
        color: '#00d2ff',
        borderColor: '#00d2ff',
        background: 'rgba(5,8,20,0.9)',
        boxShadow: '0 0 20px rgba(0,210,255,0.3)',
        marker: '◆ ',
      };
    }
    if (node.isB) {
      return {
        color: '#ff8c00',
        borderColor: '#ff8c00',
        background: 'rgba(5,8,20,0.9)',
        boxShadow: '0 0 20px rgba(255,140,0,0.3)',
        marker: '◇ ',
      };
    }
    if (node.isSelected) {
      return {
        color: '#22c55e',
        borderColor: '#22c55e',
        background: 'rgba(5,12,8,0.9)',
        boxShadow: '0 0 18px rgba(34,197,94,0.25)',
        marker: '● ',
      };
    }
    return {
      color: '#cbd5e1',
      borderColor: 'rgba(203,213,225,0.45)',
      background: 'rgba(4,6,16,0.85)',
      boxShadow: '0 0 10px rgba(148,163,184,0.18)',
      marker: '',
    };
  };

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
          position={layoutPositions?.get?.(hoveredNodeData.id) || hoveredNodeData.pos}
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

      {/* Labels for selected/hub nodes based on label mode */}
      {labelNodes.map((n) => {
        const labelStyle = getLabelStyle(n);
        return (
        <Html
          key={`label-${n.id}`}
          position={layoutPositions?.get?.(n.id) || n.pos}
          center
          zIndexRange={[100, 0]}
          className="pointer-events-none"
          style={{ transform: 'translateY(-30px)' }}
        >
          <div
            className="px-3 py-1.5 rounded-sm text-[11px] font-bold tracking-widest uppercase backdrop-blur-md whitespace-nowrap border"
            style={{
              color: labelStyle.color,
              borderColor: labelStyle.borderColor,
              background: labelStyle.background,
              boxShadow: labelStyle.boxShadow,
            }}
          >
            {labelStyle.marker}{n.name}
          </div>
        </Html>
      )})}
    </>
  );
}
