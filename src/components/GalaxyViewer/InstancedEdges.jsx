// InstancedEdges.jsx — GPU-efficient edge rendering with data flow particles
import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useGalaxyDispatch } from './store';
import { getAdj, getGraphData } from './graphEngine';

// Animated edge material with data flow particles
const edgeVertexShader = `
  attribute vec3 instanceColorAttr;
  attribute float instanceOpacityAttr;
  attribute float instanceFlowSpeed;

  varying vec3 vColor;
  varying float vOpacity;
  varying float vFlow;
  varying vec2 vUv;

  void main() {
    vColor = instanceColorAttr;
    vOpacity = instanceOpacityAttr;
    vFlow = instanceFlowSpeed;
    vUv = uv;
    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const edgeFragmentShader = `
  varying vec3 vColor;
  varying float vOpacity;
  varying float vFlow;
  varying vec2 vUv;
  uniform float uTime;

  void main() {
    // Data flow particles along the edge
    float flow = fract(vUv.y * 4.0 - uTime * vFlow * 0.5);
    float particle = smoothstep(0.0, 0.15, flow) * smoothstep(0.4, 0.15, flow);
    float base = 0.3;
    float brightness = base + particle * 0.7 * vFlow;

    vec3 finalColor = vColor * brightness;
    float alpha = vOpacity * (base + particle * 0.5);

    gl_FragColor = vec4(finalColor, alpha);
  }
`;

export default function InstancedEdges({ edges }) {
  const activeRef = useRef();
  const bgRef = useRef();
  const dispatch = useGalaxyDispatch();

  // Separate background (faint) edges from active edges
  const { bgEdges, activeEdges } = useMemo(() => {
    const bg = [];
    const active = [];
    edges.forEach(e => {
      if (e.role === 'background') bg.push(e);
      else active.push(e);
    });
    return { bgEdges: bg, activeEdges: active };
  }, [edges]);

  // ─── Background edges: simple LineSegments ───────────────────────────
  const bgGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(bgEdges.length * 6);
    const colors = new Float32Array(bgEdges.length * 6);
    const tempC = new THREE.Color();

    bgEdges.forEach((e, i) => {
      positions[i * 6] = e.start[0];
      positions[i * 6 + 1] = e.start[1];
      positions[i * 6 + 2] = e.start[2];
      positions[i * 6 + 3] = e.end[0];
      positions[i * 6 + 4] = e.end[1];
      positions[i * 6 + 5] = e.end[2];

      tempC.set(e.color);
      colors[i * 6] = tempC.r;
      colors[i * 6 + 1] = tempC.g;
      colors[i * 6 + 2] = tempC.b;
      colors[i * 6 + 3] = tempC.r;
      colors[i * 6 + 4] = tempC.g;
      colors[i * 6 + 5] = tempC.b;
    });

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geo;
  }, [bgEdges]);

  // ─── Active edges: InstancedMesh cylinders with flow shader ──────────
  const activeCount = activeEdges.length;

  const { cylinderGeo, activeMatl, colorArr, opacityArr, flowArr } = useMemo(() => {
    const cGeo = new THREE.CylinderGeometry(0.015, 0.015, 1, 4);
    cGeo.translate(0, 0.5, 0); // pivot at bottom
    cGeo.rotateX(Math.PI / 2); // align to Z axis

    const mat = new THREE.ShaderMaterial({
      vertexShader: edgeVertexShader,
      fragmentShader: edgeFragmentShader,
      uniforms: { uTime: { value: 0 } },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    return {
      cylinderGeo: cGeo,
      activeMatl: mat,
      colorArr: new Float32Array(Math.max(activeCount, 1) * 3),
      opacityArr: new Float32Array(Math.max(activeCount, 1)),
      flowArr: new Float32Array(Math.max(activeCount, 1)),
    };
  }, [activeCount]);

  // Position and orient each cylinder between start/end points
  useEffect(() => {
    if (!activeRef.current || activeCount === 0) return;
    const mesh = activeRef.current;
    const tempStart = new THREE.Vector3();
    const tempEnd = new THREE.Vector3();
    const tempDir = new THREE.Vector3();
    const tempQuat = new THREE.Quaternion();
    const tempScaleV = new THREE.Vector3();
    const up = new THREE.Vector3(0, 0, 1);

    activeEdges.forEach((e, i) => {
      tempStart.set(e.start[0], e.start[1], e.start[2]);
      tempEnd.set(e.end[0], e.end[1], e.end[2]);
      tempDir.subVectors(tempEnd, tempStart);
      const len = tempDir.length();
      tempDir.normalize();

      // Orient cylinder from start to end
      tempQuat.setFromUnitVectors(up, tempDir);
      tempScaleV.set(
        e.lineWidth || 1,
        e.lineWidth || 1,
        len
      );
      tempMatrix.compose(tempStart, tempQuat, tempScaleV);
      mesh.setMatrixAt(i, tempMatrix);

      // Set per-instance attributes
      const tempC = new THREE.Color(e.color);
      colorArr[i * 3] = tempC.r;
      colorArr[i * 3 + 1] = tempC.g;
      colorArr[i * 3 + 2] = tempC.b;
      opacityArr[i] = e.opacity;
      flowArr[i] = e.role === 'path' || e.role === 'direct' ? 2.0 :
                   e.role === 'bridge' ? 1.0 : 0.5;
    });

    mesh.instanceMatrix.needsUpdate = true;

    // Set attributes
    const geo = mesh.geometry;
    geo.setAttribute('instanceColorAttr',
      new THREE.InstancedBufferAttribute(colorArr, 3));
    geo.setAttribute('instanceOpacityAttr',
      new THREE.InstancedBufferAttribute(opacityArr, 1));
    geo.setAttribute('instanceFlowSpeed',
      new THREE.InstancedBufferAttribute(flowArr, 1));
  }, [activeEdges, activeCount]);

  // Animate
  useFrame(({ clock }) => {
    if (activeMatl) activeMatl.uniforms.uTime.value = clock.getElapsedTime();
  });

  const tempMatrix = useMemo(() => new THREE.Matrix4(), []);

  // Edge click handler — maps instanceId back to edge data
  const handleEdgeClick = (e) => {
    e.stopPropagation();
    if (e.instanceId !== undefined && activeEdges[e.instanceId]) {
      const edge = activeEdges[e.instanceId];
      // Look up drug names from the node data
      const graphData = getGraphData();
      const startNode = graphData.nodes.find(n => n.id === edge.startId);
      const endNode = graphData.nodes.find(n => n.id === edge.endId);
      dispatch({
        type: 'SET_SELECTED_EDGE',
        payload: {
          ...edge,
          startName: startNode?.name || edge.startId,
          endName: endNode?.name || edge.endId,
        },
      });
    }
  };

  return (
    <>
      {/* Background edges — simple lines */}
      {bgEdges.length > 0 && (
        <lineSegments geometry={bgGeometry} frustumCulled={false}>
          <lineBasicMaterial
            vertexColors
            transparent
            opacity={0.06}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </lineSegments>
      )}

      {/* Active edges — instanced cylinders with flow animation */}
      {activeCount > 0 && (
        <instancedMesh
          ref={activeRef}
          args={[cylinderGeo, activeMatl, activeCount]}
          frustumCulled={false}
          onClick={handleEdgeClick}
        />
      )}
    </>
  );
}
