// GalaxyViewer/index.jsx — Main orchestrator for the GNN Galaxy Viewer
import React, { useMemo, useEffect, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { Stars } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';

import { GalaxyProvider, useGalaxy, useGalaxyDispatch } from './store';
import {
  buildNodeDict, findDrugByName, computeSubgraph,
  shortestPath as computeShortestPath, rawGnnData,
} from './graphEngine';
import InstancedNodes from './InstancedNodes';
import InstancedEdges from './InstancedEdges';
import HopShells from './HopShells';
import CameraController from './CameraController';

// Overlays
import HUD from './overlays/HUD';
import SearchBar from './overlays/SearchBar';
import Legend from './overlays/Legend';
import NodeDetailPanel from './overlays/NodeDetailPanel';
import DrugComparisonPanel from './overlays/DrugComparisonPanel';
import HopSlider from './overlays/HopSlider';

// ─── Inner scene (needs Canvas context) ─────────────────────────────────
function GalaxyScene({ nodeDict, nodes, edges, hasDrugs, isMobile, drugAId, drugBId, maxHops }) {
  return (
    <>
      <color attach="background" args={['#03050a']} />
      <fog attach="fog" args={['#03050a', 35, 100]} />

      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <pointLight position={[20, 20, 20]} intensity={0.8} color="#ffffff" />
      <pointLight position={[-20, -15, -20]} intensity={0.3} color="#8B5CF6" />
      <pointLight position={[15, -20, 15]} intensity={0.3} color="#EC4899" />

      {/* Starfield */}
      <Stars
        radius={120}
        depth={50}
        count={isMobile ? 3000 : 6000}
        factor={4}
        saturation={0.3}
        fade
        speed={0.8}
      />

      {/* Nodes */}
      <InstancedNodes nodes={nodes} hasDrugs={hasDrugs} />

      {/* Edges */}
      <InstancedEdges edges={edges} />

      {/* Hop boundary shells */}
      <HopShells nodes={nodes} drugAId={drugAId} drugBId={drugBId} maxHops={maxHops} />

      {/* Camera */}
      <CameraController isMobile={isMobile} />

      {/* Post-processing */}
      <EffectComposer>
        <Bloom
          luminanceThreshold={0.15}
          luminanceSmoothing={0.9}
          intensity={1.2}
          mipmapBlur
        />
        <Vignette
          offset={0.3}
          darkness={0.7}
        />
      </EffectComposer>
    </>
  );
}

// ─── Graph data manager (connects store to graph engine) ────────────────
function GraphDataManager({ drugs, nodeDict, onComputed }) {
  const { maxHops } = useGalaxy();
  const dispatch = useGalaxyDispatch();

  useEffect(() => {
    // Match external drug props to graph nodes
    const nameA = drugs?.[0]?.name || null;
    const nameB = drugs?.[1]?.name || null;
    const drugAId = findDrugByName(nodeDict, nameA);
    const drugBId = findDrugByName(nodeDict, nameB);

    if (nameA && drugAId && nodeDict[drugAId]) {
      dispatch({ type: 'SELECT_DRUG_A', payload: nodeDict[drugAId] });
    }
    if (nameB && drugBId && nodeDict[drugBId]) {
      dispatch({ type: 'SELECT_DRUG_B', payload: nodeDict[drugBId] });
    }
  }, [drugs?.[0]?.name, drugs?.[1]?.name]);

  return null;
}

// ─── Main viewer with state-driven graph computation ────────────────────
function GalaxyViewerInner({ drugs, isMobile }) {
  const { drugA, drugB, maxHops } = useGalaxy();
  const dispatch = useGalaxyDispatch();

  // Build node dictionary (stable, only computed once)
  const nodeDict = useMemo(() => buildNodeDict(0.25), []);

  // Sync external drugs prop → store
  useEffect(() => {
    const nameA = drugs?.[0]?.name || null;
    const nameB = drugs?.[1]?.name || null;
    const aId = findDrugByName(nodeDict, nameA);
    const bId = findDrugByName(nodeDict, nameB);

    if (nameA && aId && nodeDict[aId]) {
      dispatch({ type: 'SELECT_DRUG_A', payload: nodeDict[aId] });
    } else if (!nameA) {
      dispatch({ type: 'SELECT_DRUG_A', payload: null });
    }

    if (nameB && bId && nodeDict[bId]) {
      dispatch({ type: 'SELECT_DRUG_B', payload: nodeDict[bId] });
    } else if (!nameB) {
      dispatch({ type: 'SELECT_DRUG_B', payload: null });
    }
  }, [drugs?.[0]?.name, drugs?.[1]?.name]);

  // Compute subgraph whenever selection or hops change
  const { nodes, edges, hasDrugs, drugAId, drugBId } = useMemo(() => {
    const aId = drugA?.id || null;
    const bId = drugB?.id || null;
    const result = computeSubgraph(nodeDict, aId, bId, maxHops);
    return {
      nodes: result.nodes,
      edges: result.edges,
      hasDrugs: aId !== null || bId !== null,
      drugAId: aId,
      drugBId: bId,
      ...result,
    };
  }, [drugA?.id, drugB?.id, maxHops, nodeDict]);

  // Update stats and shortest path in store
  useEffect(() => {
    const aId = drugA?.id || null;
    const bId = drugB?.id || null;
    const adj = rawGnnData.adj || {};

    const path = computeShortestPath(adj, aId, bId);
    dispatch({ type: 'SET_SHORTEST_PATH', payload: path });

    // Compute stats
    const visibleNodes = nodes.filter(n =>
      n.hopA <= maxHops || n.hopB <= maxHops || (!drugA && !drugB)
    ).length;

    dispatch({
      type: 'SET_STATS',
      payload: {
        totalNodes: rawGnnData.nodes.length,
        totalEdges: Object.values(adj).reduce((sum, arr) => sum + arr.length, 0) / 2,
        visibleNodes: hasDrugs ? visibleNodes : rawGnnData.nodes.length,
        visibleEdges: edges.length,
        pathLength: path.length > 0 ? path.length - 1 : -1,
        sharedNeighbors: nodes.filter(n => n.hopA <= maxHops && n.hopB <= maxHops && !n.isA && !n.isB).length,
      },
    });
  }, [drugA?.id, drugB?.id, maxHops, nodes, edges]);

  return (
    <div
      className="w-full h-full relative overflow-hidden rounded-xl border border-white/5"
      style={{ minHeight: isMobile ? '400px' : '650px', background: '#03050a' }}
    >
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 8, 30], fov: 60 }}
        dpr={[1, Math.min(window.devicePixelRatio, 2)]}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      >
        <GalaxyScene
          nodeDict={nodeDict}
          nodes={nodes}
          edges={edges}
          hasDrugs={hasDrugs}
          isMobile={isMobile}
          drugAId={drugAId}
          drugBId={drugBId}
          maxHops={maxHops}
        />
      </Canvas>

      {/* Overlay UI */}
      <SearchBar nodeDict={nodeDict} />
      <HUD />
      <Legend isMobile={isMobile} />
      <NodeDetailPanel />
      <DrugComparisonPanel />
      <HopSlider />
    </div>
  );
}

// ─── Export: wraps everything in provider ──────────────────────────────
export default function GNNGalaxyViewer({ drugs, isMobile }) {
  return (
    <GalaxyProvider>
      <GalaxyViewerInner drugs={drugs} isMobile={isMobile} />
    </GalaxyProvider>
  );
}
