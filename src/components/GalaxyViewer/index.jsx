// GalaxyViewer/index.jsx — Main orchestrator for the GNN Galaxy Viewer V2
import React, { useMemo, useEffect, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { Stars } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';

import { GalaxyProvider, useGalaxy, useGalaxyDispatch } from './store';
import {
  buildNodeDict, findDrugByName, computeSubgraph,
  shortestPath as computeShortestPath, rawGnnData, applyFilters,
} from './graphEngine';
import InstancedNodes from './InstancedNodes';
import InstancedEdges from './InstancedEdges';
import HopShells from './HopShells';
import CameraController from './CameraController';
import ClusterBubbles from './ClusterBubbles';
import PathParticles from './PathParticles';

// Layouts
import { computeRadialPositions } from './layouts/RadialLayout';
import { computeClusterPositions } from './layouts/ClusterLayout';
import { computePathPositions } from './layouts/PathLayout';

// Overlays
import HUD from './overlays/HUD';
import SearchBar from './overlays/SearchBar';
import Legend from './overlays/Legend';
import NodeDetailPanel from './overlays/NodeDetailPanel';
import DrugComparisonPanel from './overlays/DrugComparisonPanel';
import HopSlider from './overlays/HopSlider';
import Toolbar from './overlays/Toolbar';
import EdgeDetailPanel from './overlays/EdgeDetailPanel';
import FilterPanel from './overlays/FilterPanel';

// Data enrichment
import { enrichDrug, enrichInteraction } from './dataEnrichment';

// ─── Inner scene (needs Canvas context) ─────────────────────────────────
function GalaxyScene({
  nodeDict, nodes, edges, hasDrugs, isMobile,
  drugAId, drugBId, maxHops,
  layoutPositions, clusterData, showEdges, shortestPath,
  nodeVisibility,
}) {
  const { viewMode } = useGalaxy();

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

      {/* Nodes — with layout position transitions */}
      <InstancedNodes
        nodes={nodes}
        hasDrugs={hasDrugs}
        layoutPositions={layoutPositions}
        nodeVisibility={nodeVisibility}
      />

      {/* Edges */}
      {showEdges && <InstancedEdges edges={edges} />}

      {/* Hop boundary shells (only in galaxy/radial modes) */}
      {(viewMode === 'galaxy' || viewMode === 'radial') && (
        <HopShells nodes={nodes} drugAId={drugAId} drugBId={drugBId} maxHops={maxHops} />
      )}

      {/* Cluster bubbles (only in cluster mode) */}
      <ClusterBubbles
        visible={viewMode === 'cluster'}
        clusterCenters={clusterData?.clusterCenters}
        groups={clusterData?.groups}
      />

      {/* Path particles (when path exists and path/galaxy mode) */}
      <PathParticles
        visible={shortestPath?.length >= 2 && (viewMode === 'path' || viewMode === 'galaxy')}
        path={shortestPath}
        nodeDict={nodeDict}
      />

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

// ─── Main viewer with state-driven graph computation ────────────────────
function GalaxyViewerInner({ drugs, result, isMobile }) {
  const { drugA, drugB, maxHops, viewMode, showEdges, shortestPath, showFilters, filters } = useGalaxy();
  const dispatch = useGalaxyDispatch();
  const containerRef = useRef(null);
  const [filtersOpen, setFiltersOpen] = useState(false);

  // Build node dictionary (stable, only computed once)
  const nodeDict = useMemo(() => buildNodeDict(0.35), []);

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
    };
  }, [drugA?.id, drugB?.id, maxHops, nodeDict]);

  // Apply filters to nodes and edges
  const { nodeVisibility, filteredEdges } = useMemo(() => {
    return applyFilters(nodes, edges, filters);
  }, [nodes, edges, filters]);

  // Compute shortest path
  const computedPath = useMemo(() => {
    const aId = drugA?.id || null;
    const bId = drugB?.id || null;
    const adj = rawGnnData.adj || {};
    return computeShortestPath(adj, aId, bId);
  }, [drugA?.id, drugB?.id]);

  // Update store with path and stats
  useEffect(() => {
    dispatch({ type: 'SET_SHORTEST_PATH', payload: computedPath });

    const adj = rawGnnData.adj || {};
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
        pathLength: computedPath.length > 0 ? computedPath.length - 1 : -1,
        sharedNeighbors: nodes.filter(n => n.hopA <= maxHops && n.hopB <= maxHops && !n.isA && !n.isB).length,
      },
    });
  }, [drugA?.id, drugB?.id, maxHops, nodes, edges, computedPath]);

  // ─── Data enrichment — fetch from API on drug selection ─────────────
  useEffect(() => {
    if (drugA?.name) {
      dispatch({ type: 'SET_ENRICHMENT_LOADING', payload: { drugA: true } });
      enrichDrug(drugA.name).then(data => {
        dispatch({ type: 'SET_ENRICHED_DRUG_A', payload: data });
      });
    } else {
      dispatch({ type: 'SET_ENRICHED_DRUG_A', payload: null });
    }
  }, [drugA?.name]);

  useEffect(() => {
    if (drugB?.name) {
      dispatch({ type: 'SET_ENRICHMENT_LOADING', payload: { drugB: true } });
      enrichDrug(drugB.name).then(data => {
        dispatch({ type: 'SET_ENRICHED_DRUG_B', payload: data });
      });
    } else {
      dispatch({ type: 'SET_ENRICHED_DRUG_B', payload: null });
    }
  }, [drugB?.name]);

  useEffect(() => {
    if (drugA?.name && drugB?.name) {
      dispatch({ type: 'SET_ENRICHMENT_LOADING', payload: { interaction: true } });
      enrichInteraction(drugA.name, drugB.name).then(data => {
        dispatch({ type: 'SET_ENRICHED_INTERACTION', payload: data });
      });
    } else {
      dispatch({ type: 'SET_ENRICHED_INTERACTION', payload: null });
    }
  }, [drugA?.name, drugB?.name]);

  // Sync prediction result from Dashboard
  useEffect(() => {
    if (result) {
      dispatch({ type: 'SET_PREDICTION_RESULT', payload: result });
    }
  }, [result]);

  // ─── Keyboard shortcuts ────────────────────────────────────────────
  useEffect(() => {
    const handleKey = (e) => {
      // Don't trigger shortcuts when typing in an input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      const viewModes = { '1': 'galaxy', '2': 'radial', '3': 'cluster', '4': 'path' };
      if (viewModes[e.key]) {
        dispatch({ type: 'SET_VIEW_MODE', payload: viewModes[e.key] });
      } else if (e.key === 'Escape') {
        dispatch({ type: 'CLEAR_SELECTION' });
        setFiltersOpen(false);
      } else if (e.key === 'e' || e.key === 'E') {
        dispatch({ type: 'TOGGLE_EDGES' });
      } else if (e.key === 'l' || e.key === 'L') {
        dispatch({ type: 'TOGGLE_LABELS' });
      } else if (e.key === 'f' || e.key === 'F') {
        setFiltersOpen(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  // ─── Layout computation ─────────────────────────────────────────────
  const clusterData = useMemo(() => {
    if (viewMode !== 'cluster') return null;
    return computeClusterPositions(nodes);
  }, [viewMode, nodes]);

  const layoutPositions = useMemo(() => {
    const adj = rawGnnData.adj || {};
    const aId = drugA?.id || null;
    const bId = drugB?.id || null;

    switch (viewMode) {
      case 'radial':
        return computeRadialPositions(nodes, aId, bId, adj, maxHops);
      case 'cluster':
        return clusterData?.positions || null;
      case 'path':
        return computePathPositions(nodes, computedPath, adj);
      case 'galaxy':
      default:
        return null; // null = use default T-SNE positions (node.pos)
    }
  }, [viewMode, nodes, drugA?.id, drugB?.id, maxHops, computedPath, clusterData]);

  return (
    <div
      ref={containerRef}
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
          edges={filteredEdges}
          hasDrugs={hasDrugs}
          isMobile={isMobile}
          drugAId={drugAId}
          drugBId={drugBId}
          maxHops={maxHops}
          layoutPositions={layoutPositions}
          clusterData={clusterData}
          showEdges={showEdges}
          shortestPath={computedPath}
          nodeVisibility={nodeVisibility}
        />
      </Canvas>

      {/* Overlay UI */}
      <Toolbar
        onToggleFilters={() => setFiltersOpen(!filtersOpen)}
        showFilters={filtersOpen}
        canvasRef={containerRef}
      />
      <SearchBar nodeDict={nodeDict} />
      <FilterPanel isOpen={filtersOpen} onClose={() => setFiltersOpen(false)} />
      <HUD />
      <Legend isMobile={isMobile} />
      <NodeDetailPanel />
      <EdgeDetailPanel />
      <DrugComparisonPanel />
      <HopSlider />
    </div>
  );
}

// ─── Export: wraps everything in provider ──────────────────────────────
export default function GNNGalaxyViewer({ drugs, result, isMobile }) {
  return (
    <GalaxyProvider>
      <GalaxyViewerInner drugs={drugs} result={result} isMobile={isMobile} />
    </GalaxyProvider>
  );
}
