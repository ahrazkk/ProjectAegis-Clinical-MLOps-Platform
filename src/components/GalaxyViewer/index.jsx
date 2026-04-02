// GalaxyViewer/index.jsx — Main orchestrator for the GNN Galaxy Viewer V2
// Now with dynamic data loading from Neo4j AuraDB via API
import React, { useMemo, useEffect, useRef, useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { Stars } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';

import { GalaxyProvider, useGalaxy, useGalaxyDispatch } from './store';
import {
  buildNodeDict, findDrugByName, computeSubgraph,
  shortestPath as computeShortestPath, applyFilters,
  setGraphData, getAdj, buildEdgeMetaIndex,
} from './graphEngine';
import { loadGraphData, clearGraphCache, getPerformanceLimits, setPerformanceLimits } from './graphDataService';
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
import EmbeddingInsightsPanel from './overlays/EmbeddingInsightsPanel';

// Data enrichment
import { enrichDrug, enrichInteraction } from './dataEnrichment';

// ─── Loading screen ─────────────────────────────────────────────────────────
function LoadingScreen({ status, error, onRetry }) {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center rounded-xl border border-white/5"
      style={{ minHeight: '650px', background: '#03050a' }}>
      <div className="text-center space-y-4">
        {error ? (
          <>
            <div className="w-12 h-12 mx-auto rounded-full bg-red-500/20 flex items-center justify-center">
              <svg className="w-6 h-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-red-400 text-sm font-mono">{error}</p>
            <button
              onClick={onRetry}
              className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white/80 text-sm rounded-lg transition-colors"
            >
              Retry
            </button>
          </>
        ) : (
          <>
            <div className="relative w-16 h-16 mx-auto">
              <div className="absolute inset-0 rounded-full border-2 border-purple-500/30 animate-ping" />
              <div className="absolute inset-2 rounded-full border-2 border-cyan-400/50 animate-spin" style={{ animationDuration: '2s' }} />
              <div className="absolute inset-4 rounded-full bg-purple-500/20 animate-pulse" />
            </div>
            <p className="text-white/60 text-sm font-mono tracking-wider">{status}</p>
            <div className="flex items-center gap-1 justify-center">
              <div className="w-1.5 h-1.5 bg-cyan-400/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-1.5 h-1.5 bg-cyan-400/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-1.5 h-1.5 bg-cyan-400/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Data source badge ──────────────────────────────────────────────────────
function DataSourceBadge({ source, stats }) {
  const isLive = source === 'api';
  const coordinateSources = stats?.coordinateSources || null;
  const coordSummary = coordinateSources
    ? Object.entries(coordinateSources)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([kind, count]) => `${kind}:${count}`)
      .join(', ')
    : null;

  return (
    <div className="absolute bottom-2 right-2 z-10 flex items-center gap-1.5 px-2 py-1 rounded-md bg-black/60 backdrop-blur-sm border border-white/10">
      <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`} />
      <span className="text-[10px] font-mono text-white/50">
        {isLive ? 'LIVE' : 'STATIC'} | {stats?.fetchedNodes || 0} nodes | {stats?.fetchedEdges || 0} edges{coordSummary ? ` | coords ${coordSummary}` : ''}
      </span>
    </div>
  );
}

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
        visible={shortestPath?.length >= 2 && (viewMode === 'path' || viewMode === 'galaxy' || viewMode === 'focus')}
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
function GalaxyViewerInner({ drugs, result, polypharmacyResult, isMobile }) {
  const { drugA, drugB, maxHops, viewMode, showEdges, filters } = useGalaxy();
  const dispatch = useGalaxyDispatch();
  const containerRef = useRef(null);
  const [filtersOpen, setFiltersOpen] = useState(false);

  // ─── Dynamic data loading state ─────────────────────────────────────
  const [loading, setLoading] = useState(true);
  const [loadingStatus, setLoadingStatus] = useState('Connecting to Neo4j AuraDB...');
  const [loadError, setLoadError] = useState(null);
  const [dataSource, setDataSource] = useState(null); // 'api' | 'static'
  const [dataStats, setDataStats] = useState(null);
  const [dataVersion, setDataVersion] = useState(0); // Increments on data reload

  // Load graph data on mount
  const loadData = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    setLoadingStatus('Connecting to Neo4j AuraDB...');

    try {
      setLoadingStatus('Fetching graph topology...');
      const data = await loadGraphData();

      // Set the graph data in the engine module
      setGraphData(data);
      buildEdgeMetaIndex();

      setDataSource(data.source);
      setDataStats(data.stats);
      setLoadingStatus('Building visualization...');

      // Small delay to let React render the loading state update
      await new Promise(r => setTimeout(r, 100));

      setDataVersion(v => v + 1);
      setLoading(false);

      if (data.apiError) {
        console.warn('[GalaxyViewer] Using fallback data:', data.apiError);
      }
    } catch (err) {
      console.error('[GalaxyViewer] Failed to load graph data:', err);
      setLoadError(`Failed to load graph data: ${err.message}`);
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // Build node dictionary (recomputed when data changes)
  const nodeDict = useMemo(() => buildNodeDict(0.35), [dataVersion]);

  const selectedDrugIds = useMemo(() => {
    if (loading || !Array.isArray(drugs) || Object.keys(nodeDict).length === 0) return [];

    const ids = [];
    const seen = new Set();
    drugs.forEach((drug) => {
      const name = typeof drug === 'string' ? drug : drug?.name;
      if (!name) return;
      const id = findDrugByName(nodeDict, name);
      if (id && !seen.has(id)) {
        ids.push(id);
        seen.add(id);
      }
    });

    return ids;
  }, [drugs, loading, nodeDict]);

  const interactionPairs = useMemo(() => {
    if (loading || !polypharmacyResult || !Array.isArray(polypharmacyResult.interactions)) return [];

    const pairs = [];
    const seen = new Set();

    polypharmacyResult.interactions.forEach((interaction) => {
      const srcName = interaction?.source || interaction?.drug_a;
      const tgtName = interaction?.target || interaction?.drug_b;
      if (!srcName || !tgtName) return;

      const srcId = findDrugByName(nodeDict, srcName);
      const tgtId = findDrugByName(nodeDict, tgtName);
      if (!srcId || !tgtId || srcId === tgtId) return;

      const key = srcId < tgtId ? `${srcId}-${tgtId}` : `${tgtId}-${srcId}`;
      if (seen.has(key)) return;

      seen.add(key);
      pairs.push([srcId, tgtId]);
    });

    return pairs;
  }, [loading, polypharmacyResult, nodeDict]);

  // Sync external drugs prop → store
  useEffect(() => {
    if (loading) return;
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
  }, [drugs?.[0]?.name, drugs?.[1]?.name, loading, dataVersion]);

  // Compute subgraph whenever selection or hops change
  const { nodes, edges, hasDrugs, drugAId, drugBId, subgraphStats } = useMemo(() => {
    if (loading || Object.keys(nodeDict).length === 0) {
      return {
        nodes: [],
        edges: [],
        hasDrugs: false,
        drugAId: null,
        drugBId: null,
        subgraphStats: null,
      };
    }
    const aId = drugA?.id || null;
    const bId = drugB?.id || null;
    const result = computeSubgraph(nodeDict, aId, bId, maxHops, selectedDrugIds, interactionPairs, viewMode, filters);
    return {
      nodes: result.nodes,
      edges: result.edges,
      hasDrugs: selectedDrugIds.length > 0,
      drugAId: aId,
      drugBId: bId,
      subgraphStats: result.stats,
    };
  }, [drugA?.id, drugB?.id, maxHops, nodeDict, loading, dataVersion, selectedDrugIds, interactionPairs, viewMode, filters]);

  // Apply filters to nodes and edges
  const { nodeVisibility, filteredEdges } = useMemo(() => {
    if (nodes.length === 0) return { nodeVisibility: new Map(), filteredEdges: [] };
    return applyFilters(nodes, edges, filters);
  }, [nodes, edges, filters]);

  // Compute shortest path
  const computedPath = useMemo(() => {
    if (loading) return [];
    const aId = drugA?.id || null;
    const bId = drugB?.id || null;
    const adj = getAdj();
    return computeShortestPath(adj, aId, bId);
  }, [drugA?.id, drugB?.id, loading, dataVersion]);

  // Update store with path and stats
  useEffect(() => {
    if (loading) return;
    dispatch({ type: 'SET_SHORTEST_PATH', payload: computedPath });

    const adj = getAdj();
    const visibleNodes = nodes.filter(n => nodeVisibility.get(n.id) !== false).length;

    const totalNodes = Object.keys(nodeDict).length;
    const totalEdges = Object.values(adj).reduce((sum, arr) => sum + arr.length, 0) / 2;

    dispatch({
      type: 'SET_STATS',
      payload: {
        totalNodes,
        totalEdges,
        visibleNodes: hasDrugs ? visibleNodes : totalNodes,
        visibleEdges: filteredEdges.length,
        pathLength: computedPath.length > 0 ? computedPath.length - 1 : -1,
        sharedNeighbors: subgraphStats?.sharedNeighbors || 0,
        selectedCount: subgraphStats?.selectedCount || 0,
        interactionPairCount: subgraphStats?.interactionPairCount || 0,
        focusConnectorEdges: subgraphStats?.focusConnectorEdges || 0,
        focusPathCount: subgraphStats?.focusPathCount || 0,
        focusMode: Boolean(subgraphStats?.focusMode),
      },
    });
  }, [
    drugA?.id,
    drugB?.id,
    maxHops,
    nodes,
    filteredEdges,
    nodeVisibility,
    computedPath,
    loading,
    dataVersion,
    hasDrugs,
    subgraphStats,
  ]);

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

      const viewModes = { '1': 'galaxy', '2': 'embedding', '3': 'radial', '4': 'cluster', '5': 'path', '6': 'focus' };
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
      } else if (e.key === 'r' || e.key === 'R') {
        // Refresh data from API
        clearGraphCache();
        loadData();
      }
    };

    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [loadData]);

  // ─── Layout computation ─────────────────────────────────────────────
  const clusterData = useMemo(() => {
    if (viewMode !== 'cluster' || nodes.length === 0) return null;
    return computeClusterPositions(nodes);
  }, [viewMode, nodes]);

  const layoutPositions = useMemo(() => {
    if (nodes.length === 0) return null;
    const adj = getAdj();
    const aId = drugA?.id || null;
    const bId = drugB?.id || null;

    switch (viewMode) {
      case 'embedding':
        return null;
      case 'radial':
        return computeRadialPositions(nodes, aId, bId, adj, maxHops);
      case 'cluster':
        return clusterData?.positions || null;
      case 'path':
        return computePathPositions(nodes, computedPath, adj);
      case 'focus':
        return null;
      case 'galaxy':
      default:
        return null; // null = use default positions (from API or T-SNE)
    }
  }, [viewMode, nodes, drugA?.id, drugB?.id, maxHops, computedPath, clusterData]);

  // ─── Loading state ──────────────────────────────────────────────────
  if (loading) {
    return <LoadingScreen status={loadingStatus} />;
  }

  if (loadError && Object.keys(nodeDict).length === 0) {
    return <LoadingScreen error={loadError} onRetry={loadData} />;
  }

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
      <EmbeddingInsightsPanel nodes={nodes} />
      <HopSlider />
      <DataSourceBadge source={dataSource} stats={dataStats} />
    </div>
  );
}

// ─── Export: wraps everything in provider ──────────────────────────────
export default function GNNGalaxyViewer({ drugs, result, polypharmacyResult, isMobile }) {
  return (
    <GalaxyProvider>
      <GalaxyViewerInner
        drugs={drugs}
        result={result}
        polypharmacyResult={polypharmacyResult}
        isMobile={isMobile}
      />
    </GalaxyProvider>
  );
}
