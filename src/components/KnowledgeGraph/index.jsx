// KnowledgeGraph/index.jsx - Biological mechanism explorer with explainability overlays

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Loader2,
  WifiOff,
  RefreshCw,
  Maximize2,
  AlertTriangle,
  Info,
  Filter,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  FlaskConical,
  Target,
  HeartPulse,
  X,
} from 'lucide-react';

import { fetchDrugBiology, fetchMechanismMap, clearBiologyCache } from './biologyService';
import { buildMechanismGraph, computeRadialLayout } from './mechanismGraphEngine';

import DrugNode from './nodes/DrugNode';
import EnzymeNode from './nodes/EnzymeNode';
import TargetNode from './nodes/TargetNode';
import SideEffectNode from './nodes/SideEffectNode';
import BiologyEdge from './edges/BiologyEdge';
import ConflictPanel from './overlays/ConflictPanel';
import NodeTooltip from './overlays/NodeTooltip';
import MechanismLegend from './overlays/MechanismLegend';

const NODE_RENDERERS = {
  drug: DrugNode,
  enzyme: EnzymeNode,
  target: TargetNode,
  side_effect: SideEffectNode,
};

const SOURCE_LABELS = {
  'biology-api': 'Biology API',
  'mechanism-api': 'Mechanism API',
  'drug-info-api': 'Drug Info API',
  'interaction-info-api': 'Interaction Info API',
  offline: 'Offline CYP fallback',
  none: 'No source data',
};

const FILTER_NODE_TYPES = ['enzyme', 'target', 'side_effect'];

function getSeverityColor(severity) {
  const normalized = String(severity || '').toLowerCase();
  if (normalized === 'major' || normalized === 'severe' || normalized === 'critical' || normalized === 'high') {
    return '#ef4444';
  }
  if (normalized === 'moderate') return '#eab308';
  if (normalized === 'minor' || normalized === 'low') return '#22c55e';
  return '#6b7280';
}

function formatRelativeTime(timestamp) {
  if (!timestamp) return 'not loaded';
  const diffSec = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (diffSec < 5) return 'just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  return `${diffHr}h ago`;
}

function normalizeStorageKey(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// Same axial coordinate system as node layout. Nodes sit inside these cells.
function HexGridBackground({ width, height, occupiedCells, isLightTheme = false }) {
  const minDim = Math.min(width, height);
  const hexSize = Math.max(50, minDim * 0.09);
  const cx = width / 2;
  const cy = height / 2;
  const gridRadius = 8;
  const palette = isLightTheme
    ? {
      occupiedFill: 'rgba(15,23,42,0.06)',
      idleFill: 'rgba(15,23,42,0.018)',
      occupiedStroke: 'rgba(15,23,42,0.28)',
      idleStroke: 'rgba(15,23,42,0.11)',
    }
    : {
      occupiedFill: 'rgba(255,255,255,0.045)',
      idleFill: 'rgba(255,255,255,0.013)',
      occupiedStroke: 'rgba(148,163,184,0.22)',
      idleStroke: 'rgba(148,163,184,0.085)',
    };

  const hexPoints = (px, py, s) => {
    return Array.from({ length: 6 }, (_, i) => {
      const angle = (Math.PI / 3) * i - Math.PI / 6;
      return `${px + s * Math.cos(angle)},${py + s * Math.sin(angle)}`;
    }).join(' ');
  };

  const hexes = [];
  for (let q = -gridRadius; q <= gridRadius; q++) {
    const r1 = Math.max(-gridRadius, -q - gridRadius);
    const r2 = Math.min(gridRadius, -q + gridRadius);
    for (let r = r1; r <= r2; r++) {
      const x = cx + hexSize * Math.sqrt(3) * (q + r / 2);
      const y = cy + hexSize * (3 / 2) * r;
      if (x < -hexSize * 2 || x > width + hexSize * 2 || y < -hexSize * 2 || y > height + hexSize * 2) continue;

      const isOccupied = occupiedCells?.has(`${q},${r}`);
      hexes.push(
        <polygon
          key={`${q},${r}`}
          points={hexPoints(x, y, hexSize - 1)}
          fill={isOccupied ? palette.occupiedFill : palette.idleFill}
          stroke={isOccupied ? palette.occupiedStroke : palette.idleStroke}
          strokeWidth={isOccupied ? 0.7 : 0.55}
        />
      );
    }
  }

  return <g>{hexes}</g>;
}

export default function KnowledgeGraphView({ drugs = [], result, polypharmacyResult, isMobile }) {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [isLightTheme, setIsLightTheme] = useState(() => {
    if (typeof document === 'undefined') return false;
    return document.documentElement.classList.contains('light-theme');
  });

  // Data state
  const [drug1Bio, setDrug1Bio] = useState(null);
  const [drug2Bio, setDrug2Bio] = useState(null);
  const [mechanismMapData, setMechanismMapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dataSource, setDataSource] = useState('');
  const [loadError, setLoadError] = useState('');
  const [lastLoadedAt, setLastLoadedAt] = useState(null);
  const [reloadNonce, setReloadNonce] = useState(0);

  // UI state
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const [activePairIndex, setActivePairIndex] = useState(0);
  const [showInsights, setShowInsights] = useState(!isMobile);
  const [showHelp, setShowHelp] = useState(!isMobile);
  const [filters, setFilters] = useState({
    enzyme: true,
    target: true,
    side_effect: true,
    onlyConflicts: false,
  });

  // Zoom/pan
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });
  const skipViewportPersistRef = useRef(false);

  // Measure container
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) setDimensions({ width, height });
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (isMobile) {
      setShowInsights(false);
      setShowHelp(false);
    }
  }, [isMobile]);

  useEffect(() => {
    if (typeof document === 'undefined') return;

    const root = document.documentElement;
    const syncTheme = () => {
      setIsLightTheme(root.classList.contains('light-theme'));
    };

    syncTheme();
    const observer = new MutationObserver(syncTheme);
    observer.observe(root, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  const pairCandidates = useMemo(() => {
    if (!polypharmacyResult || !Array.isArray(polypharmacyResult.interactions)) return [];

    const dedup = new Map();
    polypharmacyResult.interactions.forEach((interaction) => {
      const drugA = String(interaction?.source || interaction?.drug_a || '').trim();
      const drugB = String(interaction?.target || interaction?.drug_b || '').trim();
      if (!drugA || !drugB) return;

      const key = [drugA.toLowerCase(), drugB.toLowerCase()].sort().join('::');
      const riskScore = Number(interaction?.risk_score || 0);
      const existing = dedup.get(key);

      if (!existing || riskScore > existing.riskScore) {
        dedup.set(key, {
          drugA,
          drugB,
          riskScore,
          severity: String(interaction?.severity || interaction?.risk_level || 'unknown').toLowerCase(),
        });
      }
    });

    return Array.from(dedup.values()).sort((a, b) => b.riskScore - a.riskScore);
  }, [polypharmacyResult]);

  useEffect(() => {
    setActivePairIndex(0);
  }, [pairCandidates.length, drugs?.length]);

  useEffect(() => {
    if (activePairIndex >= pairCandidates.length && pairCandidates.length > 0) {
      setActivePairIndex(0);
    }
  }, [activePairIndex, pairCandidates.length]);

  const fallbackDrug1Name = drugs[0]?.name || drugs[0] || null;
  const fallbackDrug2Name = drugs[1]?.name || drugs[1] || null;
  const activePair = pairCandidates[activePairIndex] || null;

  // Active pair names (polypharmacy pair if available, else first two selected)
  const drug1Name = activePair?.drugA || fallbackDrug1Name;
  const drug2Name = activePair?.drugB || fallbackDrug2Name;

  const viewportStorageKey = useMemo(() => {
    if (!drug1Name) return null;
    const a = normalizeStorageKey(drug1Name);
    const b = normalizeStorageKey(drug2Name || 'single');
    return `kg:viewport:${a}::${b}`;
  }, [drug1Name, drug2Name]);

  useEffect(() => {
    if (!viewportStorageKey) return;
    skipViewportPersistRef.current = true;

    try {
      const raw = window.sessionStorage.getItem(viewportStorageKey);
      if (!raw) {
        setZoom(1);
        setPan({ x: 0, y: 0 });
        return;
      }

      const parsed = JSON.parse(raw);
      const nextZoom = Number(parsed?.zoom);
      const nextX = Number(parsed?.pan?.x);
      const nextY = Number(parsed?.pan?.y);

      setZoom(Number.isFinite(nextZoom) ? Math.max(0.3, Math.min(3, nextZoom)) : 1);
      setPan({
        x: Number.isFinite(nextX) ? nextX : 0,
        y: Number.isFinite(nextY) ? nextY : 0,
      });
    } catch {
      setZoom(1);
      setPan({ x: 0, y: 0 });
    }
  }, [viewportStorageKey]);

  useEffect(() => {
    if (!viewportStorageKey) return;
    if (skipViewportPersistRef.current) {
      skipViewportPersistRef.current = false;
      return;
    }

    try {
      window.sessionStorage.setItem(
        viewportStorageKey,
        JSON.stringify({ zoom, pan })
      );
    } catch {
      // Ignore storage failures (private mode, quota, etc.)
    }
  }, [viewportStorageKey, zoom, pan]);

  useEffect(() => {
    if (!drug1Name) {
      setDrug1Bio(null);
      setDrug2Bio(null);
      setMechanismMapData(null);
      setDataSource('');
      setLoadError('');
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setLoadError('');

      try {
        const bio1 = await fetchDrugBiology(drug1Name);
        if (cancelled) return;

        setDrug1Bio(bio1);
        let resolvedSource = bio1?.source || 'offline';

        if (drug2Name) {
          const [bio2, mmap] = await Promise.all([
            fetchDrugBiology(drug2Name),
            fetchMechanismMap(drug1Name, drug2Name),
          ]);
          if (cancelled) return;

          setDrug2Bio(bio2);
          setMechanismMapData(mmap);
          resolvedSource = mmap?.source || bio2?.source || resolvedSource;
        } else {
          setDrug2Bio(null);
          setMechanismMapData(null);
        }

        setDataSource(resolvedSource || 'offline');
        setLastLoadedAt(Date.now());
      } catch (error) {
        if (cancelled) return;

        setLoadError(error?.message || 'Failed to load mechanism data.');
        setDrug1Bio(null);
        setDrug2Bio(null);
        setMechanismMapData(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [drug1Name, drug2Name, reloadNonce]);

  // Incorporate pair result data into mechanism map for 2-drug mode.
  const enrichedMechanismMap = useMemo(() => {
    if (!mechanismMapData) return null;
    const mmap = { ...mechanismMapData };

    // For N-way analysis, prefer pair-specific mechanism data over summary result payload.
    const shouldMergePairwiseResult = pairCandidates.length === 0;

    if (result && shouldMergePairwiseResult) {
      if (!mmap.interaction?.mechanism && result.mechanism_hypothesis) {
        mmap.interaction = {
          ...mmap.interaction,
          mechanism: result.mechanism_hypothesis,
          severity: result.severity || mmap.interaction?.severity,
        };
      }
      if (result.affected_systems && (!mmap.affected_systems || mmap.affected_systems.length === 0)) {
        mmap.affected_systems = result.affected_systems;
      }
    }

    return mmap;
  }, [mechanismMapData, result, pairCandidates.length]);

  // Build graph
  const graphData = useMemo(() => {
    if (!drug1Bio) return { nodes: [], edges: [], conflicts: [] };
    return buildMechanismGraph(drug1Bio, drug2Bio, enrichedMechanismMap);
  }, [drug1Bio, drug2Bio, enrichedMechanismMap]);

  // Compute layout
  const layoutResult = useMemo(() => {
    if (graphData.nodes.length === 0) return { positions: new Map(), occupiedCells: new Set() };
    return computeRadialLayout(graphData.nodes, dimensions.width, dimensions.height);
  }, [graphData.nodes, dimensions]);

  const layout = layoutResult.positions;
  const occupiedCells = layoutResult.occupiedCells;
  const cellSize = layoutResult.hexSize || 50;

  const conflictNodeIds = useMemo(() => {
    return new Set(graphData.conflicts.map((c) => c.nodeId));
  }, [graphData.conflicts]);

  const nodeCounts = useMemo(() => {
    const counts = { enzyme: 0, target: 0, side_effect: 0 };
    graphData.nodes.forEach((node) => {
      if (node.type in counts) counts[node.type] += 1;
    });
    return counts;
  }, [graphData.nodes]);

  const visibleNodes = useMemo(() => {
    return graphData.nodes.filter((node) => {
      if (node.type === 'drug') return true;
      if (!FILTER_NODE_TYPES.includes(node.type)) return true;
      if (!filters[node.type]) return false;
      if (filters.onlyConflicts) return node.isConflict === true;
      return true;
    });
  }, [graphData.nodes, filters]);

  const visibleNodeIds = useMemo(() => {
    return new Set(visibleNodes.map((node) => node.id));
  }, [visibleNodes]);

  const visibleEdges = useMemo(() => {
    return graphData.edges.filter((edge) => {
      if (!visibleNodeIds.has(edge.source) || !visibleNodeIds.has(edge.target)) return false;
      if (filters.onlyConflicts) {
        return conflictNodeIds.has(edge.source) || conflictNodeIds.has(edge.target);
      }
      return true;
    });
  }, [graphData.edges, visibleNodeIds, filters.onlyConflicts, conflictNodeIds]);

  const nodeLookup = useMemo(() => {
    return new Map(graphData.nodes.map((node) => [node.id, node]));
  }, [graphData.nodes]);

  const selectedNodeConflicts = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.conflicts.filter((conflict) => conflict.nodeId === selectedNode.id);
  }, [selectedNode, graphData.conflicts]);

  const selectedNodeRelations = useMemo(() => {
    if (!selectedNode) return [];

    const related = graphData.edges
      .filter((edge) => edge.source === selectedNode.id || edge.target === selectedNode.id)
      .map((edge, index) => {
        const isOutbound = edge.source === selectedNode.id;
        const neighborId = isOutbound ? edge.target : edge.source;
        const neighbor = nodeLookup.get(neighborId);
        return {
          id: `${edge.source}-${edge.target}-${edge.type}-${index}`,
          direction: isOutbound ? 'outbound' : 'inbound',
          neighborLabel: neighbor?.label || neighborId,
          neighborType: neighbor?.type || 'unknown',
          type: edge.type,
          label: edge.label,
          confidence: Number(edge.confidence),
          evidenceSource: edge.evidence_source,
        };
      })
      .sort((a, b) => {
        const aConf = Number.isFinite(a.confidence) ? a.confidence : -1;
        const bConf = Number.isFinite(b.confidence) ? b.confidence : -1;
        return bConf - aConf;
      });

    return related;
  }, [selectedNode, graphData.edges, nodeLookup]);

  useEffect(() => {
    if (selectedNode && !visibleNodeIds.has(selectedNode.id)) setSelectedNode(null);
    if (hoveredNode && !visibleNodeIds.has(hoveredNode.id)) setHoveredNode(null);
  }, [selectedNode, hoveredNode, visibleNodeIds]);

  const averageEdgeConfidence = useMemo(() => {
    if (visibleEdges.length === 0) return 0;
    const scores = visibleEdges
      .map((edge) => Number(edge.confidence))
      .filter((value) => Number.isFinite(value));
    if (scores.length === 0) return 0;
    return scores.reduce((sum, value) => sum + value, 0) / scores.length;
  }, [visibleEdges]);

  const lowConfidenceEdges = useMemo(() => {
    return visibleEdges.filter((edge) => Number.isFinite(Number(edge.confidence)) && Number(edge.confidence) < 0.65).length;
  }, [visibleEdges]);

  const explainabilityRows = useMemo(() => {
    const rows = [];

    const cypConflicts = graphData.conflicts
      .filter((conflict) => conflict.type === 'cyp' && (conflict.risk_level === 'high' || conflict.risk_level === 'moderate'))
      .slice(0, 3);
    cypConflicts.forEach((conflict) => {
      rows.push(
        `${conflict.enzyme}: ${
          conflict.risk || `${conflict.drug1_role || 'unknown'} + ${conflict.drug2_role || 'unknown'} interaction pattern`
        }`
      );
    });

    const targetConflict = graphData.conflicts.find((conflict) => conflict.type === 'target');
    if (targetConflict) {
      rows.push(
        `Shared target ${targetConflict.gene || targetConflict.target || 'unknown'} can amplify pharmacodynamic effects.`
      );
    }

    if (enrichedMechanismMap?.interaction?.mechanism) {
      rows.push(`Known mechanism: ${String(enrichedMechanismMap.interaction.mechanism).slice(0, 140)}`);
    }

    if (rows.length === 0) {
      rows.push('No high-confidence conflict pathways were detected for this pair in the current evidence set.');
    }

    return rows;
  }, [graphData.conflicts, enrichedMechanismMap]);

  // Mouse handlers
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    setZoom((z) => Math.max(0.3, Math.min(3, z * (e.deltaY > 0 ? 0.9 : 1.1))));
  }, []);

  const handleMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    setIsPanning(true);
    panStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  }, [pan]);

  const handleMouseMove = useCallback((e) => {
    if (isPanning) {
      setPan({ x: e.clientX - panStart.current.x, y: e.clientY - panStart.current.y });
    }
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setHoverPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }
  }, [isPanning]);

  const handleMouseUp = useCallback(() => setIsPanning(false), []);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const showPairNavigator = pairCandidates.length > 1;
  const pairLabel = activePair
    ? `${activePair.drugA} <-> ${activePair.drugB}`
    : (drug1Name && drug2Name ? `${drug1Name} <-> ${drug2Name}` : 'Select two drugs');

  const refresh = useCallback(() => {
    clearBiologyCache();
    setLoadError('');
    setReloadNonce((prev) => prev + 1);
  }, []);

  const isOffline = dataSource === 'offline' || dataSource === 'none';
  const isLive = dataSource.includes('api');
  const hasCachedSegment = Boolean(drug1Bio?._cached || drug2Bio?._cached || mechanismMapData?._cached);
  const sourceLabel = SOURCE_LABELS[dataSource] || dataSource || 'Unknown source';
  const freshnessLabel = formatRelativeTime(lastLoadedAt);

  const hasGraphData = graphData.nodes.length > 0;
  const hasVisibleGraph = visibleNodes.length > 0;
  const hasVisibleConflicts = visibleNodes.some((node) => node.isConflict);

  const conflictSummary = enrichedMechanismMap?.conflict_summary || {
    cyp_conflicts: 0,
    target_overlaps: 0,
    shared_side_effects: 0,
    overall_risk: 'low',
  };

  const filtersTop = showPairNavigator ? 95 : 56;
  const helpPanelBottom = isMobile ? 12 : 188;
  const graphCanvasStyle = isLightTheme
    ? {
      background:
        'radial-gradient(circle at 16% 18%, rgba(59,130,246,0.10) 0%, rgba(59,130,246,0) 42%), radial-gradient(circle at 88% 82%, rgba(14,165,233,0.08) 0%, rgba(14,165,233,0) 46%), linear-gradient(180deg, #f8fafc 0%, #edf2f7 54%, #e6ebf2 100%)',
    }
    : {
      background:
        'radial-gradient(circle at 14% 16%, rgba(56,189,248,0.12) 0%, rgba(56,189,248,0) 40%), radial-gradient(circle at 88% 82%, rgba(99,102,241,0.10) 0%, rgba(99,102,241,0) 44%), linear-gradient(180deg, #0b1220 0%, #0e1728 55%, #121d32 100%)',
    };
  const controlButtonClass = isLightTheme ? 'text-slate-500 hover:text-slate-900' : 'text-white/30 hover:text-white/60';
  const activeControlClass = isLightTheme ? 'text-cyan-700' : 'text-cyan-300';
  const helpSurfaceClass = isLightTheme
    ? 'bg-white/86 border-slate-500/35 shadow-lg shadow-slate-900/10'
    : 'bg-black/80 border-white/10';
  const helpTitleClass = isLightTheme ? 'text-cyan-700' : 'text-cyan-300/80';
  const helpTextClass = isLightTheme ? 'text-slate-700/80' : 'text-white/50';

  // Empty state: no drugs selected
  if (!drug1Name) {
    return (
      <div ref={containerRef} className="w-full h-full relative overflow-hidden" style={graphCanvasStyle}>
        <svg width="100%" height="100%" className="absolute inset-0">
          <HexGridBackground
            width={dimensions.width}
            height={dimensions.height}
            occupiedCells={occupiedCells}
            isLightTheme={isLightTheme}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <InstructionalDiagram isLightTheme={isLightTheme} />
        </div>
      </div>
    );
  }

  // Loading state for first load
  if (loading && !drug1Bio) {
    return (
      <div ref={containerRef} className="w-full h-full relative overflow-hidden" style={graphCanvasStyle}>
        <svg width="100%" height="100%" className="absolute inset-0">
          <HexGridBackground
            width={dimensions.width}
            height={dimensions.height}
            occupiedCells={occupiedCells}
            isLightTheme={isLightTheme}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="w-6 h-6 text-purple-400 animate-spin mx-auto mb-2" />
            <div className={`text-[10px] font-mono ${isLightTheme ? 'text-slate-700' : 'text-white/50'}`}>
              Loading mechanism graph...
            </div>
            <div className={`text-[8px] mt-1 ${isLightTheme ? 'text-slate-600/80' : 'text-white/30'}`}>
              Retrieving CYP pathways, targets, and side effects
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Hard error state when no graph data loaded
  if (!loading && loadError && !hasGraphData) {
    return (
      <LoadErrorState
        containerRef={containerRef}
        dimensions={dimensions}
        occupiedCells={occupiedCells}
        isLightTheme={isLightTheme}
        graphCanvasStyle={graphCanvasStyle}
        error={loadError}
        onRetry={refresh}
      />
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden select-none"
      style={graphCanvasStyle}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* SVG Canvas */}
      <svg
        width={dimensions.width}
        height={dimensions.height}
        className="w-full h-full"
        style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
      >
        <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
          <HexGridBackground
            width={dimensions.width}
            height={dimensions.height}
            occupiedCells={occupiedCells}
            isLightTheme={isLightTheme}
          />

          {visibleEdges.map((edge, i) => {
            const sourcePos = layout.get(edge.source);
            const targetPos = layout.get(edge.target);
            if (!sourcePos || !targetPos) return null;
            return (
              <BiologyEdge
                key={`e-${i}`}
                edge={edge}
                x1={sourcePos.x}
                y1={sourcePos.y}
                x2={targetPos.x}
                y2={targetPos.y}
                isConflict={conflictNodeIds.has(edge.source) || conflictNodeIds.has(edge.target)}
              />
            );
          })}

          {['side_effect', 'target', 'enzyme', 'drug'].map((type) =>
            visibleNodes
              .filter((node) => node.type === type)
              .map((node) => {
                const pos = layout.get(node.id);
                if (!pos) return null;
                const Renderer = NODE_RENDERERS[type];
                return (
                  <Renderer
                    key={node.id}
                    node={node}
                    x={pos.x}
                    y={pos.y}
                    cellSize={cellSize}
                    isSelected={selectedNode?.id === node.id}
                    onClick={setSelectedNode}
                    onHover={setHoveredNode}
                  />
                );
              })
          )}
        </g>
      </svg>

      {showInsights && graphData.conflicts.length > 0 && (
        <ConflictPanel
          conflicts={graphData.conflicts}
          conflictSummary={enrichedMechanismMap?.conflict_summary}
          interaction={enrichedMechanismMap?.interaction}
        />
      )}

      <MechanismLegend isOffline={isOffline} isLightTheme={isLightTheme} />

      <NodeTooltip
        node={hoveredNode}
        x={hoverPos.x}
        y={hoverPos.y}
        containerRect={dimensions}
      />

      {/* Top control bar */}
      <div className="absolute top-2 left-3 right-3 z-30 flex items-center justify-between pointer-events-none">
        <div className="pointer-events-auto flex items-center gap-2">
          {isLive && (
            <span className="text-[7px] font-mono text-emerald-400/90 bg-emerald-400/10 border border-emerald-400/25 px-1.5 py-0.5 rounded flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              LIVE
            </span>
          )}
          {isOffline && (
            <span className="text-[7px] font-mono text-yellow-400/80 bg-yellow-400/10 border border-yellow-400/20 px-1.5 py-0.5 rounded flex items-center gap-1">
              <WifiOff className="w-2.5 h-2.5" />
              FALLBACK
            </span>
          )}

          <span className="text-[7px] font-mono text-cyan-200/90 bg-cyan-400/10 border border-cyan-400/20 px-1.5 py-0.5 rounded">
            {sourceLabel} | {hasCachedSegment ? 'cache' : 'fresh'} | {freshnessLabel}
          </span>

          {drug1Name && drug2Name && (
            <span className="text-[7px] font-mono text-purple-200/90 bg-purple-400/10 border border-purple-400/20 px-1.5 py-0.5 rounded">
              {pairLabel}
            </span>
          )}

          {!drug2Name && (
            <span className="text-[7px] font-mono text-white/55 bg-white/5 border border-white/15 px-1.5 py-0.5 rounded">
              Single-drug mode
            </span>
          )}

          {loadError && (
            <button
              onClick={refresh}
              className="text-[7px] font-mono text-red-300 bg-red-400/10 border border-red-400/25 px-1.5 py-0.5 rounded hover:bg-red-400/15"
              title="Retry loading graph data"
            >
              Error loading data - retry
            </button>
          )}
        </div>

        <div className="pointer-events-auto flex items-center gap-1">
          <button
            onClick={() => setShowHelp((value) => !value)}
            className={`p-1 transition-colors ${showHelp ? activeControlClass : controlButtonClass}`}
            title="Toggle quick guide"
          >
            <BookOpen className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setShowInsights((value) => !value)}
            className={`p-1 transition-colors ${showInsights ? activeControlClass : controlButtonClass}`}
            title="Toggle insight panel"
          >
            <Info className="w-3.5 h-3.5" />
          </button>
          <button onClick={refresh} className={`p-1 transition-colors ${controlButtonClass}`} title="Refresh data">
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button onClick={resetView} className={`p-1 transition-colors ${controlButtonClass}`} title="Reset view">
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {showPairNavigator && (
        <PairExplorer
          pairs={pairCandidates}
          activeIndex={activePairIndex}
          onChange={setActivePairIndex}
        />
      )}

      <GraphFilterBar
        filters={filters}
        setFilters={setFilters}
        nodeCounts={nodeCounts}
        style={{ top: `${filtersTop}px` }}
      />

      {showInsights && (
        <GraphInsightsPanel
          pairLabel={pairLabel}
          conflictSummary={conflictSummary}
          explainabilityRows={explainabilityRows}
          interaction={enrichedMechanismMap?.interaction}
          averageEdgeConfidence={averageEdgeConfidence}
          lowConfidenceEdges={lowConfidenceEdges}
          loading={loading}
        />
      )}

      {selectedNode && (
        <NodeDetailDrawer
          node={selectedNode}
          relations={selectedNodeRelations}
          conflicts={selectedNodeConflicts}
          onClose={() => setSelectedNode(null)}
          isMobile={Boolean(isMobile)}
        />
      )}

      {/* Empty graph or over-filtered view message */}
      {!loading && (!hasVisibleGraph || (filters.onlyConflicts && !hasVisibleConflicts)) && (
        <NoGraphDataState
          onlyConflicts={filters.onlyConflicts}
          hasGraphData={hasGraphData}
          onClearConflictFilter={() => setFilters((prev) => ({ ...prev, onlyConflicts: false }))}
        />
      )}

      {/* Bottom stats */}
      <div className="absolute bottom-2 right-3 z-20 pointer-events-none">
        <span className={`text-[7px] font-mono ${isLightTheme ? 'text-slate-600/80' : 'text-white/25'}`}>
          {visibleNodes.length}/{graphData.nodes.length} nodes | {visibleEdges.length}/{graphData.edges.length} edges
          {graphData.conflicts.length > 0 && (
            <span className="text-red-400/70"> | {graphData.conflicts.length} conflicts</span>
          )}
        </span>
      </div>

      {/* Interaction severity banner */}
      {enrichedMechanismMap?.interaction?.severity && (
        <div className="absolute top-10 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
          <InteractionBanner
            interaction={enrichedMechanismMap.interaction}
            conflictSummary={enrichedMechanismMap.conflict_summary}
          />
        </div>
      )}

      {showHelp && !isMobile && (
        <div className="absolute left-3 z-20 pointer-events-none max-w-[300px]" style={{ bottom: `${helpPanelBottom}px` }}>
          <div className={`backdrop-blur-md border rounded-lg px-3 py-2 ${helpSurfaceClass}`}>
            <div className={`text-[8px] uppercase tracking-wider mb-1 font-mono ${helpTitleClass}`}>How to read this graph</div>
            <div className={`text-[8px] leading-relaxed space-y-1 ${helpTextClass}`}>
              <p>1. Drug nodes anchor the pair; connected enzymes, targets, and side effects explain mechanisms.</p>
              <p>2. Red outlines indicate conflict-relevant biology.</p>
              <p>3. Edge labels include confidence percentages from available evidence.</p>
              <p>4. Use filters to isolate conflict pathways and reduce visual noise.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function PairExplorer({ pairs, activeIndex, onChange }) {
  if (!pairs || pairs.length <= 1) return null;

  const activePair = pairs[activeIndex] || null;
  const activeColor = getSeverityColor(activePair?.severity || 'unknown');

  return (
    <div className="absolute top-10 left-3 right-3 z-30 pointer-events-none">
      <div className="pointer-events-auto bg-black/80 backdrop-blur-md border border-white/10 rounded-lg px-2.5 py-2">
        <div className="flex items-center justify-between mb-1.5">
          <div className="text-[8px] text-white/45 uppercase tracking-wider font-mono">Pair Explorer</div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onChange((activeIndex - 1 + pairs.length) % pairs.length)}
              className="p-1 text-white/50 hover:text-white/80 transition-colors"
              title="Previous pair"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span className="text-[8px] text-white/55 font-mono">
              {activeIndex + 1}/{pairs.length}
            </span>
            <button
              onClick={() => onChange((activeIndex + 1) % pairs.length)}
              className="p-1 text-white/50 hover:text-white/80 transition-colors"
              title="Next pair"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        <div className="flex gap-1.5 overflow-x-auto pb-0.5">
          {pairs.map((pair, index) => {
            const selected = index === activeIndex;
            const color = getSeverityColor(pair.severity);
            return (
              <button
                key={`${pair.drugA}-${pair.drugB}-${index}`}
                onClick={() => onChange(index)}
                className={`min-w-[130px] text-left border rounded px-2 py-1 transition-colors ${
                  selected
                    ? 'border-white/30 bg-white/10'
                    : 'border-white/10 bg-white/[0.03] hover:bg-white/[0.06]'
                }`}
              >
                <div className="text-[8px] text-white/70 font-mono truncate">{pair.drugA}{' <-> '}{pair.drugB}</div>
                <div className="text-[7px] font-mono mt-0.5" style={{ color }}>
                  {Math.round(pair.riskScore * 100)}% risk | {pair.severity || 'unknown'}
                </div>
              </button>
            );
          })}
        </div>

        {activePair && (
          <div className="mt-1.5 pt-1.5 border-t border-white/5 text-[7px] font-mono" style={{ color: activeColor }}>
            Active pair: {activePair.drugA}{' <-> '}{activePair.drugB}
          </div>
        )}
      </div>
    </div>
  );
}

function GraphFilterBar({ filters, setFilters, nodeCounts, style }) {
  const toggleFilter = (key) => {
    setFilters((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="absolute left-3 z-30 pointer-events-none" style={style}>
      <div className="pointer-events-auto bg-black/75 backdrop-blur-md border border-white/10 rounded-lg px-2 py-2 flex items-center gap-1.5">
        <span className="text-[8px] text-white/45 uppercase tracking-wider font-mono flex items-center gap-1">
          <Filter className="w-3 h-3" />
          Filters
        </span>

        <FilterToggle
          label="Enzymes"
          count={nodeCounts.enzyme}
          enabled={filters.enzyme}
          onClick={() => toggleFilter('enzyme')}
          accent="text-cyan-300"
        />

        <FilterToggle
          label="Targets"
          count={nodeCounts.target}
          enabled={filters.target}
          onClick={() => toggleFilter('target')}
          accent="text-purple-300"
        />

        <FilterToggle
          label="Effects"
          count={nodeCounts.side_effect}
          enabled={filters.side_effect}
          onClick={() => toggleFilter('side_effect')}
          accent="text-pink-300"
        />

        <button
          onClick={() => toggleFilter('onlyConflicts')}
          className={`px-1.5 py-1 text-[8px] font-mono border rounded transition-colors ${
            filters.onlyConflicts
              ? 'text-red-300 border-red-300/40 bg-red-400/10'
              : 'text-white/45 border-white/15 hover:text-white/70'
          }`}
        >
          Conflicts only
        </button>
      </div>
    </div>
  );
}

function FilterToggle({ label, count, enabled, onClick, accent }) {
  return (
    <button
      onClick={onClick}
      className={`px-1.5 py-1 text-[8px] font-mono border rounded transition-colors ${
        enabled
          ? `border-white/30 bg-white/10 ${accent}`
          : 'border-white/15 text-white/45 hover:text-white/70'
      }`}
    >
      {label} ({count})
    </button>
  );
}

function GraphInsightsPanel({
  pairLabel,
  conflictSummary,
  explainabilityRows,
  interaction,
  averageEdgeConfidence,
  lowConfidenceEdges,
  loading,
}) {
  const riskColor = getSeverityColor(conflictSummary?.overall_risk || 'low');

  return (
    <div className="absolute left-3 top-[148px] z-20 pointer-events-none max-w-[360px]">
      <div className="pointer-events-auto bg-black/80 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 shadow-2xl">
        <div className="flex items-center justify-between mb-2">
          <div className="text-[9px] text-white/50 uppercase tracking-wider font-mono">Mechanism Insights</div>
          <span className="text-[8px] font-mono px-1.5 py-0.5 rounded" style={{ color: riskColor, background: `${riskColor}1A` }}>
            {String(conflictSummary?.overall_risk || 'low').toUpperCase()} RISK
          </span>
        </div>

        <div className="text-[8px] text-white/60 font-mono mb-2 truncate">{pairLabel}</div>

        <div className="grid grid-cols-3 gap-1 mb-2">
          <MetricPill icon={FlaskConical} label="CYP" value={String(conflictSummary?.cyp_conflicts || 0)} tone="text-cyan-300" />
          <MetricPill icon={Target} label="Target" value={String(conflictSummary?.target_overlaps || 0)} tone="text-purple-300" />
          <MetricPill icon={HeartPulse} label="Effects" value={String(conflictSummary?.shared_side_effects || 0)} tone="text-pink-300" />
        </div>

        <div className="mb-2 border border-white/10 rounded px-2 py-1.5 bg-white/[0.03]">
          <div className="text-[7px] text-white/45 uppercase tracking-wider font-mono mb-1">Evidence confidence</div>
          <div className="flex items-center justify-between text-[8px] font-mono">
            <span className="text-white/60">Mean edge confidence</span>
            <span className="text-emerald-300/90">{Math.round(averageEdgeConfidence * 100)}%</span>
          </div>
          <div className="flex items-center justify-between text-[8px] font-mono mt-0.5">
            <span className="text-white/60">Low-confidence edges</span>
            <span className="text-yellow-300/90">{lowConfidenceEdges}</span>
          </div>
        </div>

        <div className="border border-white/10 rounded px-2 py-1.5 bg-white/[0.03]">
          <div className="text-[7px] text-white/45 uppercase tracking-wider font-mono mb-1">Why this pair is flagged</div>
          <div className="space-y-1">
            {explainabilityRows.slice(0, 3).map((row, index) => (
              <p key={`${row}-${index}`} className="text-[8px] text-white/58 leading-relaxed">
                {index + 1}. {row}
              </p>
            ))}
          </div>
        </div>

        {interaction?.severity && (
          <div className="mt-2 pt-2 border-t border-white/8">
            <div className="text-[7px] text-white/45 uppercase tracking-wider font-mono mb-0.5">Known interaction severity</div>
            <div className="text-[8px] font-mono" style={{ color: getSeverityColor(interaction.severity) }}>
              {String(interaction.severity).toUpperCase()}
            </div>
          </div>
        )}

        {loading && (
          <div className="mt-2 text-[7px] text-white/35 font-mono">Refreshing evidence...</div>
        )}
      </div>
    </div>
  );
}

function NodeDetailDrawer({ node, relations, conflicts, onClose, isMobile }) {
  const typeLabel = String(node.type || 'node').replace('_', ' ');
  const nodeTone = node.isConflict ? 'text-red-300' : 'text-cyan-200';
  const safeRelations = Array.isArray(relations) ? relations : [];
  const safeConflicts = Array.isArray(conflicts) ? conflicts : [];

  return (
    <div
      className={`absolute z-40 pointer-events-none ${
        isMobile ? 'left-3 right-3 bottom-3' : 'right-3 top-[250px] w-[320px]'
      }`}
    >
      <div className="pointer-events-auto bg-black/85 backdrop-blur-md border border-white/12 rounded-lg shadow-2xl max-h-[48vh] overflow-y-auto">
        <div className="px-3 py-2 border-b border-white/8 flex items-center justify-between">
          <div>
            <div className={`text-[10px] font-mono font-bold ${nodeTone}`}>{node.label || 'Selected node'}</div>
            <div className="text-[8px] text-white/45 uppercase tracking-wider">{typeLabel}</div>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-white/40 hover:text-white/70 transition-colors"
            title="Close detail panel"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="px-3 py-2 space-y-2">
          <div className="border border-white/10 rounded px-2 py-1.5 bg-white/[0.03]">
            <div className="text-[7px] text-white/45 uppercase tracking-wider font-mono mb-1">Node metadata</div>
            {node.type === 'drug' && (
              <div className="text-[8px] text-white/60 space-y-1">
                <div>Therapeutic class: {node.therapeutic_class || 'Unknown'}</div>
                <div>Slot: {node.slot || 'N/A'}</div>
              </div>
            )}
            {node.type === 'enzyme' && (
              <div className="text-[8px] text-white/60 space-y-1">
                {Object.entries(node.enzymeRoles || {}).map(([slot, role]) => (
                  <div key={`${slot}-${role}`}>{slot === 'A' ? 'Drug A' : 'Drug B'} role: {role}</div>
                ))}
                {Object.keys(node.enzymeRoles || {}).length === 0 && <div>No enzyme role data</div>}
              </div>
            )}
            {node.type === 'target' && (
              <div className="text-[8px] text-white/60 space-y-1">
                <div>Gene: {node.gene || 'Unknown'}</div>
                <div>Action: {node.action || 'unknown'}</div>
              </div>
            )}
            {node.type === 'side_effect' && (
              <div className="text-[8px] text-white/60 space-y-1">
                <div>Organ system: {node.organ_system || 'Unknown'}</div>
                <div>
                  Severity: {typeof node.severity === 'number' ? `${Math.round(node.severity * 100)}%` : (node.severity || 'unknown')}
                </div>
              </div>
            )}
          </div>

          {safeConflicts.length > 0 && (
            <div className="border border-red-400/20 rounded px-2 py-1.5 bg-red-500/[0.05]">
              <div className="text-[7px] text-red-300/85 uppercase tracking-wider font-mono mb-1">Conflict evidence</div>
              <div className="space-y-1.5">
                {safeConflicts.map((conflict) => (
                  <div key={conflict.id || `${conflict.type}-${conflict.nodeId}`} className="text-[8px] text-white/62 leading-relaxed">
                    <div className="text-red-300/85">{String(conflict.type || 'conflict').toUpperCase()}</div>
                    <div>{conflict.risk || conflict.name || 'Conflict detected on this node.'}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="border border-white/10 rounded px-2 py-1.5 bg-white/[0.03]">
            <div className="text-[7px] text-white/45 uppercase tracking-wider font-mono mb-1">
              Connected pathways ({safeRelations.length})
            </div>
            {safeRelations.length === 0 && (
              <div className="text-[8px] text-white/45">No direct relationships available.</div>
            )}
            {safeRelations.length > 0 && (
              <div className="space-y-1">
                {safeRelations.slice(0, 8).map((relation) => {
                  const confidenceText = Number.isFinite(relation.confidence)
                    ? `${Math.round(relation.confidence * 100)}%`
                    : 'n/a';
                  return (
                    <div key={relation.id} className="text-[8px] text-white/62 leading-relaxed">
                      <span className="text-white/35">{relation.direction === 'outbound' ? '->' : '<-'}</span>{' '}
                      {relation.neighborLabel}{' '}
                      <span className="text-white/35">[{relation.type}]</span>{' '}
                      <span className="text-emerald-300/80">{confidenceText}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricPill({ icon: Icon, label, value, tone }) {
  return (
    <div className="border border-white/10 rounded px-2 py-1 bg-white/[0.03]">
      <div className="flex items-center gap-1 text-[7px] text-white/45 font-mono uppercase tracking-wider">
        <Icon className="w-2.5 h-2.5" />
        {label}
      </div>
      <div className={`text-[9px] font-mono mt-0.5 ${tone}`}>{value}</div>
    </div>
  );
}

function LoadErrorState({ containerRef, dimensions, occupiedCells, isLightTheme, graphCanvasStyle, error, onRetry }) {
  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden" style={graphCanvasStyle}>
      <svg width="100%" height="100%" className="absolute inset-0">
        <HexGridBackground
          width={dimensions.width}
          height={dimensions.height}
          occupiedCells={occupiedCells}
          isLightTheme={isLightTheme}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center px-4">
        <div
          className="max-w-md backdrop-blur-md border border-red-400/30 rounded-lg px-4 py-3 text-center"
          style={{ background: isLightTheme ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.8)' }}
        >
          <AlertTriangle className="w-5 h-5 text-red-400 mx-auto mb-2" />
          <div className="text-[11px] text-red-300 uppercase tracking-wider font-mono mb-1">Unable to render mechanism graph</div>
          <p className={`text-[9px] leading-relaxed mb-3 ${isLightTheme ? 'text-slate-700/85' : 'text-white/60'}`}>
            {error || 'Data retrieval failed.'}
          </p>
          <button
            onClick={onRetry}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[9px] font-mono border border-red-300/40 text-red-200 bg-red-400/10 hover:bg-red-400/20 transition-colors rounded"
          >
            <RefreshCw className="w-3 h-3" />
            Retry data load
          </button>
        </div>
      </div>
    </div>
  );
}

function NoGraphDataState({ onlyConflicts, hasGraphData, onClearConflictFilter }) {
  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none px-6">
      <div className="max-w-sm bg-black/70 backdrop-blur-md border border-white/10 rounded-lg px-4 py-3 text-center pointer-events-auto">
        <div className="text-[10px] text-white/65 uppercase tracking-wider font-mono mb-1">
          {onlyConflicts ? 'No conflicts in current filter' : 'No graph entities to display'}
        </div>
        <p className="text-[8px] text-white/45 leading-relaxed mb-2">
          {onlyConflicts
            ? 'This pair has no visible conflict-marked nodes under the current filter mode.'
            : (hasGraphData
              ? 'Current filters hide all visible entities.'
              : 'Biological relationships were not returned for this pair. Try a different pair or refresh data.')}
        </p>
        {onlyConflicts && (
          <button
            onClick={onClearConflictFilter}
            className="px-2 py-1 text-[8px] font-mono border border-white/20 text-white/70 hover:text-white hover:border-white/35 rounded transition-colors"
          >
            Show all entities
          </button>
        )}
      </div>
    </div>
  );
}

function InteractionBanner({ interaction, conflictSummary }) {
  const severity = interaction.severity?.toLowerCase() || 'unknown';
  const color = severity === 'major' || severity === 'severe' || severity === 'critical'
    ? '#ef4444'
    : severity === 'moderate'
      ? '#eab308'
      : severity === 'minor'
        ? '#22c55e'
        : '#6b7280';

  return (
    <div className="bg-black/80 backdrop-blur-md border rounded-lg px-4 py-2 max-w-md" style={{ borderColor: `${color}40` }}>
      <div className="flex items-center gap-2 mb-1">
        <AlertTriangle className="w-3.5 h-3.5" style={{ color }} />
        <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color }}>
          {severity} interaction
        </span>
      </div>
      {interaction.mechanism && (
        <p className="text-[8px] text-white/50 leading-relaxed line-clamp-2">{interaction.mechanism}</p>
      )}
      {conflictSummary && (
        <div className="flex gap-3 mt-1.5 pt-1.5 border-t border-white/5">
          <span className="text-[7px] font-mono text-cyan-400/60">{conflictSummary.cyp_conflicts || 0} CYP conflicts</span>
          <span className="text-[7px] font-mono text-purple-400/60">{conflictSummary.target_overlaps || 0} shared targets</span>
          <span className="text-[7px] font-mono text-pink-400/60">{conflictSummary.shared_side_effects || 0} shared effects</span>
        </div>
      )}
    </div>
  );
}

function InstructionalDiagram({ isLightTheme = false }) {
  return (
    <div className="text-center max-w-xs px-4">
      <svg width={260} height={160} viewBox="0 0 260 160" className="mx-auto mb-3 opacity-50">
        <rect x={15} y={55} width={70} height={34} rx={8} fill="#00d2ff08" stroke="#00d2ff" strokeWidth={1} />
        <text x={50} y={75} textAnchor="middle" fill="#00d2ff" fontSize={9} fontFamily="monospace" fontWeight="bold">Warfarin</text>

        <rect x={175} y={55} width={70} height={34} rx={8} fill="#ff8c0008" stroke="#ff8c00" strokeWidth={1} />
        <text x={210} y={75} textAnchor="middle" fill="#ff8c00" fontSize={9} fontFamily="monospace" fontWeight="bold">Aspirin</text>

        <polygon points="130,18 144,25 144,39 130,46 116,39 116,25" fill="#ef444410" stroke="#ef4444" strokeWidth={1.5} />
        <text x={130} y={35} textAnchor="middle" fill="#ef4444" fontSize={7} fontFamily="monospace" fontWeight="bold">CYP2C9</text>

        <circle cx={130} cy={95} r={16} fill="#a855f708" stroke="#a855f7" strokeWidth={1} />
        <text x={130} y={98} textAnchor="middle" fill="#a855f7" fontSize={7} fontFamily="monospace">COX-1</text>

        <polygon points="130,125 138,133 130,141 122,133" fill="#eab30808" stroke="#eab308" strokeWidth={1} />
        <text x={130} y={155} textAnchor="middle" fill="#eab308" fontSize={6} fontFamily="monospace">Bleeding</text>

        <path d="M 85 60 Q 105 40 116 32" fill="none" stroke="#3b82f6" strokeWidth={0.8} opacity={0.5} />
        <path d="M 175 60 Q 155 40 144 32" fill="none" stroke="#ef4444" strokeWidth={0.8} strokeDasharray="4,2" opacity={0.5} />
        <path d="M 85 85 Q 105 90 114 92" fill="none" stroke="#a855f7" strokeWidth={0.8} opacity={0.4} />
        <path d="M 175 85 Q 155 90 146 92" fill="none" stroke="#a855f7" strokeWidth={0.8} opacity={0.4} />
        <line x1={130} y1={111} x2={130} y2={125} stroke="#6b7280" strokeWidth={0.6} opacity={0.3} />

        <polygon points="130,18 144,25 144,39 130,46 116,39 116,25" fill="none" stroke="#ef4444" strokeWidth={2} opacity={0.3}>
          <animate attributeName="opacity" values="0.1;0.4;0.1" dur="2s" repeatCount="indefinite" />
        </polygon>
      </svg>

      <div className={`text-[11px] font-mono mb-1 ${isLightTheme ? 'text-slate-700/85' : 'text-white/50'}`}>
        Biological Mechanism Map
      </div>
      <div className={`text-[9px] leading-relaxed ${isLightTheme ? 'text-slate-600/80' : 'text-white/25'}`}>
        Select two drugs to inspect shared CYP enzymes, protein targets, and side effects.
        The map highlights conflict pathways and edge confidence.
      </div>
    </div>
  );
}
