// KnowledgeGraph/index.jsx — Biological Mechanism Map
// Shows WHY drugs interact: shared CYP enzymes, protein targets, side effects.
// Uses hex grid background aesthetic + radial biology graph.

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Loader2, WifiOff, RefreshCw, Maximize2, AlertTriangle } from 'lucide-react';

import { fetchDrugBiology, fetchMechanismMap, clearBiologyCache } from './biologyService';
import { buildMechanismGraph, computeRadialLayout, SEVERITY_COLORS } from './mechanismGraphEngine';

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

// ─── Hex grid background ────────────────────────────────────────────────────
// Same axial coordinate system as node layout. Nodes sit INSIDE these cells.
function HexGridBackground({ width, height, occupiedCells }) {
  const minDim = Math.min(width, height);
  const hexSize = Math.max(50, minDim * 0.09);
  const cx = width / 2;
  const cy = height / 2;
  const GRID_RAD = 8;

  const hexPoints = (px, py, s) => {
    return Array.from({ length: 6 }, (_, i) => {
      const angle = (Math.PI / 3) * i - Math.PI / 6;
      return `${px + s * Math.cos(angle)},${py + s * Math.sin(angle)}`;
    }).join(' ');
  };

  const hexes = [];
  for (let q = -GRID_RAD; q <= GRID_RAD; q++) {
    const r1 = Math.max(-GRID_RAD, -q - GRID_RAD);
    const r2 = Math.min(GRID_RAD, -q + GRID_RAD);
    for (let r = r1; r <= r2; r++) {
      const x = cx + hexSize * Math.sqrt(3) * (q + r / 2);
      const y = cy + hexSize * (3 / 2) * r;
      if (x < -hexSize * 2 || x > width + hexSize * 2 || y < -hexSize * 2 || y > height + hexSize * 2) continue;

      const isOccupied = occupiedCells?.has(`${q},${r}`);
      hexes.push(
        <polygon
          key={`${q},${r}`}
          points={hexPoints(x, y, hexSize - 1)}
          fill={isOccupied ? 'rgba(255,255,255,0.025)' : 'rgba(255,255,255,0.008)'}
          stroke={isOccupied ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.04)'}
          strokeWidth={0.5}
        />
      );
    }
  }

  return <g>{hexes}</g>;
}

export default function KnowledgeGraphView({ drugs = [], result, polypharmacyResult, isMobile }) {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Data state
  const [drug1Bio, setDrug1Bio] = useState(null);
  const [drug2Bio, setDrug2Bio] = useState(null);
  const [mechanismMapData, setMechanismMapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dataSource, setDataSource] = useState('');

  // UI state
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const [activePairIndex, setActivePairIndex] = useState(0);

  // Zoom/pan
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });

  // Measure container
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) setDimensions({ width, height });
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
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

  // Extract active pair names (polypharmacy pair if available, else first two selected)
  const drug1Name = activePair?.drugA || fallbackDrug1Name;
  const drug2Name = activePair?.drugB || fallbackDrug2Name;

  // Fetch biology data when drugs change
  useEffect(() => {
    if (!drug1Name) {
      setDrug1Bio(null);
      setDrug2Bio(null);
      setMechanismMapData(null);
      setDataSource('');
      return;
    }

    let cancelled = false;
    async function load() {
      setLoading(true);

      // Fetch drug 1
      const bio1 = await fetchDrugBiology(drug1Name);
      if (cancelled) return;
      setDrug1Bio(bio1);
      setDataSource(bio1?.source || 'offline');

      // Fetch drug 2 + mechanism map
      if (drug2Name) {
        const [bio2, mmap] = await Promise.all([
          fetchDrugBiology(drug2Name),
          fetchMechanismMap(drug1Name, drug2Name),
        ]);
        if (cancelled) return;
        setDrug2Bio(bio2);
        setMechanismMapData(mmap);
        setDataSource(mmap?.source || bio2?.source || bio1?.source || 'offline');
      } else {
        setDrug2Bio(null);
        setMechanismMapData(null);
      }

      if (!cancelled) setLoading(false);
    }

    load();
    return () => { cancelled = true; };
  }, [drug1Name, drug2Name]);

  // Also incorporate prediction result into mechanism map
  const enrichedMechanismMap = useMemo(() => {
    if (!mechanismMapData) return null;
    const mmap = { ...mechanismMapData };

    // For N-way analysis, prefer pair-specific mechanism data over summary result payload.
    const shouldMergePairwiseResult = pairCandidates.length === 0;

    // Merge prediction result data if available
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
    return new Set(graphData.conflicts.map(c => c.nodeId));
  }, [graphData.conflicts]);

  // Mouse handlers
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    setZoom(z => Math.max(0.3, Math.min(3, z * (e.deltaY > 0 ? 0.9 : 1.1))));
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

  const resetView = useCallback(() => { setZoom(1); setPan({ x: 0, y: 0 }); }, []);

  const showPairNavigator = pairCandidates.length > 1;
  const pairLabel = activePair
    ? `${activePair.drugA} <-> ${activePair.drugB}`
    : (drug1Name && drug2Name ? `${drug1Name} <-> ${drug2Name}` : 'Select two drugs');

  const refresh = useCallback(() => {
    clearBiologyCache();
    setDrug1Bio(null);
    setDrug2Bio(null);
    setMechanismMapData(null);
  }, []);

  const isOffline = dataSource === 'offline' || dataSource === 'none';
  const isLive = dataSource.includes('api');

  // ─── No drugs selected ────────────────────────────────────────────────────
  if (!drug1Name) {
    return (
      <div ref={containerRef} className="w-full h-full relative overflow-hidden">
        <svg width="100%" height="100%" className="absolute inset-0">
          <HexGridBackground width={dimensions.width} height={dimensions.height} occupiedCells={occupiedCells} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <InstructionalDiagram />
        </div>
      </div>
    );
  }

  // ─── Loading ──────────────────────────────────────────────────────────────
  if (loading && !drug1Bio) {
    return (
      <div ref={containerRef} className="w-full h-full relative overflow-hidden">
        <svg width="100%" height="100%" className="absolute inset-0">
          <HexGridBackground width={dimensions.width} height={dimensions.height} occupiedCells={occupiedCells} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="w-6 h-6 text-purple-400 animate-spin mx-auto mb-2" />
            <div className="text-[10px] text-white/40 font-mono">Loading biology data...</div>
          </div>
        </div>
      </div>
    );
  }

  // ─── Main render ──────────────────────────────────────────────────────────
  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden select-none"
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
        {/* Everything moves together with pan + zoom */}
        <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
          {/* Hex grid background — inside transform so it moves with nodes */}
          <HexGridBackground width={dimensions.width} height={dimensions.height} occupiedCells={occupiedCells} />
          {/* Edges */}
          {graphData.edges.map((edge, i) => {
            const sp = layout.get(edge.source);
            const tp = layout.get(edge.target);
            if (!sp || !tp) return null;
            return (
              <BiologyEdge
                key={`e-${i}`}
                edge={edge}
                x1={sp.x} y1={sp.y}
                x2={tp.x} y2={tp.y}
                isConflict={conflictNodeIds.has(edge.source) || conflictNodeIds.has(edge.target)}
              />
            );
          })}

          {/* Nodes — render back to front */}
          {['side_effect', 'target', 'enzyme', 'drug'].map(type =>
            graphData.nodes
              .filter(n => n.type === type)
              .map(node => {
                const pos = layout.get(node.id);
                if (!pos) return null;
                const Renderer = NODE_RENDERERS[type];
                return (
                  <Renderer
                    key={node.id}
                    node={node}
                    x={pos.x} y={pos.y}
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

      {/* Conflict Panel */}
      <ConflictPanel
        conflicts={graphData.conflicts}
        conflictSummary={enrichedMechanismMap?.conflict_summary}
        interaction={enrichedMechanismMap?.interaction}
      />

      {/* Legend */}
      <MechanismLegend isOffline={isOffline} />

      {/* Tooltip */}
      <NodeTooltip
        node={hoveredNode}
        x={hoverPos.x}
        y={hoverPos.y}
        containerRect={dimensions}
      />

      {/* Top bar: data source + controls */}
      <div className="absolute top-2 left-3 right-3 z-20 flex items-center justify-between pointer-events-none">
        {/* Data source badge */}
        <div className="pointer-events-auto flex items-center gap-2">
          {isLive && (
            <span className="text-[7px] font-mono text-emerald-400/80 bg-emerald-400/10 border border-emerald-400/20 px-1.5 py-0.5 rounded flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              LIVE
            </span>
          )}
          {isOffline && (
            <span className="text-[7px] font-mono text-yellow-400/80 bg-yellow-400/10 border border-yellow-400/20 px-1.5 py-0.5 rounded flex items-center gap-1">
              <WifiOff className="w-2.5 h-2.5" />
              CYP DATA ONLY
            </span>
          )}
          {loading && <Loader2 className="w-3 h-3 text-purple-400 animate-spin" />}

          {drug1Name && drug2Name && (
            <span className="text-[7px] font-mono text-cyan-300/90 bg-cyan-400/10 border border-cyan-400/20 px-1.5 py-0.5 rounded">
              {pairLabel}
            </span>
          )}

          {showPairNavigator && (
            <div className="flex items-center gap-1 bg-white/5 border border-white/10 rounded px-1 py-0.5">
              <button
                onClick={() => setActivePairIndex(prev => (prev - 1 + pairCandidates.length) % pairCandidates.length)}
                className="px-1 text-[9px] text-white/60 hover:text-white"
                title="Previous interaction pair"
              >
                {'<'}
              </button>
              <span className="text-[7px] font-mono text-white/50">
                pair {activePairIndex + 1}/{pairCandidates.length}
              </span>
              <button
                onClick={() => setActivePairIndex(prev => (prev + 1) % pairCandidates.length)}
                className="px-1 text-[9px] text-white/60 hover:text-white"
                title="Next interaction pair"
              >
                {'>'}
              </button>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="pointer-events-auto flex items-center gap-1">
          <button onClick={refresh} className="p-1 text-white/30 hover:text-white/60 transition-colors" title="Refresh">
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
          <button onClick={resetView} className="p-1 text-white/30 hover:text-white/60 transition-colors" title="Reset view">
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Bottom stats */}
      <div className="absolute bottom-2 right-3 z-20 pointer-events-none">
        <span className="text-[7px] font-mono text-white/20">
          {graphData.nodes.length} nodes · {graphData.edges.length} edges
          {graphData.conflicts.length > 0 && (
            <span className="text-red-400/60"> · {graphData.conflicts.length} conflicts</span>
          )}
        </span>
      </div>

      {/* Interaction severity banner (when 2 drugs) */}
      {enrichedMechanismMap?.interaction?.severity && (
        <div className="absolute top-10 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
          <InteractionBanner
            interaction={enrichedMechanismMap.interaction}
            conflictSummary={enrichedMechanismMap.conflict_summary}
            drug1={drug1Name}
            drug2={drug2Name}
          />
        </div>
      )}
    </div>
  );
}

// ─── Interaction banner ─────────────────────────────────────────────────────
function InteractionBanner({ interaction, conflictSummary, drug1, drug2 }) {
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

// ─── Instructional diagram ──────────────────────────────────────────────────
function InstructionalDiagram() {
  return (
    <div className="text-center max-w-xs px-4">
      <svg width={260} height={160} viewBox="0 0 260 160" className="mx-auto mb-3 opacity-50">
        {/* Drug A */}
        <rect x={15} y={55} width={70} height={34} rx={8} fill="#00d2ff08" stroke="#00d2ff" strokeWidth={1} />
        <text x={50} y={75} textAnchor="middle" fill="#00d2ff" fontSize={9} fontFamily="monospace" fontWeight="bold">Warfarin</text>

        {/* Drug B */}
        <rect x={175} y={55} width={70} height={34} rx={8} fill="#ff8c0008" stroke="#ff8c00" strokeWidth={1} />
        <text x={210} y={75} textAnchor="middle" fill="#ff8c00" fontSize={9} fontFamily="monospace" fontWeight="bold">Aspirin</text>

        {/* CYP2C9 - shared enzyme (conflict) */}
        <polygon points="130,18 144,25 144,39 130,46 116,39 116,25" fill="#ef444410" stroke="#ef4444" strokeWidth={1.5} />
        <text x={130} y={35} textAnchor="middle" fill="#ef4444" fontSize={7} fontFamily="monospace" fontWeight="bold">CYP2C9</text>

        {/* COX-1 target */}
        <circle cx={130} cy={95} r={16} fill="#a855f708" stroke="#a855f7" strokeWidth={1} />
        <text x={130} y={98} textAnchor="middle" fill="#a855f7" fontSize={7} fontFamily="monospace">COX-1</text>

        {/* Bleeding side effect */}
        <polygon points="130,125 138,133 130,141 122,133" fill="#eab30808" stroke="#eab308" strokeWidth={1} />
        <text x={130} y={155} textAnchor="middle" fill="#eab308" fontSize={6} fontFamily="monospace">Bleeding</text>

        {/* Edges */}
        <path d="M 85 60 Q 105 40 116 32" fill="none" stroke="#3b82f6" strokeWidth={0.8} opacity={0.5} />
        <path d="M 175 60 Q 155 40 144 32" fill="none" stroke="#ef4444" strokeWidth={0.8} strokeDasharray="4,2" opacity={0.5} />
        <path d="M 85 85 Q 105 90 114 92" fill="none" stroke="#a855f7" strokeWidth={0.8} opacity={0.4} />
        <path d="M 175 85 Q 155 90 146 92" fill="none" stroke="#a855f7" strokeWidth={0.8} opacity={0.4} />
        <line x1={130} y1={111} x2={130} y2={125} stroke="#6b7280" strokeWidth={0.6} opacity={0.3} />

        {/* Conflict glow on CYP2C9 */}
        <polygon points="130,18 144,25 144,39 130,46 116,39 116,25" fill="none" stroke="#ef4444" strokeWidth={2} opacity={0.3}>
          <animate attributeName="opacity" values="0.1;0.4;0.1" dur="2s" repeatCount="indefinite" />
        </polygon>
      </svg>

      <div className="text-[11px] text-white/50 font-mono mb-1">Biological Mechanism Map</div>
      <div className="text-[9px] text-white/25 leading-relaxed">
        Select drugs to visualize shared CYP enzymes, protein targets, and side effects.
        Conflicts are highlighted automatically.
      </div>
    </div>
  );
}
