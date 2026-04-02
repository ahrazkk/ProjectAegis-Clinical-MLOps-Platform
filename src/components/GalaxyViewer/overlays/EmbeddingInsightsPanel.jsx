import React, { useEffect, useMemo, useState } from 'react';
import { Activity, BarChart3, Download, Maximize2, Minimize2, Ruler, Search, Sigma, Sparkles } from 'lucide-react';
import { useGalaxy } from '../store';
import { getAdj, getEdgeMeta } from '../graphEngine';

function euclideanDistance(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function mean(values) {
  if (!values || values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdDev(values, valueMean = null) {
  if (!values || values.length === 0) return 0;
  const m = valueMean === null ? mean(values) : valueMean;
  const variance = values.reduce((sum, value) => sum + ((value - m) ** 2), 0) / values.length;
  return Math.sqrt(variance);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function csvEscape(value) {
  const text = String(value ?? '');
  if (text.includes(',') || text.includes('"') || text.includes('\n')) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export default function EmbeddingInsightsPanel({ nodes = [] }) {
  const { viewMode, selectedNode, filters } = useGalaxy();
  const [activeNodeId, setActiveNodeId] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const adj = getAdj();
  const kValue = clamp(Math.round(filters.embeddingK || 8), 2, 20);
  const analysisK = Math.max(10, kValue);

  const allNodes = useMemo(() => {
    return (Array.isArray(nodes) ? nodes : []).filter(n => Array.isArray(n?.pos));
  }, [nodes]);

  const regimenNodes = useMemo(() => {
    const byId = new Map();

    allNodes.forEach((node) => {
      if (node.isA || node.isB || node.isSelected) {
        byId.set(node.id, node);
      }
    });

    const clicked = selectedNode ? allNodes.find(n => n.id === selectedNode.id) : null;
    if (clicked && !byId.has(clicked.id)) {
      byId.set(clicked.id, { ...clicked, isClicked: true });
    }

    const list = Array.from(byId.values());
    list.sort((a, b) => {
      const rank = (node) => {
        if (node.isA) return 0;
        if (node.isB) return 1;
        if (node.isSelected) return 2;
        return 3;
      };
      const diff = rank(a) - rank(b);
      if (diff !== 0) return diff;
      return String(a.name || '').localeCompare(String(b.name || ''));
    });

    return list;
  }, [allNodes, selectedNode]);

  useEffect(() => {
    if (selectedNode?.id && allNodes.some(n => n.id === selectedNode.id)) {
      setActiveNodeId(selectedNode.id);
      return;
    }

    if (activeNodeId && allNodes.some(n => n.id === activeNodeId)) {
      return;
    }

    if (regimenNodes.length > 0) {
      setActiveNodeId(regimenNodes[0].id);
    } else {
      setActiveNodeId(null);
    }
  }, [selectedNode?.id, regimenNodes, allNodes, activeNodeId]);

  const analysisById = useMemo(() => {
    const result = {};
    if (!allNodes || allNodes.length === 0) return result;

    const degreeList = allNodes.map((n) => (adj[n.id] || []).length).sort((a, b) => a - b);

    const targets = new Map();
    regimenNodes.forEach(n => targets.set(n.id, n));
    if (activeNodeId) {
      const activeNode = allNodes.find(n => n.id === activeNodeId);
      if (activeNode) targets.set(activeNode.id, activeNode);
    }

    targets.forEach((anchor) => {
      const graphNeighborSet = new Set(adj[anchor.id] || []);

      const rows = allNodes
        .filter((n) => n.id !== anchor.id)
        .map((n) => {
          const distance = euclideanDistance(anchor.pos, n.pos);
          const isDirectGraphNeighbor = graphNeighborSet.has(n.id);
          const edgeMeta = isDirectGraphNeighbor ? getEdgeMeta(anchor.id, n.id) : null;

          return {
            id: n.id,
            name: n.name,
            category: n.category || 'Other',
            type: n.type || 'Unknown',
            distance,
            isDirectGraphNeighbor,
            severity: edgeMeta?.severity || 'none',
          };
        })
        .sort((a, b) => a.distance - b.distance);

      const topKRows = rows.slice(0, Math.min(analysisK, rows.length));
      const topDistances = topKRows.map((r) => r.distance);
      const meanDistance = mean(topDistances);
      const distanceStd = stdDev(topDistances, meanDistance);

      const sameClassRows = topKRows.filter((r) => r.category === (anchor.category || 'Other'));
      const diffClassRows = topKRows.filter((r) => r.category !== (anchor.category || 'Other'));
      const a = mean(sameClassRows.map((r) => r.distance));
      const b = mean(diffClassRows.map((r) => r.distance));
      const silhouetteLocal = (a > 0 && b > 0) ? ((b - a) / Math.max(a, b)) : 0;

      const overlapCount = topKRows.filter((r) => r.isDirectGraphNeighbor).length;
      const overlapRate = topKRows.length > 0 ? overlapCount / topKRows.length : 0;
      const degree = (adj[anchor.id] || []).length;
      const degreeRank = degreeList.filter((d) => d <= degree).length;
      const degreePercentile = degreeList.length > 0 ? (degreeRank / degreeList.length) : 0;

      const expectedOverlapCount = topKRows.length * (degree / Math.max(1, allNodes.length - 1));
      const overlapEnrichment = expectedOverlapCount > 0 ? overlapCount / expectedOverlapCount : 0;

      const graphNeighborRanks = [];
      rows.forEach((row, index) => {
        if (row.isDirectGraphNeighbor) graphNeighborRanks.push(index + 1);
      });
      const meanGraphRank = graphNeighborRanks.length > 0 ? mean(graphNeighborRanks.slice(0, 20)) : 0;

      result[anchor.id] = {
        anchor,
        rows,
        topKRows,
        metrics: {
          degree,
          degreePercentile,
          meanDistance,
          distanceStd,
          localDensity: meanDistance > 0 ? 1 / meanDistance : 0,
          classPurity: topKRows.length > 0 ? sameClassRows.length / topKRows.length : 0,
          graphOverlap: overlapRate,
          overlapCount,
          expectedOverlapCount,
          overlapEnrichment,
          silhouetteLocal,
          meanGraphRank,
        },
      };
    });

    return result;
  }, [allNodes, adj, regimenNodes, activeNodeId, analysisK]);

  const activeAnalysis = activeNodeId ? analysisById[activeNodeId] : null;
  const nearestRows = activeAnalysis
    ? activeAnalysis.rows.slice(0, isExpanded ? 40 : 12)
    : [];

  const searchResults = useMemo(() => {
    const query = String(searchQuery || '').trim().toLowerCase();
    if (query.length < 2) return [];

    return allNodes
      .filter((node) => String(node.name || '').toLowerCase().includes(query))
      .slice(0, 8);
  }, [searchQuery, allNodes]);

  const exportNearestCsv = () => {
    if (!activeAnalysis || activeAnalysis.rows.length === 0) return;

    const header = [
      'rank',
      'focus_drug',
      'neighbor_drug',
      'distance_3d',
      'neighbor_category',
      'direct_graph_edge',
      'edge_severity',
      `top_${analysisK}_member`,
    ];

    const rows = activeAnalysis.rows.slice(0, 150).map((n, index) => [
      index + 1,
      activeAnalysis.anchor.name || activeAnalysis.anchor.id,
      n.name,
      n.distance.toFixed(6),
      n.category,
      n.isDirectGraphNeighbor ? 'yes' : 'no',
      n.severity,
      index < analysisK ? 'yes' : 'no',
    ]);

    const csv = [header, ...rows]
      .map((row) => row.map(csvEscape).join(','))
      .join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `embedding-nearest-${(activeAnalysis.anchor.name || activeAnalysis.anchor.id || 'drug').replace(/\s+/g, '_').toLowerCase()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportRegimenSummaryCsv = () => {
    if (!regimenNodes.length) return;

    const header = [
      'drug',
      'role',
      'degree',
      'degree_percentile',
      `mean_${analysisK}nn_distance`,
      `std_${analysisK}nn_distance`,
      `purity_${analysisK}`,
      `graph_overlap_${analysisK}`,
      `expected_graph_overlap_${analysisK}`,
      'overlap_enrichment',
      `silhouette_local_${analysisK}`,
      'mean_graph_neighbor_rank',
    ];

    const rows = regimenNodes
      .map((node) => {
        const analysis = analysisById[node.id];
        if (!analysis) return null;

        const role = node.isA ? 'A' : node.isB ? 'B' : node.isSelected ? 'S' : 'clicked';
        return [
          node.name,
          role,
          analysis.metrics.degree,
          (analysis.metrics.degreePercentile * 100).toFixed(2),
          analysis.metrics.meanDistance.toFixed(6),
          analysis.metrics.distanceStd.toFixed(6),
          analysis.metrics.classPurity.toFixed(6),
          analysis.metrics.graphOverlap.toFixed(6),
          analysis.metrics.expectedOverlapCount.toFixed(6),
          analysis.metrics.overlapEnrichment.toFixed(6),
          analysis.metrics.silhouetteLocal.toFixed(6),
          analysis.metrics.meanGraphRank.toFixed(6),
        ];
      })
      .filter(Boolean);

    const csv = [header, ...rows]
      .map((row) => row.map(csvEscape).join(','))
      .join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'embedding-regimen-summary.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const containerClass = isExpanded
    ? 'absolute inset-x-3 top-16 bottom-3 z-30 pointer-events-none'
    : 'absolute right-3 top-16 z-20 pointer-events-none';

  const panelClass = isExpanded
    ? 'h-full w-full max-w-none'
    : 'w-[min(420px,calc(100vw-1.5rem))] max-h-[calc(100vh-8rem)]';

  if (viewMode !== 'embedding') return null;

  return (
    <div className={containerClass}>
      <div className={`${panelClass} bg-black/80 backdrop-blur-md border border-cyan-500/20 rounded-lg shadow-2xl overflow-hidden pointer-events-auto flex flex-col`}>
        <div className="px-3 py-2 border-b border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-cyan-300" />
            <span className="text-[10px] uppercase tracking-wider text-cyan-200 font-mono font-bold">Embedding Insights</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={exportRegimenSummaryCsv}
              disabled={regimenNodes.length === 0}
              className="flex items-center gap-1 px-2 py-1 rounded text-[8px] font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/25 hover:bg-cyan-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
              title="Export regimen-level metrics"
            >
              <Download className="w-3 h-3" />
              Summary
            </button>
            <button
              onClick={exportNearestCsv}
              disabled={!activeAnalysis || activeAnalysis.rows.length === 0}
              className="flex items-center gap-1 px-2 py-1 rounded text-[8px] font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/25 hover:bg-cyan-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
              title="Export nearest-neighbor distances"
            >
              <Download className="w-3 h-3" />
              Nearest
            </button>
            <button
              onClick={() => setIsExpanded((prev) => !prev)}
              className="p-1.5 rounded border border-white/15 bg-white/[0.02] text-white/70 hover:text-cyan-200 hover:border-cyan-400/30"
              title={isExpanded ? 'Minimize insights panel' : 'Maximize insights panel'}
            >
              {isExpanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        <div className={`p-3 min-h-0 flex-1 ${isExpanded ? 'overflow-hidden' : 'overflow-y-auto'}`}>
          {!activeAnalysis && (
            <div className="text-[10px] text-white/45 leading-relaxed">
              Select a drug sphere to inspect local manifold metrics and nearest-neighbor evidence.
            </div>
          )}

          {activeAnalysis && (
            <div className={`grid gap-3 min-h-0 ${isExpanded ? 'md:grid-cols-[360px_1fr] h-full' : 'grid-cols-1'}`}>
              <div className={`${isExpanded ? 'space-y-3 min-h-0 overflow-y-auto pr-1' : 'space-y-3'}`}>
                <div className="border border-white/10 rounded-md p-2.5 bg-white/[0.02]">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <div>
                      <div className="text-[11px] text-white/85 font-semibold truncate">{activeAnalysis.anchor.name}</div>
                      <div className="text-[8px] text-white/35 font-mono uppercase">
                        Edge model: {(filters.embeddingEdgeMode || 'knn').toUpperCase()} {(filters.embeddingEdgeMode || 'knn') === 'knn' ? `(k=${kValue})` : ''}
                      </div>
                    </div>
                    <BarChart3 className="w-4 h-4 text-cyan-300" />
                  </div>

                  <div className="relative mb-2">
                    <Search className="w-3 h-3 text-white/35 absolute left-2 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Analyze any drug..."
                      className="w-full pl-7 pr-2 py-1.5 rounded bg-black/45 border border-white/15 text-[8px] text-white/80 placeholder:text-white/30 font-mono outline-none focus:border-cyan-500/40"
                    />
                    {searchResults.length > 0 && (
                      <div className="absolute left-0 right-0 mt-1 rounded border border-white/15 bg-black/90 shadow-xl max-h-[140px] overflow-y-auto scrollbar-thin z-40">
                        {searchResults.map((node) => (
                          <button
                            key={`search-${node.id}`}
                            onClick={() => {
                              setActiveNodeId(node.id);
                              setSearchQuery('');
                            }}
                            className="w-full text-left px-2 py-1.5 text-[8px] text-white/75 hover:bg-white/[0.08] font-mono truncate"
                            title={node.name}
                          >
                            {node.name}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="flex flex-wrap gap-1">
                    {regimenNodes.map((node) => {
                      const active = node.id === activeAnalysis.anchor.id;
                      const role = node.isA ? 'A' : node.isB ? 'B' : node.isSelected ? 'S' : 'C';
                      return (
                        <button
                          key={node.id}
                          onClick={() => setActiveNodeId(node.id)}
                          className={`px-2 py-1 rounded text-[8px] font-mono border transition-colors ${
                            active
                              ? 'bg-cyan-500/15 text-cyan-200 border-cyan-500/40'
                              : 'bg-white/[0.02] text-white/65 border-white/15 hover:bg-white/[0.05]'
                          }`}
                          title={node.name}
                        >
                          {role}:{node.name}
                        </button>
                      );
                    })}
                  </div>

                  <div className="grid grid-cols-2 gap-1.5 pt-2">
                    <MetricChip icon={LinkGlyph} label="Degree" value={`${activeAnalysis.metrics.degree} (${(activeAnalysis.metrics.degreePercentile * 100).toFixed(1)}p)`} />
                    <MetricChip icon={Ruler} label={`Mean ${analysisK}NN Dist`} value={activeAnalysis.metrics.meanDistance.toFixed(4)} />
                    <MetricChip icon={Sigma} label={`Purity@${analysisK}`} value={`${(activeAnalysis.metrics.classPurity * 100).toFixed(1)}%`} />
                    <MetricChip icon={Activity} label={`Overlap@${analysisK}`} value={`${(activeAnalysis.metrics.graphOverlap * 100).toFixed(1)}%`} />
                    <MetricChip icon={Activity} label="Overlap Enrichment" value={activeAnalysis.metrics.overlapEnrichment.toFixed(2)} />
                    <MetricChip icon={Ruler} label="Silhouette Local" value={activeAnalysis.metrics.silhouetteLocal.toFixed(3)} />
                    <MetricChip icon={Ruler} label="Density rho" value={activeAnalysis.metrics.localDensity.toFixed(3)} />
                    <MetricChip icon={Ruler} label="Mean Graph Rank" value={activeAnalysis.metrics.meanGraphRank.toFixed(1)} />
                  </div>
                </div>

                <div className="border border-white/10 rounded-md p-2.5 bg-white/[0.02]">
                  <div className="text-[8px] text-white/35 uppercase tracking-wider font-mono mb-1.5">Research Formulas</div>
                  <div className="space-y-1 text-[8px] text-white/60 font-mono leading-relaxed">
                    <div>rho_k(x) = 1 / mean distance to k nearest neighbors</div>
                    <div>purity@k = same-class neighbors in kNN divided by k</div>
                    <div>overlap@k = latent kNN neighbors shared with graph neighbors, divided by k</div>
                    <div>enrichment = observed_overlap / expected_overlap_random</div>
                    <div>s_local = (b - a) / max(a, b)</div>
                    <div className="text-white/40 pt-1">
                      where a is mean same-class distance and b is mean different-class distance in local neighborhood.
                    </div>
                    <div className="text-white/45 pt-1">
                      Interpretation: high purity with positive silhouette indicates coherent pharmacologic neighborhoods, while high overlap and enrichment indicate agreement between latent geometry and known interaction topology.
                    </div>
                  </div>
                </div>

                <div className="border border-white/10 rounded-md p-2.5 bg-white/[0.02]">
                  <div className="text-[8px] text-white/35 uppercase tracking-wider font-mono mb-1.5">Regimen Comparison</div>
                  <div className="space-y-1">
                    {regimenNodes.map((node) => {
                      const analysis = analysisById[node.id];
                      if (!analysis) return null;

                      return (
                        <div key={`summary-${node.id}`} className="grid grid-cols-[1fr_56px_56px_56px] gap-1 items-center text-[8px] font-mono px-1.5 py-1 rounded bg-white/[0.02]">
                          <button
                            onClick={() => setActiveNodeId(node.id)}
                            className="text-left text-white/75 truncate hover:text-cyan-200"
                            title={node.name}
                          >
                            {node.name}
                          </button>
                          <span className="text-right text-cyan-300">{analysis.metrics.meanDistance.toFixed(3)}</span>
                          <span className="text-right text-emerald-300">{(analysis.metrics.classPurity * 100).toFixed(0)}%</span>
                          <span className="text-right text-purple-300">{analysis.metrics.silhouetteLocal.toFixed(2)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div className="border border-white/10 rounded-md p-2.5 bg-white/[0.02] min-h-0 flex flex-col">
                <div className="text-[8px] text-white/35 uppercase tracking-wider font-mono mb-1.5">
                  Nearest Drugs ({nearestRows.length} shown, k={analysisK})
                </div>
                <div className={`${isExpanded ? 'min-h-0 flex-1' : 'max-h-[220px]'} overflow-y-auto scrollbar-thin`}>
                  <div className="grid grid-cols-[24px_1fr_62px_62px_68px] gap-1 px-1.5 py-1 text-[7px] text-white/35 uppercase tracking-wider font-mono border-b border-white/10 sticky top-0 bg-black/85 z-10">
                    <span>Rank</span>
                    <span>Drug</span>
                    <span className="text-right">Dist</span>
                    <span className="text-right">Type</span>
                    <span className="text-right">Severity</span>
                  </div>
                  <div className="space-y-0.5 mt-1">
                    {nearestRows.map((row, index) => (
                      <div key={`${row.id}-${index}`} className="grid grid-cols-[24px_1fr_62px_62px_68px] gap-1 px-1.5 py-1 rounded text-[8px] font-mono bg-white/[0.02] hover:bg-white/[0.05]">
                        <span className="text-white/45">{index + 1}</span>
                        <span className="text-white/80 truncate" title={`${row.name} (${row.category})`}>{row.name}</span>
                        <span className="text-right text-cyan-300">{row.distance.toFixed(4)}</span>
                        <span className="text-right" style={{ color: row.isDirectGraphNeighbor ? 'rgba(34,197,94,0.95)' : 'rgba(148,163,184,0.85)' }}>
                          {row.isDirectGraphNeighbor ? 'graph' : 'latent'}
                        </span>
                        <span className="text-right text-white/60 truncate">{row.severity}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LinkGlyph() {
  return <span className="text-[9px]">#</span>;
}

function MetricChip({ icon: Icon, label, value }) {
  return (
    <div className="rounded border border-white/10 bg-white/[0.03] px-2 py-1.5">
      <div className="flex items-center gap-1 text-white/35">
        {typeof Icon === 'function' ? <Icon className="w-3 h-3" /> : null}
        <span className="text-[7px] font-mono uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-[10px] text-white/85 font-mono mt-0.5">{value}</div>
    </div>
  );
}
