// NodeDetailPanel.jsx — Enriched drug info panel on click
import React from 'react';
import { X, Link2, Layers, Tag, Beaker, AlertCircle, Activity } from 'lucide-react';
import { useGalaxy, useGalaxyDispatch } from '../store';
import { getAdj, getGraphData } from '../graphEngine';

export default function NodeDetailPanel() {
  const { selectedNode, drugA, drugB, enrichedDrugA, enrichedDrugB, enrichmentLoading, predictionResult } = useGalaxy();
  const dispatch = useGalaxyDispatch();

  if (!selectedNode) return null;

  const adj = getAdj();
  const neighbors = adj[selectedNode.id] || [];
  const degree = neighbors.length;

  const hopInfo = [];
  if (selectedNode.hopA < Infinity) hopInfo.push({ label: 'Hops from A', value: selectedNode.hopA, color: '#00d2ff' });
  if (selectedNode.hopB < Infinity) hopInfo.push({ label: 'Hops from B', value: selectedNode.hopB, color: '#ff8c00' });

  const close = () => dispatch({ type: 'SET_SELECTED_NODE', payload: null });

  const selectAsDrug = (slot) => {
    dispatch({ type: slot === 'A' ? 'SELECT_DRUG_A' : 'SELECT_DRUG_B', payload: selectedNode });
    close();
  };

  // Check if this node has enriched data
  const isNodeDrugA = selectedNode.id === drugA?.id;
  const isNodeDrugB = selectedNode.id === drugB?.id;
  const enrichedData = isNodeDrugA ? enrichedDrugA : isNodeDrugB ? enrichedDrugB : null;
  const isLoading = isNodeDrugA ? enrichmentLoading.drugA : isNodeDrugB ? enrichmentLoading.drugB : false;

  // Severity from prediction result
  const severity = predictionResult?.severity;
  const severityColors = {
    minor: '#22c55e',
    moderate: '#eab308',
    major: '#f97316',
    critical: '#ef4444',
  };

  return (
    <div className="absolute top-12 right-3 pointer-events-none z-30">
    <div className="w-60 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg shadow-2xl overflow-hidden animate-[slideInRight_0.3s_ease] pointer-events-auto">
      {/* Header */}
      <div className="px-3 py-2.5 border-b border-white/5 flex items-start justify-between">
        <div>
          <h3 className="text-[12px] font-bold text-white tracking-wide">{selectedNode.name}</h3>
          <p className="text-[9px] text-white/30 font-mono mt-0.5">{selectedNode.category}</p>
        </div>
        <div className="flex items-center gap-1.5">
          {severity && (isNodeDrugA || isNodeDrugB) && (
            <span
              className="text-[7px] font-mono uppercase px-1.5 py-0.5 rounded"
              style={{
                color: severityColors[severity] || '#fff',
                background: `${severityColors[severity] || '#fff'}15`,
              }}
            >
              {severity}
            </span>
          )}
          <button onClick={close} className="text-white/30 hover:text-white/60 transition-colors">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Info */}
      <div className="px-3 py-2 space-y-2">
        {/* Type */}
        <div className="flex items-center gap-2">
          <Tag className="w-3 h-3 text-purple-400" />
          <span className="text-[9px] text-white/40">Type:</span>
          <span className="text-[9px] text-white/70">{selectedNode.type || 'Unknown'}</span>
        </div>

        {/* Degree */}
        <div className="flex items-center gap-2">
          <Link2 className="w-3 h-3 text-purple-400" />
          <span className="text-[9px] text-white/40">Connections:</span>
          <span className="text-[9px] text-white/70 font-mono">{degree}</span>
        </div>

        {/* Hop distances */}
        {hopInfo.map((h, i) => (
          <div key={i} className="flex items-center gap-2">
            <Layers className="w-3 h-3" style={{ color: h.color }} />
            <span className="text-[9px] text-white/40">{h.label}:</span>
            <span className="text-[9px] font-mono font-bold" style={{ color: h.color }}>{h.value}</span>
          </div>
        ))}

        {/* Enriched Data — CYP Enzymes */}
        {isLoading && (
          <div className="border-t border-white/5 pt-2 space-y-1.5 animate-pulse">
            <div className="h-3 bg-white/5 rounded w-3/4" />
            <div className="h-3 bg-white/5 rounded w-1/2" />
          </div>
        )}

        {enrichedData && !isLoading && (
          <>
            {/* CYP Metabolism */}
            {enrichedData.cyp_metabolism && (
              <div className="border-t border-white/5 pt-2">
                <div className="flex items-center gap-1.5 mb-1">
                  <Beaker className="w-3 h-3 text-cyan-400" />
                  <span className="text-[8px] text-white/40 uppercase tracking-wider">CYP Enzymes</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {(enrichedData.cyp_metabolism.substrates || []).slice(0, 4).map((cyp, i) => (
                    <span key={i} className="text-[7px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                      {cyp}
                    </span>
                  ))}
                  {(enrichedData.cyp_metabolism.inhibitors || []).slice(0, 3).map((cyp, i) => (
                    <span key={`inh-${i}`} className="text-[7px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20 font-mono">
                      {cyp} (inh)
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Side Effects */}
            {enrichedData.side_effects && enrichedData.side_effects.length > 0 && (
              <div className="border-t border-white/5 pt-2">
                <div className="flex items-center gap-1.5 mb-1">
                  <AlertCircle className="w-3 h-3 text-yellow-400" />
                  <span className="text-[8px] text-white/40 uppercase tracking-wider">Side Effects</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {enrichedData.side_effects.slice(0, 5).map((se, i) => (
                    <span key={i} className="text-[7px] px-1.5 py-0.5 rounded bg-white/5 text-white/40 font-mono">
                      {typeof se === 'string' ? se : se.name || se.effect || 'Unknown'}
                    </span>
                  ))}
                  {enrichedData.side_effects.length > 5 && (
                    <span className="text-[7px] text-white/20 italic">
                      +{enrichedData.side_effects.length - 5} more
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* FAERS Data */}
            {enrichedData.faers_data && enrichedData.faers_data.total_reports > 0 && (
              <div className="border-t border-white/5 pt-2">
                <div className="flex items-center gap-1.5 mb-1">
                  <Activity className="w-3 h-3 text-pink-400" />
                  <span className="text-[8px] text-white/40 uppercase tracking-wider">
                    FAERS Reports: {enrichedData.faers_data.total_reports.toLocaleString()}
                  </span>
                </div>
              </div>
            )}
          </>
        )}

        {/* Actions */}
        <div className="flex gap-1.5 pt-1 border-t border-white/5">
          <button
            onClick={() => selectAsDrug('A')}
            className="flex-1 bg-[#00d2ff]/10 border border-[#00d2ff]/20 text-[#00d2ff] text-[8px] py-1 rounded hover:bg-[#00d2ff]/20 transition-colors font-mono"
          >
            Set as Drug A
          </button>
          <button
            onClick={() => selectAsDrug('B')}
            className="flex-1 bg-[#ff8c00]/10 border border-[#ff8c00]/20 text-[#ff8c00] text-[8px] py-1 rounded hover:bg-[#ff8c00]/20 transition-colors font-mono"
          >
            Set as Drug B
          </button>
        </div>

        {/* Neighbor preview */}
        {neighbors.length > 0 && (
          <div className="pt-1 border-t border-white/5">
            <p className="text-[8px] text-white/25 uppercase tracking-wider mb-1">Neighbors ({Math.min(neighbors.length, 6)} of {neighbors.length})</p>
            <div className="space-y-0.5 max-h-24 overflow-y-auto scrollbar-thin">
              {neighbors.slice(0, 6).map(nId => {
                const neighbor = getGraphData().nodes.find(n => n.id === nId);
                return neighbor ? (
                  <button
                    key={nId}
                    onClick={() => {
                      const fullNeighbor = getGraphData().nodes.find(n => n.id === nId);
                      if (fullNeighbor) dispatch({ type: 'SET_SELECTED_NODE', payload: { ...fullNeighbor, category: selectedNode.category } });
                    }}
                    className="block w-full text-left text-[8px] text-white/40 hover:text-white/60 cursor-pointer truncate font-mono"
                  >
                    · {neighbor.name}
                  </button>
                ) : null;
              })}
              {neighbors.length > 6 && (
                <div className="text-[7px] text-white/20 italic">+{neighbors.length - 6} more</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
    </div>
  );
}
