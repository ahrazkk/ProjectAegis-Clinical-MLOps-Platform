// NodeDetailPanel.jsx — Drug info panel on click
import React from 'react';
import { X, Link2, Layers, Tag } from 'lucide-react';
import { useGalaxy, useGalaxyDispatch } from '../store';
import { rawGnnData } from '../graphEngine';

export default function NodeDetailPanel() {
  const { selectedNode, drugA, drugB } = useGalaxy();
  const dispatch = useGalaxyDispatch();

  if (!selectedNode) return null;

  const adj = rawGnnData.adj || {};
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

  return (
    <div className="absolute top-12 right-3 w-56 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg shadow-2xl overflow-hidden z-30 animate-[slideInRight_0.3s_ease]">
      {/* Header */}
      <div className="px-3 py-2.5 border-b border-white/5 flex items-start justify-between">
        <div>
          <h3 className="text-[12px] font-bold text-white tracking-wide">{selectedNode.name}</h3>
          <p className="text-[9px] text-white/30 font-mono mt-0.5">{selectedNode.category}</p>
        </div>
        <button onClick={close} className="text-white/30 hover:text-white/60 transition-colors">
          <X className="w-3.5 h-3.5" />
        </button>
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
                const neighbor = rawGnnData.nodes.find(n => n.id === nId);
                return neighbor ? (
                  <div key={nId} className="text-[8px] text-white/40 hover:text-white/60 cursor-pointer truncate font-mono">
                    · {neighbor.name}
                  </div>
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
  );
}
