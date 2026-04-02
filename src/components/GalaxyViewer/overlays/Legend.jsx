// Legend.jsx — Interactive legend with GNN hop explanation
import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Info } from 'lucide-react';
import { useGalaxy } from '../store';

export default function Legend({ isMobile }) {
  const [expanded, setExpanded] = useState(!isMobile);
  const { viewMode } = useGalaxy();

  return (
    <div className="absolute bottom-3 left-3 pointer-events-none">
      <div className="bg-black/70 border border-white/5 backdrop-blur-md rounded-lg shadow-xl overflow-hidden pointer-events-auto" style={{ maxWidth: isMobile ? '200px' : '240px' }}>
        {/* Header */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center justify-between px-3 py-2 text-white/60 hover:text-white/80 transition-colors"
        >
          <span className="text-[10px] font-bold tracking-wider uppercase font-mono">GNN Latent Space</span>
          {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
        </button>

        {expanded && (
          <div className="px-3 pb-3 space-y-2">
            {/* Color legend */}
            <div className="space-y-1.5">
              <LegendItem color="#00d2ff" glow label="Source Drug (A)" sub="Hub node" />
              <LegendItem color="#ff8c00" glow label="Target Drug (B)" sub="Hub node" />
              <LegendItem color="#22c55e" glow label="Selected Regimen" sub="Additional chosen drugs" />
              <LegendItem color="#a855f7" label="Bridge Nodes" sub="Shared neighbors" />
              <LegendItem color="#ef4444" isLine label="Direct DDI / Path" sub="Known interaction" />
              <LegendItem color="#475569" dim label="Background" sub="Full graph" />
            </div>

            {/* GNN explanation */}
            <div className="border-t border-white/5 pt-2 mt-2">
              <div className="flex items-start gap-1.5">
                <Info className="w-3 h-3 text-purple-400 mt-0.5 flex-shrink-0" />
                <p className="text-[8px] text-white/30 leading-relaxed">
                  {viewMode === 'embedding'
                    ? (
                      <>
                        Embedding mode shows the full latent atlas: each sphere is a drug position in model space.
                        Faint gray links show sparse global structure, while colored links highlight selected-drug relationships.
                      </>
                    )
                    : (
                      <>
                        The GraphSAGE GNN aggregates information from selected drugs out to <span className="text-purple-300">3 hops</span>.
                        In multi-drug regimens, hop distance is computed from any selected anchor to expose shared interaction neighborhoods.
                        Focus mode keeps direct selected links and prefers alternate connector routes when similar-length paths exist.
                      </>
                    )}
                </p>
              </div>
            </div>

            {/* Hop explanation */}
            <div className="space-y-1">
              <HopLabel hop={1} desc="Direct partners" />
              <HopLabel hop={2} desc="Partners of partners" />
              <HopLabel hop={3} desc="Full receptive field" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function LegendItem({ color, label, sub, glow, isLine, dim }) {
  return (
    <div className="flex items-center gap-2">
      {isLine ? (
        <div className="w-4 h-0.5 flex-shrink-0 rounded-full" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
      ) : (
        <div
          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
          style={{
            background: dim ? color : color,
            opacity: dim ? 0.3 : 1,
            boxShadow: glow ? `0 0 8px ${color}` : 'none',
          }}
        />
      )}
      <div>
        <span className="text-[9px] text-white/60">{label}</span>
        {sub && <span className="text-[7px] text-white/25 ml-1">{sub}</span>}
      </div>
    </div>
  );
}

function HopLabel({ hop, desc }) {
  const opacity = 1 - (hop - 1) * 0.25;
  return (
    <div className="flex items-center gap-2">
      <div className="w-4 flex justify-center">
        <span className="text-[8px] font-mono text-purple-400" style={{ opacity }}>{hop}H</span>
      </div>
      <span className="text-[8px] text-white/25">{desc}</span>
    </div>
  );
}
