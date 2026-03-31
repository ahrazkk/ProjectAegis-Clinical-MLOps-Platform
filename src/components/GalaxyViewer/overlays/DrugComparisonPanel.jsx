// DrugComparisonPanel.jsx — Side-by-side comparison when 2 drugs selected
import React from 'react';
import { ArrowRight, GitBranch, Layers, Link2 } from 'lucide-react';
import { useGalaxy } from '../store';
import { getAdj } from '../graphEngine';

export default function DrugComparisonPanel() {
  const { drugA, drugB, stats, shortestPath } = useGalaxy();

  if (!drugA || !drugB) return null;

  const adj = getAdj();
  const degreeA = (adj[drugA.id] || []).length;
  const degreeB = (adj[drugB.id] || []).length;

  return (
    <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
      <div className="bg-black/75 backdrop-blur-md border border-white/8 rounded-lg shadow-2xl px-4 py-3 min-w-[320px] pointer-events-auto">
        {/* Drug comparison */}
        <div className="flex items-center gap-4">
          {/* Drug A */}
          <div className="flex-1 text-right">
            <div className="text-[11px] font-bold text-[#00d2ff] tracking-wide">{drugA.name}</div>
            <div className="text-[8px] text-white/25 font-mono mt-0.5">{drugA.category}</div>
            <div className="text-[8px] text-white/20 font-mono">{degreeA} connections</div>
          </div>

          {/* Path indicator */}
          <div className="flex flex-col items-center gap-0.5">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-[#00d2ff]" />
              <div className="flex items-center gap-0.5">
                {shortestPath.length > 0 ? (
                  <>
                    {Array.from({ length: Math.min(shortestPath.length - 2, 3) }).map((_, i) => (
                      <React.Fragment key={i}>
                        <div className="w-1 h-px bg-red-400/60" />
                        <div className="w-1 h-1 rounded-full bg-purple-400/60" />
                      </React.Fragment>
                    ))}
                    <div className="w-1 h-px bg-red-400/60" />
                  </>
                ) : (
                  <div className="w-6 h-px bg-white/10" />
                )}
              </div>
              <div className="w-2 h-2 rounded-full bg-[#ff8c00]" />
            </div>
            <span className="text-[7px] font-mono text-white/25">
              {stats.pathLength >= 0 ? `${stats.pathLength} hops` : 'no path'}
            </span>
          </div>

          {/* Drug B */}
          <div className="flex-1 text-left">
            <div className="text-[11px] font-bold text-[#ff8c00] tracking-wide">{drugB.name}</div>
            <div className="text-[8px] text-white/25 font-mono mt-0.5">{drugB.category}</div>
            <div className="text-[8px] text-white/20 font-mono">{degreeB} connections</div>
          </div>
        </div>

        {/* Stats row */}
        <div className="flex justify-center gap-6 mt-2 pt-2 border-t border-white/5">
          <StatChip icon={<GitBranch className="w-3 h-3" />} label="Bridges" value={stats.sharedNeighbors} color="#a855f7" />
          <StatChip icon={<Layers className="w-3 h-3" />} label="Path" value={stats.pathLength >= 0 ? stats.pathLength : '∞'} color="#ef4444" />
          <StatChip icon={<Link2 className="w-3 h-3" />} label="Active Edges" value={stats.visibleEdges} color="#8B5CF6" />
        </div>
      </div>
    </div>
  );
}

function StatChip({ icon, label, value, color }) {
  return (
    <div className="flex items-center gap-1.5">
      <span style={{ color }} className="opacity-60">{icon}</span>
      <div>
        <div className="text-[10px] font-mono font-bold" style={{ color }}>{value}</div>
        <div className="text-[7px] text-white/20 uppercase">{label}</div>
      </div>
    </div>
  );
}
