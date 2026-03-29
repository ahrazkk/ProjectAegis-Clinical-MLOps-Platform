// HUD.jsx — Telemetry-style stats overlay matching YOLO aesthetic
import React from 'react';
import { useGalaxy } from '../store';

export default function HUD() {
  const { stats, maxHops, drugA, drugB, viewMode } = useGalaxy();

  return (
    <div className="absolute top-12 right-3 pointer-events-none flex flex-col items-end gap-1.5">
      {/* Status badge */}
      <div className="bg-black/70 border border-purple-500/30 text-purple-300 px-2.5 py-1 rounded text-[9px] uppercase tracking-[0.2em] shadow-[0_0_12px_rgba(139,92,246,0.15)] backdrop-blur-sm font-mono flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
        GraphSAGE-3L · T-SNE
      </div>

      {/* Stats grid */}
      <div className="bg-black/60 border border-white/5 backdrop-blur-sm rounded px-2.5 py-2 font-mono text-[8px] leading-relaxed min-w-[140px]">
        <div className="flex justify-between text-white/30">
          <span>NODES</span>
          <span className="text-white/60">{stats.totalNodes.toLocaleString()}</span>
        </div>
        <div className="flex justify-between text-white/30">
          <span>EDGES</span>
          <span className="text-white/60">{stats.totalEdges.toLocaleString()}</span>
        </div>
        <div className="flex justify-between text-white/30 border-t border-white/5 mt-1 pt-1">
          <span>VISIBLE</span>
          <span className="text-purple-300">{stats.visibleNodes.toLocaleString()}</span>
        </div>
        <div className="flex justify-between text-white/30">
          <span>ACT_EDGES</span>
          <span className="text-purple-300">{stats.visibleEdges.toLocaleString()}</span>
        </div>
        <div className="flex justify-between text-white/30">
          <span>HOP_DEPTH</span>
          <span className="text-cyan-300">{maxHops}</span>
        </div>
        {stats.pathLength >= 0 && (
          <div className="flex justify-between text-white/30 border-t border-white/5 mt-1 pt-1">
            <span>PATH_LEN</span>
            <span className="text-red-300">{stats.pathLength}</span>
          </div>
        )}
        {stats.sharedNeighbors > 0 && (
          <div className="flex justify-between text-white/30">
            <span>BRIDGES</span>
            <span className="text-purple-300">{stats.sharedNeighbors}</span>
          </div>
        )}
      </div>

      {/* Mode indicator */}
      <div className="bg-black/50 border border-white/5 backdrop-blur-sm rounded px-2 py-1 font-mono text-[7px] text-white/25 uppercase tracking-widest">
        {viewMode}
      </div>
    </div>
  );
}
