// HopSlider.jsx — 1-3 hop distance slider control
import React from 'react';
import { useGalaxy, useGalaxyDispatch } from '../store';

export default function HopSlider() {
  const { maxHops, drugA, drugB } = useGalaxy();
  const dispatch = useGalaxyDispatch();

  // Only show when drugs are selected
  if (!drugA && !drugB) return null;

  const hops = [1, 2, 3];

  return (
    <div className="absolute bottom-16 left-3 pointer-events-none">
      <div className="bg-black/70 backdrop-blur-md border border-white/5 rounded-lg px-3 py-2 shadow-xl pointer-events-auto">
        <div className="text-[8px] text-white/30 font-mono uppercase tracking-wider mb-2">Hop Depth</div>

        <div className="flex items-center gap-1">
          {hops.map(h => (
            <button
              key={h}
              onClick={() => dispatch({ type: 'SET_MAX_HOPS', payload: h })}
              className={`
                w-8 h-8 rounded flex items-center justify-center text-[11px] font-mono font-bold transition-all duration-200
                ${maxHops === h
                  ? 'bg-purple-500/20 border border-purple-500/40 text-purple-300 shadow-[0_0_10px_rgba(139,92,246,0.2)]'
                  : 'bg-white/3 border border-white/5 text-white/25 hover:text-white/40 hover:bg-white/5'
                }
              `}
            >
              {h}
            </button>
          ))}
        </div>

        {/* Hop description */}
        <div className="text-[7px] text-white/20 font-mono mt-1.5">
          {maxHops === 1 && 'Direct interaction partners'}
          {maxHops === 2 && 'Partners of partners'}
          {maxHops === 3 && 'Full GNN receptive field'}
        </div>
      </div>
    </div>
  );
}
