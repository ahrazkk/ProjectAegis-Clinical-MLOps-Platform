// Toolbar.jsx — View mode switcher, toggles, export, reset
import React, { useRef } from 'react';
import {
  Globe, Atom, Target, Layers, Route, SlidersHorizontal,
  Tag, GitBranch, Camera, RotateCcw, Maximize2,
} from 'lucide-react';
import { useGalaxy, useGalaxyDispatch } from '../store';

const viewModes = [
  { id: 'galaxy', label: 'Galaxy', icon: Globe, tip: 'T-SNE embedding space' },
  { id: 'embedding', label: 'Embedding', icon: Atom, tip: 'All-drug latent space (full atlas)' },
  { id: 'radial', label: 'Radial', icon: Target, tip: 'Concentric hop rings' },
  { id: 'cluster', label: 'Cluster', icon: Layers, tip: 'Therapeutic class groups' },
  { id: 'path', label: 'Path', icon: Route, tip: 'Shortest path subway map' },
  { id: 'focus', label: 'Focus', icon: GitBranch, tip: 'Least-hop connector across selected drugs' },
];

export default function Toolbar({ onToggleFilters, showFilters, canvasRef }) {
  const { viewMode, showLabels, showEdges, drugA, drugB, shortestPath } = useGalaxy();
  const dispatch = useGalaxyDispatch();

  const handleScreenshot = () => {
    if (!canvasRef?.current) return;
    const canvas = canvasRef.current.querySelector('canvas');
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `galaxy-viewer-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const handleReset = () => {
    dispatch({ type: 'CLEAR_SELECTION' });
    dispatch({ type: 'SET_VIEW_MODE', payload: 'galaxy' });
  };

  const handleFullscreen = () => {
    const el = canvasRef?.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      el.requestFullscreen?.();
    }
  };

  const canUsePath = shortestPath && shortestPath.length >= 2;
  const canUseFocus = Boolean(drugA || drugB);

  return (
    <div className="absolute top-2 left-2 right-2 z-20 pointer-events-none flex items-center gap-1 flex-wrap">
      {/* View mode buttons */}
      <div className="flex items-center bg-black/60 backdrop-blur-md border border-white/5 rounded-lg overflow-hidden pointer-events-auto">
        {viewModes.map(mode => {
          const Icon = mode.icon;
          const isActive = viewMode === mode.id;
          const isDisabled =
            (mode.id === 'path' && !canUsePath) ||
            (mode.id === 'focus' && !canUseFocus);

          return (
            <button
              key={mode.id}
              onClick={() => !isDisabled && dispatch({ type: 'SET_VIEW_MODE', payload: mode.id })}
              disabled={isDisabled}
              title={mode.tip}
              className={`
                flex items-center gap-1.5 px-2.5 py-1.5 text-[9px] font-mono uppercase tracking-wider transition-all
                ${isActive
                  ? 'bg-purple-500/20 text-purple-300 border-b-2 border-purple-400'
                  : isDisabled
                    ? 'text-white/10 cursor-not-allowed'
                    : 'text-white/35 hover:text-white/60 hover:bg-white/5'
                }
              `}
            >
              <Icon className="w-3 h-3" />
              <span className="hidden sm:inline">{mode.label}</span>
            </button>
          );
        })}
      </div>

      {/* Separator */}
      <div className="w-px h-5 bg-white/5 mx-0.5" />

      {/* Filter toggle */}
      <button
        onClick={onToggleFilters}
        className={`
          pointer-events-auto flex items-center gap-1 px-2 py-1.5 rounded-lg text-[9px] font-mono uppercase tracking-wider transition-all
          border backdrop-blur-md
          ${showFilters
            ? 'bg-purple-500/15 border-purple-500/30 text-purple-300'
            : 'bg-black/60 border-white/5 text-white/35 hover:text-white/60'
          }
        `}
      >
        <SlidersHorizontal className="w-3 h-3" />
        <span className="hidden sm:inline">Filters</span>
      </button>

      {/* Toggles */}
      <button
        onClick={() => dispatch({ type: 'TOGGLE_LABELS' })}
        className={`
          pointer-events-auto flex items-center gap-1 px-2 py-1.5 rounded-lg text-[9px] font-mono uppercase tracking-wider transition-all border backdrop-blur-md
          ${showLabels !== 'none'
            ? 'bg-black/60 border-white/10 text-white/50'
            : 'bg-black/60 border-white/5 text-white/20'
          }
        `}
        title={`Toggle labels (current: ${showLabels})`}
      >
        <Tag className="w-3 h-3" />
        <span className="hidden sm:inline">{showLabels === 'none' ? 'off' : showLabels}</span>
      </button>

      <button
        onClick={() => dispatch({ type: 'TOGGLE_EDGES' })}
        className={`
          pointer-events-auto flex items-center gap-1 px-2 py-1.5 rounded-lg text-[9px] font-mono transition-all border backdrop-blur-md
          ${showEdges
            ? 'bg-black/60 border-white/10 text-white/50'
            : 'bg-black/60 border-white/5 text-white/20'
          }
        `}
        title="Toggle edges"
      >
        <GitBranch className="w-3 h-3" />
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Right side actions */}
      <div className="flex items-center gap-1 pointer-events-auto">
        <button
          onClick={handleScreenshot}
          className="p-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/5 text-white/30 hover:text-white/60 transition-colors"
          title="Screenshot"
        >
          <Camera className="w-3 h-3" />
        </button>
        <button
          onClick={handleReset}
          className="p-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/5 text-white/30 hover:text-white/60 transition-colors"
          title="Reset"
        >
          <RotateCcw className="w-3 h-3" />
        </button>
        <button
          onClick={handleFullscreen}
          className="p-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/5 text-white/30 hover:text-white/60 transition-colors"
          title="Fullscreen"
        >
          <Maximize2 className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}
