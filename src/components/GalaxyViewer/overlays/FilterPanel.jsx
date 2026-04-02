// FilterPanel.jsx — Collapsible filter controls for the Galaxy Viewer
import React from 'react';
import { X, Filter } from 'lucide-react';
import { useGalaxy, useGalaxyDispatch } from '../store';
import { CATEGORY_COLORS, CATEGORY_LIST } from '../graphEngine';

export default function FilterPanel({ isOpen, onClose }) {
  const { filters, viewMode } = useGalaxy();
  const dispatch = useGalaxyDispatch();

  if (!isOpen) return null;

  const updateFilter = (key, value) => {
    dispatch({ type: 'SET_FILTERS', payload: { [key]: value } });
  };

  const severityLevels = ['minor', 'moderate', 'major', 'critical', 'unknown'];
  const activeSeverity = filters.severityLevels || severityLevels;

  const toggleSeverity = (level) => {
    const current = new Set(activeSeverity);
    if (current.has(level)) {
      current.delete(level);
    } else {
      current.add(level);
    }
    updateFilter('severityLevels', Array.from(current));
  };

  const visibleClasses = filters.visibleClasses; // null = all visible
  const toggleClass = (cat) => {
    let current;
    if (visibleClasses === null) {
      // Start with all enabled, remove this one
      current = new Set(CATEGORY_LIST.filter(c => c !== cat));
    } else {
      current = new Set(visibleClasses);
      if (current.has(cat)) {
        current.delete(cat);
      } else {
        current.add(cat);
      }
    }
    // If all are enabled, set back to null
    if (current.size === CATEGORY_LIST.length) {
      updateFilter('visibleClasses', null);
    } else {
      updateFilter('visibleClasses', Array.from(current));
    }
  };

  const isClassVisible = (cat) => {
    if (visibleClasses === null) return true;
    return visibleClasses.includes(cat);
  };

  const activeFilterCount = (
    (activeSeverity.length < severityLevels.length ? 1 : 0) +
    (visibleClasses !== null ? 1 : 0) +
    (filters.minDegree > 0 ? 1 : 0) +
    (filters.edgeDensity < 1.0 ? 1 : 0) +
    (filters.embeddingEdgeMode !== 'knn' ? 1 : 0) +
    (filters.embeddingK !== 8 ? 1 : 0)
  );

  return (
    <div className="absolute top-12 left-3 pointer-events-none z-20">
      <div className="w-56 bg-black/85 backdrop-blur-md border border-white/10 rounded-lg shadow-2xl overflow-hidden pointer-events-auto animate-[slideInLeft_0.3s_ease]">
        {/* Header */}
        <div className="px-3 py-2 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Filter className="w-3 h-3 text-purple-400" />
            <span className="text-[10px] font-bold text-white/70 uppercase tracking-wider font-mono">Filters</span>
            {activeFilterCount > 0 && (
              <span className="text-[8px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded-full font-mono">
                {activeFilterCount}
              </span>
            )}
          </div>
          <button onClick={onClose} className="text-white/30 hover:text-white/60 transition-colors">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="px-3 py-2.5 space-y-3 max-h-[400px] overflow-y-auto scrollbar-thin">
          {/* Severity Filter */}
          <div>
            <div className="text-[8px] text-white/30 uppercase tracking-wider font-mono mb-1.5">Severity</div>
            <div className="flex flex-wrap gap-1">
              {severityLevels.map(level => {
                const isActive = activeSeverity.includes(level);
                const colors = {
                  minor: '#22c55e',
                  moderate: '#eab308',
                  major: '#f97316',
                  critical: '#ef4444',
                  unknown: '#64748b',
                };
                return (
                  <button
                    key={level}
                    onClick={() => toggleSeverity(level)}
                    className="text-[8px] px-2 py-1 rounded font-mono uppercase transition-all"
                    style={{
                      background: isActive ? `${colors[level]}15` : 'rgba(255,255,255,0.02)',
                      color: isActive ? colors[level] : 'rgba(255,255,255,0.2)',
                      border: `1px solid ${isActive ? `${colors[level]}40` : 'rgba(255,255,255,0.05)'}`,
                    }}
                  >
                    {level}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Therapeutic Class Filter */}
          <div>
            <div className="text-[8px] text-white/30 uppercase tracking-wider font-mono mb-1.5">Drug Class</div>
            <div className="space-y-0.5 max-h-[180px] overflow-y-auto scrollbar-thin">
              {CATEGORY_LIST.map(cat => {
                const visible = isClassVisible(cat);
                return (
                  <button
                    key={cat}
                    onClick={() => toggleClass(cat)}
                    className="w-full flex items-center gap-2 px-1.5 py-1 rounded text-left transition-all hover:bg-white/5"
                  >
                    <div
                      className="w-2 h-2 rounded-full flex-shrink-0 transition-opacity"
                      style={{
                        background: CATEGORY_COLORS[cat],
                        opacity: visible ? 1 : 0.15,
                      }}
                    />
                    <span
                      className="text-[9px] font-mono transition-colors"
                      style={{ color: visible ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.15)' }}
                    >
                      {cat}
                    </span>
                  </button>
                );
              })}
            </div>
            <div className="flex gap-1 mt-1">
              <button
                onClick={() => updateFilter('visibleClasses', null)}
                className="text-[7px] text-purple-400/60 hover:text-purple-400 font-mono transition-colors"
              >
                Select All
              </button>
              <span className="text-white/10">|</span>
              <button
                onClick={() => updateFilter('visibleClasses', [])}
                className="text-[7px] text-purple-400/60 hover:text-purple-400 font-mono transition-colors"
              >
                Deselect All
              </button>
            </div>
          </div>

          {/* Minimum Degree Slider */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[8px] text-white/30 uppercase tracking-wider font-mono">Min Connections</span>
              <span className="text-[9px] text-purple-300 font-mono">{filters.minDegree || 0}</span>
            </div>
            <input
              type="range"
              min={0}
              max={50}
              step={1}
              value={filters.minDegree || 0}
              onChange={e => updateFilter('minDegree', parseInt(e.target.value))}
              className="w-full h-1 bg-white/5 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
          </div>

          {/* Edge Density Slider */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[8px] text-white/30 uppercase tracking-wider font-mono">Edge Density</span>
              <span className="text-[9px] text-purple-300 font-mono">{Math.round((filters.edgeDensity || 1) * 100)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={Math.round((filters.edgeDensity || 1) * 100)}
              onChange={e => updateFilter('edgeDensity', parseInt(e.target.value) / 100)}
              className="w-full h-1 bg-white/5 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
          </div>

          {/* Embedding mode controls */}
          {viewMode === 'embedding' && (
            <>
              <div className="border-t border-white/5 pt-2">
                <div className="text-[8px] text-white/30 uppercase tracking-wider font-mono mb-1.5">Embedding Edge Model</div>
                <div className="grid grid-cols-2 gap-1">
                  {[{ id: 'knn', label: 'KNN' }, { id: 'graph', label: 'Graph' }].map(mode => {
                    const active = (filters.embeddingEdgeMode || 'knn') === mode.id;
                    return (
                      <button
                        key={mode.id}
                        onClick={() => updateFilter('embeddingEdgeMode', mode.id)}
                        className="text-[8px] px-2 py-1 rounded font-mono uppercase transition-all"
                        style={{
                          background: active ? 'rgba(34,211,238,0.12)' : 'rgba(255,255,255,0.02)',
                          color: active ? 'rgba(103,232,249,0.95)' : 'rgba(255,255,255,0.25)',
                          border: `1px solid ${active ? 'rgba(34,211,238,0.35)' : 'rgba(255,255,255,0.06)'}`,
                        }}
                      >
                        {mode.label}
                      </button>
                    );
                  })}
                </div>
                <p className="text-[7px] text-white/25 mt-1 leading-relaxed">
                  KNN reveals geometric manifold neighborhoods. Graph uses known interaction topology.
                </p>
              </div>

              {(filters.embeddingEdgeMode || 'knn') === 'knn' && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[8px] text-white/30 uppercase tracking-wider font-mono">KNN Degree (k)</span>
                    <span className="text-[9px] text-cyan-300 font-mono">{filters.embeddingK || 8}</span>
                  </div>
                  <input
                    type="range"
                    min={2}
                    max={20}
                    step={1}
                    value={filters.embeddingK || 8}
                    onChange={e => updateFilter('embeddingK', parseInt(e.target.value, 10))}
                    className="w-full h-1 bg-white/5 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                  />
                </div>
              )}
            </>
          )}

          {/* Reset Filters */}
          {activeFilterCount > 0 && (
            <button
              onClick={() => dispatch({
                type: 'SET_FILTERS',
                payload: {
                  severityLevels,
                  visibleClasses: null,
                  minDegree: 0,
                  edgeDensity: 1.0,
                  embeddingEdgeMode: 'knn',
                  embeddingK: 8,
                },
              })}
              className="w-full text-[8px] text-center py-1.5 rounded bg-white/5 text-white/30 hover:text-white/60 hover:bg-white/8 transition-all font-mono uppercase tracking-wider"
            >
              Reset All Filters
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
