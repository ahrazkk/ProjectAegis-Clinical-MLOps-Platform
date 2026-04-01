// LayerToggle.jsx — Toggle circulatory/heatmap/skeleton layers
import React from 'react';
import { Heart, Thermometer, Bone, Eye } from 'lucide-react';

const LAYERS = [
  { key: 'circulatory', icon: Heart, label: 'Vessels', color: '#ef4444' },
  { key: 'heatmap', icon: Thermometer, label: 'Heat Map', color: '#f97316' },
  { key: 'skeleton', icon: Bone, label: 'Skeleton', color: '#94a3b8' },
];

export default function LayerToggle({ layers = {}, onToggle, isMobile = false }) {
  return (
    <div className={`
      absolute z-10 flex gap-1
      ${isMobile ? 'top-2 right-2' : 'top-4 right-4'}
    `}>
      {LAYERS.map(({ key, icon: Icon, label, color }) => {
        const active = layers[key] !== false; // Default to on
        return (
          <button
            key={key}
            onClick={() => onToggle?.(key)}
            className={`
              flex items-center gap-1 px-2 py-1 rounded border transition-all
              ${active
                ? 'border-white/15 bg-white/5'
                : 'border-white/5 bg-transparent opacity-40'
              }
              hover:bg-white/10
            `}
            title={`${active ? 'Hide' : 'Show'} ${label}`}
          >
            <Icon
              className="w-3 h-3"
              style={{ color: active ? color : '#475569' }}
            />
            {!isMobile && (
              <span
                className="text-[9px] font-mono uppercase tracking-wider"
                style={{ color: active ? color : '#475569' }}
              >
                {label}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
