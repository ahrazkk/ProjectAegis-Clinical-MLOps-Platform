// CirculatoryOverlay.jsx — Animated circulatory system SVG paths
// Major arteries (red) and veins (blue) with animated dash flow.

import React from 'react';
import { CIRCULATORY } from '../organRegistry';

const ARTERY_PATHS = [
  { key: 'aorta', d: CIRCULATORY.aorta, label: 'Aorta' },
  { key: 'renalLeft', d: CIRCULATORY.renalLeft },
  { key: 'renalRight', d: CIRCULATORY.renalRight },
  { key: 'carotidLeft', d: CIRCULATORY.carotidLeft },
  { key: 'carotidRight', d: CIRCULATORY.carotidRight },
  { key: 'pulmonaryLeft', d: CIRCULATORY.pulmonaryLeft },
  { key: 'pulmonaryRight', d: CIRCULATORY.pulmonaryRight },
];

const VEIN_PATHS = [
  { key: 'venaCava', d: CIRCULATORY.venaCava, label: 'Vena Cava' },
  { key: 'hepaticPortal', d: CIRCULATORY.hepaticPortal, label: 'Hepatic Portal' },
];

export default function CirculatoryOverlay({ visible = true, activeOrgans = [] }) {
  if (!visible) return null;

  const activeSet = new Set(activeOrgans.map(o => typeof o === 'string' ? o : o.id));

  // Highlight paths that connect to active organs
  const isPathActive = (key) => {
    if (key.includes('renal')) return activeSet.has('kidney');
    if (key.includes('carotid')) return activeSet.has('brain');
    if (key.includes('pulmonary')) return activeSet.has('lungs');
    if (key.includes('hepatic')) return activeSet.has('liver') || activeSet.has('gi');
    return activeSet.size > 0; // Aorta/vena cava active if anything is
  };

  return (
    <g className="circulatory-overlay">
      {/* Arteries — red, animated dash suggesting blood flow */}
      {ARTERY_PATHS.map(({ key, d }) => {
        const active = isPathActive(key);
        return (
          <path
            key={key}
            d={d}
            fill="none"
            stroke={active ? 'rgba(239, 68, 68, 0.5)' : 'rgba(239, 68, 68, 0.15)'}
            strokeWidth={key === 'aorta' ? 2 : 1.2}
            strokeLinecap="round"
            strokeDasharray="6,4"
            style={{
              transition: 'stroke 0.5s',
            }}
          >
            <animate
              attributeName="stroke-dashoffset"
              values="0;-20"
              dur={active ? '0.8s' : '2s'}
              repeatCount="indefinite"
            />
          </path>
        );
      })}

      {/* Veins — blue, opposite flow direction */}
      {VEIN_PATHS.map(({ key, d }) => {
        const active = isPathActive(key);
        return (
          <path
            key={key}
            d={d}
            fill="none"
            stroke={active ? 'rgba(59, 130, 246, 0.5)' : 'rgba(59, 130, 246, 0.15)'}
            strokeWidth={key === 'venaCava' ? 2 : 1.2}
            strokeLinecap="round"
            strokeDasharray="6,4"
            style={{
              transition: 'stroke 0.5s',
            }}
          >
            <animate
              attributeName="stroke-dashoffset"
              values="0;20"
              dur={active ? '0.8s' : '2s'}
              repeatCount="indefinite"
            />
          </path>
        );
      })}
    </g>
  );
}
