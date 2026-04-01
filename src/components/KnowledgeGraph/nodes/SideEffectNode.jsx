// SideEffectNode.jsx — Hexagonal side effect node, fits inside grid cell
import React from 'react';
import { SEVERITY_COLORS } from '../mechanismGraphEngine';

function hexPoints(r) {
  return Array.from({ length: 6 }, (_, i) => {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    return `${r * Math.cos(angle)},${r * Math.sin(angle)}`;
  }).join(' ');
}

export default function SideEffectNode({ node, x, y, isSelected, onClick, onHover, cellSize }) {
  const r = (cellSize || 50) * 0.42;
  const severity = node.severity || 0;
  const isConflict = node.isConflict;

  let color = '#6b7280';
  if (typeof severity === 'number') {
    color = severity > 0.6 ? SEVERITY_COLORS.high : severity > 0.3 ? SEVERITY_COLORS.moderate : SEVERITY_COLORS.low;
  } else if (typeof severity === 'string') {
    color = SEVERITY_COLORS[severity] || '#6b7280';
  }

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onClick={() => onClick?.(node)}
      onMouseEnter={() => onHover?.(node)}
      onMouseLeave={() => onHover?.(null)}
      className="cursor-pointer"
    >
      {isConflict && (
        <polygon points={hexPoints(r + 3)} fill="none" stroke="#ef4444" strokeWidth={1.5}>
          <animate attributeName="opacity" values="0.2;0.7;0.2" dur="1.5s" repeatCount="indefinite" />
        </polygon>
      )}

      <polygon
        points={hexPoints(r)}
        fill={`${color}12`}
        stroke={color}
        strokeWidth={isConflict ? 1.5 : 0.8}
      />

      <text
        x={0}
        y={1}
        textAnchor="middle"
        dominantBaseline="central"
        fill={color}
        fontSize={Math.max(5, r * 0.3)}
        fontWeight="bold"
        fontFamily="monospace"
      >
        {(node.label || '').length > 10 ? node.label.slice(0, 9) + '…' : node.label}
      </text>
    </g>
  );
}
