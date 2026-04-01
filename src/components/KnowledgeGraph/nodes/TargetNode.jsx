// TargetNode.jsx — Hexagonal protein target node, fits inside grid cell
import React from 'react';
import { TARGET_ACTION_COLORS } from '../mechanismGraphEngine';

function hexPoints(r) {
  return Array.from({ length: 6 }, (_, i) => {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    return `${r * Math.cos(angle)},${r * Math.sin(angle)}`;
  }).join(' ');
}

export default function TargetNode({ node, x, y, isSelected, onClick, onHover, cellSize }) {
  const r = (cellSize || 50) * 0.48;
  const action = node.action || 'unknown';
  const color = TARGET_ACTION_COLORS[action] || TARGET_ACTION_COLORS.unknown;
  const isConflict = node.isConflict;

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onClick={() => onClick?.(node)}
      onMouseEnter={() => onHover?.(node)}
      onMouseLeave={() => onHover?.(null)}
      className="cursor-pointer"
    >
      {isConflict && (
        <polygon points={hexPoints(r + 4)} fill="none" stroke="#ef4444" strokeWidth={2}>
          <animate attributeName="opacity" values="0.2;0.7;0.2" dur="1.5s" repeatCount="indefinite" />
        </polygon>
      )}

      <polygon
        points={hexPoints(r)}
        fill={`${color}12`}
        stroke={color}
        strokeWidth={isConflict ? 2 : 1}
      />

      <text
        x={0}
        y={-2}
        textAnchor="middle"
        dominantBaseline="central"
        fill={color}
        fontSize={Math.max(6, r * 0.35)}
        fontWeight="bold"
        fontFamily="monospace"
      >
        {(node.gene || node.label || '').slice(0, 8)}
      </text>

      <text
        x={0}
        y={r * 0.4}
        textAnchor="middle"
        fill="rgba(255,255,255,0.25)"
        fontSize={Math.max(5, r * 0.25)}
        fontFamily="monospace"
      >
        {action}
      </text>
    </g>
  );
}
