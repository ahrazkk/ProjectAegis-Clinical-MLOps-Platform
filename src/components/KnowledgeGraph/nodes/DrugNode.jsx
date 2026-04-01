// DrugNode.jsx — Hexagonal drug node, sized to fit inside a hex grid cell
import React from 'react';

function hexPoints(r) {
  return Array.from({ length: 6 }, (_, i) => {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    return `${r * Math.cos(angle)},${r * Math.sin(angle)}`;
  }).join(' ');
}

export default function DrugNode({ node, x, y, isSelected, onClick, onHover, cellSize }) {
  // Node hex is ~70% of the cell size so it sits clearly inside the grid cell
  const r = (cellSize || 50) * 0.68;
  const color = node.color || '#00d2ff';
  const isConflict = node.isConflict;

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onClick={() => onClick?.(node)}
      onMouseEnter={() => onHover?.(node)}
      onMouseLeave={() => onHover?.(null)}
      className="cursor-pointer"
    >
      {(isConflict || isSelected) && (
        <polygon
          points={hexPoints(r + 3)}
          fill="none"
          stroke={isConflict ? '#ef4444' : color}
          strokeWidth={2}
          opacity={0.4}
        >
          <animate attributeName="opacity" values="0.2;0.6;0.2" dur="2s" repeatCount="indefinite" />
        </polygon>
      )}

      <polygon
        points={hexPoints(r)}
        fill={`${color}15`}
        stroke={color}
        strokeWidth={1.5}
      />

      <text
        x={0}
        y={-2}
        textAnchor="middle"
        dominantBaseline="central"
        fill={color}
        fontSize={Math.max(8, r * 0.3)}
        fontWeight="bold"
        fontFamily="monospace"
      >
        {node.label?.length > 12 ? node.label.slice(0, 11) + '…' : node.label}
      </text>

      <text
        x={0}
        y={r * 0.38}
        textAnchor="middle"
        fill="rgba(255,255,255,0.3)"
        fontSize={Math.max(6, r * 0.2)}
        fontFamily="monospace"
      >
        {node.therapeutic_class || (node.slot === 'A' ? 'Drug A' : 'Drug B')}
      </text>
    </g>
  );
}
