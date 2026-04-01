// EnzymeNode.jsx — Hexagonal CYP enzyme node, fits inside grid cell
import React from 'react';
import { ENZYME_ROLE_COLORS } from '../mechanismGraphEngine';

function hexPoints(r) {
  return Array.from({ length: 6 }, (_, i) => {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    return `${r * Math.cos(angle)},${r * Math.sin(angle)}`;
  }).join(' ');
}

export default function EnzymeNode({ node, x, y, isSelected, onClick, onHover, cellSize }) {
  const r = (cellSize || 50) * 0.55;
  const isConflict = node.isConflict;

  const roles = node.enzymeRoles || {};
  const allRoles = Object.values(roles);
  let borderColor = ENZYME_ROLE_COLORS.substrate;
  if (allRoles.includes('inhibitor')) borderColor = ENZYME_ROLE_COLORS.inhibitor;
  if (allRoles.includes('inducer')) borderColor = ENZYME_ROLE_COLORS.inducer;
  if (allRoles.length > 1 && new Set(allRoles).size > 1) borderColor = ENZYME_ROLE_COLORS.mixed;

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
        fill={`${borderColor}15`}
        stroke={borderColor}
        strokeWidth={isConflict ? 2 : 1.5}
      />

      <text
        x={0}
        y={1}
        textAnchor="middle"
        dominantBaseline="central"
        fill={borderColor}
        fontSize={Math.max(7, r * 0.33)}
        fontWeight="bold"
        fontFamily="monospace"
      >
        {node.label}
      </text>

      {/* Role badges below the hex */}
      {Object.entries(roles).map(([slot, role], i) => {
        const badgeY = r + 8;
        const badgeX = Object.keys(roles).length > 1 ? (i === 0 ? -12 : 12) : 0;
        return (
          <g key={slot} transform={`translate(${badgeX}, ${badgeY})`}>
            <rect x={-10} y={-5} width={20} height={10} rx={3}
              fill={`${slot === 'A' ? '#00d2ff' : '#ff8c00'}20`}
              stroke={slot === 'A' ? '#00d2ff' : '#ff8c00'}
              strokeWidth={0.5} />
            <text x={0} y={1} textAnchor="middle" dominantBaseline="central"
              fill={slot === 'A' ? '#00d2ff' : '#ff8c00'} fontSize={6} fontFamily="monospace">
              {role.slice(0, 3).toUpperCase()}
            </text>
          </g>
        );
      })}
    </g>
  );
}
