// BiologyEdge.jsx — Hex-grid-aligned edges
// Edges follow the 3 hex axes (0°, 60°, 120°) with at most one angular bend.
import React from 'react';
import { EDGE_STYLES } from '../mechanismGraphEngine';

// Snap an angle to the nearest hex axis direction (every 60°)
function snapToHexAngle(dx, dy) {
  const angle = Math.atan2(dy, dx);
  // 6 hex directions: 0, 60, 120, 180, 240, 300 degrees
  const snapped = Math.round(angle / (Math.PI / 3)) * (Math.PI / 3);
  return snapped;
}

// Build a path from (x1,y1) to (x2,y2) using hex-axis-aligned segments
function hexPath(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.sqrt(dx * dx + dy * dy);

  // If very short or nearly aligned to a hex axis, go straight
  if (dist < 10) return `M ${x1} ${y1} L ${x2} ${y2}`;

  // Try direct line first — check if it's close to a hex axis
  const angle = Math.atan2(dy, dx);
  const hexStep = Math.PI / 3;
  const remainder = ((angle % hexStep) + hexStep) % hexStep;
  const isAligned = remainder < 0.15 || remainder > hexStep - 0.15;

  if (isAligned) {
    return `M ${x1} ${y1} L ${x2} ${y2}`;
  }

  // Not aligned — use a two-segment path with a bend point
  // Pick the hex axis closest to the line from source to target
  const primaryAngle = snapToHexAngle(dx, dy);

  // Travel along the primary axis until we can turn to reach target on another axis
  const cos1 = Math.cos(primaryAngle);
  const sin1 = Math.sin(primaryAngle);

  // Project target onto primary axis to find how far to travel
  const projLen = dx * cos1 + dy * sin1;

  // Bend point: travel along primary axis
  const bx = x1 + cos1 * projLen;
  const by = y1 + sin1 * projLen;

  // Check if bend point is close enough to target
  const remDx = x2 - bx;
  const remDy = y2 - by;
  const remDist = Math.sqrt(remDx * remDx + remDy * remDy);

  if (remDist < 5) {
    return `M ${x1} ${y1} L ${x2} ${y2}`;
  }

  // Use a midpoint bend approach: go halfway on primary axis, then switch
  const midFrac = 0.5;
  const mx = x1 + dx * midFrac;
  const my = y1;  // horizontal first

  // Snap the midpoint to create clean angles
  // Strategy: go horizontally to mid-x, then diagonally to target
  // This creates clean 0° + 60° or 0° + 120° bends

  // Actually simplest: go along one hex axis to align with target on another
  // Horizontal first (0°), then angled
  if (Math.abs(dx) > Math.abs(dy) * 0.5) {
    // Go horizontal partway, then angle to target
    const bendX = x1 + dx * 0.6;
    const bendY = y1;
    return `M ${x1} ${y1} L ${bendX} ${bendY} L ${x2} ${y2}`;
  } else {
    // Go angled first, then horizontal
    const bendX = x2;
    const bendY = y1 + dy * 0.6;
    return `M ${x1} ${y1} L ${bendX} ${bendY} L ${x2} ${y2}`;
  }
}

export default function BiologyEdge({ edge, x1, y1, x2, y2, isConflict }) {
  const style = EDGE_STYLES[edge.type] || EDGE_STYLES.causes;
  const effectiveStyle = isConflict ? EDGE_STYLES.conflict : style;

  const path = hexPath(x1, y1, x2, y2);

  // Label at midpoint
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;

  return (
    <g>
      <path
        d={path}
        fill="none"
        stroke={effectiveStyle.stroke}
        strokeWidth={effectiveStyle.width}
        strokeDasharray={effectiveStyle.dasharray === 'none' ? undefined : effectiveStyle.dasharray}
        opacity={isConflict ? 0.8 : 0.4}
        strokeLinejoin="round"
      />
      {edge.label && (
        <text
          x={mx}
          y={my - 6}
          textAnchor="middle"
          fill={effectiveStyle.stroke}
          fontSize={6}
          fontFamily="monospace"
          opacity={0.5}
        >
          {edge.label}
        </text>
      )}
      {isConflict && (
        <path
          d={path}
          fill="none"
          stroke="#ef4444"
          strokeWidth={1}
          strokeDasharray="4,4"
          opacity={0.6}
          strokeLinejoin="round"
        >
          <animate attributeName="stroke-dashoffset" from="0" to="16" dur="1s" repeatCount="indefinite" />
        </path>
      )}
    </g>
  );
}
