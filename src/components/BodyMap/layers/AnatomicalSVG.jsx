// AnatomicalSVG.jsx — Detailed anatomical body with 10 organ systems, skeletal hints, organ glow
// All coordinates in 400×900 viewBox. Organs glow and pulse based on severity.

import React, { useState, useMemo } from 'react';
import { ORGAN_SYSTEMS, BODY_OUTLINE, SKELETAL_HINTS, getSeverityColor } from '../organRegistry';

// ─── SVG Filter definitions for organ glow ─────────────────────────────────
function GlowFilters() {
  return (
    <defs>
      {/* Soft glow for affected organs */}
      <filter id="organ-glow-mild" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <filter id="organ-glow-moderate" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <filter id="organ-glow-severe" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="8" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="blur" />
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>

      {/* Scan line pattern */}
      <pattern id="scan-lines" width="4" height="4" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0" x2="4" y2="0" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
      </pattern>

      {/* Grid pattern for body interior */}
      <pattern id="body-grid" width="10" height="10" patternUnits="userSpaceOnUse">
        <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(100,200,255,0.04)" strokeWidth="0.5" />
      </pattern>
    </defs>
  );
}

// ─── Body outline (head, torso, arms) ──────────────────────────────────────
function BodyOutline() {
  return (
    <g className="body-outline">
      {Object.entries(BODY_OUTLINE).map(([key, path]) => (
        <path
          key={key}
          d={path}
          fill="none"
          stroke="rgba(100, 180, 255, 0.15)"
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}
      {/* Inner body fill with grid */}
      <path
        d={BODY_OUTLINE.torso}
        fill="url(#body-grid)"
        opacity={0.5}
      />
    </g>
  );
}

// ─── Skeletal hints (dashed overlay) ───────────────────────────────────────
function SkeletalOverlay({ visible = true }) {
  if (!visible) return null;

  return (
    <g className="skeletal-hints" opacity={0.25}>
      {/* Ribcage */}
      {SKELETAL_HINTS.ribs.map((rib, i) => (
        <path
          key={`rib-${i}`}
          d={rib}
          fill="none"
          stroke="rgba(150, 200, 255, 0.3)"
          strokeWidth="0.8"
          strokeDasharray="3,3"
        />
      ))}
      {/* Spine */}
      <path
        d={SKELETAL_HINTS.spine}
        fill="none"
        stroke="rgba(150, 200, 255, 0.25)"
        strokeWidth="1"
        strokeDasharray="2,4"
      />
      {/* Pelvis */}
      <path
        d={SKELETAL_HINTS.pelvis}
        fill="none"
        stroke="rgba(150, 200, 255, 0.2)"
        strokeWidth="0.8"
        strokeDasharray="3,3"
      />
      {/* Skull */}
      <path
        d={SKELETAL_HINTS.skull}
        fill="none"
        stroke="rgba(150, 200, 255, 0.2)"
        strokeWidth="0.8"
        strokeDasharray="2,3"
      />
    </g>
  );
}

// ─── Single organ rendering ────────────────────────────────────────────────
function OrganShape({ organKey, organDef, severity, isHovered, isSelected, onHover, onClick }) {
  if (!organDef.paths || organDef.isOverlay) return null;

  const colorInfo = getSeverityColor(severity);
  const isAffected = severity > 0;

  // Pick glow filter based on severity
  const glowFilter = severity > 0.7
    ? 'url(#organ-glow-severe)'
    : severity > 0.4
      ? 'url(#organ-glow-moderate)'
      : severity > 0
        ? 'url(#organ-glow-mild)'
        : 'none';

  const fillOpacity = isAffected
    ? 0.15 + severity * 0.35
    : isHovered ? 0.08 : 0.03;

  const strokeOpacity = isAffected
    ? 0.5 + severity * 0.5
    : isHovered ? 0.4 : 0.15;

  return (
    <g
      className="organ-group cursor-pointer"
      filter={isAffected ? glowFilter : 'none'}
      onMouseEnter={() => onHover?.(organKey)}
      onMouseLeave={() => onHover?.(null)}
      onClick={() => onClick?.(organKey)}
    >
      {organDef.paths.map((path, i) => (
        <path
          key={`${organKey}-${i}`}
          d={path}
          fill={isAffected ? colorInfo.fill : '#1e293b'}
          fillOpacity={fillOpacity}
          stroke={isAffected ? colorInfo.fill : 'rgba(100, 180, 255, 0.25)'}
          strokeOpacity={strokeOpacity}
          strokeWidth={organDef.isSecondary ? 0.8 : 1.2}
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{
            transition: 'fill-opacity 0.4s, stroke-opacity 0.4s, stroke 0.4s, fill 0.4s',
          }}
        />
      ))}

      {/* Pulse animation for severely affected organs */}
      {severity > 0.6 && organDef.paths[0] && (
        <path
          d={organDef.paths[0]}
          fill="none"
          stroke={colorInfo.fill}
          strokeWidth="2"
          strokeOpacity="0"
        >
          <animate
            attributeName="stroke-opacity"
            values="0;0.6;0"
            dur="2s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="stroke-width"
            values="2;4;2"
            dur="2s"
            repeatCount="indefinite"
          />
        </path>
      )}

      {/* Organ label */}
      {(isHovered || (isAffected && !organDef.isSecondary)) && (
        <text
          x={organDef.center.x}
          y={organDef.center.y - (organDef.paths ? 0 : 10)}
          textAnchor="middle"
          dominantBaseline="central"
          fill={isAffected ? colorInfo.fill : 'rgba(180, 210, 255, 0.6)'}
          fontSize="8"
          fontFamily="monospace"
          fontWeight="bold"
          letterSpacing="0.5"
          style={{ pointerEvents: 'none', textTransform: 'uppercase' }}
        >
          {organDef.shortName}
        </text>
      )}

      {/* Severity indicator dot */}
      {isAffected && !organDef.isSecondary && (
        <circle
          cx={organDef.center.x + 20}
          cy={organDef.center.y - 15}
          r="3"
          fill={colorInfo.fill}
          opacity={0.8}
        >
          {severity > 0.6 && (
            <animate attributeName="r" values="3;5;3" dur="1.5s" repeatCount="indefinite" />
          )}
        </circle>
      )}
    </g>
  );
}

// ─── Overlay organ indicators (blood, skin) ────────────────────────────────
function OverlayOrgan({ organKey, organDef, severity, isHovered, onHover, onClick }) {
  if (!organDef.isOverlay || severity <= 0) return null;

  const colorInfo = getSeverityColor(severity);

  // Blood: show as vascular highlights
  if (organKey === 'blood') {
    return (
      <g
        className="cursor-pointer"
        onMouseEnter={() => onHover?.(organKey)}
        onMouseLeave={() => onHover?.(null)}
        onClick={() => onClick?.(organKey)}
        opacity={0.3 + severity * 0.4}
      >
        {/* Aorta highlight */}
        <path
          d="M 215,245 C 215,235 225,225 235,225 C 245,225 248,230 248,235 L 248,350"
          fill="none"
          stroke={colorInfo.fill}
          strokeWidth="3"
          strokeDasharray="5,3"
        >
          <animate attributeName="stroke-dashoffset" values="0;-16" dur="1s" repeatCount="indefinite" />
        </path>
        <text x={organDef.center.x} y={organDef.center.y - 25} textAnchor="middle"
          fill={colorInfo.fill} fontSize="7" fontFamily="monospace" fontWeight="bold">
          {isHovered ? 'BLOOD / HEMATOLOGICAL' : 'BLOOD'}
        </text>
      </g>
    );
  }

  // Skin: show as body outline highlight
  if (organKey === 'skin') {
    return (
      <g
        className="cursor-pointer"
        onMouseEnter={() => onHover?.(organKey)}
        onMouseLeave={() => onHover?.(null)}
        onClick={() => onClick?.(organKey)}
      >
        <path
          d={BODY_OUTLINE.torso}
          fill="none"
          stroke={colorInfo.fill}
          strokeWidth="3"
          strokeOpacity={0.2 + severity * 0.3}
          strokeDasharray="8,4"
        >
          <animate attributeName="stroke-dashoffset" values="0;-24" dur="2s" repeatCount="indefinite" />
        </path>
        <text x={organDef.center.x} y={organDef.center.y + 80} textAnchor="middle"
          fill={colorInfo.fill} fontSize="7" fontFamily="monospace" fontWeight="bold">
          {isHovered ? 'SKIN / DERMATOLOGICAL' : 'SKIN'}
        </text>
      </g>
    );
  }

  return null;
}

// ─── Main AnatomicalSVG Component ──────────────────────────────────────────
export default function AnatomicalSVG({ organs = {}, onOrganClick, onOrganHover, selectedOrgan, showSkeleton = true }) {
  const [hoveredOrgan, setHoveredOrgan] = useState(null);

  const handleHover = (organKey) => {
    setHoveredOrgan(organKey);
    onOrganHover?.(organKey);
  };

  const handleClick = (organKey) => {
    onOrganClick?.(organKey);
  };

  // Separate overlay organs from path-based organs
  const { pathOrgans, overlayOrgans } = useMemo(() => {
    const path = [];
    const overlay = [];
    for (const [key, def] of Object.entries(ORGAN_SYSTEMS)) {
      if (def.isOverlay) overlay.push([key, def]);
      else if (def.isSecondary) path.push([key, def]); // Render secondary first (behind)
      else path.push([key, def]);
    }
    // Sort: secondary organs first, then primary
    path.sort((a, b) => {
      if (a[1].isSecondary && !b[1].isSecondary) return -1;
      if (!a[1].isSecondary && b[1].isSecondary) return 1;
      return 0;
    });
    return { pathOrgans: path, overlayOrgans: overlay };
  }, []);

  return (
    <svg
      viewBox="0 0 400 780"
      className="w-full h-full"
      style={{ maxHeight: '100%' }}
    >
      <GlowFilters />

      {/* Scan line overlay for tech feel */}
      <rect width="400" height="780" fill="url(#scan-lines)" opacity={0.5} />

      {/* Body outline */}
      <BodyOutline />

      {/* Skeletal hints */}
      <SkeletalOverlay visible={showSkeleton} />

      {/* Path-based organs (secondary first, then primary) */}
      {pathOrgans.map(([key, def]) => (
        <OrganShape
          key={key}
          organKey={key}
          organDef={def}
          severity={organs[key]?.severity || 0}
          isHovered={hoveredOrgan === key}
          isSelected={selectedOrgan === key}
          onHover={handleHover}
          onClick={handleClick}
        />
      ))}

      {/* Overlay organs (blood, skin) */}
      {overlayOrgans.map(([key, def]) => (
        <OverlayOrgan
          key={key}
          organKey={key}
          organDef={def}
          severity={organs[key]?.severity || 0}
          isHovered={hoveredOrgan === key}
          onHover={handleHover}
          onClick={handleClick}
        />
      ))}

      {/* Cross-hair markers at organ centers (tech aesthetic) */}
      {Object.entries(ORGAN_SYSTEMS).map(([key, def]) => {
        const sev = organs[key]?.severity || 0;
        if (sev <= 0 || def.isOverlay || def.isSecondary) return null;
        const c = def.center;
        const colorInfo = getSeverityColor(sev);
        return (
          <g key={`marker-${key}`} opacity={0.4}>
            <line x1={c.x - 6} y1={c.y} x2={c.x - 2} y2={c.y} stroke={colorInfo.fill} strokeWidth="0.5" />
            <line x1={c.x + 2} y1={c.y} x2={c.x + 6} y2={c.y} stroke={colorInfo.fill} strokeWidth="0.5" />
            <line x1={c.x} y1={c.y - 6} x2={c.x} y2={c.y - 2} stroke={colorInfo.fill} strokeWidth="0.5" />
            <line x1={c.x} y1={c.y + 2} x2={c.x} y2={c.y + 6} stroke={colorInfo.fill} strokeWidth="0.5" />
          </g>
        );
      })}
    </svg>
  );
}
