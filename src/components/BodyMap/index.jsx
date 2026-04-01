// BodyMap/index.jsx — V2 Orchestrator: layered anatomical drug effect visualization
// Stack: HeatMapCanvas → CirculatoryOverlay → AnatomicalSVG → OrganDetailPanel + Overlays
// Data: enriched from affectedSystems + drugInfoCache + interactionEvidence + CYP data

import React, { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, AlertCircle } from 'lucide-react';

import { enrichBodyMapData } from './bodyMapDataService';
import AnatomicalSVG from './layers/AnatomicalSVG';
import HeatMapCanvas from './layers/HeatMapCanvas';
import CirculatoryOverlay from './layers/CirculatoryOverlay';
import OrganDetailPanel from './overlays/OrganDetailPanel';
import SeverityLegend from './overlays/SeverityLegend';
import LayerToggle from './overlays/LayerToggle';

// ─── Empty state ───────────────────────────────────────────────────────────
function EmptyState({ hasDrugs, hasResult }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center px-6 py-4 bg-[#0a0f1a]/80 backdrop-blur-sm rounded-lg border border-white/5"
      >
        <div className="flex items-center justify-center gap-2 mb-2">
          {!hasDrugs ? (
            <Activity className="w-4 h-4 text-slate-500" />
          ) : (
            <Zap className="w-4 h-4 text-amber-500/60" />
          )}
          <span className="text-xs text-slate-400 font-mono uppercase tracking-wider">
            {!hasDrugs ? 'Body Map' : 'Awaiting Analysis'}
          </span>
        </div>
        <p className="text-[10px] text-slate-600 max-w-[200px]">
          {!hasDrugs
            ? 'Add drugs to visualize physiological effects'
            : 'Run interaction analysis to visualize affected organ systems'
          }
        </p>
      </motion.div>
    </div>
  );
}

// ─── Title bar ─────────────────────────────────────────────────────────────
function TitleBar({ drugCount, dataQuality, isMobile }) {
  return (
    <div className={`
      absolute z-10 flex items-center gap-2
      ${isMobile ? 'top-2 left-2' : 'top-4 left-4'}
    `}>
      <div className="flex items-center gap-1.5 px-2 py-1 bg-[#0a0f1a]/80 backdrop-blur-sm rounded border border-white/5">
        <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
        <span className="text-[9px] text-cyan-400/80 font-mono uppercase tracking-widest">
          Physiological Impact
        </span>
        {drugCount > 0 && (
          <span className="text-[9px] text-slate-600 font-mono">
            {drugCount} drug{drugCount > 1 ? 's' : ''}
          </span>
        )}
      </div>
    </div>
  );
}

// ─── Affected systems summary bar ──────────────────────────────────────────
function AffectedSummary({ organs, isMobile }) {
  const affected = useMemo(() => {
    return Object.entries(organs)
      .filter(([_, o]) => o.severity > 0)
      .sort((a, b) => b[1].severity - a[1].severity)
      .slice(0, isMobile ? 3 : 6);
  }, [organs, isMobile]);

  if (affected.length === 0) return null;

  return (
    <div className={`
      absolute z-10 flex items-center gap-1.5
      ${isMobile ? 'top-2 left-2 right-2' : 'top-4 left-52'}
    `}>
      {!isMobile && (
        <AlertCircle className="w-3 h-3 text-amber-500/60 flex-shrink-0" />
      )}
      {affected.map(([key, organ]) => {
        const sev = organ.severity;
        const color = sev > 0.7 ? '#ef4444' : sev > 0.4 ? '#f97316' : '#eab308';
        return (
          <span
            key={key}
            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-white/[0.03] border border-white/5"
          >
            <span className="w-1 h-1 rounded-full" style={{ background: color }} />
            <span className="text-[8px] font-mono text-slate-400 uppercase">{key}</span>
          </span>
        );
      })}
    </div>
  );
}

// ─── Main BodyMap Component ────────────────────────────────────────────────
export default function BodyMap({
  affectedSystems = {},
  drugs = [],
  drugInfoCache = {},
  interactionEvidence = null,
  polypharmacyResult = null,
  result = null,
  isMobile = false,
}) {
  const [selectedOrgan, setSelectedOrgan] = useState(null);
  const [layers, setLayers] = useState({
    circulatory: true,
    heatmap: true,
    skeleton: true,
  });

  // Enrich data from all sources
  const enriched = useMemo(() => {
    return enrichBodyMapData({
      affectedSystems,
      drugs,
      drugInfoCache,
      interactionEvidence,
      polypharmacyResult,
    });
  }, [affectedSystems, drugs, drugInfoCache, interactionEvidence, polypharmacyResult]);

  const handleOrganClick = useCallback((organKey) => {
    setSelectedOrgan(prev => prev === organKey ? null : organKey);
  }, []);

  const handleOrganHover = useCallback(() => {
    // Future: could show tooltip or highlight connections
  }, []);

  const handleLayerToggle = useCallback((layerKey) => {
    setLayers(prev => ({ ...prev, [layerKey]: !prev[layerKey] }));
  }, []);

  const handleCloseDetail = useCallback(() => {
    setSelectedOrgan(null);
  }, []);

  const hasDrugs = drugs.length > 0;
  const hasResult = Object.keys(affectedSystems).length > 0 || !enriched.isEmpty;

  return (
    <div
      className="relative w-full h-full overflow-hidden"
      style={{ background: '#03050a' }}
    >
      {/* Subtle background grid */}
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(rgba(100, 200, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(100, 200, 255, 0.03) 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px',
        }}
      />

      {/* Title */}
      <TitleBar drugCount={drugs.length} dataQuality={enriched.dataQuality} isMobile={isMobile} />

      {/* Layer toggles */}
      <LayerToggle layers={layers} onToggle={handleLayerToggle} isMobile={isMobile} />

      {/* Empty state */}
      {enriched.isEmpty && <EmptyState hasDrugs={hasDrugs} hasResult={hasResult} />}

      {/* Visual stack */}
      <div className="relative w-full h-full flex items-center justify-center">
        {/* Heat map (behind everything) */}
        <HeatMapCanvas
          organs={enriched.organs}
          visible={layers.heatmap && !enriched.isEmpty}
        />

        {/* SVG layers */}
        <div className="relative w-full h-full flex items-center justify-center" style={{ maxWidth: isMobile ? '100%' : '450px' }}>
          {/* Main anatomical SVG with circulatory overlay injected */}
          <svg
            viewBox="0 0 400 780"
            className="w-full h-full"
            style={{ maxHeight: '100%', opacity: enriched.isEmpty ? 0.3 : 1, transition: 'opacity 0.5s' }}
          >
            {/* Import definitions from AnatomicalSVG's filter defs */}
            <defs>
              <filter id="organ-glow-mild" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              <filter id="organ-glow-moderate" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              <filter id="organ-glow-severe" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="8" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="blur" /><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              <pattern id="scan-lines" width="4" height="4" patternUnits="userSpaceOnUse">
                <line x1="0" y1="0" x2="4" y2="0" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
              </pattern>
              <pattern id="body-grid" width="10" height="10" patternUnits="userSpaceOnUse">
                <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(100,200,255,0.04)" strokeWidth="0.5" />
              </pattern>
            </defs>

            {/* Scan lines for tech feel */}
            <rect width="400" height="780" fill="url(#scan-lines)" opacity={0.5} />

            {/* Circulatory overlay (rendered in same SVG for coordinate alignment) */}
            <CirculatoryOverlay
              visible={layers.circulatory && !enriched.isEmpty}
              activeOrgans={enriched.drugPathway?.activeOrgans || []}
            />

            {/* The anatomical body is rendered inline here to share the SVG context */}
            <AnatomicalSVGInline
              organs={enriched.organs}
              selectedOrgan={selectedOrgan}
              showSkeleton={layers.skeleton}
              onOrganClick={handleOrganClick}
              onOrganHover={handleOrganHover}
            />
          </svg>
        </div>
      </div>

      {/* Affected systems summary */}
      {!enriched.isEmpty && !isMobile && (
        <AffectedSummary organs={enriched.organs} isMobile={isMobile} />
      )}

      {/* Severity legend */}
      <SeverityLegend dataQuality={enriched.dataQuality} isMobile={isMobile} />

      {/* Organ detail panel */}
      <AnimatePresence>
        {selectedOrgan && enriched.organs[selectedOrgan] && (
          <OrganDetailPanel
            organKey={selectedOrgan}
            organData={enriched.organs[selectedOrgan]}
            onClose={handleCloseDetail}
            isMobile={isMobile}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Inline SVG body (avoids nested <svg> issues) ─────────────────────────
// This renders the same content as AnatomicalSVG but as a <g> group
import { ORGAN_SYSTEMS, BODY_OUTLINE, SKELETAL_HINTS, getSeverityColor } from './organRegistry';

function AnatomicalSVGInline({ organs = {}, selectedOrgan, showSkeleton = true, onOrganClick, onOrganHover }) {
  const [hoveredOrgan, setHoveredOrgan] = useState(null);

  const handleHover = (key) => {
    setHoveredOrgan(key);
    onOrganHover?.(key);
  };

  // Separate and sort organs
  const sortedOrgans = useMemo(() => {
    return Object.entries(ORGAN_SYSTEMS)
      .filter(([_, def]) => def.paths && !def.isOverlay)
      .sort((a, b) => {
        if (a[1].isSecondary && !b[1].isSecondary) return -1;
        if (!a[1].isSecondary && b[1].isSecondary) return 1;
        return 0;
      });
  }, []);

  const overlayOrgans = useMemo(() => {
    return Object.entries(ORGAN_SYSTEMS).filter(([_, def]) => def.isOverlay);
  }, []);

  return (
    <g>
      {/* Body outline */}
      {Object.entries(BODY_OUTLINE).map(([key, path]) => (
        <path
          key={`outline-${key}`}
          d={path}
          fill="none"
          stroke="rgba(100, 180, 255, 0.15)"
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}
      <path d={BODY_OUTLINE.torso} fill="url(#body-grid)" opacity={0.5} />

      {/* Skeletal hints */}
      {showSkeleton && (
        <g opacity={0.25}>
          {SKELETAL_HINTS.ribs.map((rib, i) => (
            <path key={`rib-${i}`} d={rib} fill="none" stroke="rgba(150, 200, 255, 0.3)" strokeWidth="0.8" strokeDasharray="3,3" />
          ))}
          <path d={SKELETAL_HINTS.spine} fill="none" stroke="rgba(150, 200, 255, 0.25)" strokeWidth="1" strokeDasharray="2,4" />
          <path d={SKELETAL_HINTS.pelvis} fill="none" stroke="rgba(150, 200, 255, 0.2)" strokeWidth="0.8" strokeDasharray="3,3" />
          <path d={SKELETAL_HINTS.skull} fill="none" stroke="rgba(150, 200, 255, 0.2)" strokeWidth="0.8" strokeDasharray="2,3" />
        </g>
      )}

      {/* Path-based organs */}
      {sortedOrgans.map(([key, def]) => {
        const severity = organs[key]?.severity || 0;
        const isAffected = severity > 0;
        const isHovered = hoveredOrgan === key;
        const colorInfo = getSeverityColor(severity);

        const glowFilter = severity > 0.7 ? 'url(#organ-glow-severe)'
          : severity > 0.4 ? 'url(#organ-glow-moderate)'
          : severity > 0 ? 'url(#organ-glow-mild)' : 'none';

        const fillOpacity = isAffected ? 0.15 + severity * 0.35 : isHovered ? 0.08 : 0.03;
        const strokeOpacity = isAffected ? 0.5 + severity * 0.5 : isHovered ? 0.4 : 0.15;

        return (
          <g
            key={key}
            className="cursor-pointer"
            filter={isAffected ? glowFilter : 'none'}
            onMouseEnter={() => handleHover(key)}
            onMouseLeave={() => handleHover(null)}
            onClick={() => onOrganClick?.(key)}
          >
            {def.paths.map((path, i) => (
              <path
                key={`${key}-${i}`}
                d={path}
                fill={isAffected ? colorInfo.fill : '#1e293b'}
                fillOpacity={fillOpacity}
                stroke={isAffected ? colorInfo.fill : 'rgba(100, 180, 255, 0.25)'}
                strokeOpacity={strokeOpacity}
                strokeWidth={def.isSecondary ? 0.8 : 1.2}
                strokeLinejoin="round"
                strokeLinecap="round"
                style={{ transition: 'fill-opacity 0.4s, stroke-opacity 0.4s, stroke 0.4s, fill 0.4s' }}
              />
            ))}

            {/* Pulse animation for severe organs */}
            {severity > 0.6 && def.paths[0] && (
              <path d={def.paths[0]} fill="none" stroke={colorInfo.fill} strokeWidth="2" strokeOpacity="0">
                <animate attributeName="stroke-opacity" values="0;0.6;0" dur="2s" repeatCount="indefinite" />
                <animate attributeName="stroke-width" values="2;4;2" dur="2s" repeatCount="indefinite" />
              </path>
            )}

            {/* Organ label */}
            {(isHovered || (isAffected && !def.isSecondary)) && (
              <text
                x={def.center.x}
                y={def.center.y}
                textAnchor="middle"
                dominantBaseline="central"
                fill={isAffected ? colorInfo.fill : 'rgba(180, 210, 255, 0.6)'}
                fontSize="8"
                fontFamily="monospace"
                fontWeight="bold"
                letterSpacing="0.5"
                style={{ pointerEvents: 'none', textTransform: 'uppercase' }}
              >
                {def.shortName}
              </text>
            )}

            {/* Severity dot */}
            {isAffected && !def.isSecondary && (
              <circle cx={def.center.x + 20} cy={def.center.y - 15} r="3" fill={colorInfo.fill} opacity={0.8}>
                {severity > 0.6 && (
                  <animate attributeName="r" values="3;5;3" dur="1.5s" repeatCount="indefinite" />
                )}
              </circle>
            )}

            {/* Cross-hair markers */}
            {isAffected && !def.isSecondary && (
              <g opacity={0.4}>
                <line x1={def.center.x - 6} y1={def.center.y} x2={def.center.x - 2} y2={def.center.y} stroke={colorInfo.fill} strokeWidth="0.5" />
                <line x1={def.center.x + 2} y1={def.center.y} x2={def.center.x + 6} y2={def.center.y} stroke={colorInfo.fill} strokeWidth="0.5" />
                <line x1={def.center.x} y1={def.center.y - 6} x2={def.center.x} y2={def.center.y - 2} stroke={colorInfo.fill} strokeWidth="0.5" />
                <line x1={def.center.x} y1={def.center.y + 2} x2={def.center.x} y2={def.center.y + 6} stroke={colorInfo.fill} strokeWidth="0.5" />
              </g>
            )}
          </g>
        );
      })}

      {/* Overlay organs (blood, skin) */}
      {overlayOrgans.map(([key, def]) => {
        const severity = organs[key]?.severity || 0;
        if (severity <= 0) return null;
        const isHovered = hoveredOrgan === key;
        const colorInfo = getSeverityColor(severity);

        if (key === 'blood') {
          return (
            <g key={key} className="cursor-pointer"
              onMouseEnter={() => handleHover(key)}
              onMouseLeave={() => handleHover(null)}
              onClick={() => onOrganClick?.(key)}
              opacity={0.3 + severity * 0.4}
            >
              <path
                d="M 215,245 C 215,235 225,225 235,225 C 245,225 248,230 248,235 L 248,350"
                fill="none" stroke={colorInfo.fill} strokeWidth="3" strokeDasharray="5,3"
              >
                <animate attributeName="stroke-dashoffset" values="0;-16" dur="1s" repeatCount="indefinite" />
              </path>
              <text x={def.center.x} y={def.center.y - 25} textAnchor="middle"
                fill={colorInfo.fill} fontSize="7" fontFamily="monospace" fontWeight="bold">
                {isHovered ? 'BLOOD / HEMATOLOGICAL' : 'BLOOD'}
              </text>
            </g>
          );
        }

        if (key === 'skin') {
          return (
            <g key={key} className="cursor-pointer"
              onMouseEnter={() => handleHover(key)}
              onMouseLeave={() => handleHover(null)}
              onClick={() => onOrganClick?.(key)}
            >
              <path d={BODY_OUTLINE.torso} fill="none" stroke={colorInfo.fill}
                strokeWidth="3" strokeOpacity={0.2 + severity * 0.3} strokeDasharray="8,4"
              >
                <animate attributeName="stroke-dashoffset" values="0;-24" dur="2s" repeatCount="indefinite" />
              </path>
              <text x={def.center.x} y={def.center.y + 80} textAnchor="middle"
                fill={colorInfo.fill} fontSize="7" fontFamily="monospace" fontWeight="bold">
                {isHovered ? 'SKIN / DERMATOLOGICAL' : 'SKIN'}
              </text>
            </g>
          );
        }

        return null;
      })}
    </g>
  );
}
