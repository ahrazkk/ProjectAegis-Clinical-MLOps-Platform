// BodyMap/index.jsx — V2 Orchestrator: layered anatomical drug effect visualization
// Stack: HeatMapCanvas → CirculatoryOverlay → AnatomicalSVG → OrganDetailPanel + Overlays
// Data: enriched from affectedSystems + drugInfoCache + interactionEvidence + CYP data

import React, { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, AlertCircle, ListFilter, Eye, EyeOff } from 'lucide-react';

import { enrichBodyMapData } from './bodyMapDataService';
import { ORGAN_SYSTEMS, BODY_OUTLINE, SKELETAL_HINTS, ANATOMY_LANDMARKS, getSeverityColor } from './organRegistry';
import HeatMapCanvas from './layers/HeatMapCanvas';
import CirculatoryOverlay from './layers/CirculatoryOverlay';
import SegmentedBodyFigure from './layers/SegmentedBodyFigure';
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

// ─── Organ navigator: quick-jump + filter ──────────────────────────────────
function OrganNavigator({
  organs,
  selectedOrgan,
  onSelect,
  showOnlyAffected,
  onToggleShowOnlyAffected,
  isMobile,
}) {
  const ranked = useMemo(() => {
    return Object.entries(organs)
      .filter(([_, o]) => (o?.severity || 0) > 0)
      .sort((a, b) => (b[1].severity || 0) - (a[1].severity || 0));
  }, [organs]);

  if (ranked.length === 0) return null;

  const top = ranked.slice(0, isMobile ? 4 : 6);

  return (
    <div className={`absolute z-10 ${isMobile ? 'left-2 right-2 bottom-16' : 'right-4 top-14 w-56'}`}>
      <div className="bg-[#0a0f1a]/85 backdrop-blur-sm border border-white/10 rounded-lg p-2">
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-1.5">
            <ListFilter className="w-3 h-3 text-cyan-400/80" />
            <span className="text-[9px] text-slate-400 font-mono uppercase tracking-wider">Organ Navigator</span>
          </div>
          <button
            onClick={onToggleShowOnlyAffected}
            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-white/10 hover:bg-white/5 transition-colors"
            title={showOnlyAffected ? 'Show all organs' : 'Show only affected organs'}
          >
            {showOnlyAffected ? (
              <EyeOff className="w-3 h-3 text-amber-400/80" />
            ) : (
              <Eye className="w-3 h-3 text-cyan-400/80" />
            )}
          </button>
        </div>

        <div className="space-y-1">
          {top.map(([key, organ]) => {
            const severity = organ?.severity || 0;
            const organDef = ORGAN_SYSTEMS[key];
            const organLabel = organDef?.shortName || key;
            const tone = severity > 0.7
              ? 'from-red-500/50 to-red-500/10 text-red-300 border-red-500/30'
              : severity > 0.4
                ? 'from-orange-500/50 to-orange-500/10 text-orange-300 border-orange-500/30'
                : 'from-yellow-500/40 to-yellow-500/10 text-yellow-300 border-yellow-500/30';

            return (
              <button
                key={key}
                onClick={() => onSelect?.(key)}
                className={`w-full text-left p-1.5 rounded border bg-gradient-to-r transition-all ${tone} ${selectedOrgan === key ? 'ring-1 ring-cyan-400/50' : ''}`}
                title={organDef?.name || organLabel}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[9px] font-mono uppercase tracking-wider truncate">{organLabel}</span>
                  <span className="text-[9px] font-mono">{Math.round(severity * 100)}%</span>
                </div>
                <div className="mt-1 h-1 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-cyan-300/70"
                    style={{ width: `${Math.max(6, severity * 100)}%` }}
                  />
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── Clinical + research utility strip ────────────────────────────────────
function ClinicalIntelPanel({ organs, interactionEvidence, isMobile }) {
  const ranked = useMemo(() => {
    return Object.entries(organs)
      .filter(([_, value]) => (value?.severity || 0) > 0)
      .sort((a, b) => (b[1]?.severity || 0) - (a[1]?.severity || 0));
  }, [organs]);

  const affectedCount = ranked.length;
  const avgBurden = affectedCount
    ? ranked.reduce((sum, [_, value]) => sum + (value?.severity || 0), 0) / affectedCount
    : 0;
  const maxSeverity = ranked[0]?.[1]?.severity || 0;
  const topSystemKey = ranked[0]?.[0] || null;

  const summary = interactionEvidence?.evidence_summary && typeof interactionEvidence.evidence_summary === 'object'
    ? interactionEvidence.evidence_summary
    : {};
  const weightedSupport = Number(summary.weighted_support_score);
  const weightedUncertainty = Number(summary.weighted_uncertainty_score);
  const disagreementLevel = String(summary?.disagreement?.level || 'none');

  const recommendation = maxSeverity > 0.7 || avgBurden > 0.55
    ? {
      label: 'Escalate Monitoring',
      tone: 'text-red-300 border-red-500/40 bg-red-500/10',
    }
    : maxSeverity > 0.4 || avgBurden > 0.3
      ? {
        label: 'Focused Review',
        tone: 'text-amber-300 border-amber-500/40 bg-amber-500/10',
      }
      : {
        label: 'Routine Surveillance',
        tone: 'text-cyan-300 border-cyan-500/40 bg-cyan-500/10',
      };

  const fmtPercent = (value) => (Number.isFinite(value) ? `${Math.round(value * 100)}%` : 'N/A');
  const topSystemLabel = topSystemKey ? (ORGAN_SYSTEMS[topSystemKey]?.name || topSystemKey) : 'No active system';

  return (
    <div className={`absolute z-10 ${isMobile ? 'left-2 right-2 bottom-20' : 'left-1/2 -translate-x-1/2 bottom-4 w-[430px]'}`}>
      <div className="bg-[#081024]/88 backdrop-blur-md border border-cyan-500/15 rounded-lg p-2.5 shadow-[0_0_30px_rgba(30,80,160,0.22)]">
        <div className="flex items-center justify-between gap-2 mb-2">
          <span className="text-[9px] uppercase tracking-[0.2em] text-cyan-300/80 font-mono">Clinical Systems Lens</span>
          <span className={`text-[8px] px-1.5 py-0.5 border uppercase tracking-wider ${recommendation.tone}`}>{recommendation.label}</span>
        </div>

        <div className="grid grid-cols-2 gap-2 mb-2">
          <div className="p-1.5 border border-white/10 bg-white/[0.02] rounded">
            <p className="text-[8px] text-slate-500 uppercase tracking-wider">Affected Systems</p>
            <p className="text-[11px] text-slate-200 font-mono">{affectedCount}</p>
          </div>
          <div className="p-1.5 border border-white/10 bg-white/[0.02] rounded">
            <p className="text-[8px] text-slate-500 uppercase tracking-wider">System Burden Index</p>
            <p className="text-[11px] text-slate-200 font-mono">{fmtPercent(avgBurden)}</p>
          </div>
          <div className="p-1.5 border border-white/10 bg-white/[0.02] rounded">
            <p className="text-[8px] text-slate-500 uppercase tracking-wider">Evidence Support</p>
            <p className="text-[11px] text-slate-200 font-mono">{fmtPercent(weightedSupport)}</p>
          </div>
          <div className="p-1.5 border border-white/10 bg-white/[0.02] rounded">
            <p className="text-[8px] text-slate-500 uppercase tracking-wider">Uncertainty</p>
            <p className="text-[11px] text-slate-200 font-mono">{fmtPercent(weightedUncertainty)}</p>
          </div>
        </div>

        <div className="flex items-center justify-between gap-2 mb-1.5">
          <p className="text-[8px] text-slate-500 uppercase tracking-wider">Dominant System</p>
          <p className="text-[8px] text-slate-500 uppercase tracking-wider">Disagreement: {disagreementLevel}</p>
        </div>
        <p className="text-[10px] text-cyan-100/90 font-mono truncate">{topSystemLabel}</p>

        {affectedCount > 0 && (
          <div className="mt-2 space-y-1">
            {ranked.slice(0, 3).map(([key, value]) => {
              const severity = value?.severity || 0;
              const label = ORGAN_SYSTEMS[key]?.shortName || key;
              const tone = severity > 0.7 ? 'bg-red-400/70' : severity > 0.4 ? 'bg-orange-400/70' : 'bg-yellow-300/70';
              return (
                <div key={key} className="flex items-center gap-2">
                  <span className="w-[64px] text-[8px] text-slate-400 uppercase tracking-wider font-mono truncate">{label}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                    <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.max(5, severity * 100)}%` }} />
                  </div>
                  <span className="text-[8px] text-slate-400 font-mono">{fmtPercent(severity)}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Systems intelligence rail (left panel) ───────────────────────────────
function SystemsIntelligenceRail({
  organs,
  selectedOrgan,
  hoveredOrgan,
  onSelect,
  onHover,
  interactionEvidence,
  isMobile,
}) {
  const summary = interactionEvidence?.evidence_summary && typeof interactionEvidence.evidence_summary === 'object'
    ? interactionEvidence.evidence_summary
    : {};

  const weightedSupport = Number(summary.weighted_support_score);
  const weightedUncertainty = Number(summary.weighted_uncertainty_score);
  const disagreementLevel = String(summary?.disagreement?.level || 'none');

  const systems = useMemo(() => {
    return Object.entries(organs)
      .map(([key, value]) => {
        const severity = value?.severity || 0;
        const sideEffectSignals = value?.sideEffects?.length || 0;
        const detailSignals = value?.details?.length || 0;
        const faersCount = value?.faersCount || 0;
        const signalScore = Math.min(1, (sideEffectSignals + detailSignals) / 8);
        const realWorldScore = Math.min(1, faersCount / 40);
        const systemScore = Math.min(1, (severity * 0.68) + (signalScore * 0.22) + (realWorldScore * 0.1));

        return {
          key,
          name: ORGAN_SYSTEMS[key]?.name || key,
          shortName: ORGAN_SYSTEMS[key]?.shortName || key,
          severity,
          signalCount: sideEffectSignals + detailSignals,
          faersCount,
          systemScore,
        };
      })
      .sort((a, b) => b.systemScore - a.systemScore);
  }, [organs]);

  const affectedCount = systems.filter((s) => s.severity > 0).length;
  const topSystem = systems[0];

  const fmtPercent = (value) => (Number.isFinite(value) ? `${Math.round(value * 100)}%` : 'N/A');

  return (
    <div className={`absolute z-20 ${isMobile ? 'left-2 right-2 top-14 max-h-[34%]' : 'left-4 top-20 w-[320px] max-h-[74%]'}`}>
      <div className="h-full rounded-xl border border-cyan-500/20 bg-[#071022]/90 backdrop-blur-md shadow-[0_0_35px_rgba(25,90,170,0.28)] overflow-hidden">
        <div className="px-3 py-2 border-b border-cyan-500/20 bg-gradient-to-r from-cyan-500/10 via-blue-500/5 to-transparent">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[9px] uppercase tracking-[0.2em] text-cyan-300/90 font-mono">Systems Intelligence</p>
            <p className="text-[8px] text-slate-400 font-mono">{affectedCount} active</p>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-1.5">
            <div className="rounded border border-white/10 bg-white/[0.02] px-1.5 py-1">
              <p className="text-[7px] text-slate-500 uppercase tracking-wider">Top</p>
              <p className="text-[8px] text-cyan-200 truncate font-mono">{topSystem?.shortName || 'N/A'}</p>
            </div>
            <div className="rounded border border-white/10 bg-white/[0.02] px-1.5 py-1">
              <p className="text-[7px] text-slate-500 uppercase tracking-wider">Support</p>
              <p className="text-[8px] text-cyan-200 font-mono">{fmtPercent(weightedSupport)}</p>
            </div>
            <div className="rounded border border-white/10 bg-white/[0.02] px-1.5 py-1">
              <p className="text-[7px] text-slate-500 uppercase tracking-wider">Uncertainty</p>
              <p className="text-[8px] text-cyan-200 font-mono">{fmtPercent(weightedUncertainty)}</p>
            </div>
          </div>
          <p className="mt-1.5 text-[7px] text-slate-500 uppercase tracking-wider">Disagreement: {disagreementLevel}</p>
        </div>

        <div className={`px-2 py-2 overflow-y-auto ${isMobile ? 'max-h-[140px]' : 'max-h-[440px]'}`}>
          <div className="space-y-1.5">
            {systems.map((system) => {
              const isSelected = selectedOrgan === system.key;
              const isHovered = hoveredOrgan === system.key;
              const tone = system.severity > 0.7
                ? 'from-red-500/20 to-red-500/5 border-red-500/35 text-red-200'
                : system.severity > 0.4
                  ? 'from-orange-500/20 to-orange-500/5 border-orange-500/35 text-orange-200'
                  : system.severity > 0
                    ? 'from-amber-500/20 to-amber-500/5 border-amber-500/35 text-amber-200'
                    : 'from-slate-500/10 to-slate-500/5 border-white/10 text-slate-300';

              return (
                <button
                  key={system.key}
                  onClick={() => onSelect?.(system.key)}
                  onMouseEnter={() => onHover?.(system.key)}
                  onMouseLeave={() => onHover?.(null)}
                  onFocus={() => onHover?.(system.key)}
                  onBlur={() => onHover?.(null)}
                  className={`w-full text-left rounded-md border bg-gradient-to-r p-2 transition-all ${tone} ${isSelected ? 'ring-1 ring-cyan-400/55 shadow-[0_0_12px_rgba(56,189,248,0.25)]' : isHovered ? 'ring-1 ring-cyan-400/40 border-cyan-400/45 shadow-[0_0_10px_rgba(56,189,248,0.2)]' : 'hover:border-cyan-400/35'}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[9px] uppercase tracking-wider font-mono truncate">{system.name}</span>
                    <span className="text-[8px] font-mono">{Math.round(system.systemScore * 100)}%</span>
                  </div>

                  <div className="mt-1.5 h-1.5 rounded-full bg-black/25 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-cyan-300/80"
                      style={{ width: `${Math.max(4, system.systemScore * 100)}%` }}
                    />
                  </div>

                  <div className="mt-1.5 flex items-center justify-between text-[7px] text-slate-400 font-mono uppercase tracking-wider">
                    <span>Impact {Math.round(system.severity * 100)}%</span>
                    <span>Signals {system.signalCount}</span>
                    <span>RWE {system.faersCount}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Right-side evidence panel ────────────────────────────────────────────
function RightEvidencePanel({
  organs,
  selectedOrgan,
  interactionEvidence,
  systemEvidence = {},
  evidenceLens = 'clinical',
  onChangeLens,
  onRunUpliftAction,
  upliftActionStatus = {},
  onRunPriorityUplift,
  priorityUpliftStatus = {},
  isMobile,
}) {
  if (isMobile) return null;

  const rankedKeys = Object.keys(organs)
    .sort((a, b) => ((organs[b]?.severity || 0) - (organs[a]?.severity || 0)));

  const activeKey = (selectedOrgan && organs[selectedOrgan]) ? selectedOrgan : rankedKeys[0];
  const activeOrgan = activeKey ? organs[activeKey] : null;
  const activeMeta = activeKey ? ORGAN_SYSTEMS[activeKey] : null;
  const activeProfile = activeKey ? systemEvidence?.[activeKey] : null;

  const evidenceChain = Array.isArray(interactionEvidence?.evidence_chain)
    ? interactionEvidence.evidence_chain.slice(0, 3)
    : [];

  const highlights = activeProfile?.evidenceHighlights || [];
  const uncertaintyReasons = activeProfile?.uncertaintyReasons || [];
  const uncertaintyTopDrivers = activeProfile?.uncertaintyTopDrivers || [];
  const uncertaintyDecomposition = activeProfile?.uncertaintyDecomposition || {};
  const monitoringFocus = activeProfile?.monitoringFocus || [];
  const sources = activeProfile?.sources || [];
  const sourceBreakdown = activeProfile?.sourceBreakdown || {};
  const sourceRecencyByCategory = activeProfile?.sourceRecencyByCategory || {};

  const impact = Number(activeOrgan?.severity);
  const confidence = Number(activeProfile?.confidenceScore);
  const certainty = Number(activeProfile?.certaintyScore);
  const confidencePotential = Number(activeProfile?.confidencePotentialScore);
  const sourceReliability = Number(activeProfile?.sourceReliabilityScore);
  const sourceConsistency = Number(activeProfile?.sourceConsistencyTrend);
  const sourceQuality = Number(activeProfile?.sourceQualityScore);
  const recencyScore = Number(activeProfile?.recencyScore);
  const support = Number(activeProfile?.support);
  const uncertainty = Number(activeProfile?.uncertainty);
  const coverage = Number(activeProfile?.coverage);
  const disagreement = String(activeProfile?.disagreementLevel || 'none');
  const confidenceBand = String(activeProfile?.confidenceBand || 'unknown');
  const freshnessLabel = String(activeProfile?.freshnessLabel || 'unknown');
  const evidenceCount = Number(activeProfile?.evidenceCount || 0);
  const recommendation = String(activeProfile?.recommendation || 'Routine surveillance');
  const confidenceUpliftPlan = activeProfile?.confidenceUpliftPlan || [];
  const priorityRunIsRunning = priorityUpliftStatus?.state === 'running';
  const priorityRunSummary = priorityUpliftStatus?.summary && typeof priorityUpliftStatus.summary === 'object'
    ? priorityUpliftStatus.summary
    : null;
  const formatPercentPointDelta = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 'N/A';
    const signed = numeric > 0 ? '+' : '';
    return `${signed}${(numeric * 100).toFixed(1)}pp`;
  };
  const formatCountDelta = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 'N/A';
    const signed = numeric > 0 ? '+' : '';
    return `${signed}${numeric.toFixed(0)}`;
  };
  const formatRefreshTime = (value) => {
    if (!value) return 'N/A';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const sourceBreakdownRows = Object.entries(sourceBreakdown)
    .map(([key, value]) => ({ key, value: Number(value) || 0 }))
    .sort((a, b) => b.value - a.value);

  const sourceRecencyRows = Object.entries(sourceRecencyByCategory)
    .map(([key, value]) => ({ key, value: Number(value) || 0 }))
    .sort((a, b) => b.value - a.value);

  const sourceLabelMap = {
    knowledgeGraph: 'Knowledge Graph',
    literature: 'Literature',
    realWorld: 'Real-world',
    mechanistic: 'Mechanistic',
    modelSignals: 'Model Signals',
  };

  const uncertaintyBreakdownRows = Object.entries(uncertaintyDecomposition)
    .map(([key, value]) => ({ key, value: Number(value) || 0 }))
    .sort((a, b) => b.value - a.value);

  const uncertaintyLabelMap = {
    dataSparsity: 'Data sparsity',
    sourceDisagreement: 'Source disagreement',
    recencyRisk: 'Recency risk',
    crossSourceVariance: 'Cross-source variance',
    realWorldGap: 'RWE gap',
  };

  const fmt = (value) => (Number.isFinite(value) ? `${Math.round(value * 100)}%` : 'N/A');

  const confidenceTone = confidenceBand === 'high'
    ? 'text-emerald-300 border-emerald-500/35 bg-emerald-500/10'
    : confidenceBand === 'medium'
      ? 'text-amber-300 border-amber-500/35 bg-amber-500/10'
      : 'text-red-300 border-red-500/35 bg-red-500/10';

  const freshnessTone = freshnessLabel === 'fresh'
    ? 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10'
    : freshnessLabel === 'recent'
      ? 'text-cyan-300 border-cyan-500/30 bg-cyan-500/10'
      : freshnessLabel === 'stale'
        ? 'text-amber-300 border-amber-500/30 bg-amber-500/10'
        : 'text-slate-300 border-white/15 bg-white/[0.03]';

  const disagreementTone = disagreement === 'high'
    ? 'text-red-300'
    : disagreement === 'medium'
      ? 'text-amber-300'
      : disagreement === 'low'
        ? 'text-cyan-300'
        : 'text-slate-400';

  return (
    <div className="absolute z-20 right-4 top-24 bottom-4 w-[320px]">
      <div className="h-full rounded-xl border border-cyan-500/20 bg-[#071022]/90 backdrop-blur-md shadow-[0_0_38px_rgba(25,90,170,0.28)] overflow-hidden flex flex-col">
        <div className="px-3 py-2 border-b border-cyan-500/20 bg-gradient-to-r from-cyan-500/10 to-transparent">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[9px] uppercase tracking-[0.2em] text-cyan-300/90 font-mono">Evidence Command</p>
            <div className="flex items-center gap-1 rounded border border-white/10 bg-white/[0.02] p-0.5">
              <button
                onClick={() => onChangeLens?.('clinical')}
                className={`px-1.5 py-0.5 text-[7px] uppercase tracking-wider font-mono rounded ${evidenceLens === 'clinical' ? 'bg-cyan-500/20 text-cyan-200' : 'text-slate-400 hover:text-slate-300'}`}
              >
                Clinical
              </button>
              <button
                onClick={() => onChangeLens?.('research')}
                className={`px-1.5 py-0.5 text-[7px] uppercase tracking-wider font-mono rounded ${evidenceLens === 'research' ? 'bg-cyan-500/20 text-cyan-200' : 'text-slate-400 hover:text-slate-300'}`}
              >
                Research
              </button>
            </div>
          </div>
          <p className="mt-1 text-[10px] text-slate-300 font-mono truncate">{activeMeta?.name || 'No active system selected'}</p>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="p-2 border-b border-white/10">
            <p className="text-[8px] text-slate-500 uppercase tracking-wider mb-1">Risk Snapshot</p>
            <div className="grid grid-cols-2 gap-1.5">
              <div className="rounded border border-white/10 bg-white/[0.02] px-1.5 py-1">
                <p className="text-[7px] text-slate-500 uppercase tracking-wider">Impact</p>
                <p className="text-[9px] text-cyan-200 font-mono">{fmt(impact)}</p>
              </div>
              <div className={`rounded border px-1.5 py-1 ${confidenceTone}`}>
                <p className="text-[7px] uppercase tracking-wider">Confidence</p>
                <p className="text-[9px] font-mono">{confidenceBand.toUpperCase()} · {fmt(confidence)}</p>
              </div>
              <div className="rounded border border-white/10 bg-white/[0.02] px-1.5 py-1">
                <p className="text-[7px] text-slate-500 uppercase tracking-wider">Evidence Nodes</p>
                <p className="text-[9px] text-cyan-200 font-mono">{evidenceCount}</p>
              </div>
              <div className={`rounded border px-1.5 py-1 ${freshnessTone}`}>
                <p className="text-[7px] uppercase tracking-wider">Freshness</p>
                <p className="text-[9px] font-mono">{freshnessLabel.toUpperCase()}</p>
              </div>
            </div>
            <div className="mt-1.5 grid grid-cols-2 gap-1.5">
              <div className="rounded border border-white/10 bg-white/[0.02] px-1.5 py-1">
                <p className="text-[7px] text-slate-500 uppercase tracking-wider">Certainty</p>
                <p className="text-[9px] text-cyan-200 font-mono">{fmt(certainty)}</p>
              </div>
              <div className="rounded border border-cyan-500/25 bg-cyan-500/8 px-1.5 py-1">
                <p className="text-[7px] text-cyan-300 uppercase tracking-wider">Confidence Potential</p>
                <p className="text-[9px] text-cyan-100 font-mono">{fmt(confidencePotential)}</p>
              </div>
            </div>
            <div className="mt-1.5 flex items-center justify-between gap-2 text-[7px] uppercase tracking-wider font-mono">
              <span className="text-slate-500">Recommendation</span>
              <span className="text-cyan-200">{recommendation}</span>
            </div>
          </div>

          <div className="px-2 py-2 border-b border-white/10">
            <p className="text-[8px] text-slate-500 uppercase tracking-wider mb-1">Why This System Is Highlighted</p>
            <div className="space-y-1">
              {highlights.length > 0 ? highlights.map((entry, index) => (
                <div key={`${entry.title || 'evidence'}-${index}`} className="rounded border border-white/10 bg-white/[0.02] px-1.5 py-1">
                  <div className="flex items-center justify-between gap-1">
                    <p className="text-[8px] text-slate-200 uppercase tracking-wider font-mono truncate">{entry.title || 'Evidence signal'}</p>
                    <span className="text-[7px] text-cyan-300 font-mono">{fmt(entry.score)}</span>
                  </div>
                  <div className="mt-0.5 flex items-center justify-between gap-1">
                    <span className="text-[7px] text-slate-500 font-mono uppercase">{entry.source || 'signal'}</span>
                    {evidenceLens === 'research' && (
                      <span className="text-[7px] text-slate-500 truncate">{entry.detail || 'No additional note'}</span>
                    )}
                  </div>
                </div>
              )) : (
                <p className="text-[8px] text-slate-500 font-mono">No organ-specific evidence highlights available yet.</p>
              )}
            </div>
          </div>

          <div className="px-2 py-2 border-b border-white/10">
            <p className="text-[8px] text-slate-500 uppercase tracking-wider mb-1">How Sure Are We</p>
            <div className="space-y-1.5">
              <div>
                <div className="flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-500">
                  <span>Support</span>
                  <span>{fmt(support)}</span>
                </div>
                <div className="mt-0.5 h-1 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-emerald-300/80" style={{ width: `${Math.max(4, (Number.isFinite(support) ? support : 0) * 100)}%` }} />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-500">
                  <span>Uncertainty</span>
                  <span>{fmt(uncertainty)}</span>
                </div>
                <div className="mt-0.5 h-1 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-amber-300/80" style={{ width: `${Math.max(4, (Number.isFinite(uncertainty) ? uncertainty : 0) * 100)}%` }} />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-500">
                  <span>Source Reliability</span>
                  <span>{fmt(sourceReliability)}</span>
                </div>
                <div className="mt-0.5 h-1 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-cyan-300/80" style={{ width: `${Math.max(4, (Number.isFinite(sourceReliability) ? sourceReliability : 0) * 100)}%` }} />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-500">
                  <span>Recency Score</span>
                  <span>{fmt(recencyScore)}</span>
                </div>
                <div className="mt-0.5 h-1 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-blue-300/80" style={{ width: `${Math.max(4, (Number.isFinite(recencyScore) ? recencyScore : 0) * 100)}%` }} />
                </div>
              </div>
            </div>
            <div className="mt-1.5 flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-500">
              <span>Coverage: {fmt(coverage)}</span>
              <span className={disagreementTone}>Disagreement: {disagreement}</span>
            </div>
            <div className="mt-1 flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-500">
              <span>Source quality: {fmt(sourceQuality)}</span>
              <span>Consistency: {fmt(sourceConsistency)}</span>
            </div>
            {sources.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {sources.slice(0, 5).map((source) => (
                  <span key={source} className="px-1 py-0.5 rounded border border-white/10 bg-white/[0.02] text-[7px] uppercase tracking-wider text-slate-400 font-mono">
                    {source}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="px-2 py-2">
            <p className="text-[8px] text-slate-500 uppercase tracking-wider mb-1">Monitor Next</p>
            <div className="space-y-1">
              {monitoringFocus.slice(0, 3).map((item, index) => (
                <div key={`${item}-${index}`} className="rounded border border-cyan-500/15 bg-cyan-500/5 px-1.5 py-1">
                  <p className="text-[8px] text-cyan-100/90 leading-relaxed">{item}</p>
                </div>
              ))}
            </div>

            {confidenceUpliftPlan.length > 0 && (
              <div className="mt-2 rounded border border-emerald-500/25 bg-emerald-500/5 px-1.5 py-1.5">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <p className="text-[7px] text-emerald-300 uppercase tracking-wider font-mono">Confidence Uplift Plan</p>
                  {onRunPriorityUplift && (
                    <button
                      onClick={onRunPriorityUplift}
                      disabled={priorityRunIsRunning}
                      className="px-1.5 py-0.5 rounded border border-emerald-500/30 text-[7px] text-emerald-200 uppercase tracking-wider font-mono hover:bg-emerald-500/10 transition-colors disabled:opacity-50"
                    >
                      {priorityRunIsRunning ? 'Running...' : 'Run Priority'}
                    </button>
                  )}
                </div>
                {onRunUpliftAction && (
                  <p className="text-[7px] text-slate-500 mb-1 leading-relaxed">
                    Execute runs live evidence refresh and re-scores confidence for this regimen.
                  </p>
                )}
                {priorityUpliftStatus?.message && (
                  <p className="text-[7px] text-slate-400 mb-1 leading-relaxed">{priorityUpliftStatus.message}</p>
                )}
                {priorityRunSummary && (
                  <div className="mb-1 rounded border border-emerald-500/15 bg-black/20 px-1 py-1 text-[7px] text-slate-400 font-mono leading-relaxed">
                    <p>Support: {(priorityRunSummary.beforeSupport * 100).toFixed(1)}% to {(priorityRunSummary.afterSupport * 100).toFixed(1)}% ({formatPercentPointDelta(priorityRunSummary.deltaSupport)})</p>
                    <p>Uncertainty: {(priorityRunSummary.beforeUncertainty * 100).toFixed(1)}% to {(priorityRunSummary.afterUncertainty * 100).toFixed(1)}% ({formatPercentPointDelta(priorityRunSummary.deltaUncertainty)})</p>
                    <p>Coverage: {(priorityRunSummary.beforeCoverage * 100).toFixed(1)}% to {(priorityRunSummary.afterCoverage * 100).toFixed(1)}% ({formatPercentPointDelta(priorityRunSummary.deltaCoverage)})</p>
                    <p>Sources: {priorityRunSummary.beforeSources} to {priorityRunSummary.afterSources} ({formatCountDelta(priorityRunSummary.deltaSources)})</p>
                    <p>Evidence nodes: {priorityRunSummary.beforeEvidenceNodes} to {priorityRunSummary.afterEvidenceNodes} ({formatCountDelta(priorityRunSummary.deltaEvidenceNodes)})</p>
                    <p>Clinical signals: {priorityRunSummary.beforeSideEffectSignals} to {priorityRunSummary.afterSideEffectSignals} ({formatCountDelta(priorityRunSummary.deltaSideEffectSignals)})</p>
                    <p>FAERS reports: {priorityRunSummary.beforeFaersReports} to {priorityRunSummary.afterFaersReports} ({formatCountDelta(priorityRunSummary.deltaFaersReports)})</p>
                    <p>Refresh: {formatRefreshTime(priorityRunSummary.refreshedAt)}</p>
                    {Array.isArray(priorityRunSummary.actionDiagnostics) && priorityRunSummary.actionDiagnostics.length > 0 && (
                      <p>
                        Actions: {priorityRunSummary.actionDiagnostics
                          .map((action) => `${action.key}(${Array.isArray(action.changedFields) && action.changedFields.length > 0 ? action.changedFields.join('/') : 'none'})`)
                          .join(', ')}
                      </p>
                    )}
                    {priorityRunSummary.noObservableMetricChange && (
                      <p className="mt-0.5 text-slate-500 leading-relaxed">
                        {priorityRunSummary.note || 'No measurable metric change was detected after refresh.'}
                      </p>
                    )}
                  </div>
                )}
                <div className="space-y-1">
                  {confidenceUpliftPlan.map((step) => {
                    const priorityTone = step.priority === 'high'
                      ? 'text-red-300 border-red-500/25 bg-red-500/10'
                      : step.priority === 'medium'
                        ? 'text-amber-300 border-amber-500/25 bg-amber-500/10'
                        : 'text-cyan-300 border-cyan-500/25 bg-cyan-500/10';
                    const stepStatus = upliftActionStatus?.[step.key] || {};
                    const isRunning = stepStatus.state === 'running';
                    const isSuccess = stepStatus.state === 'success';
                    const isError = stepStatus.state === 'error';

                    return (
                      <div key={step.key} className="rounded border border-white/10 bg-black/20 px-1.5 py-1">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[8px] text-slate-200 leading-relaxed">{step.action}</span>
                          <span className="text-[7px] text-emerald-300 font-mono whitespace-nowrap">+{fmt(step.expectedGain)}</span>
                        </div>
                        <div className="mt-0.5">
                          <span className={`inline-flex px-1 py-0.5 rounded border text-[7px] uppercase tracking-wider font-mono ${priorityTone}`}>
                            {step.priority} priority
                          </span>
                          {onRunUpliftAction && (
                            <button
                              onClick={() => onRunUpliftAction(step.key)}
                              disabled={isRunning}
                              className="ml-1 inline-flex px-1 py-0.5 rounded border border-cyan-500/30 text-[7px] uppercase tracking-wider font-mono text-cyan-200 hover:bg-cyan-500/10 transition-colors disabled:opacity-50"
                            >
                              {isRunning ? 'Running...' : isSuccess ? 'Run Again' : 'Execute'}
                            </button>
                          )}
                          {step.currentState && (
                            <p className="mt-1 text-[7px] text-slate-500 leading-relaxed">{step.currentState}</p>
                          )}
                          {stepStatus?.message && (
                            <p className={`mt-0.5 text-[7px] leading-relaxed ${isError ? 'text-red-300' : isSuccess ? 'text-emerald-300' : 'text-slate-500'}`}>
                              {stepStatus.message}
                            </p>
                          )}
                          {isError && (
                            <p className="mt-0.5 text-[7px] text-red-300 uppercase tracking-wider">Action failed</p>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {uncertaintyReasons.length > 0 && (
              <div className="mt-2 rounded border border-amber-500/20 bg-amber-500/5 px-1.5 py-1.5">
                <p className="text-[7px] text-amber-300 uppercase tracking-wider font-mono mb-1">Uncertainty Reasons</p>
                <div className="space-y-1">
                  {uncertaintyReasons.slice(0, 3).map((reason, index) => (
                    <p key={`${reason}-${index}`} className="text-[8px] text-slate-300 leading-relaxed">{reason}</p>
                  ))}
                </div>
              </div>
            )}

            {evidenceLens === 'research' && evidenceChain.length > 0 && (
              <div className="mt-2 rounded border border-white/10 bg-white/[0.02] px-1.5 py-1.5">
                <p className="text-[7px] text-slate-500 uppercase tracking-wider font-mono mb-1">Evidence Chain</p>
                <div className="space-y-1">
                  {evidenceChain.map((item, index) => (
                    <div key={`chain-${index}`} className="rounded border border-white/10 bg-black/20 px-1.5 py-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[7px] text-cyan-300 uppercase tracking-wider font-mono">{item?.source || item?.source_name || 'source'}</span>
                        <span className="text-[7px] text-slate-500 font-mono">{fmt(Number(item?.support ?? item?.strength))}</span>
                      </div>
                      <p className="mt-0.5 text-[8px] text-slate-300 leading-relaxed">
                        {item?.claim || item?.summary || item?.evidence || item?.name || 'Evidence item'}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {evidenceLens === 'research' && sourceBreakdownRows.length > 0 && (
              <div className="mt-2 rounded border border-white/10 bg-white/[0.02] px-1.5 py-1.5">
                <p className="text-[7px] text-slate-500 uppercase tracking-wider font-mono mb-1">Source Signal Mix</p>
                <div className="space-y-1">
                  {sourceBreakdownRows.slice(0, 5).map((row) => (
                    <div key={row.key}>
                      <div className="flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-400">
                        <span>{sourceLabelMap[row.key] || row.key}</span>
                        <span>{fmt(row.value)}</span>
                      </div>
                      <div className="mt-0.5 h-1 rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full rounded-full bg-cyan-300/75" style={{ width: `${Math.max(4, row.value * 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {evidenceLens === 'research' && sourceRecencyRows.length > 0 && (
              <div className="mt-2 rounded border border-cyan-500/20 bg-cyan-500/5 px-1.5 py-1.5">
                <p className="text-[7px] text-cyan-300 uppercase tracking-wider font-mono mb-1">Source Recency Decay</p>
                <div className="space-y-1">
                  {sourceRecencyRows.slice(0, 5).map((row) => (
                    <div key={row.key}>
                      <div className="flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-400">
                        <span>{sourceLabelMap[row.key] || row.key}</span>
                        <span>{fmt(row.value)}</span>
                      </div>
                      <div className="mt-0.5 h-1 rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full rounded-full bg-cyan-300/75" style={{ width: `${Math.max(4, row.value * 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {evidenceLens === 'research' && uncertaintyBreakdownRows.length > 0 && (
              <div className="mt-2 rounded border border-amber-500/25 bg-amber-500/5 px-1.5 py-1.5">
                <p className="text-[7px] text-amber-300 uppercase tracking-wider font-mono mb-1">Uncertainty Decomposition</p>
                <div className="space-y-1">
                  {uncertaintyBreakdownRows.slice(0, 4).map((row) => (
                    <div key={row.key}>
                      <div className="flex items-center justify-between text-[7px] uppercase tracking-wider font-mono text-slate-400">
                        <span>{uncertaintyLabelMap[row.key] || row.key}</span>
                        <span>{fmt(row.value)}</span>
                      </div>
                      <div className="mt-0.5 h-1 rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full rounded-full bg-amber-300/75" style={{ width: `${Math.max(4, row.value * 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>

                {uncertaintyTopDrivers.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {uncertaintyTopDrivers.map((driver) => (
                      <span key={driver.key} className="px-1 py-0.5 rounded border border-amber-500/20 bg-amber-500/10 text-[7px] text-amber-200 uppercase tracking-wider font-mono">
                        {driver.label}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
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
  onRunUpliftAction,
  upliftActionStatus = {},
  onRunPriorityUplift,
  priorityUpliftStatus = {},
  isMobile = false,
}) {
  const [selectedOrgan, setSelectedOrgan] = useState(null);
  const [hoveredOrgan, setHoveredOrgan] = useState(null);
  const [showOnlyAffected, setShowOnlyAffected] = useState(false);
  const [evidenceLens, setEvidenceLens] = useState('clinical');
  const [layers, setLayers] = useState({
    circulatory: true,
    heatmap: true,
    skeleton: false,
  });

  // Enrich data from all sources
  const enriched = useMemo(() => {
    return enrichBodyMapData({
      affectedSystems,
      drugs,
      drugInfoCache,
      interactionEvidence,
      polypharmacyResult,
      result,
    });
  }, [affectedSystems, drugs, drugInfoCache, interactionEvidence, polypharmacyResult, result]);

  const handleOrganClick = useCallback((organKey) => {
    setSelectedOrgan(prev => prev === organKey ? null : organKey);
  }, []);

  const handleLayerToggle = useCallback((layerKey) => {
    setLayers((prev) => ({ ...prev, [layerKey]: !prev[layerKey] }));
  }, []);

  const handleToggleAffectedOnly = useCallback(() => {
    setShowOnlyAffected((prev) => !prev);
  }, []);

  const handleCloseDetail = useCallback(() => {
    setSelectedOrgan(null);
  }, []);


  const hasDrugs = drugs.length > 0;
  const hasResult = Object.keys(affectedSystems).length > 0 || !enriched.isEmpty;

  return (
    <div
      className="relative w-full h-full overflow-hidden"
      style={{ background: 'radial-gradient(ellipse at top, #0a1529 0%, #050b15 45%, #02040a 100%)' }}
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

      {/* Layer controls for overlay stack */}
      <LayerToggle layers={layers} onToggle={handleLayerToggle} isMobile={isMobile} />

      {/* Left-side systems intelligence rail */}
      {!enriched.isEmpty && (
        <SystemsIntelligenceRail
          organs={enriched.organs}
          selectedOrgan={selectedOrgan}
          hoveredOrgan={hoveredOrgan}
          onSelect={handleOrganClick}
          onHover={setHoveredOrgan}
          interactionEvidence={interactionEvidence}
          isMobile={isMobile}
        />
      )}

      {/* Compact navigator on mobile */}
      {!enriched.isEmpty && isMobile && (
        <OrganNavigator
          organs={enriched.organs}
          selectedOrgan={selectedOrgan}
          onSelect={handleOrganClick}
          showOnlyAffected={showOnlyAffected}
          onToggleShowOnlyAffected={handleToggleAffectedOnly}
          isMobile={isMobile}
        />
      )}

      {/* Fill right side with evidence intelligence */}
      {!enriched.isEmpty && (
        <RightEvidencePanel
          organs={enriched.organs}
          selectedOrgan={selectedOrgan}
          interactionEvidence={interactionEvidence}
          systemEvidence={enriched.systemEvidence}
          evidenceLens={evidenceLens}
          onChangeLens={setEvidenceLens}
          onRunUpliftAction={onRunUpliftAction}
          upliftActionStatus={upliftActionStatus}
          onRunPriorityUplift={onRunPriorityUplift}
          priorityUpliftStatus={priorityUpliftStatus}
          isMobile={isMobile}
        />
      )}

      {/* Empty state */}
      {enriched.isEmpty && <EmptyState hasDrugs={hasDrugs} hasResult={hasResult} />}

      {/* Visual stack */}
      <div className={`relative w-full h-full flex items-center ${isMobile ? 'justify-center' : 'justify-center pl-[290px] pr-[290px]'}`}>
        {/* Heat map (background) */}
        <HeatMapCanvas
          organs={enriched.organs}
          visible={layers.heatmap && !enriched.isEmpty}
        />

        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute left-1/2 top-1/2 h-[560px] w-[360px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-400/12 blur-3xl" />
          <div className="absolute left-1/2 top-1/2 h-[420px] w-[260px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-blue-500/10 blur-2xl" />
        </div>

        <div className="relative z-[3] w-full h-full flex items-center justify-center">
          <SegmentedBodyFigure
            organs={enriched.organs}
            selectedOrgan={selectedOrgan}
            hoveredOrgan={hoveredOrgan}
            onSelectOrgan={setSelectedOrgan}
            showOnlyAffected={showOnlyAffected}
            showCirculatory={layers.circulatory && !enriched.isEmpty}
            showSkeleton={layers.skeleton}
            isMobile={isMobile}
          />
        </div>
      </div>

      {/* Top affected summary */}
      {!enriched.isEmpty && !isMobile && (
        <AffectedSummary organs={enriched.organs} isMobile={isMobile} />
      )}

      {/* Bottom center research strip */}
      {!enriched.isEmpty && (
        <ClinicalIntelPanel
          organs={enriched.organs}
          interactionEvidence={interactionEvidence}
          isMobile={isMobile}
        />
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
const SYSTEM_REGION_LAYOUT = {
  brain: { radius: 24, labelDy: -34 },
  endocrine: { radius: 12, offsetY: -14, labelDy: -24 },
  lungs: { variant: 'paired', radius: 22, spread: 36, labelDy: -34 },
  heart: { radius: 16, offsetX: 18, offsetY: 8, labelDy: -26 },
  liver: { radius: 26, offsetX: -16, offsetY: 10, labelDy: -36 },
  gi: { radius: 28, offsetY: 38, labelDy: -40 },
  kidney: { variant: 'paired', radius: 15, spread: 50, offsetY: 30, labelDy: -30 },
  blood: { variant: 'halo', radius: 74, offsetY: 12, labelDy: -86, strokeWidth: 2.2 },
  skin: { variant: 'halo', radius: 96, offsetY: 24, labelDy: -110, strokeWidth: 1.7, dashed: true },
  musculoskeletal: { radius: 34, offsetY: 98, labelDy: -42 },
};

function AnatomicalSVGInline({
  organs = {},
  selectedOrgan,
  showSkeleton = true,
  showOnlyAffected = false,
  renderSilhouette = true,
  onOrganClick,
  onOrganHover,
}) {
  const [hoveredOrgan, setHoveredOrgan] = useState(null);
  const [hoveredAnchor, setHoveredAnchor] = useState(null);

  const handleHover = (key, anchor = null) => {
    setHoveredOrgan(key);
    setHoveredAnchor(anchor);
    onOrganHover?.(key);
  };

  const regionSystems = useMemo(() => {
    return Object.entries(ORGAN_SYSTEMS)
      .sort((a, b) => {
        const severityA = organs[a[0]]?.severity || 0;
        const severityB = organs[b[0]]?.severity || 0;
        const affectedA = severityA > 0;
        const affectedB = severityB > 0;
        if (affectedA !== affectedB) return affectedA ? 1 : -1;
        return severityA - severityB;
      });
  }, [organs]);

  const hoveredDef = hoveredOrgan ? ORGAN_SYSTEMS[hoveredOrgan] : null;
  const hoveredSeverity = hoveredOrgan ? (organs[hoveredOrgan]?.severity || 0) : 0;
  const hasAffectedOrgans = Object.values(organs || {}).some((organ) => (organ?.severity || 0) > 0);

  const silhouetteOpacity = hasAffectedOrgans ? 0.32 : 0.46;
  const coreOpacity = hasAffectedOrgans ? 0.16 : 0.26;
  const outlineAlpha = hasAffectedOrgans ? 0.26 : 0.44;
  const gridOpacity = hasAffectedOrgans ? 0.22 : 0.34;

  return (
    <g>
      {renderSilhouette && (
        <>
          {/* Human silhouette depth layers */}
          <path d={BODY_OUTLINE.torso} fill="url(#body-surface-gradient)" opacity={silhouetteOpacity} filter="url(#body-soft-shadow)" />
          <path d={BODY_OUTLINE.torso} fill="url(#body-core-gradient)" opacity={coreOpacity} />

          {/* Body outline */}
          {Object.entries(BODY_OUTLINE).map(([key, path]) => (
            <path
              key={`outline-${key}`}
              d={path}
              fill="none"
              stroke={`rgba(134, 195, 255, ${outlineAlpha})`}
              strokeWidth={key === 'torso' ? 1.8 : 1.4}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          <path d={BODY_OUTLINE.torso} fill="url(#body-grid)" opacity={gridOpacity} />
        </>
      )}

      {/* Skeletal hints */}
      {showSkeleton && renderSilhouette && (
        <g opacity={0.24}>
          {SKELETAL_HINTS.ribs.map((rib, i) => (
            <path key={`rib-${i}`} d={rib} fill="none" stroke="rgba(150, 200, 255, 0.3)" strokeWidth="0.8" strokeDasharray="3,3" />
          ))}
          <path d={SKELETAL_HINTS.spine} fill="none" stroke="rgba(150, 200, 255, 0.25)" strokeWidth="1" strokeDasharray="2,4" />
          <path d={SKELETAL_HINTS.pelvis} fill="none" stroke="rgba(150, 200, 255, 0.2)" strokeWidth="0.8" strokeDasharray="3,3" />
          <path d={SKELETAL_HINTS.skull} fill="none" stroke="rgba(150, 200, 255, 0.2)" strokeWidth="0.8" strokeDasharray="2,3" />
        </g>
      )}

      {/* Anatomical landmarks for realism */}
      {renderSilhouette && (
        <g opacity={0.22}>
          {Object.entries(ANATOMY_LANDMARKS).map(([key, path]) => (
            <path key={`landmark-${key}`} d={path} fill="none" stroke="rgba(165, 212, 255, 0.28)" strokeWidth="0.7" />
          ))}
        </g>
      )}

      {/* Region-based clinical system bubbles */}
      {regionSystems.map(([key, def]) => {
        const severity = organs[key]?.severity || 0;
        const isAffected = severity > 0;
        const isHovered = hoveredOrgan === key;
        const isSelected = selectedOrgan === key;
        const colorInfo = getSeverityColor(severity);
        const layout = SYSTEM_REGION_LAYOUT[key] || {};
        const isHalo = layout.variant === 'halo';
        const radius = layout.radius || 18;
        const anchorX = def.center.x + (layout.offsetX || 0);
        const anchorY = def.center.y + (layout.offsetY || 0);
        const labelX = anchorX + (layout.labelDx || 0);
        const labelY = anchorY + (layout.labelDy || -(radius + 10));

        const nodes = layout.variant === 'paired'
          ? [
            { cx: anchorX - (layout.spread || 34) / 2, cy: anchorY, r: radius },
            { cx: anchorX + (layout.spread || 34) / 2, cy: anchorY, r: radius },
          ]
          : [{ cx: anchorX, cy: anchorY, r: radius }];

        if (showOnlyAffected && !isAffected) return null;

        const glowFilter = severity > 0.7 ? 'url(#organ-glow-severe)'
          : severity > 0.4 ? 'url(#organ-glow-moderate)'
          : severity > 0 ? 'url(#organ-glow-mild)' : 'none';

        const fillColor = isHalo
          ? 'none'
          : (isAffected ? colorInfo.fill : 'rgba(76, 126, 206, 0.22)');
        const fillOpacity = isHalo
          ? 0
          : (isAffected ? 0.2 + severity * 0.26 : isHovered || isSelected ? 0.12 : 0.06);
        const strokeColor = isSelected
          ? 'rgba(56, 221, 255, 0.96)'
          : (isAffected ? colorInfo.fill : 'rgba(120, 171, 242, 0.34)');
        const strokeOpacity = isHalo
          ? (isAffected ? 0.42 + severity * 0.45 : isHovered ? 0.28 : 0.14)
          : (isAffected ? 0.7 + severity * 0.24 : isHovered ? 0.46 : 0.24);
        const strokeWidth = isSelected
          ? (layout.strokeWidth || (isHalo ? 2 : 1.3)) + 0.7
          : (layout.strokeWidth || (isHalo ? 2 : 1.3));

        return (
          <g
            key={key}
            className="cursor-pointer"
            onMouseEnter={() => handleHover(key, { x: anchorX, y: anchorY })}
            onMouseLeave={() => handleHover(null, null)}
            onClick={() => onOrganClick?.(key)}
          >
            {nodes.map((node, index) => (
              <circle
                key={`${key}-${index}`}
                cx={node.cx}
                cy={node.cy}
                r={node.r}
                fill={fillColor}
                fillOpacity={fillOpacity}
                stroke={strokeColor}
                strokeOpacity={strokeOpacity}
                strokeWidth={strokeWidth}
                strokeDasharray={layout.dashed ? '5,4' : undefined}
                filter={isAffected ? glowFilter : 'none'}
                style={{ transition: 'fill-opacity 0.35s, stroke-opacity 0.35s, stroke 0.35s' }}
              />
            ))}

            {isSelected && nodes.map((node, index) => (
              <circle
                key={`${key}-selected-${index}`}
                cx={node.cx}
                cy={node.cy}
                r={node.r + 6}
                fill="none"
                stroke="rgba(56, 221, 255, 0.72)"
                strokeWidth="1.2"
                strokeDasharray="4,3"
              >
                <animate attributeName="stroke-dashoffset" values="0;-14" dur="2s" repeatCount="indefinite" />
              </circle>
            ))}

            {(isHovered || isSelected) && (
              <line
                x1={anchorX}
                y1={anchorY - radius}
                x2={labelX}
                y2={labelY + 3}
                stroke="rgba(120, 188, 255, 0.45)"
                strokeWidth="0.9"
                strokeDasharray="2,2"
              />
            )}

            {/* Pulse animation for severe organs */}
            {severity > 0.6 && nodes.map((node, index) => (
              <circle key={`${key}-pulse-${index}`} cx={node.cx} cy={node.cy} r={node.r} fill="none" stroke={colorInfo.fill} strokeWidth="1.6" strokeOpacity="0">
                <animate attributeName="r" values={`${node.r};${node.r + 4};${node.r}`} dur="1.8s" repeatCount="indefinite" />
                <animate attributeName="stroke-opacity" values="0;0.55;0" dur="1.8s" repeatCount="indefinite" />
              </circle>
            ))}

            {/* Region label */}
            {(isHovered || isAffected || isSelected) && (
              <text
                x={labelX}
                y={labelY}
                textAnchor="middle"
                dominantBaseline="central"
                fill={isAffected ? colorInfo.fill : 'rgba(180, 210, 255, 0.6)'}
                fontSize={isSelected ? '8.5' : '7.5'}
                fontFamily="monospace"
                fontWeight="bold"
                letterSpacing="0.4"
                style={{ pointerEvents: 'none', textTransform: 'uppercase' }}
              >
                {def.shortName}
              </text>
            )}

            {/* Severity dot */}
            {isAffected && (
              <circle cx={labelX + 22} cy={labelY - 2} r="2.8" fill={colorInfo.fill} opacity={0.85}>
                {severity > 0.6 && (
                  <animate attributeName="r" values="2.8;4;2.8" dur="1.5s" repeatCount="indefinite" />
                )}
              </circle>
            )}
          </g>
        );
      })}

      {/* Hover tooltip */}
      {hoveredDef && (
        <g>
          {(() => {
            const anchor = hoveredAnchor || hoveredDef.center;
            const boxX = Math.min(308, Math.max(16, anchor.x + 18));
            const boxY = Math.min(748, Math.max(16, anchor.y - 22));
            const textX = Math.min(345, Math.max(22, anchor.x + 24));
            const nameY = Math.min(762, Math.max(29, anchor.y - 6));
            const impactY = Math.min(772, Math.max(39, anchor.y + 4));
            return (
              <>
          <rect
            x={boxX}
            y={boxY}
            width="74"
            height="24"
            rx="4"
            fill="rgba(6, 11, 20, 0.88)"
            stroke="rgba(145, 200, 255, 0.35)"
            strokeWidth="0.8"
          />
          <text
            x={textX}
            y={nameY}
            fill="rgba(192, 223, 255, 0.9)"
            fontSize="8"
            fontFamily="monospace"
          >
            {hoveredDef.shortName}
          </text>
          <text
            x={textX}
            y={impactY}
            fill={getSeverityColor(hoveredSeverity).fill}
            fontSize="7"
            fontFamily="monospace"
          >
            {Math.round(hoveredSeverity * 100)}% impact
          </text>
              </>
            );
          })()}
        </g>
      )}
    </g>
  );
}
