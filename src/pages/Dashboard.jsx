import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Plus,
  Trash2,
  Zap,
  AlertTriangle,
  Shield,
  Activity,
  Send,
  Loader2,
  ChevronLeft,
  Settings,
  Bell,
  User,
  Sparkles,
  Network,
  Heart,
  Brain,
  Database,
  Beaker,
  FileText,
  X,
  Check,
  AlertCircle,
  TrendingUp,
  GitBranch,
  GitCompare,
  RefreshCw,
  ExternalLink,
  Microscope,
  Pill,
  Target,
  Layers,
  Box,
  Hexagon,
  BarChart3,
  Lightbulb,
  Sun,
  Moon,
  Menu,
  ChevronUp,
  ChevronDown,
  MessageCircle,
  Home,
  FlaskConical,
  Camera,
  ScanLine
} from 'lucide-react';
import { useSystemLogs } from '../hooks/useSystemLogs';
import { useTheme } from '../hooks/useTheme';
import { searchDrugs, predictDDI, analyzePolypharmacy, analyzePolypharmacyDigitalTwin, sendChatMessage, checkHealth, getDrugInfo, getInteractionInfo, getDatabaseStats, computeCalibrationMetrics } from '../services/api';
import { DEMO_DRUG_GROUPS, readDemoModeSetting } from '../config/demoMode';
import GNNGalaxyViewer from '../components/GalaxyViewer';
import MoleculeViewer2D from '../components/MoleculeViewer2D';
import BodyMap from '../components/BodyMap';
import KnowledgeGraphView from '../components/KnowledgeGraph';
import PolypharmacyDigitalTwin from '../components/PolypharmacyDigitalTwin';
import RiskGauge from '../components/RiskGauge';
import StatsDashboard from '../components/StatsDashboard';
import DrugComparison from '../components/DrugComparison';
import TherapeuticAlternatives from '../components/TherapeuticAlternatives';
import { DrugScanner } from '../components/DrugScanner';

// Debounce hook
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const handler = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(handler);
  }, [value, delay]);
  return debouncedValue;
}

const SELECTED_DRUGS_STORAGE_KEY = 'aegis:selectedDrugs:v1';

function pickBestSearchMatch(results, requestedName) {
  if (!Array.isArray(results) || results.length === 0) return null;

  const target = String(requestedName || '').trim().toLowerCase();
  if (!target) return results[0];

  const exact = results.find((item) => String(item?.name || '').trim().toLowerCase() === target);
  if (exact) return exact;

  const startsWith = results.find((item) => String(item?.name || '').trim().toLowerCase().startsWith(target));
  if (startsWith) return startsWith;

  const contains = results.find((item) => String(item?.name || '').trim().toLowerCase().includes(target));
  if (contains) return contains;

  return results[0];
}

function loadStoredSelectedDrugs() {
  if (typeof window === 'undefined') return [];

  try {
    const raw = window.localStorage.getItem(SELECTED_DRUGS_STORAGE_KEY);
    if (!raw) return [];

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    const seen = new Set();
    const sanitized = [];

    parsed.forEach((drug) => {
      if (!drug || typeof drug !== 'object') return;
      if (!drug.name || typeof drug.name !== 'string') return;

      const stableId = String(drug.drugbank_id || drug.id || drug.name).toLowerCase();
      if (seen.has(stableId)) return;

      seen.add(stableId);
      sanitized.push(drug);
    });

    return sanitized;
  } catch {
    return [];
  }
}

function normalizeMetaValue(value) {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  if (typeof value !== 'string') return null;

  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function formatPercent(score) {
  if (typeof score !== 'number' || !Number.isFinite(score)) return 'N/A';
  return `${(score * 100).toFixed(1)}%`;
}

const DEFAULT_CALIBRATION_CSV = [
  'label,raw_score,calibrated_score',
  '0,0.40,0.10',
  '0,0.35,0.08',
  '1,0.60,0.90',
  '1,0.55,0.88',
  '1,0.70,0.95',
  '0,0.30,0.12',
].join('\n');

function parseCalibrationCsv(csvText) {
  const lines = csvText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) {
    throw new Error('Provide at least one CSV row of label/raw_score/calibrated_score data.');
  }

  const labels = [];
  const rawScores = [];
  const calibratedScores = [];

  lines.forEach((line, idx) => {
    const cols = line.split(',').map((x) => x.trim());
    if (cols.length < 3) {
      throw new Error(`Line ${idx + 1} is invalid. Expected 3 comma-separated values.`);
    }

    // Allow a header row such as: label,raw_score,calibrated_score
    if (
      idx === 0 &&
      cols[0].toLowerCase().includes('label') &&
      cols[1].toLowerCase().includes('raw')
    ) {
      return;
    }

    const label = Number(cols[0]);
    const raw = Number(cols[1]);
    const calibrated = Number(cols[2]);

    if (!Number.isInteger(label) || (label !== 0 && label !== 1)) {
      throw new Error(`Line ${idx + 1}: label must be 0 or 1.`);
    }
    if (!Number.isFinite(raw) || raw < 0 || raw > 1) {
      throw new Error(`Line ${idx + 1}: raw_score must be between 0 and 1.`);
    }
    if (!Number.isFinite(calibrated) || calibrated < 0 || calibrated > 1) {
      throw new Error(`Line ${idx + 1}: calibrated_score must be between 0 and 1.`);
    }

    labels.push(label);
    rawScores.push(raw);
    calibratedScores.push(calibrated);
  });

  if (!labels.length) {
    throw new Error('No calibration rows found after header parsing.');
  }

  return { labels, rawScores, calibratedScores };
}

function formatMetric(value, digits = 4) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'N/A';
  return value.toFixed(digits);
}

function getCalibrationRunErrorMessage(err) {
  const rawMessage = String(err?.message || 'Failed to run calibration QA');
  const compactMessage = rawMessage.trim();

  const isCalibration404 =
    /Page not found at\s*\/api\/v1\/calibration\/metrics\//i.test(compactMessage)
    || /calibration\/metrics/i.test(compactMessage) && /status:\s*404/i.test(compactMessage)
    || /Route not found:\s*\/api\/v1\/calibration\/metrics\//i.test(compactMessage);

  if (isCalibration404) {
    return 'Calibration endpoint is missing on the running backend. Restart backend (or rebuild container) so /api/v1/calibration/metrics/ is loaded.';
  }

  if (compactMessage.startsWith('<!DOCTYPE html>') || compactMessage.startsWith('<html')) {
    return 'Backend returned an HTML error page. Verify backend is running the latest code and route configuration.';
  }

  return compactMessage;
}

function getCalibrationQualityBand(eceValue) {
  if (typeof eceValue !== 'number' || !Number.isFinite(eceValue)) {
    return {
      label: 'Unknown',
      tone: 'text-theme-muted border-theme/30 bg-theme-panel/60',
    };
  }

  if (eceValue <= 0.03) {
    return {
      label: 'Excellent',
      tone: 'text-risk-low border-risk-low/40 bg-risk-low/10',
    };
  }
  if (eceValue <= 0.08) {
    return {
      label: 'Good',
      tone: 'text-theme-accent border-theme-accent/40 bg-theme-accent/10',
    };
  }
  if (eceValue <= 0.15) {
    return {
      label: 'Needs Review',
      tone: 'text-risk-medium border-risk-medium/40 bg-risk-medium/10',
    };
  }

  return {
    label: 'Poor',
    tone: 'text-risk-high border-risk-high/40 bg-risk-high/10',
  };
}

function extractBinSeries(section) {
  const bins = Array.isArray(section?.bins) ? section.bins : [];
  return bins
    .filter((bin) => Number(bin?.count) > 0)
    .map((bin) => ({
      index: Number(bin.bin_index),
      count: Number(bin.count),
      start: Number(bin.start),
      end: Number(bin.end),
      confidence: Number(bin.avg_confidence),
      accuracy: Number(bin.empirical_accuracy),
      gap: Number(bin.gap),
    }))
    .filter((bin) => Number.isFinite(bin.confidence) && Number.isFinite(bin.accuracy));
}

function buildReliabilityLinePath(points, width, height, padding) {
  if (!points.length) return '';

  const minX = padding;
  const maxX = width - padding;
  const minY = padding;
  const maxY = height - padding;

  return points
    .map((point, idx) => {
      const x = minX + point.confidence * (maxX - minX);
      const y = maxY - point.accuracy * (maxY - minY);
      return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

function ReliabilityCurveChart({ rawSection, calibratedSection }) {
  const width = 360;
  const height = 240;
  const padding = 30;

  const rawPoints = extractBinSeries(rawSection);
  const calibratedPoints = extractBinSeries(calibratedSection);

  const rawPath = buildReliabilityLinePath(rawPoints, width, height, padding);
  const calibratedPath = buildReliabilityLinePath(calibratedPoints, width, height, padding);
  const diagonalPath = `M ${padding} ${height - padding} L ${width - padding} ${padding}`;

  const mapPoint = (point) => {
    const x = padding + point.confidence * (width - 2 * padding);
    const y = height - padding - point.accuracy * (height - 2 * padding);
    return { x, y };
  };

  return (
    <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[9px] text-theme-muted uppercase tracking-wider">Reliability Curve</span>
        <span className="text-[8px] text-theme-dim uppercase tracking-wider">Confidence vs Empirical Accuracy</span>
      </div>

      <div className="border border-theme/40 bg-theme-primary/80">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-52" role="img" aria-label="Reliability curve chart">
          {/* Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const x = padding + tick * (width - 2 * padding);
            const y = height - padding - tick * (height - 2 * padding);
            return (
              <g key={tick}>
                <line x1={x} y1={padding} x2={x} y2={height - padding} stroke="currentColor" className="text-theme/10" strokeWidth="1" />
                <line x1={padding} y1={y} x2={width - padding} y2={y} stroke="currentColor" className="text-theme/10" strokeWidth="1" />
              </g>
            );
          })}

          {/* Axes */}
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="currentColor" className="text-theme/60" strokeWidth="1.2" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="currentColor" className="text-theme/60" strokeWidth="1.2" />

          {/* Perfect calibration diagonal */}
          <path d={diagonalPath} fill="none" stroke="currentColor" className="text-theme-muted/60" strokeWidth="1.2" strokeDasharray="5 4" />

          {/* Raw and calibrated lines */}
          {rawPath && <path d={rawPath} fill="none" stroke="#f59e0b" strokeWidth="2.2" />}
          {calibratedPath && <path d={calibratedPath} fill="none" stroke="#22d3ee" strokeWidth="2.2" />}

          {/* Raw points */}
          {rawPoints.map((point) => {
            const p = mapPoint(point);
            return (
              <circle key={`raw-${point.index}`} cx={p.x} cy={p.y} r="3.3" fill="#f59e0b" opacity="0.95" />
            );
          })}

          {/* Calibrated points */}
          {calibratedPoints.map((point) => {
            const p = mapPoint(point);
            return (
              <circle key={`cal-${point.index}`} cx={p.x} cy={p.y} r="3.1" fill="#22d3ee" opacity="0.95" />
            );
          })}
        </svg>
      </div>

      <div className="mt-2 flex flex-wrap gap-2 text-[8px] uppercase tracking-wider">
        <span className="px-2 py-1 border border-theme-muted/40 text-theme-muted">Dashed = Perfect</span>
        <span className="px-2 py-1 border border-risk-medium/40 text-risk-medium">Raw</span>
        <span className="px-2 py-1 border border-theme-accent/40 text-theme-accent">Calibrated</span>
      </div>
    </div>
  );
}

function CalibrationBinTable({ title, bins, tone }) {
  const rows = extractBinSeries({ bins });

  return (
    <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[9px] text-theme-muted uppercase tracking-wider">{title}</span>
        <span className={`text-[8px] uppercase tracking-wider ${tone}`}>{rows.length} bins with data</span>
      </div>

      {rows.length === 0 ? (
        <p className="text-[10px] text-theme-dim">No populated bins.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[9px]">
            <thead>
              <tr className="text-theme-muted uppercase tracking-wider border-b border-theme/20">
                <th className="text-left py-1 pr-2">Bin</th>
                <th className="text-right py-1 px-2">N</th>
                <th className="text-right py-1 px-2">Conf</th>
                <th className="text-right py-1 px-2">Acc</th>
                <th className="text-right py-1 pl-2">Gap</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${title}-${row.index}`} className="border-b border-theme/10 text-theme-secondary">
                  <td className="py-1 pr-2">{formatMetric(row.start, 2)}-{formatMetric(row.end, 2)}</td>
                  <td className="py-1 px-2 text-right">{row.count}</td>
                  <td className="py-1 px-2 text-right">{formatMetric(row.confidence, 3)}</td>
                  <td className="py-1 px-2 text-right">{formatMetric(row.accuracy, 3)}</td>
                  <td className="py-1 pl-2 text-right">{formatMetric(row.gap, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function getTransparencyData(result) {
  if (!result || typeof result !== 'object') return null;

  const provenance = result.provenance && typeof result.provenance === 'object'
    ? result.provenance
    : {};
  const explanation = result.explanation && typeof result.explanation === 'object'
    ? result.explanation
    : {};
  const calibration = explanation.calibration && typeof explanation.calibration === 'object'
    ? explanation.calibration
    : {};

  const rawScore = typeof result.raw_score === 'number' ? result.raw_score : null;
  const calibratedScore = typeof result.calibrated_score === 'number'
    ? result.calibrated_score
    : (typeof result.risk_score === 'number' ? result.risk_score : null);

  const modelVersion = normalizeMetaValue(provenance.model_version) || normalizeMetaValue(explanation.model_version);
  const modelUsed = normalizeMetaValue(provenance.model_used) || normalizeMetaValue(result.source);
  const predictionPath = normalizeMetaValue(provenance.prediction_path);

  const calibrationMethod = normalizeMetaValue(provenance.calibration_method)
    || normalizeMetaValue(calibration.method);
  const calibrationVersion = normalizeMetaValue(provenance.calibration_version)
    || normalizeMetaValue(calibration.version);
  const fallbackReason = normalizeMetaValue(provenance.fallback_reason)
    || normalizeMetaValue(explanation.fallback_reason);

  const hasPanelData =
    rawScore !== null ||
    calibratedScore !== null ||
    modelVersion ||
    modelUsed ||
    predictionPath ||
    calibrationMethod ||
    calibrationVersion ||
    fallbackReason;

  if (!hasPanelData) return null;

  let calibrationDelta = null;
  if (rawScore !== null && calibratedScore !== null) {
    calibrationDelta = calibratedScore - rawScore;
  }

  return {
    rawScore,
    calibratedScore,
    calibrationDelta,
    modelVersion,
    modelUsed,
    predictionPath,
    calibrationMethod,
    calibrationVersion,
    fallbackReason,
  };
}

function PredictionTransparencyPanel({ result, isMobile = false }) {
  const data = getTransparencyData(result);
  if (!data) return null;

  const labelClass = isMobile ? 'text-[10px]' : 'text-[9px]';
  const valueClass = isMobile ? 'text-xs' : 'text-[10px]';

  return (
    <div className="p-4 border border-theme-accent/30 bg-theme-accent/5 relative">
      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-theme-accent/70"></div>
      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-theme-accent/70"></div>

      <div className="flex items-center gap-2 mb-3">
        <Layers className="w-3.5 h-3.5 text-theme-accent" />
        <span className="text-[10px] text-theme-muted uppercase tracking-widest">Model Transparency</span>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between gap-3">
          <span className={`${labelClass} text-theme-muted uppercase tracking-wider`}>Raw Model Score</span>
          <span className={`${valueClass} text-theme-secondary`}>{formatPercent(data.rawScore)}</span>
        </div>

        <div className="flex items-center justify-between gap-3">
          <span className={`${labelClass} text-theme-muted uppercase tracking-wider`}>Calibrated Score</span>
          <span className={`${valueClass} text-theme-accent`}>{formatPercent(data.calibratedScore)}</span>
        </div>

        {data.calibrationDelta !== null && (
          <div className="flex items-center justify-between gap-3">
            <span className={`${labelClass} text-theme-muted uppercase tracking-wider`}>Calibration Shift</span>
            <span className={`${valueClass} ${data.calibrationDelta === 0 ? 'text-theme-secondary' : 'text-risk-medium'}`}>
              {data.calibrationDelta > 0 ? '+' : ''}{(data.calibrationDelta * 100).toFixed(1)} pp
            </span>
          </div>
        )}

        {data.modelVersion && (
          <div className="flex items-center justify-between gap-3">
            <span className={`${labelClass} text-theme-muted uppercase tracking-wider`}>Model Version</span>
            <span className={`${valueClass} text-theme-secondary truncate max-w-[180px] text-right`}>{data.modelVersion}</span>
          </div>
        )}

        {data.modelUsed && (
          <div className="flex items-center justify-between gap-3">
            <span className={`${labelClass} text-theme-muted uppercase tracking-wider`}>Model Used</span>
            <span className={`${valueClass} text-theme-secondary truncate max-w-[180px] text-right`}>{data.modelUsed}</span>
          </div>
        )}

        {data.predictionPath && (
          <div className="flex items-center justify-between gap-3">
            <span className={`${labelClass} text-theme-muted uppercase tracking-wider`}>Prediction Path</span>
            <span className={`${valueClass} text-theme-secondary truncate max-w-[180px] text-right`}>{data.predictionPath}</span>
          </div>
        )}

        {(data.calibrationMethod || data.calibrationVersion) && (
          <div className="flex items-center justify-between gap-3">
            <span className={`${labelClass} text-theme-muted uppercase tracking-wider`}>Calibration</span>
            <span className={`${valueClass} text-theme-secondary text-right`}>
              {[data.calibrationMethod, data.calibrationVersion].filter(Boolean).join(' / ')}
            </span>
          </div>
        )}
      </div>

      {data.fallbackReason && (
        <div className="mt-3 pt-3 border-t border-theme-accent/20">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-risk-medium mt-0.5" />
            <div>
              <span className="text-[9px] text-risk-medium uppercase tracking-wider block">Fallback Applied</span>
              <p className="text-[10px] text-theme-secondary mt-1">{data.fallbackReason}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function getEvidenceStrengthMeta(score) {
  const numeric = Number(score);
  if (!Number.isFinite(numeric)) {
    return {
      label: 'Unknown',
      tone: 'border-theme text-theme-muted',
    };
  }

  if (numeric >= 0.75) {
    return {
      label: 'Strong',
      tone: 'border-risk-high/40 text-risk-high bg-risk-high/10',
    };
  }

  if (numeric >= 0.45) {
    return {
      label: 'Moderate',
      tone: 'border-risk-medium/40 text-risk-medium bg-risk-medium/10',
    };
  }

  return {
    label: 'Weak',
    tone: 'border-risk-low/40 text-risk-low bg-risk-low/10',
  };
}

function EvidenceChainTimeline({ interactionEvidence, compact = false }) {
  const evidenceChain = Array.isArray(interactionEvidence?.evidence_chain)
    ? interactionEvidence.evidence_chain
    : [];

  if (evidenceChain.length === 0) return null;

  const maxItems = compact ? 3 : 5;
  const visibleItems = evidenceChain.slice(0, maxItems);
  const remainingCount = Math.max(0, evidenceChain.length - visibleItems.length);

  return (
    <div className="p-4 border border-theme-accent/30 relative bg-theme-accent/5">
      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-theme-accent"></div>
      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-theme-accent"></div>

      <div className="flex items-center gap-2 mb-3">
        <GitBranch className="w-3.5 h-3.5 text-theme-accent" />
        <span className="text-[10px] text-theme-muted uppercase tracking-widest">Why This Interaction</span>
        <span className="ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border border-theme-accent/40 text-theme-accent">
          Evidence Chain
        </span>
      </div>

      <div className="space-y-2.5">
        {visibleItems.map((item, index) => {
          const strength = getEvidenceStrengthMeta(item?.strength_score);
          const sourceLabel = item?.source?.label || item?.source?.id || 'Unknown Source';
          const claim = typeof item?.claim === 'string' ? item.claim : 'Evidence item available.';
          const caveat = Array.isArray(item?.caveats) ? item.caveats[0] : null;

          return (
            <div key={`evidence-${index}`} className="p-2.5 border border-theme/20 bg-theme-primary/70">
              <div className="flex items-start gap-2">
                <div className="mt-0.5">
                  {item?.supports_interaction ? (
                    <Check className="w-3 h-3 text-risk-low" />
                  ) : (
                    <AlertCircle className="w-3 h-3 text-risk-medium" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center flex-wrap gap-1.5 mb-1">
                    <span className="text-[8px] text-theme-dim uppercase tracking-wider">Step {item?.step || index + 1}</span>
                    <span className="text-[8px] text-theme-muted uppercase tracking-wider">{sourceLabel}</span>
                    <span className={`px-1.5 py-0.5 border text-[8px] uppercase tracking-wider ${strength.tone}`}>
                      {strength.label}
                    </span>
                  </div>

                  <p className="text-[11px] text-theme-secondary leading-relaxed">{claim}</p>

                  {caveat && (
                    <p className="text-[9px] text-risk-medium mt-1">Caveat: {caveat}</p>
                  )}

                  {item?.freshness?.update_frequency && (
                    <p className="text-[8px] text-theme-dim mt-1 uppercase tracking-wider">
                      Freshness: {item.freshness.update_frequency}
                      {item?.freshness?.expected_lag ? ` / lag ${item.freshness.expected_lag}` : ''}
                    </p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {remainingCount > 0 && (
        <p className="text-[9px] text-theme-dim mt-2 uppercase tracking-wider">
          +{remainingCount} additional evidence step{remainingCount === 1 ? '' : 's'}
        </p>
      )}

      {Array.isArray(interactionEvidence?.faers_data?.caveats) && interactionEvidence.faers_data.caveats.length > 0 && (
        <div className="mt-3 pt-3 border-t border-theme-accent/20">
          <p className="text-[9px] text-theme-muted uppercase tracking-wider mb-1">FAERS Caveat</p>
          <p className="text-[10px] text-theme-secondary">{interactionEvidence.faers_data.caveats[0]}</p>
        </div>
      )}
    </div>
  );
}

function getEvidenceConfidenceTone(confidenceBand) {
  const normalized = String(confidenceBand || '').toLowerCase();
  if (normalized === 'high') {
    return 'border-risk-low/40 text-risk-low bg-risk-low/10';
  }
  if (normalized === 'moderate') {
    return 'border-risk-medium/40 text-risk-medium bg-risk-medium/10';
  }
  if (normalized === 'low') {
    return 'border-risk-high/40 text-risk-high bg-risk-high/10';
  }
  return 'border-theme text-theme-muted';
}

function getDisagreementTone(level) {
  const normalized = String(level || '').toLowerCase();
  if (normalized === 'high') {
    return 'border-risk-high/40 text-risk-high bg-risk-high/10';
  }
  if (normalized === 'moderate') {
    return 'border-risk-medium/40 text-risk-medium bg-risk-medium/10';
  }
  return 'border-theme text-theme-muted';
}

function EvidenceUncertaintyPanel({ interactionEvidence, compact = false }) {
  const summary = interactionEvidence?.evidence_summary && typeof interactionEvidence.evidence_summary === 'object'
    ? interactionEvidence.evidence_summary
    : null;

  if (!summary) return null;

  const weightedSupport = Number(summary.weighted_support_score);
  const weightedUncertainty = Number(summary.weighted_uncertainty_score);
  const confidenceBand = String(summary.confidence_band || 'unknown');
  const disagreement = summary.disagreement && typeof summary.disagreement === 'object'
    ? summary.disagreement
    : {};
  const coverage = summary.primary_source_coverage && typeof summary.primary_source_coverage === 'object'
    ? summary.primary_source_coverage
    : {};

  const reasons = Array.isArray(summary.uncertainty_reasons) ? summary.uncertainty_reasons : [];
  const visibleReasons = reasons.slice(0, compact ? 2 : 4);
  const remainingReasonCount = Math.max(0, reasons.length - visibleReasons.length);

  const hasConflict = Boolean(disagreement.has_conflict);
  const disagreementLevel = String(disagreement.level || 'none');

  const hasSummarySignals = Number.isFinite(weightedSupport)
    || Number.isFinite(weightedUncertainty)
    || reasons.length > 0
    || hasConflict;

  if (!hasSummarySignals) return null;

  return (
    <div className="p-4 border border-risk-medium/30 relative bg-risk-medium/5">
      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-risk-medium"></div>
      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-risk-medium"></div>

      <div className="flex items-center gap-2 mb-3">
        <AlertTriangle className="w-3.5 h-3.5 text-risk-medium" />
        <span className="text-[10px] text-theme-muted uppercase tracking-widest">Uncertainty Reasons</span>
        <span className={`ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border ${getEvidenceConfidenceTone(confidenceBand)}`}>
          {confidenceBand} confidence
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="p-2 border border-theme/20 bg-theme-primary/70">
          <p className="text-[8px] text-theme-muted uppercase tracking-wider">Weighted Support</p>
          <p className="text-[11px] text-theme-secondary mt-1">
            {Number.isFinite(weightedSupport) ? `${(weightedSupport * 100).toFixed(1)}%` : 'N/A'}
          </p>
        </div>
        <div className="p-2 border border-theme/20 bg-theme-primary/70">
          <p className="text-[8px] text-theme-muted uppercase tracking-wider">Weighted Uncertainty</p>
          <p className="text-[11px] text-theme-secondary mt-1">
            {Number.isFinite(weightedUncertainty) ? `${(weightedUncertainty * 100).toFixed(1)}%` : 'N/A'}
          </p>
        </div>
      </div>

      {hasConflict && (
        <div className="mb-3 p-2 border border-theme/20 bg-theme-primary/70">
          <div className="flex items-center gap-2 mb-1">
            <span className={`px-1.5 py-0.5 border text-[8px] uppercase tracking-wider ${getDisagreementTone(disagreementLevel)}`}>
              Conflict: {disagreementLevel}
            </span>
          </div>
          {disagreement.narrative && (
            <p className="text-[10px] text-theme-secondary leading-relaxed">{disagreement.narrative}</p>
          )}
        </div>
      )}

      {Number.isFinite(coverage.ratio) && (
        <p className="text-[9px] text-theme-dim uppercase tracking-wider mb-2">
          Primary Source Coverage: {(coverage.ratio * 100).toFixed(0)}%
        </p>
      )}

      {visibleReasons.length > 0 ? (
        <div className="space-y-2">
          {visibleReasons.map((reason, index) => (
            <div key={`uncertainty-reason-${index}`} className="p-2 border border-theme/20 bg-theme-primary/70">
              <p className="text-[9px] text-theme-muted uppercase tracking-wider">
                {reason?.source_label || reason?.source_id || 'Unknown Source'}
              </p>
              <p className="text-[10px] text-theme-secondary mt-1 leading-relaxed">
                {reason?.reason || 'Uncertainty signal detected.'}
              </p>
              {reason?.caveat && (
                <p className="text-[9px] text-risk-medium mt-1">Caveat: {reason.caveat}</p>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-[10px] text-theme-dim">No explicit uncertainty reasons returned for this pair.</p>
      )}

      {remainingReasonCount > 0 && (
        <p className="text-[9px] text-theme-dim mt-2 uppercase tracking-wider">
          +{remainingReasonCount} additional uncertainty reason{remainingReasonCount === 1 ? '' : 's'}
        </p>
      )}
    </div>
  );
}

function CalibrationQAPanel({ addLog, defaultExpanded = false }) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [csvInput, setCsvInput] = useState(DEFAULT_CALIBRATION_CSV);
  const [nBins, setNBins] = useState(10);
  const [nBootstrap, setNBootstrap] = useState(400);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState(null);
  const [panelError, setPanelError] = useState(null);

  const rawQuality = getCalibrationQualityBand(report?.raw?.ece);
  const calibratedQuality = getCalibrationQualityBand(report?.calibrated?.ece);

  const runCalibrationQa = async () => {
    setPanelError(null);
    setRunning(true);

    try {
      const parsed = parseCalibrationCsv(csvInput);
      const response = await computeCalibrationMetrics({
        labels: parsed.labels,
        rawScores: parsed.rawScores,
        calibratedScores: parsed.calibratedScores,
        nBins,
        nBootstrap,
        seed: 42,
      });

      setReport(response);
      if (addLog) {
        addLog(
          `Calibration QA completed: ECE improvement ${formatMetric(response?.delta?.ece_improvement)} (n=${response?.meta?.n_samples || 0})`,
          'success',
          'AI'
        );
      }
    } catch (err) {
      const message = getCalibrationRunErrorMessage(err);
      setPanelError(message);
      if (addLog) {
        addLog(`Calibration QA failed: ${message}`, 'error', 'AI');
      }
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="p-3 border border-theme bg-theme-secondary">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-3.5 h-3.5 text-theme-accent" />
          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Calibration QA (Research)</span>
        </div>
        <button
          onClick={() => setExpanded((v) => !v)}
          className="px-2 py-0.5 text-[8px] border border-theme text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors"
        >
          {expanded ? 'Hide' : 'Open'}
        </button>
      </div>

      {expanded && (
        <div className="mt-3 space-y-3">
          <p className="text-[10px] text-theme-muted leading-relaxed">
            Paste labeled calibration rows as CSV: <span className="text-theme-secondary">label,raw_score,calibrated_score</span>
          </p>

          <textarea
            value={csvInput}
            onChange={(e) => setCsvInput(e.target.value)}
            rows={7}
            className="w-full bg-theme-primary border border-theme p-2 text-[10px] font-mono text-theme-secondary placeholder:text-theme-dim focus:outline-none focus:border-theme-accent/50"
            placeholder="label,raw_score,calibrated_score"
          />

          <div className="grid grid-cols-2 gap-2">
            <label className="text-[9px] text-theme-muted uppercase tracking-wider">
              Bins
              <input
                type="number"
                min={2}
                max={30}
                value={nBins}
                onChange={(e) => setNBins(Math.max(2, Number(e.target.value) || 2))}
                className="mt-1 w-full bg-theme-primary border border-theme px-2 py-1 text-[10px] text-theme-secondary focus:outline-none focus:border-theme-accent/50"
              />
            </label>
            <label className="text-[9px] text-theme-muted uppercase tracking-wider">
              Bootstrap
              <input
                type="number"
                min={200}
                max={5000}
                step={100}
                value={nBootstrap}
                onChange={(e) => setNBootstrap(Math.max(200, Number(e.target.value) || 200))}
                className="mt-1 w-full bg-theme-primary border border-theme px-2 py-1 text-[10px] text-theme-secondary focus:outline-none focus:border-theme-accent/50"
              />
            </label>
          </div>

          <button
            onClick={runCalibrationQa}
            disabled={running}
            className={`w-full py-2 border text-[10px] uppercase tracking-wider transition-all ${
              running
                ? 'border-theme-accent/40 text-theme-accent bg-theme-accent/10 cursor-wait'
                : 'border-theme-accent text-theme-accent hover:bg-theme-accent/10'
            }`}
          >
            {running ? (
              <span className="inline-flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Running Calibration QA
              </span>
            ) : (
              'Run Calibration QA'
            )}
          </button>

          {panelError && (
            <div className="p-2 border border-risk-high/30 bg-risk-high/10 text-[10px] text-risk-high">
              {panelError}
            </div>
          )}

          {report && (
            <div className="space-y-3">
              <div className="p-3 border border-theme-accent/30 bg-theme-accent/5 space-y-2">
                <div className="flex items-center justify-between text-[9px] uppercase tracking-wider text-theme-muted">
                  <span>Samples</span>
                  <span className="text-theme-secondary">{report?.meta?.n_samples || 0}</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 border border-theme/20 bg-theme-panel/80">
                    <p className="text-[8px] text-theme-muted uppercase tracking-wider">Raw ECE</p>
                    <p className="text-xs text-theme-secondary mt-1">{formatMetric(report?.raw?.ece)}</p>
                    <span className={`inline-block mt-1 px-1.5 py-0.5 text-[8px] uppercase tracking-wider border ${rawQuality.tone}`}>
                      {rawQuality.label}
                    </span>
                  </div>
                  <div className="p-2 border border-theme/20 bg-theme-panel/80">
                    <p className="text-[8px] text-theme-muted uppercase tracking-wider">Calibrated ECE</p>
                    <p className="text-xs text-theme-accent mt-1">{formatMetric(report?.calibrated?.ece)}</p>
                    <span className={`inline-block mt-1 px-1.5 py-0.5 text-[8px] uppercase tracking-wider border ${calibratedQuality.tone}`}>
                      {calibratedQuality.label}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between text-[9px] uppercase tracking-wider text-theme-muted">
                  <span>ECE Improvement</span>
                  <span className="text-risk-low">{formatMetric(report?.delta?.ece_improvement)}</span>
                </div>
                <div className="flex items-center justify-between text-[9px] uppercase tracking-wider text-theme-muted">
                  <span>Brier Improvement</span>
                  <span className="text-risk-low">{formatMetric(report?.delta?.brier_improvement)}</span>
                </div>

                <div className="pt-2 border-t border-theme-accent/20 text-[9px] text-theme-muted leading-relaxed">
                  Calibrated ECE 95% CI: {formatMetric(report?.calibrated?.ece_confidence_interval?.lower)} to {formatMetric(report?.calibrated?.ece_confidence_interval?.upper)}
                </div>
              </div>

              <ReliabilityCurveChart rawSection={report?.raw} calibratedSection={report?.calibrated} />

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                <CalibrationBinTable title="Raw Bin Diagnostics" bins={report?.raw?.bins} tone="text-risk-medium" />
                <CalibrationBinTable title="Calibrated Bin Diagnostics" bins={report?.calibrated?.bins} tone="text-theme-accent" />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const WHAT_IF_SNAPSHOTS_STORAGE_KEY = 'aegis:what-if-snapshots:v1';

const SCENARIO_SEVERITY_RANKS = {
  no_interaction: 0,
  none: 0,
  low: 1,
  minor: 1,
  medium: 2,
  moderate: 2,
  high: 3,
  major: 3,
  severe: 4,
  critical: 5,
};

function normalizeScenarioDrug(drug) {
  if (!drug || typeof drug !== 'object') return null;
  if (!drug.name || typeof drug.name !== 'string') return null;

  const normalizedName = drug.name.trim();
  if (!normalizedName) return null;

  return {
    name: normalizedName,
    smiles: typeof drug.smiles === 'string' ? drug.smiles : undefined,
    drugbank_id: drug.drugbank_id || drug.id || undefined,
  };
}

function dedupeScenarioDrugs(drugs = []) {
  const seen = new Set();
  const normalized = [];

  drugs.forEach((drug) => {
    const cleaned = normalizeScenarioDrug(drug);
    if (!cleaned) return;

    const key = cleaned.name.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    normalized.push(cleaned);
  });

  return normalized;
}

function normalizeScenarioSeverity(value) {
  if (typeof value !== 'string') return 'no_interaction';
  const normalized = value.trim().toLowerCase().replace(/\s+/g, '_');
  return normalized || 'no_interaction';
}

function getScenarioSeverityRank(severity) {
  const normalized = normalizeScenarioSeverity(severity);
  return SCENARIO_SEVERITY_RANKS[normalized] ?? 0;
}

function getScenarioRiskLabelFromRank(rank) {
  if (rank >= 5) return 'critical';
  if (rank >= 4) return 'severe';
  if (rank >= 3) return 'high';
  if (rank >= 2) return 'medium';
  if (rank >= 1) return 'low';
  return 'no_interaction';
}

function getScenarioRiskTone(rank) {
  if (rank >= 4) return 'text-risk-high border-risk-high/40 bg-risk-high/10';
  if (rank >= 3) return 'text-risk-medium border-risk-medium/40 bg-risk-medium/10';
  if (rank >= 2) return 'text-theme-accent border-theme-accent/40 bg-theme-accent/10';
  if (rank >= 1) return 'text-theme-secondary border-theme/40 bg-theme-panel/60';
  return 'text-risk-low border-risk-low/40 bg-risk-low/10';
}

function getScenarioDeltaTone(delta) {
  if (!Number.isFinite(delta)) return 'text-theme-muted';
  if (delta >= 0.05) return 'text-risk-high';
  if (delta <= -0.05) return 'text-risk-low';
  return 'text-theme-muted';
}

function getScenarioSeverityTone(severity) {
  const rank = getScenarioSeverityRank(severity);
  if (rank >= 4) return 'text-risk-high border-risk-high/40 bg-risk-high/10';
  if (rank >= 3) return 'text-risk-medium border-risk-medium/40 bg-risk-medium/10';
  if (rank >= 2) return 'text-theme-accent border-theme-accent/40 bg-theme-accent/10';
  if (rank >= 1) return 'text-theme-secondary border-theme/40 bg-theme-panel/60';
  return 'text-risk-low border-risk-low/40 bg-risk-low/10';
}

function buildScenarioPairKey(nameA, nameB) {
  const first = String(nameA || '').trim().toLowerCase();
  const second = String(nameB || '').trim().toLowerCase();
  return [first, second].sort().join('||');
}

function buildScenarioPairs(drugs) {
  const pairs = [];
  for (let i = 0; i < drugs.length; i += 1) {
    for (let j = i + 1; j < drugs.length; j += 1) {
      pairs.push([drugs[i], drugs[j]]);
    }
  }
  return pairs;
}

function downloadScenarioFile(fileName, content, mimeType = 'text/plain;charset=utf-8') {
  if (typeof window === 'undefined') return;

  const blob = new Blob([content], { type: mimeType });
  const objectUrl = window.URL.createObjectURL(blob);

  const anchor = document.createElement('a');
  anchor.href = objectUrl;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  window.setTimeout(() => {
    window.URL.revokeObjectURL(objectUrl);
  }, 0);
}

function toCsvCell(value) {
  const text = String(value ?? '');
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function loadScenarioSnapshots() {
  if (typeof window === 'undefined') return [];

  try {
    const raw = window.localStorage.getItem(WHAT_IF_SNAPSHOTS_STORAGE_KEY);
    if (!raw) return [];

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    return parsed
      .filter((item) => item && typeof item === 'object' && Array.isArray(item.proposed) && Array.isArray(item.baseline))
      .slice(0, 8);
  } catch {
    return [];
  }
}

function formatScenarioTimestamp(isoDate) {
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) return String(isoDate || '');
  return date.toLocaleString();
}

async function evaluateScenarioRegimen(drugs) {
  const regimen = dedupeScenarioDrugs(drugs);
  const pairs = buildScenarioPairs(regimen);

  if (!pairs.length) {
    return {
      regimen,
      pairCount: 0,
      failedPairs: 0,
      interactions: [],
      significant: [],
      maxRank: 0,
      riskScore: 0,
      riskLevel: 'no_interaction',
      highRiskCount: 0,
    };
  }

  const settledResults = await Promise.allSettled(
    pairs.map(([drugA, drugB]) => predictDDI(
      { name: drugA.name, smiles: drugA.smiles },
      { name: drugB.name, smiles: drugB.smiles }
    ))
  );

  const interactions = [];
  let failedPairs = 0;

  settledResults.forEach((result, idx) => {
    const [drugA, drugB] = pairs[idx];
    if (result.status !== 'fulfilled') {
      failedPairs += 1;
      return;
    }

    const response = result.value || {};
    const provenance = response.provenance && typeof response.provenance === 'object'
      ? response.provenance
      : {};
    const normalizedSeverity = normalizeScenarioSeverity(response.severity || response.risk_level);
    const rank = getScenarioSeverityRank(normalizedSeverity);
    const numericRisk = Number(response.risk_score);
    const rawScore = Number(response.raw_score);
    const calibratedScore = Number(response.calibrated_score);

    interactions.push({
      key: buildScenarioPairKey(drugA.name, drugB.name),
      pair: `${drugA.name} + ${drugB.name}`,
      source: drugA.name,
      target: drugB.name,
      severity: normalizedSeverity,
      rank,
      riskScore: Number.isFinite(numericRisk) ? numericRisk : 0,
      confidence: Number.isFinite(response.confidence) ? response.confidence : null,
      mechanism: response.mechanism_hypothesis || null,
      rawScore: Number.isFinite(rawScore) ? rawScore : null,
      calibratedScore: Number.isFinite(calibratedScore) ? calibratedScore : null,
      modelUsed: typeof provenance.model_used === 'string' ? provenance.model_used : null,
      predictionPath: typeof provenance.prediction_path === 'string' ? provenance.prediction_path : null,
      fallbackReason: typeof provenance.fallback_reason === 'string' ? provenance.fallback_reason : null,
    });
  });

  interactions.sort((a, b) => {
    if (b.rank !== a.rank) return b.rank - a.rank;
    return b.riskScore - a.riskScore;
  });

  const significant = interactions.filter((edge) => edge.rank > 0 || edge.riskScore >= 0.25);
  const maxRank = significant.reduce((current, edge) => Math.max(current, edge.rank), 0);
  const maxRiskScore = significant.reduce((current, edge) => Math.max(current, edge.riskScore), 0);
  const density = pairs.length ? significant.length / pairs.length : 0;
  const combinedRiskScore = Math.min(1, Number((maxRiskScore * 0.75 + density * 0.25).toFixed(3)));

  return {
    regimen,
    pairCount: pairs.length,
    failedPairs,
    interactions,
    significant,
    maxRank,
    riskScore: combinedRiskScore,
    riskLevel: getScenarioRiskLabelFromRank(maxRank),
    highRiskCount: significant.filter((edge) => edge.rank >= 3 || edge.riskScore >= 0.7).length,
  };
}

function computeScenarioDelta(baselineEval, proposedEval) {
  const baselineMap = new Map(baselineEval.significant.map((edge) => [edge.key, edge]));
  const proposedMap = new Map(proposedEval.significant.map((edge) => [edge.key, edge]));

  const added = [];
  const removed = [];
  const changed = [];

  proposedMap.forEach((edge, key) => {
    const previous = baselineMap.get(key);
    if (!previous) {
      added.push(edge);
      return;
    }

    const riskChange = edge.riskScore - previous.riskScore;
    if (edge.rank !== previous.rank || Math.abs(riskChange) >= 0.05) {
      changed.push({
        ...edge,
        previousSeverity: previous.severity,
        previousRiskScore: previous.riskScore,
        riskChange,
      });
    }
  });

  baselineMap.forEach((edge, key) => {
    if (!proposedMap.has(key)) {
      removed.push(edge);
    }
  });

  const sortFn = (a, b) => {
    if (b.rank !== a.rank) return b.rank - a.rank;
    return b.riskScore - a.riskScore;
  };

  added.sort(sortFn);
  removed.sort(sortFn);
  changed.sort((a, b) => {
    const deltaA = Math.abs(a.riskChange || 0);
    const deltaB = Math.abs(b.riskChange || 0);
    return deltaB - deltaA;
  });

  return {
    added,
    removed,
    changed,
    riskDelta: proposedEval.riskScore - baselineEval.riskScore,
    interactionDelta: proposedEval.significant.length - baselineEval.significant.length,
  };
}

function clampScenarioValue(value, min = 0, max = 1) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function getMechanismTheme(mechanismText) {
  const text = String(mechanismText || '').toLowerCase();

  if (/cyp|metabol|enzyme|inhibitor|inducer/.test(text)) {
    return {
      label: 'Metabolic Collision',
      detail: 'Competing metabolism or enzyme modulation can amplify exposure.',
    };
  }
  if (/qt|arrhythm|torsade|cardiac/.test(text)) {
    return {
      label: 'Electrical Instability',
      detail: 'Combined electrophysiologic effects may elevate arrhythmia risk.',
    };
  }
  if (/bleed|platelet|anticoag|hemorrhag/.test(text)) {
    return {
      label: 'Hemostasis Strain',
      detail: 'Overlapping antiplatelet/anticoagulant pressure can increase bleeding risk.',
    };
  }
  if (/serotonin|cns|sedat|neuro|seizure/.test(text)) {
    return {
      label: 'Neurochemical Overlap',
      detail: 'Concurrent CNS or neurotransmitter effects may compound toxicity.',
    };
  }
  if (/renal|kidney|nephro|creatin/.test(text)) {
    return {
      label: 'Renal Clearance Pressure',
      detail: 'Combined renal burden may reduce clearance and increase accumulation.',
    };
  }

  return {
    label: 'Multi-Pathway Interaction',
    detail: 'The model indicates a non-trivial interaction requiring closer review.',
  };
}

function inferMechanismSystems(themeLabel, mechanismText) {
  const text = String(mechanismText || '').toLowerCase();
  const systems = [];

  if (/cardiac|qt|arrhythm|torsade/.test(text) || themeLabel === 'Electrical Instability') {
    systems.push('Cardiovascular');
  }
  if (/bleed|platelet|hemorrhag/.test(text) || themeLabel === 'Hemostasis Strain') {
    systems.push('Hematologic');
  }
  if (/renal|kidney|creatin/.test(text) || themeLabel === 'Renal Clearance Pressure') {
    systems.push('Renal');
  }
  if (/liver|hepatic|cyp|metabol/.test(text) || themeLabel === 'Metabolic Collision') {
    systems.push('Hepatic');
  }
  if (/cns|sedat|neuro|serotonin|seizure/.test(text) || themeLabel === 'Neurochemical Overlap') {
    systems.push('Neurologic');
  }

  if (!systems.length) {
    systems.push('Multi-system');
  }

  return Array.from(new Set(systems)).slice(0, 3);
}

function getUncertaintyBand(score) {
  if (score >= 0.65) return 'High';
  if (score >= 0.35) return 'Moderate';
  return 'Low';
}

function getUncertaintyTone(score) {
  if (score >= 0.65) return 'text-risk-high border-risk-high/40 bg-risk-high/10';
  if (score >= 0.35) return 'text-risk-medium border-risk-medium/40 bg-risk-medium/10';
  return 'text-risk-low border-risk-low/40 bg-risk-low/10';
}

function computeEdgeUncertainty(edge) {
  const confidencePenalty = Number.isFinite(edge?.confidence)
    ? 1 - clampScenarioValue(edge.confidence)
    : 0.45;

  let calibrationShift = 0.08;
  if (Number.isFinite(edge?.rawScore) && Number.isFinite(edge?.calibratedScore)) {
    calibrationShift = Math.abs(edge.calibratedScore - edge.rawScore);
  }

  const fallbackPenalty = edge?.fallbackReason ? 0.22 : 0;
  const severityPenalty = getScenarioSeverityRank(edge?.severity) >= 4 ? 0.06 : 0;

  const score = clampScenarioValue(
    confidencePenalty * 0.65 + calibrationShift * 0.9 + fallbackPenalty + severityPenalty
  );

  return {
    score,
    confidencePenalty,
    calibrationShift,
    fallbackPenalty,
  };
}

function getReplayEdgeFromReport(report) {
  if (!report || typeof report !== 'object') return null;

  const worseningChange = Array.isArray(report?.delta?.changed)
    ? report.delta.changed.find((edge) => Number(edge?.riskChange) > 0)
    : null;

  return report?.delta?.added?.[0]
    || worseningChange
    || report?.proposed?.significant?.[0]
    || report?.proposed?.interactions?.[0]
    || null;
}

function buildCausalReplayData(report, mutationRecommendations = []) {
  const edge = getReplayEdgeFromReport(report);
  if (!edge) return null;

  const theme = getMechanismTheme(edge.mechanism);
  const systems = inferMechanismSystems(theme.label, edge.mechanism);
  const bestMutation = mutationRecommendations.find((row) => Number(row?.riskDrop) > 0.01) || null;

  const mitigation = bestMutation
    ? `Removing ${bestMutation.removedDrug} projects a ${(bestMutation.riskDrop * 100).toFixed(1)} pp risk reduction.`
    : 'Run Mutation Engine to identify the highest-impact regimen change.';

  const steps = [
    {
      title: 'Pair Trigger',
      detail: `${edge.pair} is the current highest-priority interaction target.`,
    },
    {
      title: 'Mechanistic Driver',
      detail: `${theme.label}: ${theme.detail}`,
    },
    {
      title: 'Clinical Surface',
      detail: `Likely burden channels: ${systems.join(', ')}.`,
    },
    {
      title: 'Mitigation Move',
      detail: mitigation,
    },
  ];

  return {
    edge,
    theme,
    systems,
    steps,
  };
}

function buildScenarioCsvReport(report) {
  if (!report) return '';

  const lines = [
    'change_type,pair,severity,previous_severity,risk_score,previous_risk_score,risk_delta',
  ];

  report.delta.added.forEach((edge) => {
    lines.push([
      'added',
      toCsvCell(edge.pair),
      toCsvCell(edge.severity),
      '',
      edge.riskScore.toFixed(4),
      '',
      edge.riskScore.toFixed(4),
    ].join(','));
  });

  report.delta.removed.forEach((edge) => {
    lines.push([
      'removed',
      toCsvCell(edge.pair),
      toCsvCell(edge.severity),
      '',
      '',
      edge.riskScore.toFixed(4),
      (-edge.riskScore).toFixed(4),
    ].join(','));
  });

  report.delta.changed.forEach((edge) => {
    lines.push([
      'changed',
      toCsvCell(edge.pair),
      toCsvCell(edge.severity),
      toCsvCell(edge.previousSeverity),
      edge.riskScore.toFixed(4),
      Number(edge.previousRiskScore || 0).toFixed(4),
      Number(edge.riskChange || 0).toFixed(4),
    ].join(','));
  });

  return lines.join('\n');
}

function ScenarioInteractionTable({ title, rows, emptyLabel, includeChange = false }) {
  return (
    <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[9px] text-theme-muted uppercase tracking-wider">{title}</span>
        <span className="text-[8px] text-theme-dim uppercase tracking-wider">{rows.length} rows</span>
      </div>

      {rows.length === 0 ? (
        <p className="text-[10px] text-theme-dim">{emptyLabel}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[9px]">
            <thead>
              <tr className="border-b border-theme/20 text-theme-muted uppercase tracking-wider">
                <th className="text-left py-1 pr-2">Pair</th>
                <th className="text-right py-1 px-2">Severity</th>
                <th className="text-right py-1 px-2">Risk</th>
                {includeChange && <th className="text-right py-1 pl-2">Delta</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${title}-${row.key || row.pair}`} className="border-b border-theme/10 text-theme-secondary">
                  <td className="py-1 pr-2">{row.pair}</td>
                  <td className="py-1 px-2 text-right">
                    <span className={`inline-flex px-1.5 py-0.5 border text-[8px] uppercase tracking-wider ${getScenarioSeverityTone(row.severity)}`}>
                      {String(row.severity || 'unknown').replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className="py-1 px-2 text-right">{formatMetric(row.riskScore, 3)}</td>
                  {includeChange && (
                    <td className={`py-1 pl-2 text-right ${getScenarioDeltaTone(row.riskChange)}`}>
                      {(Number(row.riskChange || 0) > 0 ? '+' : '') + formatMetric(Number(row.riskChange || 0), 3)}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function CausalInteractionReplayPanel({ report, mutationRecommendations = [] }) {
  const replayData = useMemo(
    () => buildCausalReplayData(report, mutationRecommendations),
    [report, mutationRecommendations]
  );

  if (!report) {
    return (
      <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
        <div className="flex items-center gap-2 mb-2">
          <Brain className="w-3.5 h-3.5 text-theme-accent" />
          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Causal Interaction Replay</span>
        </div>
        <p className="text-[10px] text-theme-dim">Run a What-If comparison to generate a causal replay narrative.</p>
      </div>
    );
  }

  if (!replayData) {
    return (
      <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
        <div className="flex items-center gap-2 mb-2">
          <Brain className="w-3.5 h-3.5 text-theme-accent" />
          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Causal Interaction Replay</span>
        </div>
        <p className="text-[10px] text-theme-dim">No high-priority interaction edge available for replay.</p>
      </div>
    );
  }

  const edgeUncertainty = computeEdgeUncertainty(replayData.edge);

  return (
    <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Brain className="w-3.5 h-3.5 text-theme-accent" />
          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Causal Interaction Replay</span>
        </div>
        <span className={`px-2 py-0.5 text-[8px] border uppercase tracking-wider ${getScenarioSeverityTone(replayData.edge.severity)}`}>
          {String(replayData.edge.severity || 'unknown').replace(/_/g, ' ')}
        </span>
      </div>

      <div className="p-2 border border-theme/30 bg-theme-primary/70 text-[9px] text-theme-muted uppercase tracking-wider flex flex-wrap gap-x-3 gap-y-1">
        <span>Focus Pair: {replayData.edge.pair}</span>
        <span>Risk Score: {formatMetric(replayData.edge.riskScore, 3)}</span>
        <span className={getUncertaintyTone(edgeUncertainty.score).split(' ')[0]}>
          Uncertainty: {getUncertaintyBand(edgeUncertainty.score)} ({(edgeUncertainty.score * 100).toFixed(0)}%)
        </span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-2">
        {replayData.steps.map((step, idx) => (
          <div key={`replay-step-${idx}`} className="p-2 border border-theme/30 bg-theme-primary/70">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-4 h-4 border border-theme-accent/40 text-theme-accent text-[8px] flex items-center justify-center">
                {idx + 1}
              </span>
              <span className="text-[8px] text-theme-muted uppercase tracking-wider">{step.title}</span>
            </div>
            <p className="text-[10px] text-theme-secondary leading-relaxed">{step.detail}</p>
          </div>
        ))}
      </div>

      {(replayData.edge.modelUsed || replayData.edge.predictionPath || replayData.edge.fallbackReason) && (
        <div className="p-2 border border-theme/30 bg-theme-primary/70 text-[8px] uppercase tracking-wider text-theme-dim space-y-1">
          {replayData.edge.modelUsed && <p>Model: {replayData.edge.modelUsed}</p>}
          {replayData.edge.predictionPath && <p>Path: {replayData.edge.predictionPath}</p>}
          {replayData.edge.fallbackReason && <p className="text-risk-medium">Fallback: {replayData.edge.fallbackReason}</p>}
        </div>
      )}
    </div>
  );
}

function UncertaintyHeatLensPanel({ report }) {
  const rows = useMemo(() => {
    const edges = Array.isArray(report?.proposed?.interactions) ? report.proposed.interactions : [];
    return edges
      .map((edge) => {
        const uncertainty = computeEdgeUncertainty(edge);
        return {
          ...edge,
          uncertainty: uncertainty.score,
          confidencePenalty: uncertainty.confidencePenalty,
          calibrationShift: uncertainty.calibrationShift,
          fallbackPenalty: uncertainty.fallbackPenalty,
        };
      })
      .sort((a, b) => b.uncertainty - a.uncertainty)
      .slice(0, 16);
  }, [report]);

  if (!report) {
    return (
      <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
        <div className="flex items-center gap-2 mb-2">
          <Activity className="w-3.5 h-3.5 text-theme-accent" />
          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Uncertainty Heat Lens</span>
        </div>
        <p className="text-[10px] text-theme-dim">Run a scenario to map uncertainty hotspots across pairwise edges.</p>
      </div>
    );
  }

  const highCount = rows.filter((row) => row.uncertainty >= 0.65).length;
  const mediumCount = rows.filter((row) => row.uncertainty >= 0.35 && row.uncertainty < 0.65).length;
  const lowCount = rows.filter((row) => row.uncertainty < 0.35).length;

  return (
    <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-theme-accent" />
          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Uncertainty Heat Lens</span>
        </div>
        <span className="text-[8px] text-theme-dim uppercase tracking-wider">{rows.length} edges</span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[8px] uppercase tracking-wider">
        <div className="p-2 border border-risk-high/30 bg-risk-high/10 text-risk-high">High: {highCount}</div>
        <div className="p-2 border border-risk-medium/30 bg-risk-medium/10 text-risk-medium">Moderate: {mediumCount}</div>
        <div className="p-2 border border-risk-low/30 bg-risk-low/10 text-risk-low">Low: {lowCount}</div>
      </div>

      {rows.length === 0 ? (
        <p className="text-[10px] text-theme-dim">No pair edges available for uncertainty scoring.</p>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
          {rows.map((row) => {
            const tone = getUncertaintyTone(row.uncertainty);
            const band = getUncertaintyBand(row.uncertainty);

            return (
              <div key={`uncertainty-${row.key || row.pair}`} className="p-2 border border-theme/30 bg-theme-primary/70">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="text-[9px] text-theme-secondary uppercase tracking-wider truncate">{row.pair}</span>
                  <span className={`px-1.5 py-0.5 text-[8px] border uppercase tracking-wider ${tone}`}>
                    {band} {(row.uncertainty * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="h-1.5 bg-theme-panel border border-theme/20 overflow-hidden">
                  <div
                    className={`h-full ${row.uncertainty >= 0.65 ? 'bg-risk-high' : row.uncertainty >= 0.35 ? 'bg-risk-medium' : 'bg-risk-low'}`}
                    style={{ width: `${Math.max(4, row.uncertainty * 100)}%` }}
                  />
                </div>

                <div className="mt-1 text-[8px] text-theme-dim uppercase tracking-wider flex flex-wrap gap-x-2 gap-y-1">
                  <span>Confidence: {Number.isFinite(row.confidence) ? formatMetric(row.confidence, 2) : 'N/A'}</span>
                  <span>Shift: {formatMetric(row.calibrationShift, 3)}</span>
                  {row.fallbackPenalty > 0 && <span className="text-risk-medium">Fallback Penalty</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function WhatIfScenarioBuilder({ selectedDrugs, apiStatus, addLog }) {
  const baselineRegimen = useMemo(() => dedupeScenarioDrugs(selectedDrugs), [selectedDrugs]);
  const baselineSignature = useMemo(
    () => baselineRegimen.map((drug) => drug.name.toLowerCase()).join('|'),
    [baselineRegimen]
  );

  const [proposedRegimen, setProposedRegimen] = useState(() => dedupeScenarioDrugs(selectedDrugs));
  const [scenarioTitle, setScenarioTitle] = useState('');
  const [addQuery, setAddQuery] = useState('');
  const [addResults, setAddResults] = useState([]);
  const [addSearching, setAddSearching] = useState(false);
  const [runningComparison, setRunningComparison] = useState(false);
  const [scenarioError, setScenarioError] = useState(null);
  const [scenarioReport, setScenarioReport] = useState(null);
  const [snapshots, setSnapshots] = useState(() => loadScenarioSnapshots());
  const [mutationRunning, setMutationRunning] = useState(false);
  const [mutationError, setMutationError] = useState(null);
  const [mutationRecommendations, setMutationRecommendations] = useState([]);

  const debouncedAddQuery = useDebounce(addQuery, 250);

  useEffect(() => {
    setProposedRegimen(baselineRegimen);
    setScenarioReport(null);
    setScenarioError(null);
    setMutationRecommendations([]);
    setMutationError(null);
  }, [baselineSignature, baselineRegimen]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(WHAT_IF_SNAPSHOTS_STORAGE_KEY, JSON.stringify(snapshots));
  }, [snapshots]);

  useEffect(() => {
    let cancelled = false;

    const runSearch = async () => {
      if (apiStatus !== 'online' || !debouncedAddQuery || debouncedAddQuery.length < 2) {
        setAddResults([]);
        return;
      }

      setAddSearching(true);
      try {
        const response = await searchDrugs(debouncedAddQuery);
        const filtered = (response.results || []).filter((drug) => {
          const name = String(drug?.name || '').toLowerCase();
          return name && !proposedRegimen.some((existing) => existing.name.toLowerCase() === name);
        });

        if (!cancelled) {
          setAddResults(filtered.slice(0, 8));
        }
      } catch {
        if (!cancelled) {
          setAddResults([]);
        }
      } finally {
        if (!cancelled) {
          setAddSearching(false);
        }
      }
    };

    runSearch();
    return () => {
      cancelled = true;
    };
  }, [debouncedAddQuery, apiStatus, proposedRegimen]);

  const addDrugToProposed = useCallback((drug) => {
    const normalized = normalizeScenarioDrug(drug);
    if (!normalized) return;

    setProposedRegimen((prev) => {
      if (prev.some((item) => item.name.toLowerCase() === normalized.name.toLowerCase())) {
        return prev;
      }
      return [...prev, normalized];
    });

    setAddQuery('');
    setAddResults([]);
    setScenarioReport(null);
  }, []);

  const removeDrugFromProposed = useCallback((name) => {
    setProposedRegimen((prev) => prev.filter((drug) => drug.name.toLowerCase() !== String(name).toLowerCase()));
    setScenarioReport(null);
  }, []);

  const resetToBaseline = useCallback(() => {
    setProposedRegimen(baselineRegimen);
    setScenarioReport(null);
    setScenarioError(null);
    if (addLog) {
      addLog('What-If proposed regimen reset to current selection.', 'info', 'SYSTEM');
    }
  }, [baselineRegimen, addLog]);

  const runScenarioComparison = async () => {
    setScenarioError(null);

    if (apiStatus !== 'online') {
      setScenarioError('Backend must be online to run What-If analysis.');
      return;
    }

    if (baselineRegimen.length < 2) {
      setScenarioError('Current regimen needs at least 2 drugs to compare a scenario.');
      return;
    }

    if (proposedRegimen.length < 2) {
      setScenarioError('Proposed regimen needs at least 2 drugs to compute interaction deltas.');
      return;
    }

    setRunningComparison(true);
    try {
      const [baselineEval, proposedEval] = await Promise.all([
        evaluateScenarioRegimen(baselineRegimen),
        evaluateScenarioRegimen(proposedRegimen),
      ]);

      const delta = computeScenarioDelta(baselineEval, proposedEval);
      const report = {
        generatedAt: new Date().toISOString(),
        baselineNames: baselineEval.regimen.map((drug) => drug.name),
        proposedNames: proposedEval.regimen.map((drug) => drug.name),
        baseline: baselineEval,
        proposed: proposedEval,
        delta,
      };

      setScenarioReport(report);

      if (addLog) {
        addLog(
          `What-If complete: risk ${formatPercent(baselineEval.riskScore)} -> ${formatPercent(proposedEval.riskScore)} (${delta.riskDelta >= 0 ? '+' : ''}${(delta.riskDelta * 100).toFixed(1)} pp)`,
          delta.riskDelta > 0 ? 'warning' : 'success',
          'AI'
        );
      }
    } catch (err) {
      const message = String(err?.message || 'What-If scenario comparison failed.');
      setScenarioError(message);
      if (addLog) {
        addLog(`What-If scenario failed: ${message}`, 'error', 'AI');
      }
    } finally {
      setRunningComparison(false);
    }
  };

  const runMutationEngine = async () => {
    setMutationError(null);

    if (apiStatus !== 'online') {
      setMutationError('Backend must be online to run mutation scanning.');
      return;
    }

    if (baselineRegimen.length < 3) {
      setMutationError('Mutation engine needs at least 3 baseline drugs for meaningful leave-one-out analysis.');
      return;
    }

    setMutationRunning(true);

    try {
      const baselineSignatureFromSelection = baselineRegimen
        .map((drug) => drug.name.toLowerCase())
        .join('|');

      const reportBaselineSignature = Array.isArray(scenarioReport?.baselineNames)
        ? scenarioReport.baselineNames.map((name) => String(name).toLowerCase()).join('|')
        : '';

      const baselineEval = reportBaselineSignature === baselineSignatureFromSelection
        ? scenarioReport.baseline
        : await evaluateScenarioRegimen(baselineRegimen);

      const mutationCandidates = baselineRegimen
        .map((drug) => ({
          removedDrug: drug.name,
          regimen: baselineRegimen.filter((item) => item.name.toLowerCase() !== drug.name.toLowerCase()),
        }))
        .filter((candidate) => candidate.regimen.length >= 2);

      const evaluated = await Promise.all(
        mutationCandidates.map(async (candidate) => {
          const evaluation = await evaluateScenarioRegimen(candidate.regimen);
          return {
            ...candidate,
            evaluation,
            riskDrop: baselineEval.riskScore - evaluation.riskScore,
            interactionDrop: baselineEval.significant.length - evaluation.significant.length,
            highRiskDrop: baselineEval.highRiskCount - evaluation.highRiskCount,
          };
        })
      );

      evaluated.sort((a, b) => {
        if (b.riskDrop !== a.riskDrop) return b.riskDrop - a.riskDrop;
        if (b.highRiskDrop !== a.highRiskDrop) return b.highRiskDrop - a.highRiskDrop;
        return a.evaluation.riskScore - b.evaluation.riskScore;
      });

      setMutationRecommendations(evaluated);

      if (addLog) {
        const best = evaluated[0];
        if (best) {
          addLog(
            `Mutation scan complete: best single-step removal is ${best.removedDrug} (${(best.riskDrop * 100).toFixed(1)} pp risk drop).`,
            best.riskDrop > 0 ? 'success' : 'warning',
            'AI'
          );
        }
      }
    } catch (err) {
      const message = String(err?.message || 'Mutation engine failed to evaluate candidate regimens.');
      setMutationError(message);
      if (addLog) {
        addLog(`Mutation engine failed: ${message}`, 'error', 'AI');
      }
    } finally {
      setMutationRunning(false);
    }
  };

  const applyMutationSuggestion = (suggestion) => {
    if (!suggestion || !Array.isArray(suggestion.regimen)) return;

    setProposedRegimen(suggestion.regimen);
    setScenarioReport(null);
    setScenarioError(null);

    if (addLog) {
      addLog(`Applied mutation suggestion: removed ${suggestion.removedDrug} from proposed regimen.`, 'info', 'AI');
    }
  };

  const saveSnapshot = () => {
    if (!scenarioReport) return;

    const snapshot = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      title: scenarioTitle.trim() || `Scenario @ ${new Date().toLocaleTimeString()}`,
      createdAt: scenarioReport.generatedAt,
      baseline: scenarioReport.baselineNames,
      proposed: scenarioReport.proposedNames,
      summary: {
        baselineRisk: scenarioReport.baseline.riskScore,
        proposedRisk: scenarioReport.proposed.riskScore,
        riskDelta: scenarioReport.delta.riskDelta,
        addedInteractions: scenarioReport.delta.added.length,
        removedInteractions: scenarioReport.delta.removed.length,
      },
    };

    setSnapshots((prev) => [snapshot, ...prev].slice(0, 8));
    if (addLog) {
      addLog('What-If snapshot saved to local history.', 'success', 'SYSTEM');
    }
  };

  const loadSnapshot = (snapshot) => {
    setScenarioTitle(snapshot.title || '');
    setProposedRegimen(dedupeScenarioDrugs((snapshot.proposed || []).map((name) => ({ name }))));
    setScenarioReport(null);
    setScenarioError(null);
  };

  const deleteSnapshot = (snapshotId) => {
    setSnapshots((prev) => prev.filter((snapshot) => snapshot.id !== snapshotId));
  };

  const exportScenarioJson = () => {
    if (!scenarioReport) return;

    const payload = {
      scenario_title: scenarioTitle.trim() || 'Untitled What-If Scenario',
      generated_at: scenarioReport.generatedAt,
      baseline_regimen: scenarioReport.baselineNames,
      proposed_regimen: scenarioReport.proposedNames,
      baseline_metrics: {
        risk_score: scenarioReport.baseline.riskScore,
        risk_level: scenarioReport.baseline.riskLevel,
        significant_interactions: scenarioReport.baseline.significant.length,
      },
      proposed_metrics: {
        risk_score: scenarioReport.proposed.riskScore,
        risk_level: scenarioReport.proposed.riskLevel,
        significant_interactions: scenarioReport.proposed.significant.length,
      },
      delta: {
        risk_delta: scenarioReport.delta.riskDelta,
        interaction_delta: scenarioReport.delta.interactionDelta,
      },
      added_interactions: scenarioReport.delta.added,
      removed_interactions: scenarioReport.delta.removed,
      changed_interactions: scenarioReport.delta.changed,
    };

    downloadScenarioFile(
      `what-if-scenario-${Date.now()}.json`,
      JSON.stringify(payload, null, 2),
      'application/json;charset=utf-8'
    );
  };

  const exportScenarioCsv = () => {
    if (!scenarioReport) return;
    const csvContent = buildScenarioCsvReport(scenarioReport);
    downloadScenarioFile(`what-if-scenario-${Date.now()}.csv`, csvContent, 'text/csv;charset=utf-8');
  };

  const addedMedications = proposedRegimen.filter(
    (drug) => !baselineRegimen.some((base) => base.name.toLowerCase() === drug.name.toLowerCase())
  );
  const removedMedications = baselineRegimen.filter(
    (drug) => !proposedRegimen.some((proposed) => proposed.name.toLowerCase() === drug.name.toLowerCase())
  );

  const baselineRiskRank = scenarioReport?.baseline?.maxRank ?? 0;
  const proposedRiskRank = scenarioReport?.proposed?.maxRank ?? 0;

  return (
    <div className="p-3 border border-theme bg-theme-secondary space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <GitCompare className="w-3.5 h-3.5 text-theme-accent" />
          <span className="text-[9px] text-theme-muted uppercase tracking-wider">What-If Scenario Builder</span>
        </div>
        <span className={`px-2 py-0.5 text-[8px] border uppercase tracking-wider ${apiStatus === 'online' ? 'border-risk-low/40 text-risk-low' : 'border-risk-high/40 text-risk-high'}`}>
          {apiStatus === 'online' ? 'Model Online' : 'Backend Offline'}
        </span>
      </div>

      <p className="text-[10px] text-theme-muted leading-relaxed">
        Compare current regimen versus a hypothetical medication change, then inspect interaction delta, risk shift, and removable hotspots.
      </p>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[9px] text-theme-muted uppercase tracking-wider">Current Regimen</span>
            <span className="text-[8px] text-theme-dim uppercase tracking-wider">{baselineRegimen.length} drugs</span>
          </div>

          {baselineRegimen.length === 0 ? (
            <p className="text-[10px] text-theme-dim">No active drugs selected in the sidebar.</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {baselineRegimen.map((drug) => (
                <span key={`baseline-${drug.name}`} className="px-2 py-1 text-[9px] border border-theme text-theme-secondary uppercase tracking-wider">
                  {drug.name}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[9px] text-theme-muted uppercase tracking-wider">Proposed Regimen</span>
            <span className="text-[8px] text-theme-dim uppercase tracking-wider">{proposedRegimen.length} drugs</span>
          </div>

          {proposedRegimen.length === 0 ? (
            <p className="text-[10px] text-theme-dim">Add at least two drugs for proposed scenario analysis.</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {proposedRegimen.map((drug) => (
                <span key={`proposed-${drug.name}`} className="inline-flex items-center gap-1 px-2 py-1 text-[9px] border border-theme-accent/40 text-theme-accent uppercase tracking-wider">
                  {drug.name}
                  <button
                    onClick={() => removeDrugFromProposed(drug.name)}
                    className="text-risk-high hover:text-risk-high/80"
                    aria-label={`Remove ${drug.name} from proposed regimen`}
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_auto_auto] gap-2 items-end">
        <label className="text-[9px] text-theme-muted uppercase tracking-wider">
          Add Drug To Proposed Regimen
          <div className="relative mt-1">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-theme-muted" />
            <input
              type="text"
              value={addQuery}
              onChange={(e) => setAddQuery(e.target.value)}
              placeholder="Search drugs to add..."
              className="w-full bg-theme-primary border border-theme py-2 pl-8 pr-8 text-[10px] text-theme-secondary placeholder:text-theme-dim focus:outline-none focus:border-theme-accent/50"
              disabled={apiStatus !== 'online'}
            />
            {addSearching && <Loader2 className="absolute right-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-theme-accent animate-spin" />}
          </div>
        </label>

        <button
          onClick={resetToBaseline}
          className="h-9 px-3 border border-theme text-[9px] text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors inline-flex items-center gap-1"
        >
          <RefreshCw className="w-3 h-3" />
          Sync
        </button>

        <button
          onClick={runScenarioComparison}
          disabled={runningComparison || apiStatus !== 'online'}
          className={`h-9 px-4 border text-[9px] uppercase tracking-wider transition-all inline-flex items-center gap-2 ${runningComparison
            ? 'border-theme-accent/40 text-theme-accent bg-theme-accent/10 cursor-wait'
            : 'border-theme-accent text-theme-accent hover:bg-theme-accent/10 disabled:opacity-50'
            }`}
        >
          {runningComparison ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <GitCompare className="w-3.5 h-3.5" />}
          {runningComparison ? 'Comparing...' : 'Compare Scenario'}
        </button>
      </div>

      {addResults.length > 0 && (
        <div className="border border-theme bg-theme-primary/90 divide-y divide-theme max-h-44 overflow-y-auto">
          {addResults.map((drug) => (
            <button
              key={`whatif-${drug.drugbank_id || drug.name}`}
              onClick={() => addDrugToProposed(drug)}
              className="w-full flex items-center justify-between px-3 py-2 hover:bg-theme-secondary transition-colors"
            >
              <span className="text-[10px] text-theme-secondary uppercase tracking-wider">{drug.name}</span>
              <span className="text-[8px] text-theme-accent uppercase tracking-wider">+ add</span>
            </button>
          ))}
        </div>
      )}

      {(addedMedications.length > 0 || removedMedications.length > 0) && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
          <div className="p-2 border border-theme bg-theme-panel/70">
            <span className="text-[8px] text-theme-muted uppercase tracking-wider">Proposed Adds</span>
            <div className="mt-1.5 flex flex-wrap gap-1">
              {addedMedications.length === 0 ? (
                <span className="text-[9px] text-theme-dim">No added drugs</span>
              ) : (
                addedMedications.map((drug) => (
                  <span key={`added-med-${drug.name}`} className="px-2 py-0.5 border border-risk-medium/30 text-risk-medium text-[8px] uppercase tracking-wider">
                    {drug.name}
                  </span>
                ))
              )}
            </div>
          </div>
          <div className="p-2 border border-theme bg-theme-panel/70">
            <span className="text-[8px] text-theme-muted uppercase tracking-wider">Proposed Removes</span>
            <div className="mt-1.5 flex flex-wrap gap-1">
              {removedMedications.length === 0 ? (
                <span className="text-[9px] text-theme-dim">No removed drugs</span>
              ) : (
                removedMedications.map((drug) => (
                  <span key={`removed-med-${drug.name}`} className="px-2 py-0.5 border border-risk-low/30 text-risk-low text-[8px] uppercase tracking-wider">
                    {drug.name}
                  </span>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {scenarioError && (
        <div className="p-2 border border-risk-high/30 bg-risk-high/10 text-[10px] text-risk-high">
          {scenarioError}
        </div>
      )}

      <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-theme-accent" />
            <span className="text-[9px] text-theme-muted uppercase tracking-wider">Regimen Mutation Engine</span>
          </div>
          <button
            onClick={runMutationEngine}
            disabled={mutationRunning || apiStatus !== 'online'}
            className={`px-3 py-1.5 border text-[8px] uppercase tracking-wider transition-all inline-flex items-center gap-1 ${mutationRunning
              ? 'border-theme-accent/40 text-theme-accent bg-theme-accent/10 cursor-wait'
              : 'border-theme-accent text-theme-accent hover:bg-theme-accent/10 disabled:opacity-50'
              }`}
          >
            {mutationRunning ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
            {mutationRunning ? 'Scanning...' : 'Run Mutation Scan'}
          </button>
        </div>

        <p className="text-[10px] text-theme-muted leading-relaxed">
          Simulates single-step baseline mutations (remove one drug at a time), then ranks which removal creates the biggest risk drop.
        </p>

        {mutationError && (
          <div className="p-2 border border-risk-high/30 bg-risk-high/10 text-[10px] text-risk-high">
            {mutationError}
          </div>
        )}

        {mutationRecommendations.length > 0 && (
          <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
            {mutationRecommendations.slice(0, 8).map((row) => (
              <div key={`mutation-${row.removedDrug}`} className="p-2 border border-theme/30 bg-theme-primary/70">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <div>
                    <p className="text-[10px] text-theme-secondary uppercase tracking-wider">Remove {row.removedDrug}</p>
                    <p className="text-[8px] text-theme-dim uppercase tracking-wider mt-0.5">
                      Projected Risk: {formatPercent(row.evaluation.riskScore)} ({String(row.evaluation.riskLevel || 'unknown').replace(/_/g, ' ')})
                    </p>
                  </div>
                  <button
                    onClick={() => applyMutationSuggestion(row)}
                    className="px-2 py-1 border border-theme text-[8px] text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors"
                  >
                    Use in What-If
                  </button>
                </div>

                <div className="flex flex-wrap gap-2 text-[8px] uppercase tracking-wider">
                  <span className={`${row.riskDrop >= 0 ? 'text-risk-low' : 'text-risk-high'}`}>
                    Risk Delta: {(row.riskDrop >= 0 ? '+' : '') + (row.riskDrop * 100).toFixed(1)} pp
                  </span>
                  <span className="text-theme-muted">
                    Significant Delta: {(row.interactionDrop >= 0 ? '+' : '') + row.interactionDrop}
                  </span>
                  <span className="text-theme-muted">
                    High-Risk Delta: {(row.highRiskDrop >= 0 ? '+' : '') + row.highRiskDrop}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {scenarioReport && (
        <div className="space-y-3">
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-2">
            <div className="p-2 border border-theme/20 bg-theme-panel/80">
              <p className="text-[8px] text-theme-muted uppercase tracking-wider">Current Risk</p>
              <p className="text-sm text-theme-secondary mt-1">{formatPercent(scenarioReport.baseline.riskScore)}</p>
              <span className={`inline-block mt-1 px-1.5 py-0.5 text-[8px] border uppercase tracking-wider ${getScenarioRiskTone(baselineRiskRank)}`}>
                {scenarioReport.baseline.riskLevel.replace(/_/g, ' ')}
              </span>
            </div>

            <div className="p-2 border border-theme/20 bg-theme-panel/80">
              <p className="text-[8px] text-theme-muted uppercase tracking-wider">Proposed Risk</p>
              <p className="text-sm text-theme-accent mt-1">{formatPercent(scenarioReport.proposed.riskScore)}</p>
              <span className={`inline-block mt-1 px-1.5 py-0.5 text-[8px] border uppercase tracking-wider ${getScenarioRiskTone(proposedRiskRank)}`}>
                {scenarioReport.proposed.riskLevel.replace(/_/g, ' ')}
              </span>
            </div>

            <div className="p-2 border border-theme/20 bg-theme-panel/80">
              <p className="text-[8px] text-theme-muted uppercase tracking-wider">Risk Delta</p>
              <p className={`text-sm mt-1 ${getScenarioDeltaTone(scenarioReport.delta.riskDelta)}`}>
                {(scenarioReport.delta.riskDelta >= 0 ? '+' : '') + (scenarioReport.delta.riskDelta * 100).toFixed(1)} pp
              </p>
              <p className="text-[8px] text-theme-muted mt-1 uppercase tracking-wider">
                Interaction Delta: {(scenarioReport.delta.interactionDelta >= 0 ? '+' : '') + scenarioReport.delta.interactionDelta}
              </p>
            </div>
          </div>

          <div className="p-2 border border-theme bg-theme-panel/70 text-[9px] text-theme-muted uppercase tracking-wider flex flex-wrap gap-x-3 gap-y-1">
            <span>Generated: {formatScenarioTimestamp(scenarioReport.generatedAt)}</span>
            <span>Baseline Significant: {scenarioReport.baseline.significant.length}</span>
            <span>Proposed Significant: {scenarioReport.proposed.significant.length}</span>
            {(scenarioReport.baseline.failedPairs > 0 || scenarioReport.proposed.failedPairs > 0) && (
              <span className="text-risk-medium">
                Partial Pair Failures: {scenarioReport.baseline.failedPairs + scenarioReport.proposed.failedPairs}
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            <ScenarioInteractionTable
              title="New Interactions Introduced"
              rows={scenarioReport.delta.added.slice(0, 10)}
              emptyLabel="No new interactions introduced."
            />
            <ScenarioInteractionTable
              title="Interactions Removed"
              rows={scenarioReport.delta.removed.slice(0, 10)}
              emptyLabel="No baseline interactions removed."
            />
          </div>

          <ScenarioInteractionTable
            title="Changed Interaction Intensity"
            rows={scenarioReport.delta.changed.slice(0, 10)}
            emptyLabel="Shared pairs are stable."
            includeChange={true}
          />

          <div className="p-3 border border-theme bg-theme-panel/80">
            <div className="grid grid-cols-1 xl:grid-cols-[1fr_auto_auto_auto] gap-2 items-center">
              <input
                type="text"
                value={scenarioTitle}
                onChange={(e) => setScenarioTitle(e.target.value)}
                placeholder="Optional scenario title (for snapshots/exports)"
                className="bg-theme-primary border border-theme px-2 py-2 text-[10px] text-theme-secondary placeholder:text-theme-dim focus:outline-none focus:border-theme-accent/50"
              />
              <button
                onClick={saveSnapshot}
                className="px-3 py-2 border border-theme text-[9px] text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors"
              >
                Save Snapshot
              </button>
              <button
                onClick={exportScenarioJson}
                className="px-3 py-2 border border-theme text-[9px] text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors inline-flex items-center gap-1"
              >
                <FileText className="w-3 h-3" /> JSON
              </button>
              <button
                onClick={exportScenarioCsv}
                className="px-3 py-2 border border-theme text-[9px] text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors inline-flex items-center gap-1"
              >
                <FileText className="w-3 h-3" /> CSV
              </button>
            </div>
          </div>
        </div>
      )}

      <CausalInteractionReplayPanel
        report={scenarioReport}
        mutationRecommendations={mutationRecommendations}
      />

      <UncertaintyHeatLensPanel report={scenarioReport} />

      <div className="p-3 border border-theme bg-theme-panel/80 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Scenario Snapshots</span>
          <span className="text-[8px] text-theme-dim uppercase tracking-wider">{snapshots.length} saved</span>
        </div>

        {snapshots.length === 0 ? (
          <p className="text-[10px] text-theme-dim">No saved snapshots yet. Run a comparison and click Save Snapshot.</p>
        ) : (
          <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
            {snapshots.map((snapshot) => (
              <div key={snapshot.id} className="p-2 border border-theme/30 bg-theme-primary/70">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <p className="text-[10px] text-theme-secondary uppercase tracking-wider">{snapshot.title}</p>
                    <p className="text-[8px] text-theme-dim uppercase tracking-wider mt-0.5">{formatScenarioTimestamp(snapshot.createdAt)}</p>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => loadSnapshot(snapshot)}
                      className="px-2 py-1 text-[8px] border border-theme text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors"
                    >
                      Load
                    </button>
                    <button
                      onClick={() => deleteSnapshot(snapshot.id)}
                      className="px-2 py-1 text-[8px] border border-risk-high/30 text-risk-high uppercase tracking-wider hover:bg-risk-high/10 transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
                <p className="text-[8px] text-theme-dim uppercase tracking-wider mt-1">
                  {snapshot.baseline.length} ➜ {snapshot.proposed.length} drugs
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ResearchWorkspace({ addLog, selectedDrugs, apiStatus }) {
  return (
    <main className="flex-1 overflow-y-auto p-6 bg-theme-secondary">
      <div className="max-w-6xl mx-auto space-y-4">
        <div className="p-4 border border-theme bg-theme-panel/80 backdrop-blur-sm">
          <div className="flex items-center gap-2 mb-2">
            <FlaskConical className="w-4 h-4 text-theme-accent" />
            <h2 className="text-sm uppercase tracking-widest text-theme-primary">Research Tools Lab</h2>
          </div>
          <p className="text-xs text-theme-muted leading-relaxed">
            This space is for reliability and evidence-focused analysis tooling. Calibration QA is live here now, and future research modules can be added as separate cards.
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <div className="xl:col-span-2 space-y-4">
            <WhatIfScenarioBuilder selectedDrugs={selectedDrugs} apiStatus={apiStatus} addLog={addLog} />
            <CalibrationQAPanel addLog={addLog} defaultExpanded={true} />
          </div>

          <div className="space-y-4">
            <div className="p-4 border border-theme bg-theme-panel/80 backdrop-blur-sm">
              <h3 className="text-[10px] uppercase tracking-widest text-theme-muted mb-2">Live Tool</h3>
              <p className="text-xs text-theme-secondary">Causal Interaction Replay now narrates the interaction chain from pair trigger to mitigation action.</p>
            </div>

            <div className="p-4 border border-theme bg-theme-panel/80 backdrop-blur-sm">
              <h3 className="text-[10px] uppercase tracking-widest text-theme-muted mb-2">Mutation Engine</h3>
              <p className="text-xs text-theme-secondary">Single-step leave-one-out regimen scanner now ranks which removal best lowers projected risk.</p>
            </div>

            <div className="p-4 border border-theme bg-theme-panel/80 backdrop-blur-sm">
              <h3 className="text-[10px] uppercase tracking-widest text-theme-muted mb-2">Uncertainty Heat Lens</h3>
              <p className="text-xs text-theme-secondary">Pairwise uncertainty hotspots are now scored from confidence, calibration shift, and fallback pressure.</p>
            </div>

            <div className="p-4 border border-theme bg-theme-panel/80 backdrop-blur-sm">
              <h3 className="text-[10px] uppercase tracking-widest text-theme-muted mb-2">Calibration Intelligence</h3>
              <p className="text-xs text-theme-secondary">Reliability curve + per-bin diagnostics remain active for calibration quality review.</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const { addLog } = useSystemLogs();
  const { theme, toggleTheme } = useTheme();

  // API State
  const [apiStatus, setApiStatus] = useState('checking');
  const [error, setError] = useState(null);

  // Drug Selection State
  const [selectedDrugs, setSelectedDrugs] = useState(() => loadStoredSelectedDrugs());
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [polypharmacyResult, setPolypharmacyResult] = useState(null);
  const [digitalTwinResult, setDigitalTwinResult] = useState(null);

  // Enhanced Data State
  const [drugInfoCache, setDrugInfoCache] = useState({});
  const [interactionEvidence, setInteractionEvidence] = useState(null);
  const [isLoadingDrugInfo, setIsLoadingDrugInfo] = useState(false);

  // UI State
  const [activeTab, setActiveTab] = useState('molecules2d');
  const [showSearch, setShowSearch] = useState(false);
  const [viewMode, setViewMode] = useState('analysis'); // 'analysis' | 'stats' | 'compare' | 'research'
  const [showAlternatives, setShowAlternatives] = useState(false);
  const [showScanner, setShowScanner] = useState(false);
  const [demoModeEnabled, setDemoModeEnabled] = useState(() => readDemoModeSetting());
  const [loadingDemoGroupId, setLoadingDemoGroupId] = useState(null);

  // Chat State
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const chatEndRef = useRef(null);

  // Mobile State
  const [isMobile, setIsMobile] = useState(false);
  const [mobileView, setMobileView] = useState('drugs'); // 'drugs' | 'viz' | 'results' | 'chat'
  const [showMobileSearch, setShowMobileSearch] = useState(false);
  const [showMobileDrugPanel, setShowMobileDrugPanel] = useState(false);

  // Database status
  const [showDbWarning, setShowDbWarning] = useState(false);
  const [dbDrugCount, setDbDrugCount] = useState(null);

  const debouncedSearch = useDebounce(searchQuery, 300);

  // Detect mobile screen size
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const refreshDemoMode = () => {
      setDemoModeEnabled(readDemoModeSetting());
    };

    window.addEventListener('storage', refreshDemoMode);
    window.addEventListener('focus', refreshDemoMode);

    return () => {
      window.removeEventListener('storage', refreshDemoMode);
      window.removeEventListener('focus', refreshDemoMode);
    };
  }, []);

  // Check API health on mount
  useEffect(() => {
    const checkApi = async () => {
      addLog('Initiating system health check...', 'info', 'SYSTEM');
      try {
        await checkHealth();
        setApiStatus('online');
        addLog('Backend services online', 'success', 'API');
        
        // Check database drug count
        try {
          const stats = await getDatabaseStats();
          const drugCount = stats?.total_drugs || stats?.drugs_count || 0;
          setDbDrugCount(drugCount);
          if (drugCount < 10) {
            setShowDbWarning(true);
            addLog(`Warning: Only ${drugCount} drugs in database - Neo4j may be inactive`, 'warning', 'DATABASE');
          }
        } catch (statsErr) {
          console.warn('Could not fetch database stats:', statsErr);
        }
      } catch (err) {
        console.error('API check failed:', err);
        setApiStatus('offline');
        addLog('Failed to connect to backend services', 'error', 'API');
      }
    };
    checkApi();
  }, []);

  // Search drugs when query changes
  useEffect(() => {
    const performSearch = async () => {
      if (!debouncedSearch || debouncedSearch.length < 2) {
        setSearchResults([]);
        return;
      }

      if (apiStatus !== 'online') {
        setSearchResults([]);
        return;
      }

      setIsSearching(true);
      addLog(`Searching database for "${debouncedSearch}"...`, 'info', 'DATABASE');
      try {
        const response = await searchDrugs(debouncedSearch);
        // Filter out already selected drugs
        const filtered = (response.results || []).filter(
          drug => !selectedDrugs.some(s => s.drugbank_id === drug.drugbank_id || s.name === drug.name)
        );
        setSearchResults(filtered);
        addLog(`Found ${filtered.length} matches`, 'success', 'DATABASE');
      } catch (err) {
        console.error('Search failed:', err);
        setSearchResults([]);
        addLog(`Search query failed: ${err.message}`, 'error', 'DATABASE');
      } finally {
        setIsSearching(false);
      }
    };

    performSearch();
  }, [debouncedSearch, apiStatus, selectedDrugs]);

  // Scroll chat to bottom (only when there are messages)
  useEffect(() => {
    if (messages.length > 0) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    if (!selectedDrugs || selectedDrugs.length === 0) {
      window.localStorage.removeItem(SELECTED_DRUGS_STORAGE_KEY);
      return;
    }

    window.localStorage.setItem(SELECTED_DRUGS_STORAGE_KEY, JSON.stringify(selectedDrugs));
  }, [selectedDrugs]);

  const addDrug = async (drug) => {
    setSelectedDrugs(prev => [...prev, drug]);
    setSearchQuery('');
    setSearchResults([]);
    setShowSearch(false);
    setResult(null);
    setPolypharmacyResult(null);
    setDigitalTwinResult(null);
    setInteractionEvidence(null);
    
    // Fetch drug info (side effects, etc.)
    if (!drugInfoCache[drug.name.toLowerCase()]) {
      try {
        addLog(`Fetching drug info for ${drug.name}...`, 'info', 'DATABASE');
        const info = await getDrugInfo(drug.name, false); // Skip FAERS for speed
        setDrugInfoCache(prev => ({
          ...prev,
          [drug.name.toLowerCase()]: info
        }));
        addLog(`Loaded ${info.side_effects?.length || 0} side effects for ${drug.name}`, 'success', 'DATABASE');
      } catch (err) {
        console.warn('Could not fetch drug info:', err);
      }
    }
  };

  const resolveDemoDrugByName = async (drugName) => {
    try {
      const response = await searchDrugs(drugName);
      const results = Array.isArray(response?.results) ? response.results : [];
      const candidate = pickBestSearchMatch(results, drugName);

      if (!candidate) {
        return {
          name: drugName,
          unresolved: true,
          requestedName: drugName,
        };
      }

      return {
        ...candidate,
        requestedName: drugName,
        unresolved: false,
      };
    } catch {
      return {
        name: drugName,
        unresolved: true,
        requestedName: drugName,
      };
    }
  };

  const applyDemoDrugGroup = async (group, replaceRegimen = false) => {
    if (!group || !Array.isArray(group.drugs) || group.drugs.length === 0) return;

    if (apiStatus !== 'online') {
      addLog('Demo group loading requires backend connectivity for drug lookup.', 'warning', 'SYSTEM');
      return;
    }

    const modeSuffix = replaceRegimen ? 'replace' : 'add';
    setLoadingDemoGroupId(`${group.id}:${modeSuffix}`);

    try {
      const resolvedCandidates = await Promise.all(group.drugs.map((name) => resolveDemoDrugByName(name)));

      const baseRegimen = replaceRegimen ? [] : [...selectedDrugs];
      const seen = new Set(baseRegimen.map((drug) => String(drug?.name || '').toLowerCase()));
      const merged = [...baseRegimen];

      let addedCount = 0;
      let skippedCount = 0;
      let unresolvedCount = 0;

      resolvedCandidates.forEach((candidate) => {
        const fallbackName = String(candidate?.requestedName || candidate?.name || '').trim();
        const candidateName = String(candidate?.name || fallbackName).trim();
        if (!candidateName) {
          unresolvedCount += 1;
          return;
        }

        const key = candidateName.toLowerCase();
        if (seen.has(key)) {
          skippedCount += 1;
          return;
        }

        seen.add(key);
        merged.push(candidate.unresolved ? { name: fallbackName } : candidate);

        addedCount += 1;
        if (candidate.unresolved) {
          unresolvedCount += 1;
        }
      });

      setSelectedDrugs(merged);
      setResult(null);
      setPolypharmacyResult(null);
      setDigitalTwinResult(null);
      setInteractionEvidence(null);
      setSearchQuery('');
      setSearchResults([]);
      setShowSearch(false);

      const actionWord = replaceRegimen ? 'loaded' : 'added';
      const statusLevel = addedCount > 0 ? 'success' : 'warning';
      const extra = [
        skippedCount > 0 ? `${skippedCount} already selected` : null,
        unresolvedCount > 0 ? `${unresolvedCount} unresolved lookup` : null,
      ].filter(Boolean).join(', ');

      addLog(
        `Demo group "${group.title}" ${actionWord}: ${addedCount} drug(s)${extra ? ` (${extra})` : ''}.`,
        statusLevel,
        'SYSTEM'
      );
    } finally {
      setLoadingDemoGroupId(null);
    }
  };

  const removeDrug = (drugId) => {
    setSelectedDrugs(prev => prev.filter(d => d.drugbank_id !== drugId && d.name !== drugId));
    setResult(null);
    setPolypharmacyResult(null);
    setDigitalTwinResult(null);
    setInteractionEvidence(null);
  };

  const clearAllDrugs = () => {
    if (!selectedDrugs.length) return;

    setSelectedDrugs([]);
    setSearchQuery('');
    setSearchResults([]);
    setShowSearch(false);
    setResult(null);
    setPolypharmacyResult(null);
    setDigitalTwinResult(null);
    setInteractionEvidence(null);
    addLog('Cleared selected drug regimen', 'info', 'SYSTEM');
  };

  // Handle drug detected from camera scanner
  const handleScannedDrug = (drug) => {
    // Add the detected drug
    addDrug({
      name: drug.name,
      drugbank_id: drug.drugbank_id || drug.id,
      smiles: drug.smiles,
      ...drug
    });
    // Close the scanner
    setShowScanner(false);
    addLog(`Drug "${drug.name}" added via camera scan`, 'success', 'SCANNER');
  };

  const runAnalysis = async () => {
    if (selectedDrugs.length < 2 || apiStatus !== 'online') return;

    setIsAnalyzing(true);
    setError(null);
    setInteractionEvidence(null);
    addLog(`Starting DDI analysis for ${selectedDrugs.map(d => d.name).join(' + ')}`, 'info', 'SYSTEM');

    try {
      if (selectedDrugs.length === 2) {
        // Two-drug prediction
        addLog('Querying Macroscopic GraphSAGE model...', 'info', 'AI');
        const start = performance.now();
        const response = await predictDDI(
          { name: selectedDrugs[0].name, smiles: selectedDrugs[0].smiles },
          { name: selectedDrugs[1].name, smiles: selectedDrugs[1].smiles }
        );
        const latency = (performance.now() - start).toFixed(2);
        addLog(`Prediction received in ${latency}ms`, 'success', 'AI');
        addLog(`Risk Level: ${response.risk_level} (${response.risk_score.toFixed(2)})`, 'warning', 'AI');

        setResult(response);
        setPolypharmacyResult(null);
        setDigitalTwinResult(null);
        
        // Fetch real-world evidence in background
        addLog('Fetching real-world evidence from FDA FAERS...', 'info', 'DATABASE');
        try {
          const evidence = await getInteractionInfo(
            selectedDrugs[0].name,
            selectedDrugs[1].name,
            true
          );
          setInteractionEvidence(evidence);
          addLog(`Found ${evidence.faers_data?.total_reports || 0} FDA adverse event reports`, 'success', 'DATABASE');
        } catch (err) {
          console.warn('Could not fetch interaction evidence:', err);
          addLog('Real-world evidence unavailable', 'warning', 'DATABASE');
        }
      } else {
        // Polypharmacy analysis
        addLog('Initiating Graph Neural Network (GNN) for polypharmacy...', 'info', 'AI');
        const drugs = selectedDrugs.map(d => ({ name: d.name, smiles: d.smiles }));
        const response = await analyzePolypharmacy(drugs);
        setPolypharmacyResult(response);
        addLog(`Processed ${response.total_interactions} interaction pathways`, 'success', 'AI');

        try {
          const twinResponse = await analyzePolypharmacyDigitalTwin(drugs);
          setDigitalTwinResult(twinResponse);
          const twinScore = Math.round((twinResponse?.summary?.toxicity_score || 0) * 100);
          addLog(`Digital Twin toxicity score: ${twinScore}%`, 'info', 'AI');

          // Surface the Twin panel automatically after successful N-order analysis.
          if (drugs.length >= 3) {
            setActiveTab('polyTwin');
            if (isMobile) {
              setMobileView('viz');
            }
          }
        } catch (twinErr) {
          console.warn('Digital Twin analysis failed:', twinErr);
          setDigitalTwinResult(null);
          addLog('Digital Twin unavailable for this run', 'warning', 'AI');
        }

        // Set summary result
        if (response.interactions && response.interactions.length > 0) {
          const topInteraction = response.interactions.sort((a, b) => b.risk_score - a.risk_score)[0];
          setResult({
            drug_a: topInteraction.source,
            drug_b: topInteraction.target,
            risk_score: response.max_risk_score,
            risk_level: response.overall_risk_level,
            severity: topInteraction.severity,
            confidence: 0.85,
            mechanism_hypothesis: `${response.total_interactions} interactions detected. ${response.hub_drug} is the hub drug with ${response.hub_interaction_count} interactions.`,
            affected_systems: Object.entries(response.body_map || {}).map(([system, severity]) => ({
              system,
              severity,
              symptoms: []
            }))
          });
        }
      }
    } catch (err) {
      console.error('Analysis failed:', err);
      const details = err?.message ? ` (${err.message})` : '';
      setError(`Failed to analyze interactions. Please try again${details}.`);
      addLog(`Analysis process failed: ${err.message}`, 'error', 'SYSTEM');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || isChatLoading || apiStatus !== 'online') return;

    const userMessage = chatInput.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setChatInput('');
    setIsChatLoading(true);
    addLog('Processing natural language query...', 'info', 'AI');

    try {
      const contextDrugs = selectedDrugs.map(d => d.name);
      const response = await sendChatMessage(userMessage, contextDrugs, sessionId);
      setSessionId(response.session_id);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.response,
        sources: response.sources
      }]);
      addLog('Response generated via GraphRAG', 'success', 'AI');
    } catch (err) {
      console.error('Chat failed:', err);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        isError: true
      }]);
      addLog('Chat processing failed', 'error', 'AI');
    } finally {
      setIsChatLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'critical': return 'text-risk-critical';
      case 'high': return 'text-risk-high';
      case 'medium': return 'text-risk-medium';
      default: return 'text-risk-low';
    }
  };

  const getRiskBgColor = (riskLevel) => {
    switch (riskLevel) {
      case 'critical': return 'border-risk-critical/50 text-risk-critical';
      case 'high': return 'border-risk-high/50 text-risk-high';
      case 'medium': return 'border-risk-medium/50 text-risk-medium';
      default: return 'border-risk-low/50 text-risk-low';
    }
  };

  const getBodyMapData = () => {
    if (!result?.affected_systems) return {};
    const bodyMap = {};
    result.affected_systems.forEach(sys => {
      bodyMap[sys.system] = sys.severity || 0.5;
    });
    return bodyMap;
  };

  const DemoGroupPanel = ({ compact = false }) => {
    if (!demoModeEnabled) return null;

    return (
      <div className={`border border-theme-accent/30 bg-theme-accent/5 ${compact ? 'p-3' : 'p-4'} space-y-3`}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Database className="w-3.5 h-3.5 text-theme-accent" />
            <span className="text-[9px] text-theme-muted uppercase tracking-wider">Demo Group Loader</span>
          </div>
          <button
            onClick={() => navigate('/settings')}
            className="text-[8px] border border-theme px-2 py-0.5 text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors"
          >
            Settings
          </button>
        </div>

        <p className="text-[10px] text-theme-muted leading-relaxed">
          Curated candidate sets for quick demo workflows. Add to current regimen or replace current selection in one click.
        </p>

        <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
          {DEMO_DRUG_GROUPS.map((group) => {
            const addActionId = `${group.id}:add`;
            const replaceActionId = `${group.id}:replace`;
            const isAdding = loadingDemoGroupId === addActionId;
            const isReplacing = loadingDemoGroupId === replaceActionId;
            const isBusy = Boolean(loadingDemoGroupId);

            return (
              <div key={group.id} className="p-2 border border-theme/40 bg-theme-panel/70">
                <p className="text-[10px] text-theme-secondary uppercase tracking-wider">{group.title}</p>
                <p className="text-[10px] text-theme-dim mt-1 leading-relaxed">{group.reason}</p>
                <p className="text-[8px] text-theme-muted mt-1 uppercase tracking-wider">{group.drugs.join(' • ')}</p>

                <div className="mt-2 flex gap-1.5">
                  <button
                    onClick={() => applyDemoDrugGroup(group, false)}
                    disabled={apiStatus !== 'online' || isBusy}
                    className="flex-1 px-2 py-1 border border-theme text-[8px] text-theme-muted uppercase tracking-wider hover:text-theme-accent hover:border-theme-accent/40 transition-colors disabled:opacity-40"
                  >
                    {isAdding ? 'Adding...' : 'Add Group'}
                  </button>
                  <button
                    onClick={() => applyDemoDrugGroup(group, true)}
                    disabled={apiStatus !== 'online' || isBusy}
                    className="flex-1 px-2 py-1 border border-theme-accent/50 text-[8px] text-theme-accent uppercase tracking-wider hover:bg-theme-accent/10 transition-colors disabled:opacity-40"
                  >
                    {isReplacing ? 'Loading...' : 'Use Only'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Mobile Bottom Navigation Component
  const MobileBottomNav = () => (
    <nav className="fixed bottom-0 left-0 right-0 z-50 bg-theme-primary/95 backdrop-blur-md border-t border-theme safe-area-inset-bottom">
      <div className="flex items-center justify-around h-16">
        {[
          { id: 'drugs', icon: Pill, label: 'Drugs' },
          { id: 'viz', icon: Box, label: 'Visualize' },
          { id: 'results', icon: Activity, label: 'Results' },
          { id: 'compare', icon: GitCompare, label: 'Compare+' },
          { id: 'chat', icon: MessageCircle, label: 'Chat' },
        ].map((item) => (
          <button
            key={item.id}
            onClick={() => setMobileView(item.id)}
            className={`flex flex-col items-center justify-center flex-1 h-full transition-colors ${
              mobileView === item.id
                ? 'text-theme-accent bg-theme-accent/10'
                : 'text-theme-muted'
            }`}
          >
            <item.icon className="w-5 h-5 mb-1" />
            <span className="text-[10px] uppercase tracking-wider">{item.label}</span>
            {item.id === 'drugs' && selectedDrugs.length > 0 && (
              <span className="absolute top-2 right-1/2 translate-x-6 w-4 h-4 bg-theme-accent text-theme-primary text-[9px] flex items-center justify-center rounded-full">
                {selectedDrugs.length}
              </span>
            )}
          </button>
        ))}
      </div>
    </nav>
  );

  // Mobile Drug Card Component
  const MobileDrugCard = ({ drug, index, onRemove }) => {
    const drugInfo = drugInfoCache[drug.name.toLowerCase()];
    const hasSmiles = drug.has_smiles || (drug.smiles && drug.smiles.length > 5);
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, x: -100 }}
        className={`p-4 border backdrop-blur-sm ${!hasSmiles ? 'border-risk-medium/30 bg-risk-medium/10' : 'border-theme bg-theme-primary/80'}`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 border flex items-center justify-center text-xs font-normal uppercase ${
              !hasSmiles ? 'border-risk-high/50 text-risk-high bg-risk-high/10' : 'border-theme-accent/50 text-theme-accent'
            }`}>
              {hasSmiles ? drug.name.substring(0, 2).toUpperCase() : '⚠'}
            </div>
            <div>
              <span className="text-sm font-normal text-theme-primary block">{drug.name}</span>
              <span className="text-[10px] text-theme-muted uppercase tracking-widest">{drug.category || 'Drug'}</span>
            </div>
          </div>
          <button
            onClick={() => onRemove(drug.drugbank_id || drug.name)}
            className="p-2 border border-risk-high/30 text-risk-high"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
        
        {/* Side effects preview */}
        {drugInfo?.side_effects?.length > 0 && (
          <div className="mt-3 pt-3 border-t border-theme">
            <div className="flex flex-wrap gap-1">
              {drugInfo.side_effects.slice(0, 3).map((effect, j) => (
                <span key={j} className="px-2 py-1 text-[9px] border border-risk-medium/20 text-risk-medium/70 uppercase">
                  {effect}
                </span>
              ))}
              {drugInfo.side_effects.length > 3 && (
                <span className="px-2 py-1 text-[9px] text-theme-muted">+{drugInfo.side_effects.length - 3}</span>
              )}
            </div>
          </div>
        )}
      </motion.div>
    );
  };

  // Mobile Results View
  const MobileResultsView = () => (
    <div className="p-4 pb-24 space-y-4">
      {result ? (
        <>
          {/* Risk Card */}
          <div className={`p-5 border backdrop-blur-sm ${getRiskBgColor(result.risk_level)} relative`}>
            <div className="flex items-center gap-3 mb-4">
              {result.severity === 'no_interaction' ? (
                <Shield className="w-6 h-6 text-risk-low" />
              ) : (
                <AlertTriangle className="w-6 h-6" />
              )}
              <div>
                <p className="text-base font-normal uppercase tracking-wider">
                  {result.severity === 'no_interaction' ? 'No Significant Interaction' : `${result.risk_level || result.severity} Risk`}
                </p>
                <p className="text-xs opacity-70 mt-1">
                  {result.drug_a || selectedDrugs[0]?.name} + {result.drug_b || selectedDrugs[1]?.name}
                </p>
              </div>
            </div>
            
            {/* Risk Score Visual */}
            {result.risk_score !== undefined && (
              <div className="mb-4">
                <RiskGauge score={result.risk_score} riskLevel={result.risk_level || result.severity} />
              </div>
            )}
          </div>

          <PredictionTransparencyPanel result={result} isMobile={true} />

          {/* Mechanism */}
          {result.mechanism_hypothesis && (
            <div className="p-4 border border-theme bg-theme-primary/80 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-3">
                <Brain className="w-4 h-4 text-theme-accent" />
                <span className="text-[10px] text-theme-muted uppercase tracking-widest">Mechanism</span>
              </div>
              <p className="text-sm text-theme-secondary leading-relaxed">{result.mechanism_hypothesis}</p>
            </div>
          )}

          {/* Affected Systems */}
          {result.affected_systems?.length > 0 && (
            <div className="p-4 border border-theme bg-theme-primary/80 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-3">
                <Target className="w-4 h-4 text-risk-high" />
                <span className="text-[10px] text-theme-muted uppercase tracking-widest">Affected Systems</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {result.affected_systems.map((sys, i) => (
                  <span key={i} className="px-3 py-1.5 border border-risk-high/30 bg-risk-high/10 backdrop-blur-sm text-xs text-risk-high uppercase">
                    {sys.system || sys}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Confidence */}
          {result.confidence && (
            <div className="p-4 border border-theme bg-theme-primary/80 backdrop-blur-sm flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-theme-accent" />
                <span className="text-[10px] text-theme-muted uppercase tracking-widest">Model Confidence</span>
              </div>
              <span className="text-lg font-normal text-theme-accent">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
          )}

          <EvidenceChainTimeline interactionEvidence={interactionEvidence} compact={true} />
          <EvidenceUncertaintyPanel interactionEvidence={interactionEvidence} compact={true} />

          {/* FDA Evidence */}
          {interactionEvidence?.faers_data && (
            <div className="p-4 border border-theme-accent/30 bg-theme-accent/10 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-3">
                <Activity className="w-4 h-4 text-theme-accent" />
                <span className="text-[10px] text-theme-muted uppercase tracking-widest">FDA Real-World Evidence</span>
              </div>
              <div className="text-center py-3">
                <span className="text-3xl font-mono text-theme-accent">
                  {interactionEvidence.faers_data.total_reports?.toLocaleString() || '0'}
                </span>
                <p className="text-[10px] text-theme-muted uppercase mt-1">Adverse Event Reports</p>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center justify-center py-16 text-center bg-theme-primary/80 backdrop-blur-sm border border-theme">
          <div className="w-16 h-16 border border-theme flex items-center justify-center mb-4">
            <Activity className="w-8 h-8 text-theme-dim" />
          </div>
          <p className="text-sm text-theme-muted mb-2 uppercase tracking-wider">No Analysis Yet</p>
          <p className="text-xs text-theme-dim">Select drugs and run analysis</p>
        </div>
      )}
    </div>
  );

  // Mobile Chat View - rendered inline to prevent input focus issues
  const mobileChatContent = (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Under Construction Banner */}
      <div className="mx-4 mt-4 p-3 border border-risk-medium/40 bg-risk-medium/10 backdrop-blur-sm flex items-center gap-3">
        <AlertTriangle className="w-5 h-5 text-risk-medium flex-shrink-0" />
        <div>
          <p className="text-xs text-risk-medium font-medium uppercase tracking-wider">Under Construction</p>
          <p className="text-[10px] text-theme-muted">AI Chat is still being developed and may not respond correctly</p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 pb-24 space-y-3">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center bg-theme-primary/80 backdrop-blur-sm border border-theme p-6">
            <Sparkles className="w-8 h-8 text-theme-dim mb-3" />
            <p className="text-sm text-theme-muted">Ask about drug interactions</p>
            <p className="text-xs text-theme-dim mt-1">I can help with mechanisms, alternatives, and more</p>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] p-3 text-sm leading-relaxed break-words overflow-hidden backdrop-blur-sm ${
                msg.role === 'user'
                  ? 'border border-theme-accent/50 bg-theme-accent/10'
                  : msg.isError
                    ? 'border border-risk-high/30 bg-risk-high/10 text-risk-high'
                    : 'border border-theme bg-theme-primary/80'
              }`}>
                <div className="whitespace-pre-wrap break-words">{msg.content}</div>
              </div>
            </div>
          ))
        )}
        {isChatLoading && (
          <div className="flex justify-start">
            <div className="border border-theme bg-theme-primary/80 backdrop-blur-sm p-3">
              <Loader2 className="w-5 h-5 text-theme-accent animate-spin" />
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={handleChatSubmit} className="fixed bottom-16 left-0 right-0 p-3 border-t border-theme bg-theme-primary/95 backdrop-blur-md z-40">
        <div className="relative">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="Ask about interactions..."
            disabled={apiStatus !== 'online' || isChatLoading}
            className="w-full bg-theme-secondary border border-theme py-3 pl-4 pr-12 text-base font-mono placeholder:text-theme-dim text-theme-primary focus:outline-none focus:border-theme-accent/50"
          />
          <button
            type="submit"
            disabled={!chatInput.trim() || apiStatus !== 'online' || isChatLoading}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 border border-theme-accent/50 text-theme-accent bg-theme-primary disabled:opacity-30"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  );

  // MOBILE RENDER
  if (isMobile) {
    return (
      <div className="min-h-screen bg-theme-primary text-theme-primary font-mono">
        {/* Mobile Header */}
        <header className="h-14 border-b border-theme bg-theme-primary/95 sticky top-0 z-50 backdrop-blur-md">
          <div className="h-full px-3 flex items-center justify-between">
            <button onClick={() => navigate('/')} className="p-2 border border-theme">
              <ChevronLeft className="w-4 h-4 text-theme-muted" />
            </button>
            
            <div className="flex items-center gap-2">
              <GitBranch className="w-4 h-4 text-theme-muted" />
              <span className="text-xs uppercase tracking-widest">DDI Analysis</span>
            </div>

            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${apiStatus === 'online' ? 'bg-risk-low' : 'bg-risk-high'}`} />
              <button 
                onClick={() => setViewMode(viewMode === 'stats' ? 'analysis' : 'stats')} 
                className={`p-2 border ${viewMode === 'stats' ? 'border-theme-accent text-theme-accent' : 'border-theme text-theme-muted'}`}
              >
                <BarChart3 className="w-4 h-4" />
              </button>
              <button onClick={toggleTheme} className="p-2 border border-theme">
                {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </button>
            </div>
          </div>
        </header>

        {/* Mobile Content */}
        <div className="pb-20">
          {/* Drugs View */}
          {mobileView === 'drugs' && (
            <div className="p-4 space-y-4">
              {/* Search */}
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-theme-muted" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search drugs..."
                    className="w-full bg-theme-secondary border border-theme py-3 pl-12 pr-4 text-base font-mono placeholder:text-theme-dim focus:outline-none focus:border-theme-accent/50"
                    disabled={apiStatus !== 'online'}
                  />
                  {isSearching && <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-theme-accent animate-spin" />}
                </div>
                {/* Camera Scanner Button */}
                <button
                  onClick={() => setShowScanner(true)}
                  disabled={apiStatus !== 'online'}
                  className="px-4 py-3 border border-theme-accent bg-theme-accent/10 hover:bg-theme-accent/20 transition-colors disabled:opacity-50"
                  title="Scan drug with camera"
                >
                  <Camera className="w-5 h-5 text-theme-accent" />
                </button>
                <button
                  onClick={clearAllDrugs}
                  disabled={selectedDrugs.length === 0}
                  className="px-3 py-3 border border-risk-high/40 bg-risk-high/10 hover:bg-risk-high/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  title="Clear all selected drugs"
                >
                  <Trash2 className="w-5 h-5 text-risk-high" />
                </button>
              </div>

              {/* Search Results */}
              <AnimatePresence>
                {searchResults.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="border border-theme bg-theme-primary/90 backdrop-blur-md divide-y divide-theme max-h-64 overflow-y-auto"
                  >
                    {searchResults.map((drug, i) => (
                      <button
                        key={drug.drugbank_id || i}
                        onClick={() => addDrug(drug)}
                        className="w-full flex items-center justify-between p-4 hover:bg-theme-secondary transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 border border-theme bg-theme-primary/80 backdrop-blur-sm flex items-center justify-center">
                            <Pill className="w-5 h-5 text-theme-accent" />
                          </div>
                          <div className="text-left">
                            <span className="text-sm text-theme-primary block">{drug.name}</span>
                            <span className="text-[10px] text-theme-muted uppercase">{drug.drugbank_id || drug.category}</span>
                          </div>
                        </div>
                        <Plus className="w-5 h-5 text-theme-accent" />
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Selected Drugs Header */}
              <div className="flex items-center justify-between pt-4">
                <h2 className="text-[10px] text-theme-muted uppercase tracking-widest">Selected Drugs ({selectedDrugs.length})</h2>
                {selectedDrugs.length >= 2 && (
                  <button
                    onClick={runAnalysis}
                    disabled={isAnalyzing || apiStatus !== 'online'}
                    className={`flex items-center gap-2 px-4 py-2 border text-xs uppercase tracking-widest backdrop-blur-sm ${
                      isAnalyzing ? 'border-theme-accent/50 text-theme-accent bg-theme-accent/10 animate-pulse' : 'border-theme-accent text-theme-accent bg-theme-accent/5'
                    }`}
                  >
                    {isAnalyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                    {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                  </button>
                )}
              </div>

              <DemoGroupPanel compact={true} />

              {/* Selected Drugs List */}
              {selectedDrugs.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-center border border-dashed border-theme bg-theme-primary/60 backdrop-blur-sm">
                  <Beaker className="w-10 h-10 text-theme-dim mb-4" />
                  <p className="text-sm text-theme-muted uppercase tracking-wider">No drugs selected</p>
                  <p className="text-xs text-theme-dim mt-1">Search and add drugs above</p>
                </div>
              ) : (
                <div className="space-y-3">
                  <AnimatePresence mode="popLayout">
                    {selectedDrugs.map((drug, i) => (
                      <MobileDrugCard key={drug.drugbank_id || drug.name} drug={drug} index={i} onRemove={removeDrug} />
                    ))}
                  </AnimatePresence>
                </div>
              )}

              {selectedDrugs.length === 1 && (
                <div className="p-4 border border-theme-accent/30 bg-theme-accent/10 backdrop-blur-sm text-center">
                  <p className="text-xs text-theme-accent">Add 1 more drug to analyze interactions</p>
                </div>
              )}
            </div>
          )}

          {/* Visualization View */}
          {mobileView === 'viz' && (
            <div className="h-[calc(100vh-8rem)] overflow-hidden">
              {/* Mobile Viz Tabs */}
              <div className="flex border-b border-theme bg-theme-primary/90 backdrop-blur-md overflow-x-auto scrollbar-hide">
                {[
                  { id: 'molecules2d', label: '2D', icon: Hexagon },
                  { id: 'molecules', label: 'Galaxy', icon: Box },
                  { id: 'graph', label: 'Graph', icon: Network },
                  { id: 'body', label: 'Body', icon: Heart },
                  { id: 'polyTwin', label: 'Twin', icon: Layers },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-3 text-[10px] uppercase tracking-widest whitespace-nowrap ${
                      activeTab === tab.id ? 'text-theme-accent border-b-2 border-theme-accent bg-theme-accent/10' : 'text-theme-muted'
                    }`}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Viz Content */}
              <div className="h-[calc(100%-3rem)] overflow-hidden">
                {selectedDrugs.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center p-6 bg-theme-primary/80 backdrop-blur-sm m-4 border border-theme">
                    <Microscope className="w-12 h-12 text-theme-dim mb-4" />
                    <p className="text-sm text-theme-muted uppercase tracking-wider">No Molecules</p>
                    <p className="text-xs text-theme-dim mt-1">Add drugs to visualize structures</p>
                  </div>
                ) : (
                  <div className="h-full w-full overflow-hidden">
                    {activeTab === 'molecules2d' && <MoleculeViewer2D drugs={selectedDrugs} result={result} isMobile={true} />}
                    {activeTab === 'molecules' && (
                      <div className="h-full relative">
                        <GNNGalaxyViewer
                          drugs={selectedDrugs}
                          result={result}
                          polypharmacyResult={polypharmacyResult}
                          isMobile={true}
                        />
                      </div>
                    )}
                    {activeTab === 'graph' && (
                      <div className="h-full relative">
                        <KnowledgeGraphView drugs={selectedDrugs} result={result} polypharmacyResult={polypharmacyResult} isMobile={true} />
                      </div>
                    )}
                    {activeTab === 'body' && (
                      <div className="h-full relative">
                        <BodyMap
                          affectedSystems={getBodyMapData()}
                          drugs={selectedDrugs.map(d => d.name)}
                          drugInfoCache={drugInfoCache}
                          interactionEvidence={interactionEvidence}
                          polypharmacyResult={polypharmacyResult}
                          result={result}
                          isMobile={true}
                        />
                      </div>
                    )}
                    {activeTab === 'polyTwin' && (
                      <div className="h-full relative">
                        <PolypharmacyDigitalTwin
                          drugs={selectedDrugs}
                          twinResult={digitalTwinResult}
                          isMobile={true}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Results View */}
          {mobileView === 'results' && <MobileResultsView />}

          {/* Compare View */}
          {mobileView === 'compare' && (
            <div className="p-4 pb-24">
              {selectedDrugs.length >= 2 ? (
                <DrugComparison drugs={selectedDrugs} drugInfoCache={drugInfoCache} isMobile={true} />
              ) : (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <GitCompare className="w-12 h-12 text-theme-dim mb-4" />
                  <p className="text-sm text-theme-muted uppercase tracking-wider">Compare Drugs</p>
                  <p className="text-xs text-theme-dim mt-1">Select 2+ drugs to compare properties</p>
                </div>
              )}
            </div>
          )}

          {/* Stats View (accessible from header button) */}
          {viewMode === 'stats' && (
            <div className="fixed inset-0 z-40 bg-theme-primary/95 backdrop-blur-md pt-14 pb-20 overflow-y-auto">
              <div className="p-4">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm uppercase tracking-widest">System Statistics</h2>
                  <button onClick={() => setViewMode('analysis')} className="p-2 border border-theme">
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <StatsDashboard />
              </div>
            </div>
          )}

          {/* Chat View */}
          {mobileView === 'chat' && mobileChatContent}
        </div>

        {/* Mobile Bottom Navigation */}
        <MobileBottomNav />
      </div>
    );
  }

  // DESKTOP RENDER
  return (
    <div className="min-h-screen bg-theme-primary text-theme-primary font-mono relative transition-colors duration-300">
      {/* Top Navigation */}
      <header className="h-14 border-b border-theme bg-theme-primary/95 sticky top-0 z-50 backdrop-blur-sm">
        <div className="h-full px-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 border border-theme hover:border-theme-highlight transition-colors"
            >
              <ChevronLeft className="w-4 h-4 text-theme-muted" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 border border-theme flex items-center justify-center">
                <GitBranch className="w-4 h-4 text-theme-muted" />
              </div>
              <div>
                <h1 className="text-sm font-normal tracking-widest uppercase text-theme-primary">Drug Interaction Analysis</h1>
                <p className="text-[10px] text-theme-muted uppercase tracking-widest">Project Aegis v2.0</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Compact Stats in Header */}
            {apiStatus === 'online' && (
              <StatsDashboard compact onExpand={() => setViewMode('stats')} />
            )}

            {/* API Status */}
            <div className={`flex items-center gap-2 px-3 py-1.5 text-[10px] font-normal uppercase tracking-widest border ${apiStatus === 'online'
              ? 'border-risk-low text-risk-low'
              : apiStatus === 'checking'
                ? 'border-risk-medium text-risk-medium'
                : 'border-risk-high text-risk-high'
              }`}>
              <span className={`w-1.5 h-1.5 ${apiStatus === 'online' ? 'bg-risk-low' :
                apiStatus === 'checking' ? 'bg-risk-medium animate-pulse' : 'bg-risk-high'
                }`} />
              {apiStatus === 'online' ? 'Connected' : apiStatus === 'checking' ? 'Connecting' : 'Offline'}
            </div>

            {/* View Mode Selector */}
            <div className="flex items-center gap-1 border border-theme p-1">
              {[
                { id: 'analysis', label: 'Analysis', icon: Zap },
                { id: 'compare', label: 'Compare', icon: GitCompare },
                { id: 'stats', label: 'Stats', icon: BarChart3 },
                { id: 'research', label: 'Research', icon: FlaskConical },
              ].map(mode => (
                <button
                  key={mode.id}
                  onClick={() => setViewMode(mode.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-[10px] uppercase tracking-widest transition-all ${
                    viewMode === mode.id
                      ? 'bg-theme-accent/10 text-theme-accent border border-theme-accent/30'
                      : 'text-theme-muted hover:text-theme-secondary border border-transparent'
                  }`}
                >
                  <mode.icon className="w-3 h-3" />
                  {mode.label}
                </button>
              ))}
            </div>

            <button 
              onClick={toggleTheme}
              className="p-2 border border-theme hover:border-theme-highlight transition-colors text-theme-muted hover:text-theme-secondary"
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
            <button className="p-2 border border-theme hover:border-theme-highlight transition-colors text-theme-muted hover:text-theme-secondary">
              <Bell className="w-4 h-4" />
            </button>
            <button 
              onClick={() => navigate('/settings')}
              className="p-2 border border-theme hover:border-theme-highlight transition-colors text-theme-muted hover:text-theme-secondary">
              <Settings className="w-4 h-4" />
            </button>
            <div className="w-8 h-8 border border-theme flex items-center justify-center">
              <User className="w-4 h-4 text-theme-muted" />
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-3.5rem)]">
        {/* Left Panel - Drug Selection */}
        <aside className="w-80 border-r border-theme flex flex-col bg-theme-panel">
          <div className="p-4 border-b border-theme">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[10px] font-normal text-theme-muted uppercase tracking-widest">// Drug Regimen</h2>
              <span className="text-[10px] text-theme-muted">{selectedDrugs.length} selected</span>
            </div>

            {/* Search Input */}
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-theme-muted" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onFocus={() => setShowSearch(true)}
                  placeholder="Search drugs..."
                  className="w-full bg-theme-secondary border border-theme py-2.5 pl-10 pr-4 text-sm font-mono placeholder:text-theme-dim text-theme-primary focus:outline-none focus:border-theme-accent/50 transition-all"
                  disabled={apiStatus !== 'online'}
                />
                {isSearching && (
                  <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-theme-accent animate-spin" />
                )}
              </div>
              {/* Camera Scanner Button */}
              <button
                onClick={() => setShowScanner(true)}
                disabled={apiStatus !== 'online'}
                className="px-3 py-2.5 border border-theme-accent bg-theme-accent/10 hover:bg-theme-accent/20 transition-colors disabled:opacity-50 group"
                title="Scan drug with camera"
              >
                <Camera className="w-4 h-4 text-theme-accent group-hover:scale-110 transition-transform" />
              </button>
              <button
                onClick={clearAllDrugs}
                disabled={selectedDrugs.length === 0}
                className="px-3 py-2.5 border border-risk-high/40 bg-risk-high/10 hover:bg-risk-high/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                title="Clear all selected drugs"
              >
                <Trash2 className="w-4 h-4 text-risk-high" />
              </button>
            </div>

            {/* Search Results Dropdown */}
            <AnimatePresence>
              {showSearch && searchResults.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute left-4 right-4 mt-2 bg-theme-primary/95 backdrop-blur-md border border-theme shadow-2xl overflow-hidden z-50 max-h-64 overflow-y-auto"
                >
                  {searchResults.map((drug, i) => {
                    const hasSmiles = drug.has_smiles || (drug.smiles && drug.smiles.length > 5);
                    return (
                    <button
                      key={drug.drugbank_id || i}
                      onClick={() => addDrug(drug)}
                      className={`w-full flex items-center justify-between p-3 transition-colors border-b border-theme last:border-0 ${
                        !hasSmiles ? 'opacity-70' : ''
                      } hover:bg-theme-secondary cursor-pointer`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 border flex items-center justify-center ${
                          hasSmiles 
                            ? 'border-risk-low/50 bg-risk-low/10' 
                            : 'border-risk-high/50 bg-risk-high/10'
                        }`}>
                          <Pill className={`w-4 h-4 ${
                            hasSmiles ? 'text-risk-low' : 'text-risk-high'
                          }`} />
                        </div>
                        <div className="text-left">
                          <div className="flex items-center gap-2">
                            <span className={`text-sm font-normal ${hasSmiles ? 'text-theme-primary' : 'text-theme-muted'}`}>
                              {drug.name}
                            </span>
                            {!hasSmiles && (
                              <span className="text-[8px] px-1.5 py-0.5 border border-risk-medium/30 text-risk-medium uppercase tracking-wider">
                                No Structure
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-theme-muted uppercase tracking-wider">
                            {drug.drugbank_id || drug.category || 'Unknown'}
                          </div>
                        </div>
                      </div>
                      <Plus className={`w-4 h-4 ${hasSmiles ? 'text-theme-accent' : 'text-theme-muted'}`} />
                    </button>
                  )})}
                </motion.div>
              )}
            </AnimatePresence>

            {/* No results message */}
            {showSearch && searchQuery.length >= 2 && !isSearching && searchResults.length === 0 && apiStatus === 'online' && (
              <div className="mt-2 p-3 text-center text-[10px] text-theme-muted border border-theme uppercase tracking-wider">
                No drugs found for "{searchQuery}"
              </div>
            )}

            {apiStatus !== 'online' && (
              <div className="mt-2 p-3 text-center text-[10px] text-risk-medium border border-risk-medium/30 uppercase tracking-wider">
                API offline - search unavailable
              </div>
            )}

            <div className="mt-3">
              <DemoGroupPanel compact={true} />
            </div>

          </div>

          {/* Selected Drugs */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {selectedDrugs.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-6 bg-theme-panel/60 backdrop-blur-sm">
                <div className="w-14 h-14 border border-theme bg-theme-panel/80 backdrop-blur-sm flex items-center justify-center mb-4">
                  <Beaker className="w-6 h-6 text-theme-dim" />
                </div>
                <p className="text-xs text-theme-muted mb-2 uppercase tracking-wider">No drugs selected</p>
                <p className="text-[10px] text-theme-dim">Search and add drugs to begin analysis</p>
              </div>
            ) : (
              selectedDrugs.map((drug, i) => {
                const drugInfo = drugInfoCache[drug.name.toLowerCase()];
                const sideEffects = drugInfo?.side_effects?.slice(0, 5) || [];
                const hasSmiles = drug.has_smiles || (drug.smiles && drug.smiles.length > 5);
                
                return (
                <motion.div
                  key={drug.drugbank_id || drug.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className={`group p-3 border transition-all hover:border-theme-highlight relative bg-theme-panel/80 backdrop-blur-sm ${
                    !hasSmiles ? 'border-risk-medium/30' : 'border-theme'
                  }`}
                >
                  <div className={`absolute -top-px -left-px w-2 h-2 border-t border-l ${!hasSmiles ? 'border-risk-medium' : 'border-theme'}`}></div>
                  <div className={`absolute -bottom-px -right-px w-2 h-2 border-b border-r ${!hasSmiles ? 'border-risk-medium' : 'border-theme'}`}></div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 border flex items-center justify-center text-[10px] font-normal uppercase tracking-wider ${
                        !hasSmiles ? 'border-risk-high/50 text-risk-high bg-risk-high/10' :
                        i === 0 ? 'border-theme-accent/50 text-theme-accent' :
                        i === 1 ? 'border-theme-accent/50 text-theme-accent' :
                          'border-theme text-theme-muted'
                        }`}>
                        {hasSmiles ? drug.name.substring(0, 2).toUpperCase() : '⚠'}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-normal text-theme-primary">{drug.name}</span>
                          {!hasSmiles && (
                            <span className="text-[7px] px-1 py-0.5 border border-risk-high/30 text-risk-high uppercase">
                              No 3D
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-theme-muted uppercase tracking-widest">{drug.category || 'Drug'}</div>
                      </div>
                    </div>
                    <button
                      onClick={() => removeDrug(drug.drugbank_id || drug.name)}
                      className="p-2 border border-transparent opacity-0 group-hover:opacity-100 hover:border-risk-high/30 text-theme-muted hover:text-risk-high transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                  
                  {/* No Structure Warning */}
                  {!hasSmiles && (
                    <div className="mt-2 p-2 border border-risk-medium/20 bg-risk-medium/5">
                      <p className="text-[9px] text-risk-medium">
                        ⚠ No molecular structure available. 3D visualization limited.
                      </p>
                    </div>
                  )}
                  
                  {/* Side Effects Preview */}
                  {sideEffects.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-theme">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className="w-3 h-3 text-risk-medium" />
                        <span className="text-[9px] text-theme-muted uppercase tracking-wider">Known Side Effects</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {sideEffects.map((effect, j) => (
                          <span
                            key={j}
                            className="px-1.5 py-0.5 text-[8px] border border-risk-medium/20 text-risk-medium/70 uppercase tracking-wide"
                          >
                            {effect}
                          </span>
                        ))}
                        {drugInfo?.side_effects?.length > 5 && (
                          <span className="px-1.5 py-0.5 text-[8px] text-theme-muted">
                            +{drugInfo.side_effects.length - 5} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Data Source Indicator */}
                  {drugInfo?.sources && (
                    <div className="mt-2 flex gap-1">
                      {drugInfo.sources.map((src, j) => (
                        <span key={j} className="text-[7px] text-theme-dim uppercase tracking-wider">
                          {src}
                        </span>
                      ))}
                    </div>
                  )}
                </motion.div>
              )})
            )}
          </div>

          {/* Run Analysis Button */}
          <div className="p-4 border-t border-theme">
            <button
              onClick={runAnalysis}
              disabled={selectedDrugs.length < 2 || isAnalyzing || apiStatus !== 'online'}
              className={`w-full py-3 text-xs uppercase tracking-widest font-normal flex items-center justify-center gap-2 transition-all border ${selectedDrugs.length < 2 || apiStatus !== 'online'
                ? 'border-theme text-theme-dim cursor-not-allowed'
                : isAnalyzing
                  ? 'border-theme-accent/50 text-theme-accent animate-pulse cursor-wait'
                  : 'border-theme-accent text-theme-accent hover:bg-theme-accent/10'
                }`}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Run Analysis
                </>
              )}
            </button>
            {selectedDrugs.length < 2 && selectedDrugs.length > 0 && (
              <p className="text-[10px] text-center text-theme-muted mt-2 uppercase tracking-wider">
                Add {2 - selectedDrugs.length} more drug{2 - selectedDrugs.length > 1 ? 's' : ''} to analyze
              </p>
            )}
          </div>
        </aside>

        {/* Main Content - Conditional based on viewMode */}
        {viewMode === 'stats' ? (
          <main className="flex-1 overflow-y-auto p-6 bg-theme-secondary">
            <StatsDashboard />
          </main>
        ) : viewMode === 'compare' ? (
          <main className="flex-1 overflow-y-auto p-6 bg-theme-secondary">
            <DrugComparison 
              initialDrugs={selectedDrugs} 
              onClose={() => setViewMode('analysis')}
            />
          </main>
        ) : viewMode === 'research' ? (
          <ResearchWorkspace addLog={addLog} selectedDrugs={selectedDrugs} apiStatus={apiStatus} />
        ) : (
          <>
        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Visualization Tabs */}
          <div className="p-4 border-b border-theme flex items-center gap-1 bg-theme-panel">
            {[
              { id: 'molecules2d', label: '2D Structure', icon: Hexagon },
              { id: 'molecules', label: 'GNN Galaxy', icon: Box },
              { id: 'graph', label: 'Knowledge Graph', icon: Network },
              { id: 'body', label: 'Body Map', icon: Heart },
              { id: 'polyTwin', label: 'Poly Twin', icon: Layers },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-[10px] font-normal uppercase tracking-widest transition-all border ${activeTab === tab.id
                  ? 'border-theme-accent/50 text-theme-accent bg-theme-accent/5'
                  : 'border-transparent text-theme-muted hover:text-theme-secondary hover:border-theme'
                  }`}
              >
                <tab.icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Visualization Area */}
          <div className="flex-1 relative overflow-hidden bg-theme-primary">
            {/* Galaxy Viewer, Knowledge Graph, and Body Map always render (they have their own empty states) */}
            {activeTab === 'molecules' ? (
              <div className="h-full relative">
                <GNNGalaxyViewer
                  drugs={selectedDrugs}
                  result={result}
                  polypharmacyResult={polypharmacyResult}
                />
              </div>
            ) : activeTab === 'graph' ? (
              <div className="h-full relative">
                <KnowledgeGraphView
                  drugs={selectedDrugs}
                  result={result}
                  polypharmacyResult={polypharmacyResult}
                />
              </div>
            ) : activeTab === 'body' ? (
              <div className="h-full relative">
                <BodyMap
                  affectedSystems={getBodyMapData()}
                  drugs={selectedDrugs.map(d => d.name)}
                  drugInfoCache={drugInfoCache}
                  interactionEvidence={interactionEvidence}
                  polypharmacyResult={polypharmacyResult}
                  result={result}
                />
              </div>
            ) : activeTab === 'polyTwin' ? (
              <div className="h-full relative">
                <PolypharmacyDigitalTwin
                  drugs={selectedDrugs}
                  twinResult={digitalTwinResult}
                />
              </div>
            ) : selectedDrugs.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center animate-fade-in">
                <div className="w-16 h-16 border border-theme flex items-center justify-center mb-6 relative">
                  <div className="absolute -top-px -left-px w-3 h-3 border-t border-l border-theme"></div>
                  <div className="absolute -bottom-px -right-px w-3 h-3 border-b border-r border-theme"></div>
                  <Microscope className="w-8 h-8 text-theme-muted" />
                </div>
                <h2 className="text-sm font-normal text-theme-primary mb-2 uppercase tracking-widest">Ready for Analysis</h2>
                <p className="text-theme-muted max-w-sm text-center text-xs leading-relaxed">
                  Select drugs from the sidebar to visualize their structures and analyze potential interactions using AI.
                </p>
              </div>
            ) : (
              <>
                {activeTab === 'molecules2d' && (
                  <MoleculeViewer2D
                    drugs={selectedDrugs}
                    result={result}
                  />
                )}
              </>
            )}


          </div>
        </main>

        {/* Right Panel - Results & Chat */}
        <aside className="w-96 border-l border-theme flex flex-col bg-theme-panel">
          {/* Results Section */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-4 border-b border-theme flex items-center justify-between">
              <h2 className="text-[10px] font-normal text-theme-muted flex items-center gap-2 uppercase tracking-widest">
                <Sparkles className="w-3.5 h-3.5 text-theme-accent" />
                // Analysis Results
              </h2>
              {/* Show Alternatives toggle when there's a result */}
              {result && result.severity && ['severe', 'high', 'critical', 'major'].includes(String(result.severity).toLowerCase()) && (
                <button
                  onClick={() => setShowAlternatives(!showAlternatives)}
                  className={`flex items-center gap-1 px-2 py-1 text-[9px] uppercase tracking-wider transition-all border ${
                    showAlternatives 
                      ? 'border-risk-low/50 text-risk-low bg-risk-low/10' 
                      : 'border-theme text-theme-muted hover:text-risk-low'
                  }`}
                >
                  <Lightbulb className="w-3 h-3" />
                  Alternatives
                </button>
              )}
            </div>

            <div className="p-4">
              {error && (
                <div className="mb-4 p-4 border border-risk-high/30 relative">
                  <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-risk-high"></div>
                  <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-risk-high"></div>
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-4 h-4 text-risk-high flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs text-risk-high font-normal uppercase tracking-wider">Analysis Error</p>
                      <p className="text-[10px] text-risk-high/70 mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Therapeutic Alternatives Panel */}
              {showAlternatives && result && selectedDrugs.length >= 2 && (
                <div className="mb-4">
                  <TherapeuticAlternatives
                    drugName={selectedDrugs[0]?.name}
                    interactingWith={selectedDrugs[1]?.name}
                    severity={result.severity}
                    onSelectAlternative={(alt) => {
                      // Replace first drug with alternative
                      setSelectedDrugs([
                        { name: alt.name, drugbank_id: alt.drugbank_id, smiles: alt.smiles },
                        ...selectedDrugs.slice(1)
                      ]);
                      setResult(null);
                      setShowAlternatives(false);
                    }}
                  />
                </div>
              )}

              {result ? (
                <div className="space-y-4">
                  {/* Risk Card */}
                  <div className={`p-4 border ${getRiskBgColor(result.risk_level)} relative`}>
                    <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-current opacity-50"></div>
                    <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-current opacity-50"></div>
                    <div className="flex items-start gap-3">
                      {result.severity === 'no_interaction' ? (
                        <Shield className="w-5 h-5 text-risk-low" />
                      ) : (
                        <AlertTriangle className="w-5 h-5" />
                      )}
                      <div>
                        <p className="text-sm font-normal uppercase tracking-wider">
                          {result.severity === 'no_interaction'
                            ? 'No Significant Interaction'
                            : `${result.risk_level || result.severity} Risk`}
                        </p>
                        <p className="text-[10px] opacity-70 mt-1 uppercase tracking-wider">
                          {result.drug_a || selectedDrugs[0]?.name} + {result.drug_b || selectedDrugs[1]?.name}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Risk Score */}
                  {result.risk_score !== undefined && (
                    <div className="mb-6">
                      <RiskGauge score={result.risk_score} riskLevel={result.risk_level || result.severity} />
                    </div>
                  )}

                  <PredictionTransparencyPanel result={result} />

                  {/* Mechanism */}
                  {result.mechanism_hypothesis && (
                    <div className="p-4 border border-theme relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-theme"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-theme"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <Brain className="w-3.5 h-3.5 text-theme-accent" />
                        <span className="text-[10px] text-theme-muted uppercase tracking-widest">Mechanism</span>
                        {/* Data Source Badge */}
                        <span className={`ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border ${
                          result.source === 'knowledge_graph'
                            ? 'border-theme-accent/50 text-theme-accent bg-theme-accent/10'
                            : result.source === 'pubmedbert'
                            ? 'border-risk-medium/50 text-risk-medium bg-risk-medium/10'
                            : 'border-theme text-theme-muted'
                        }`}>
                          {result.source === 'knowledge_graph' ? '⚡ Knowledge Graph' :
                           result.source === 'pubmedbert' ? '🧠 PubMedBERT AI' : 
                           result.source || 'AI Model'}
                        </span>
                      </div>
                      <p className="text-xs text-theme-secondary leading-relaxed">
                        {result.mechanism_hypothesis}
                      </p>
                    </div>
                  )}

                  {/* Affected Systems */}
                  {result.affected_systems && result.affected_systems.length > 0 && (
                    <div className="p-4 border border-theme relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-theme"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-theme"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <Target className="w-3.5 h-3.5 text-risk-high" />
                        <span className="text-[10px] text-theme-muted uppercase tracking-widest">Affected Systems</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {result.affected_systems.map((sys, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 border border-risk-high/30 text-[10px] text-risk-high uppercase tracking-wider"
                          >
                            {sys.system || sys}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Confidence */}
                  {result.confidence && (
                    <div className="p-4 border border-theme relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-theme"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-theme"></div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <TrendingUp className="w-3.5 h-3.5 text-theme-accent" />
                          <span className="text-[10px] text-theme-muted uppercase tracking-widest">Model Confidence</span>
                        </div>
                        <span className="text-sm font-normal text-theme-accent">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Context Sentence - Shows what the model analyzed */}
                  {result.context_sentence && (
                    <div className="p-4 border border-theme relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-theme"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-theme"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <FileText className="w-3.5 h-3.5 text-risk-medium" />
                        <span className="text-[10px] text-theme-muted uppercase tracking-widest">Analysis Context</span>
                        <span className={`ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border ${
                          result.context_source?.includes('ddi_corpus')
                            ? 'border-risk-low/50 text-risk-low bg-risk-low/10'
                            : result.context_source === 'template' 
                            ? 'border-risk-medium/30 text-risk-medium/80' 
                            : result.context_source === 'rag'
                            ? 'border-risk-low/30 text-risk-low/80'
                            : result.context_source === 'user_provided'
                            ? 'border-theme-accent/30 text-theme-accent/80'
                            : 'border-theme text-theme-muted'
                        }`}>
                          {result.context_source?.includes('ddi_corpus') ? '✓ Clinical Literature' :
                           result.context_source === 'template' ? 'Template' : 
                           result.context_source === 'rag' ? 'PubMed' : 
                           result.context_source === 'user_provided' ? 'Custom' : 
                           result.context_source || 'Unknown'}
                        </span>
                      </div>
                      <p className="text-[11px] text-theme-secondary leading-relaxed italic">
                        "{result.context_sentence}"
                      </p>
                      {result.template_category && (
                        <p className="text-[9px] text-theme-muted mt-2 uppercase tracking-wider">
                          Category: {result.template_category}
                        </p>
                      )}
                    </div>
                  )}

                  <EvidenceChainTimeline interactionEvidence={interactionEvidence} />
                  <EvidenceUncertaintyPanel interactionEvidence={interactionEvidence} />

                  {/* Real-World Evidence from FDA FAERS */}
                  {interactionEvidence?.faers_data && (
                    <div className="p-4 border border-theme-accent/30 relative bg-theme-accent/5">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-theme-accent"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-theme-accent"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <Activity className="w-3.5 h-3.5 text-theme-accent" />
                        <span className="text-[10px] text-theme-muted uppercase tracking-widest">FDA Real-World Evidence</span>
                        <span className="ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border border-theme-accent/50 text-theme-accent">
                          OpenFDA FAERS
                        </span>
                      </div>
                      
                      {/* Total Reports */}
                      <div className="flex items-center justify-between mb-3 pb-3 border-b border-theme">
                        <span className="text-[10px] text-theme-muted uppercase tracking-wider">Total Adverse Event Reports</span>
                        <span className="text-lg font-mono text-theme-accent">
                          {interactionEvidence.faers_data.total_reports?.toLocaleString() || '0'}
                        </span>
                      </div>
                      
                      {/* Top Reactions Bar Chart */}
                      {interactionEvidence.faers_data.top_reactions?.length > 0 && (
                        <div>
                          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Top Reported Reactions</span>
                          <div className="mt-2 space-y-1.5">
                            {interactionEvidence.faers_data.top_reactions.slice(0, 5).map((reaction, i) => {
                              const maxCount = interactionEvidence.faers_data.top_reactions[0]?.count || 1;
                              const width = Math.max((reaction.count / maxCount) * 100, 5);
                              return (
                                <div key={i} className="flex items-center gap-2">
                                  <div className="flex-1">
                                    <div className="flex items-center justify-between mb-0.5">
                                      <span className="text-[9px] text-theme-secondary uppercase tracking-wider truncate max-w-[140px]">
                                        {reaction.reaction}
                                      </span>
                                      <span className="text-[9px] text-theme-accent font-mono">
                                        {reaction.count?.toLocaleString()}
                                      </span>
                                    </div>
                                    <div className="h-1.5 bg-theme-tertiary overflow-hidden">
                                      <div 
                                        className="h-full bg-gradient-to-r from-theme-accent to-theme-accent transition-all duration-500"
                                        style={{ width: `${width}%` }}
                                      />
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      
                      {/* Serious Outcomes */}
                      {interactionEvidence.faers_data.serious_outcomes && 
                       Object.keys(interactionEvidence.faers_data.serious_outcomes).length > 0 && (
                        <div className="mt-4 pt-3 border-t border-theme">
                          <span className="text-[9px] text-theme-muted uppercase tracking-wider">Serious Outcomes</span>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {Object.entries(interactionEvidence.faers_data.serious_outcomes)
                              .filter(([_, count]) => count > 0)
                              .slice(0, 4)
                              .map(([outcome, count], i) => (
                                <div key={i} className="px-2 py-1 border border-risk-high/30 bg-risk-high/10">
                                  <span className="text-[8px] text-risk-high uppercase tracking-wider block">
                                    {outcome.replace(/_/g, ' ')}
                                  </span>
                                  <span className="text-[10px] text-risk-high font-mono">
                                    {count.toLocaleString()}
                                  </span>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Side Effects Comparison */}
                  {selectedDrugs.length === 2 && (
                    drugInfoCache[selectedDrugs[0].name.toLowerCase()]?.side_effects?.length > 0 ||
                    drugInfoCache[selectedDrugs[1].name.toLowerCase()]?.side_effects?.length > 0
                  ) && (
                    <div className="p-4 border border-risk-medium/30 relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-risk-medium"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-risk-medium"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <AlertTriangle className="w-3.5 h-3.5 text-risk-medium" />
                        <span className="text-[10px] text-theme-muted uppercase tracking-widest">Side Effects Comparison</span>
                        <span className="ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border border-risk-medium/30 text-risk-medium">
                          SIDER
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3">
                        {selectedDrugs.slice(0, 2).map((drug, i) => {
                          const info = drugInfoCache[drug.name.toLowerCase()];
                          const effects = info?.side_effects?.slice(0, 6) || [];
                          return (
                            <div key={i}>
                              <span className={`text-[9px] uppercase tracking-wider text-theme-accent`}>
                                {drug.name}
                              </span>
                              <div className="mt-1.5 space-y-1">
                                {effects.length > 0 ? effects.map((effect, j) => (
                                  <div key={j} className="text-[8px] text-theme-muted py-0.5 px-1.5 border border-theme">
                                    {effect}
                                  </div>
                                )) : (
                                  <span className="text-[8px] text-theme-dim">No data</span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      
                      {/* Find common side effects */}
                      {(() => {
                        const effects1 = drugInfoCache[selectedDrugs[0]?.name?.toLowerCase()]?.side_effects || [];
                        const effects2 = drugInfoCache[selectedDrugs[1]?.name?.toLowerCase()]?.side_effects || [];
                        const common = effects1.filter(e => effects2.map(x => x.toLowerCase()).includes(e.toLowerCase()));
                        if (common.length === 0) return null;
                        return (
                          <div className="mt-3 pt-3 border-t border-fui-gray-500/10">
                            <span className="text-[9px] text-risk-high uppercase tracking-wider">
                              ⚠️ Shared Side Effects ({common.length})
                            </span>
                            <div className="mt-1.5 flex flex-wrap gap-1">
                              {common.slice(0, 5).map((effect, i) => (
                                <span key={i} className="text-[8px] text-risk-high px-1.5 py-0.5 border border-risk-high/30 bg-risk-high/10">
                                  {effect}
                                </span>
                              ))}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {/* Data Sources Summary */}
                  <div className="p-3 border border-theme bg-theme-secondary">
                    <span className="text-[9px] text-theme-muted uppercase tracking-wider">Data Sources</span>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {result.source && (
                        <span className="px-2 py-0.5 text-[7px] border border-theme-accent/30 text-theme-accent uppercase tracking-wider">
                          {result.source === 'knowledge_graph' ? 'Neo4j KG' : result.source === 'pubmedbert' ? 'PubMedBERT' : result.source}
                        </span>
                      )}
                      {result.context_source?.includes('ddi_corpus') && (
                        <span className="px-2 py-0.5 text-[7px] border border-risk-low/30 text-risk-low uppercase tracking-wider">
                          DDI Corpus 2013
                        </span>
                      )}
                      {interactionEvidence?.faers_data && (
                        <span className="px-2 py-0.5 text-[7px] border border-theme-accent/30 text-theme-accent uppercase tracking-wider">
                          OpenFDA FAERS
                        </span>
                      )}
                      {(drugInfoCache[selectedDrugs[0]?.name?.toLowerCase()]?.side_effects?.length > 0 ||
                        drugInfoCache[selectedDrugs[1]?.name?.toLowerCase()]?.side_effects?.length > 0) && (
                        <span className="px-2 py-0.5 text-[7px] border border-risk-medium/30 text-risk-medium uppercase tracking-wider">
                          SIDER
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-14 h-14 border border-theme flex items-center justify-center mb-4">
                    <Activity className="w-6 h-6 text-theme-dim" />
                  </div>
                  <p className="text-xs text-theme-muted mb-2 uppercase tracking-wider">No Analysis Yet</p>
                  <p className="text-[10px] text-theme-dim">Select drugs and run analysis to see results</p>
                </div>
              )}

              <div className="mt-4">
                <CalibrationQAPanel addLog={addLog} />
              </div>
            </div>
          </div>

          {/* Chat Section */}
          <div className="h-80 border-t border-theme flex flex-col">
            <div className="p-3 border-b border-theme flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h3 className="text-[10px] font-normal text-theme-muted uppercase tracking-widest">// Research Assistant</h3>
                <span className="px-2 py-0.5 border border-risk-medium/40 bg-risk-medium/10 text-risk-medium text-[8px] uppercase tracking-wider">Under Construction</span>
              </div>
              {messages.length > 0 && (
                <button
                  onClick={() => setMessages([])}
                  className="text-[10px] text-theme-muted hover:text-theme-accent transition-colors uppercase tracking-wider"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <Sparkles className="w-5 h-5 text-theme-dim mb-2" />
                  <p className="text-[10px] text-theme-muted">Ask about drug interactions, mechanisms, or alternatives</p>
                </div>
              ) : (
                messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] p-3 text-xs leading-relaxed ${msg.role === 'user'
                        ? 'border border-theme-accent/50 text-theme-primary bg-theme-accent/5'
                        : msg.isError
                          ? 'border border-risk-high/30 text-risk-high'
                          : 'border border-theme text-theme-secondary'
                        }`}
                    >
                      {msg.content}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-theme">
                          <p className="text-[10px] text-theme-muted mb-1 uppercase tracking-wider">Sources:</p>
                          {msg.sources.slice(0, 2).map((s, j) => (
                            <p key={j} className="text-[10px] text-theme-accent truncate">{s}</p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              {isChatLoading && (
                <div className="flex justify-start">
                  <div className="border border-theme p-3">
                    <Loader2 className="w-4 h-4 text-theme-accent animate-spin" />
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input */}
            <form onSubmit={handleChatSubmit} className="p-3 border-t border-theme">
              <div className="relative">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder={apiStatus === 'online' ? "Ask about this interaction..." : "Chat unavailable offline"}
                  disabled={apiStatus !== 'online' || isChatLoading}
                  className="w-full bg-theme-secondary border border-theme py-2.5 pl-4 pr-12 text-sm font-mono placeholder:text-theme-dim text-theme-primary focus:outline-none focus:border-theme-accent/50 transition-all disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!chatInput.trim() || apiStatus !== 'online' || isChatLoading}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 border border-theme-accent/50 text-theme-accent disabled:opacity-30 disabled:cursor-not-allowed hover:bg-theme-accent/10 transition-colors"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </form>
          </div>
        </aside>
        </>
        )}
      </div>

      {/* Neo4j Database Warning Modal */}
      <AnimatePresence>
        {showDbWarning && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
            onClick={() => setShowDbWarning(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="bg-theme-primary border border-risk-high/50 p-6 max-w-md w-full"
            >
              <div className="flex items-start gap-4">
                <div className="p-2 bg-risk-high/20 border border-risk-high/30">
                  <AlertTriangle className="w-6 h-6 text-risk-high" />
                </div>
                <div className="flex-1">
                  <h3 className="text-sm font-medium text-risk-high uppercase tracking-widest mb-2">
                    Database Inactive
                  </h3>
                  <p className="text-xs text-theme-muted leading-relaxed mb-4">
                    Only <span className="text-risk-high font-medium">{dbDrugCount}</span> drugs found in the database. 
                    This usually means our Neo4j Aura database has become inactive after 3 days of no activity.
                  </p>
                  <p className="text-xs text-theme-muted leading-relaxed mb-4">
                    Please <span className="text-theme-accent font-medium">refresh the page</span> or try again in a few minutes. 
                    If the issue persists, contact us at <a href="mailto:1kibriaahr@gmail.com" className="text-theme-accent hover:underline">1kibriaahr@gmail.com</a>
                  </p>
                  <div className="flex gap-3">
                    <button
                      onClick={() => window.location.reload()}
                      className="flex-1 py-2 px-4 text-xs uppercase tracking-widest bg-theme-accent/20 border border-theme-accent/50 text-theme-accent hover:bg-theme-accent/30 transition-colors"
                    >
                      Refresh Page
                    </button>
                    <button
                      onClick={() => setShowDbWarning(false)}
                      className="py-2 px-4 text-xs uppercase tracking-widest border border-theme text-theme-muted hover:border-theme-highlight transition-colors"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Drug Scanner Modal */}
      <AnimatePresence>
        {showScanner && (
          <DrugScanner
            onDrugDetected={handleScannedDrug}
            onClose={() => setShowScanner(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
