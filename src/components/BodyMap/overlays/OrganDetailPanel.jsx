// OrganDetailPanel.jsx — Click organ → slide-in panel with real side effects, drug contributions, FAERS data
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, Activity, Pill, Database, BarChart3 } from 'lucide-react';
import { ORGAN_SYSTEMS, getSeverityColor } from '../organRegistry';
import { getUniqueSideEffects } from '../bodyMapDataService';

function SeverityBadge({ severity }) {
  const info = getSeverityColor(severity);
  return (
    <span
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-mono uppercase tracking-wider"
      style={{ color: info.fill, background: `${info.fill}15`, border: `1px solid ${info.fill}30` }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: info.fill }} />
      {info.label}
    </span>
  );
}

function SideEffectRow({ se }) {
  const info = getSeverityColor(se.severity);
  return (
    <div className="flex items-center justify-between py-1 px-2 rounded bg-white/[0.02] hover:bg-white/[0.05] transition-colors">
      <div className="flex items-center gap-2 min-w-0">
        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: info.fill }} />
        <span className="text-[11px] text-slate-300 truncate">{se.name}</span>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        {se.drug && se.drug !== 'interaction' && (
          <span className="text-[9px] text-slate-500 font-mono">{se.drug}</span>
        )}
        {se.drug === 'interaction' && (
          <span className="text-[9px] text-amber-500/70 font-mono">DDI</span>
        )}
        {se.frequency && (
          <span className="text-[9px] text-slate-600 font-mono">{se.frequency}</span>
        )}
      </div>
    </div>
  );
}

export default function OrganDetailPanel({ organKey, organData, onClose, isMobile }) {
  if (!organKey || !organData) return null;

  const organDef = ORGAN_SYSTEMS[organKey];
  if (!organDef) return null;

  const severity = organData.severity || 0;
  const colorInfo = getSeverityColor(severity);
  const sideEffects = getUniqueSideEffects(organData);
  const contributions = organData.drugContributions || [];
  const details = organData.details || [];
  const faersCount = organData.faersCount || 0;

  return (
    <AnimatePresence>
      <motion.div
        key={organKey}
        initial={{ opacity: 0, x: isMobile ? 0 : 20, y: isMobile ? 20 : 0 }}
        animate={{ opacity: 1, x: 0, y: 0 }}
        exit={{ opacity: 0, x: isMobile ? 0 : 20, y: isMobile ? 20 : 0 }}
        transition={{ duration: 0.25, ease: 'easeOut' }}
        className={`
          absolute z-20 bg-[#0a0f1a]/95 backdrop-blur-xl border border-white/10 rounded-lg shadow-2xl
          ${isMobile
            ? 'bottom-0 left-0 right-0 max-h-[50vh] rounded-b-none'
            : 'right-4 top-4 bottom-4 w-72'
          }
          overflow-hidden flex flex-col
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b border-white/5">
          <div className="flex items-center gap-2 min-w-0">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ background: colorInfo.fill, boxShadow: `0 0 8px ${colorInfo.glow}` }}
            />
            <div className="min-w-0">
              <h3 className="text-xs font-bold text-white truncate">{organDef.name}</h3>
              <p className="text-[9px] text-slate-500 font-mono">{organDef.shortName}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <SeverityBadge severity={severity} />
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-white/10 transition-colors"
            >
              <X className="w-3.5 h-3.5 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-3 space-y-3 custom-scrollbar">
          {/* Severity meter */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">Severity</span>
              <span className="text-[10px] font-mono" style={{ color: colorInfo.fill }}>
                {(severity * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${severity * 100}%`,
                  background: `linear-gradient(90deg, ${colorInfo.fill}80, ${colorInfo.fill})`,
                }}
              />
            </div>
          </div>

          {/* CYP Load (liver only) */}
          {organData.cypLoad > 0 && (
            <div className="p-2 rounded bg-amber-500/5 border border-amber-500/20">
              <div className="flex items-center gap-1.5 mb-1">
                <Activity className="w-3 h-3 text-amber-500" />
                <span className="text-[9px] text-amber-500 font-mono uppercase tracking-wider">
                  CYP450 Hepatic Load
                </span>
              </div>
              <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-amber-500"
                  style={{ width: `${organData.cypLoad * 100}%` }}
                />
              </div>
              <p className="text-[9px] text-slate-500 mt-1">
                {organData.cypLoad > 0.7
                  ? 'High metabolic burden — monitor hepatic function'
                  : organData.cypLoad > 0.4
                    ? 'Moderate CYP enzyme competition detected'
                    : 'Low metabolic overlap'}
              </p>
            </div>
          )}

          {/* Drug contributions */}
          {contributions.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-1.5">
                <Pill className="w-3 h-3 text-slate-400" />
                <span className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
                  Drug Contributions
                </span>
              </div>
              <div className="space-y-1">
                {contributions.map((c, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-300 font-mono">{c.drug}</span>
                    <div className="flex items-center gap-1.5">
                      <div className="w-16 h-1 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${c.contribution * 100}%`,
                            background: i === 0 ? '#22d3ee' : '#f97316',
                          }}
                        />
                      </div>
                      <span className="text-[9px] text-slate-500 font-mono w-6 text-right">
                        {(c.contribution * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Side effects */}
          {sideEffects.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-1.5">
                <AlertTriangle className="w-3 h-3 text-slate-400" />
                <span className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">
                  Side Effects ({sideEffects.length})
                </span>
              </div>
              <div className="space-y-0.5 max-h-32 overflow-y-auto">
                {sideEffects.slice(0, 15).map((se, i) => (
                  <SideEffectRow key={i} se={se} />
                ))}
                {sideEffects.length > 15 && (
                  <p className="text-[9px] text-slate-600 text-center py-1">
                    +{sideEffects.length - 15} more
                  </p>
                )}
              </div>
            </div>
          )}

          {/* FAERS data */}
          {faersCount > 0 && (
            <div className="p-2 rounded bg-blue-500/5 border border-blue-500/20">
              <div className="flex items-center gap-1.5 mb-0.5">
                <Database className="w-3 h-3 text-blue-400" />
                <span className="text-[9px] text-blue-400 font-mono uppercase tracking-wider">
                  FDA FAERS Reports
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <BarChart3 className="w-3 h-3 text-blue-300" />
                <span className="text-[11px] text-blue-300 font-mono font-bold">
                  {faersCount.toLocaleString()}
                </span>
                <span className="text-[9px] text-slate-500">adverse event reports</span>
              </div>
            </div>
          )}

          {/* Additional details */}
          {details.map((d, i) => (
            <div key={i} className="p-2 rounded bg-white/[0.02] border border-white/5">
              <span className="text-[9px] text-slate-400 font-mono uppercase">{d.label}</span>
              <p className="text-[10px] text-slate-500 mt-0.5">{d.description}</p>
            </div>
          ))}

          {/* Empty state */}
          {sideEffects.length === 0 && contributions.length === 0 && details.length === 0 && (
            <div className="text-center py-4">
              <p className="text-[10px] text-slate-600">No detailed data available</p>
              <p className="text-[9px] text-slate-700 mt-1">Run analysis with drugs to see effects</p>
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
