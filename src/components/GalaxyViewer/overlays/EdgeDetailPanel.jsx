// EdgeDetailPanel.jsx — Interaction detail panel when an edge is clicked
import React, { useEffect, useState } from 'react';
import { X, AlertTriangle, Beaker, Activity, Shield, FileSearch } from 'lucide-react';
import { useGalaxy, useGalaxyDispatch } from '../store';
import { enrichInteraction } from '../dataEnrichment';

const SEVERITY_COLORS = {
  no_interaction: { bg: '#22c55e', text: '#22c55e', label: 'None' },
  minor: { bg: '#22c55e', text: '#22c55e', label: 'Minor' },
  moderate: { bg: '#eab308', text: '#eab308', label: 'Moderate' },
  major: { bg: '#f97316', text: '#f97316', label: 'Major' },
  critical: { bg: '#ef4444', text: '#ef4444', label: 'Critical' },
};

export default function EdgeDetailPanel() {
  const { selectedEdge, enrichedInteraction } = useGalaxy();
  const dispatch = useGalaxyDispatch();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);

  const close = () => {
    dispatch({ type: 'SET_SELECTED_EDGE', payload: null });
    setData(null);
  };

  // Fetch enrichment when edge is selected
  useEffect(() => {
    if (!selectedEdge) {
      setData(null);
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      try {
        const result = await enrichInteraction(selectedEdge.startName, selectedEdge.endName);
        setData(result);
      } catch (err) {
        console.warn('Edge enrichment failed:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedEdge?.startId, selectedEdge?.endId]);

  if (!selectedEdge) return null;

  const prediction = data?.prediction;
  const interactionInfo = data?.interactionInfo;
  const severity = prediction?.severity || selectedEdge.role;
  const severityStyle = SEVERITY_COLORS[severity] || SEVERITY_COLORS.minor;
  const riskScore = prediction?.risk_score;
  const confidence = prediction?.confidence;

  return (
    <div className="absolute top-12 right-3 pointer-events-none z-30">
      <div className="w-64 bg-black/85 backdrop-blur-md border border-white/10 rounded-lg shadow-2xl overflow-hidden pointer-events-auto animate-[slideInRight_0.3s_ease]">
        {/* Header */}
        <div className="px-3 py-2.5 border-b border-white/5">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <AlertTriangle className="w-3 h-3" style={{ color: severityStyle.text }} />
                <span
                  className="text-[8px] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded"
                  style={{ color: severityStyle.text, background: `${severityStyle.bg}15` }}
                >
                  {severityStyle.label}
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="text-[#00d2ff] font-bold">{selectedEdge.startName}</span>
                <span className="text-white/20">↔</span>
                <span className="text-[#ff8c00] font-bold">{selectedEdge.endName}</span>
              </div>
            </div>
            <button onClick={close} className="text-white/30 hover:text-white/60 transition-colors">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="px-3 py-2 space-y-2.5">
          {loading ? (
            // Skeleton loading state
            <div className="space-y-2 animate-pulse">
              <div className="h-8 bg-white/5 rounded" />
              <div className="h-4 bg-white/5 rounded w-3/4" />
              <div className="h-4 bg-white/5 rounded w-1/2" />
              <div className="h-12 bg-white/5 rounded" />
            </div>
          ) : (
            <>
              {/* Risk Score Gauge */}
              {riskScore !== undefined && (
                <div className="flex items-center gap-3">
                  <div className="relative w-10 h-10">
                    <svg viewBox="0 0 36 36" className="w-10 h-10 -rotate-90">
                      <circle cx="18" cy="18" r="15" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="3" />
                      <circle
                        cx="18" cy="18" r="15" fill="none"
                        stroke={severityStyle.text}
                        strokeWidth="3"
                        strokeDasharray={`${riskScore * 94.2} 94.2`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <span className="absolute inset-0 flex items-center justify-center text-[9px] font-mono font-bold text-white">
                      {(riskScore * 100).toFixed(0)}
                    </span>
                  </div>
                  <div>
                    <div className="text-[9px] text-white/40">Risk Score</div>
                    {confidence !== undefined && (
                      <div className="text-[8px] text-white/25 font-mono">
                        {(confidence * 100).toFixed(0)}% confidence
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Mechanism */}
              {prediction?.mechanism_hypothesis && (
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <Beaker className="w-3 h-3 text-purple-400" />
                    <span className="text-[8px] text-white/40 uppercase tracking-wider">Mechanism</span>
                  </div>
                  <p className="text-[9px] text-white/60 leading-relaxed">
                    {prediction.mechanism_hypothesis}
                  </p>
                </div>
              )}

              {/* Affected Systems */}
              {prediction?.affected_systems && prediction.affected_systems.length > 0 && (
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <Activity className="w-3 h-3 text-pink-400" />
                    <span className="text-[8px] text-white/40 uppercase tracking-wider">Affected Systems</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {prediction.affected_systems.map((sys, i) => (
                      <span
                        key={i}
                        className="text-[8px] px-1.5 py-0.5 rounded bg-white/5 text-white/50 font-mono"
                      >
                        {typeof sys === 'string' ? sys : sys.system || sys.name || 'Unknown'}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Evidence Sources */}
              <div className="border-t border-white/5 pt-2">
                <div className="flex items-center gap-1.5 mb-1">
                  <Shield className="w-3 h-3 text-cyan-400" />
                  <span className="text-[8px] text-white/40 uppercase tracking-wider">Evidence</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {interactionInfo?.known_interaction && (
                    <span className="text-[7px] px-1.5 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20 font-mono">
                      Known in Graph
                    </span>
                  )}
                  {prediction && (
                    <span className="text-[7px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-mono">
                      GNN Predicted
                    </span>
                  )}
                  {interactionInfo?.faers_data?.total_reports > 0 && (
                    <span className="text-[7px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                      FAERS: {interactionInfo.faers_data.total_reports} reports
                    </span>
                  )}
                  {!interactionInfo && !prediction && (
                    <span className="text-[7px] text-white/20 italic">
                      No enrichment data available
                    </span>
                  )}
                </div>
              </div>

              {/* Graph-only info when no API data */}
              {!prediction && !loading && (
                <div className="border-t border-white/5 pt-2">
                  <div className="flex items-center gap-1.5 mb-1">
                    <FileSearch className="w-3 h-3 text-white/30" />
                    <span className="text-[8px] text-white/40 uppercase tracking-wider">Graph Info</span>
                  </div>
                  <div className="text-[9px] text-white/40 space-y-0.5">
                    <div>Role: <span className="text-white/60 font-mono">{selectedEdge.role}</span></div>
                    <div>Type: <span className="text-white/60 font-mono">
                      {selectedEdge.role === 'path' ? 'Shortest path segment' :
                       selectedEdge.role === 'direct' ? 'Direct interaction' :
                       selectedEdge.role === 'bridge' ? 'Shared neighborhood' :
                       selectedEdge.role === 'hopA' ? 'Drug A neighborhood' :
                       selectedEdge.role === 'hopB' ? 'Drug B neighborhood' :
                       'Background connection'}
                    </span></div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
