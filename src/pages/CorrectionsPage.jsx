import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronLeft, Shield, AlertTriangle, Check, X, Trash2,
  Sparkles, Download, Loader2, Lock, ChevronDown, ChevronUp,
  RefreshCw, Brain
} from 'lucide-react';
import { useTheme } from '../hooks/useTheme';
import {
  getCorrections,
  reviewCorrection,
  deleteCorrection,
  getCorrectionStats,
  exportTrainingData,
  sendChatMessage,
} from '../services/api';

const SEVERITY_OPTIONS = [
  { value: 'none', label: 'None', color: 'text-gray-400 border-gray-600' },
  { value: 'minor', label: 'Minor', color: 'text-risk-low border-risk-low' },
  { value: 'moderate', label: 'Moderate', color: 'text-risk-medium border-risk-medium' },
  { value: 'severe', label: 'Severe', color: 'text-risk-high border-risk-high' },
  { value: 'critical', label: 'Critical', color: 'text-risk-critical border-risk-critical' },
];

const STATUS_COLORS = {
  pending: 'text-amber-400 border-amber-500/40 bg-amber-500/10',
  approved: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10',
  rejected: 'text-red-400 border-red-500/40 bg-red-500/10',
};

const CorrectionsPage = () => {
  const navigate = useNavigate();
  const { theme } = useTheme();

  // Auth
  const [isAuthed, setIsAuthed] = useState(() => sessionStorage.getItem('aegis:corrections-authed') === 'true');
  const [passwordInput, setPasswordInput] = useState('');
  const [authError, setAuthError] = useState('');

  // Data
  const [corrections, setCorrections] = useState([]);
  const [stats, setStats] = useState({ pending: 0, approved: 0, rejected: 0 });
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState('');

  // Expanded card
  const [expandedId, setExpandedId] = useState(null);
  const [editSeverity, setEditSeverity] = useState('');
  const [editEvidence, setEditEvidence] = useState('');
  const [editSource, setEditSource] = useState('');
  const [actionLoading, setActionLoading] = useState('');

  // AI Review
  const [aiReviewing, setAiReviewing] = useState({});
  const [aiResults, setAiResults] = useState({});

  const getAccessToken = () => {
    try {
      return JSON.parse(localStorage.getItem('aegis:assistant-prefs:v1') || '{}').accessToken || '';
    } catch { return ''; }
  };

  const handleLogin = (e) => {
    e.preventDefault();
    const token = getAccessToken();
    if (!token) {
      setAuthError('No access token configured. Go to Settings > Research Assistant first.');
      return;
    }
    if (passwordInput === token) {
      sessionStorage.setItem('aegis:corrections-authed', 'true');
      setIsAuthed(true);
      setAuthError('');
    } else {
      setAuthError('Invalid password');
    }
  };

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [correctionsRes, statsRes] = await Promise.all([
        getCorrections({ status: statusFilter || undefined }),
        getCorrectionStats(),
      ]);
      setCorrections(correctionsRes.corrections || []);
      setStats(statsRes || { pending: 0, approved: 0, rejected: 0 });
    } catch (err) {
      console.error('Failed to fetch corrections:', err);
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    if (isAuthed) fetchData();
  }, [isAuthed, fetchData]);

  const handleExpand = (correction) => {
    if (expandedId === correction.id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(correction.id);
    setEditSeverity(correction.corrected_severity || '');
    setEditEvidence(correction.evidence_text || '');
    setEditSource(correction.evidence_source || '');
  };

  const handleReview = async (id, newStatus) => {
    setActionLoading(id);
    try {
      await reviewCorrection(id, newStatus, getAccessToken(), {
        correctedSeverity: editSeverity,
        evidenceText: editEvidence,
        evidenceSource: editSource,
      });
      await fetchData();
      setExpandedId(null);
    } catch (err) {
      console.error('Review failed:', err);
    } finally {
      setActionLoading('');
    }
  };

  const handleDelete = async (id) => {
    setActionLoading(id);
    try {
      await deleteCorrection(id, getAccessToken());
      await fetchData();
      setExpandedId(null);
    } catch (err) {
      console.error('Delete failed:', err);
    } finally {
      setActionLoading('');
    }
  };

  const handleAiReview = async (correction) => {
    const id = correction.id;
    setAiReviewing(prev => ({ ...prev, [id]: true }));
    try {
      const prefs = JSON.parse(localStorage.getItem('aegis:assistant-prefs:v1') || '{}');
      const prompt = `Assess this drug interaction: ${correction.drug_a} and ${correction.drug_b}. ` +
        `The GNN model predicted severity "${correction.gnn_severity}" with confidence ${((correction.gnn_confidence || 0) * 100).toFixed(1)}% ` +
        `and risk score ${(correction.gnn_risk_score || 0).toFixed(2)}. ` +
        `Is this prediction accurate based on clinical literature? What should the correct severity be? Cite your sources.`;

      const response = await sendChatMessage(prompt, [correction.drug_a, correction.drug_b], null, {
        assistantMode: 'llm',
        accessToken: prefs.accessToken || '',
      });
      setAiResults(prev => ({
        ...prev,
        [id]: {
          text: response.response,
          citations: response.citations || [],
          mode: response.assistant_mode,
        },
      }));
    } catch (err) {
      setAiResults(prev => ({ ...prev, [id]: { text: 'AI review failed: ' + err.message, citations: [] } }));
    } finally {
      setAiReviewing(prev => ({ ...prev, [id]: false }));
    }
  };

  const handleAiReviewAll = async () => {
    const pending = corrections.filter(c => c.status === 'pending');
    for (const c of pending) {
      await handleAiReview(c);
    }
  };

  const handleExport = async () => {
    try {
      const data = await exportTrainingData(getAccessToken());
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `aegis_corrections_training_data_${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed:', err);
    }
  };

  const getSeverityColor = (sev) => {
    const found = SEVERITY_OPTIONS.find(s => s.value === sev);
    return found ? found.color : 'text-theme-muted border-theme';
  };

  // ---- Password Gate ----
  if (!isAuthed) {
    return (
      <div className="min-h-screen bg-theme-primary text-theme-primary font-mono relative overflow-hidden flex items-center justify-center">
        <div className="fixed inset-0 pointer-events-none opacity-20">
          <div className="absolute inset-0 bg-grid-blueprint bg-grid-40" />
        </div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="border border-theme bg-theme-panel p-8 w-full max-w-sm relative z-10"
        >
          <div className="flex items-center gap-3 mb-6">
            <Lock className="w-5 h-5 text-theme-accent" />
            <h1 className="text-sm uppercase tracking-widest text-theme-secondary">Corrections Admin</h1>
          </div>
          <p className="text-xs text-theme-muted mb-4">Enter your access token to review prediction corrections.</p>
          <form onSubmit={handleLogin} className="space-y-4">
            <input
              type="password"
              value={passwordInput}
              onChange={(e) => setPasswordInput(e.target.value)}
              placeholder="Access token"
              className="w-full bg-theme-secondary border border-theme py-2 px-3 text-sm text-theme-primary placeholder:text-theme-dim focus:outline-none focus:border-theme-accent/50"
              autoFocus
            />
            {authError && <p className="text-xs text-red-400">{authError}</p>}
            <button
              type="submit"
              className="w-full py-2 border border-theme-accent text-theme-accent text-xs uppercase tracking-widest hover:bg-theme-accent/10 transition-colors"
            >
              Authenticate
            </button>
          </form>
          <button
            onClick={() => navigate('/dashboard')}
            className="mt-4 w-full py-2 border border-theme text-theme-muted text-xs uppercase tracking-widest hover:text-theme-secondary transition-colors"
          >
            Back to Dashboard
          </button>
        </motion.div>
      </div>
    );
  }

  // ---- Main Page ----
  return (
    <div className="min-h-screen bg-theme-primary text-theme-primary font-mono relative overflow-hidden">
      <div className="fixed inset-0 pointer-events-none opacity-20">
        <div className="absolute inset-0 bg-grid-blueprint bg-grid-40" />
        <div className="absolute inset-0 bg-grid-fine bg-grid-8" />
      </div>

      {/* Header */}
      <header className="h-14 border-b border-theme bg-theme-panel sticky top-0 z-50">
        <div className="max-w-7xl mx-auto h-full flex items-center px-4 md:px-6">
          <button
            onClick={() => navigate('/dashboard')}
            className="flex items-center gap-2 text-theme-muted hover:text-theme-secondary transition-colors group"
          >
            <ChevronLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
            <span className="text-xs tracking-widest uppercase">Back to System</span>
          </button>
          <div className="ml-auto flex items-center gap-4">
            <h1 className="text-xs uppercase tracking-widest text-theme-secondary flex items-center gap-2">
              <Shield className="w-4 h-4" /> Correction Review
            </h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8 relative z-10">
        {/* Stats Bar */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          {[
            { label: 'Pending', count: stats.pending, color: 'text-amber-400 border-amber-500/40' },
            { label: 'Approved', count: stats.approved, color: 'text-emerald-400 border-emerald-500/40' },
            { label: 'Rejected', count: stats.rejected, color: 'text-red-400 border-red-500/40' },
          ].map(s => (
            <div key={s.label} className={`border ${s.color} bg-theme-panel p-4 text-center`}>
              <p className="text-2xl font-bold">{s.count}</p>
              <p className="text-[10px] uppercase tracking-widest text-theme-muted mt-1">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Actions Bar */}
        <div className="flex flex-wrap items-center gap-3 mb-6">
          {/* Filter Tabs */}
          <div className="flex items-center gap-1 border border-theme p-1">
            {['', 'pending', 'approved', 'rejected'].map(f => (
              <button
                key={f}
                onClick={() => setStatusFilter(f)}
                className={`px-3 py-1.5 text-[10px] uppercase tracking-widest transition-all ${
                  statusFilter === f
                    ? 'bg-theme-accent/10 text-theme-accent border border-theme-accent/30'
                    : 'text-theme-muted hover:text-theme-secondary'
                }`}
              >
                {f || 'All'}
              </button>
            ))}
          </div>

          <button onClick={fetchData} className="flex items-center gap-1.5 px-3 py-1.5 border border-theme text-[10px] uppercase tracking-widest text-theme-muted hover:text-theme-secondary transition-colors">
            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </button>

          {stats.pending > 0 && (
            <button
              onClick={handleAiReviewAll}
              className="flex items-center gap-1.5 px-3 py-1.5 border border-cyan-500/40 text-[10px] uppercase tracking-widest text-cyan-400 hover:bg-cyan-500/10 transition-colors"
            >
              <Brain className="w-3 h-3" /> AI Review All Pending
            </button>
          )}

          {stats.approved > 0 && (
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 px-3 py-1.5 border border-emerald-500/40 text-[10px] uppercase tracking-widest text-emerald-400 hover:bg-emerald-500/10 transition-colors ml-auto"
            >
              <Download className="w-3 h-3" /> Export Training Data
            </button>
          )}
        </div>

        {/* Corrections List */}
        {loading && corrections.length === 0 ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 text-theme-accent animate-spin" />
          </div>
        ) : corrections.length === 0 ? (
          <div className="border border-theme bg-theme-panel p-12 text-center">
            <Shield className="w-10 h-10 text-theme-dim mx-auto mb-3" />
            <p className="text-sm text-theme-muted uppercase tracking-widest">No Corrections Found</p>
            <p className="text-xs text-theme-dim mt-2">System operating within confidence thresholds</p>
          </div>
        ) : (
          <div className="space-y-3">
            {corrections.map(c => (
              <div key={c.id} className="border border-theme bg-theme-panel">
                {/* Card Header */}
                <button
                  onClick={() => handleExpand(c)}
                  className="w-full p-4 flex items-center gap-4 text-left hover:bg-theme-primary/30 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-bold text-theme-primary uppercase">{c.drug_a}</span>
                      <span className="text-xs text-theme-dim">+</span>
                      <span className="text-sm font-bold text-theme-primary uppercase">{c.drug_b}</span>
                    </div>
                    <div className="flex items-center gap-3 text-[10px] text-theme-muted">
                      <span>GNN: <span className={getSeverityColor(c.gnn_severity)}>{c.gnn_severity || 'unknown'}</span></span>
                      <span>Risk: {(c.gnn_risk_score || 0).toFixed(2)}</span>
                      <span>Conf: {((c.gnn_confidence || 0) * 100).toFixed(1)}%</span>
                      {c.corrected_severity && (
                        <span>Corrected: <span className={getSeverityColor(c.corrected_severity)}>{c.corrected_severity}</span></span>
                      )}
                    </div>
                  </div>
                  <span className={`px-2 py-0.5 text-[10px] uppercase tracking-wider border ${STATUS_COLORS[c.status] || ''}`}>
                    {c.status}
                  </span>
                  {c.evidence_source === 'auto-capture:low-confidence' && (
                    <span className="px-2 py-0.5 text-[10px] uppercase tracking-wider border border-amber-500/30 text-amber-400 bg-amber-500/5">Auto</span>
                  )}
                  {expandedId === c.id ? <ChevronUp className="w-4 h-4 text-theme-muted" /> : <ChevronDown className="w-4 h-4 text-theme-muted" />}
                </button>

                {/* Expanded Review Panel */}
                <AnimatePresence>
                  {expandedId === c.id && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="border-t border-theme overflow-hidden"
                    >
                      <div className="p-4 space-y-4">
                        {/* AI Review Result */}
                        {aiResults[c.id] && (
                          <div className="border border-cyan-800/50 bg-cyan-950/20 p-3">
                            <p className="text-[10px] text-cyan-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                              <Brain className="w-3 h-3" /> AI Assessment
                            </p>
                            <div className="text-xs text-theme-secondary whitespace-pre-wrap">{aiResults[c.id].text}</div>
                            {aiResults[c.id].citations?.length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {aiResults[c.id].citations.map((cit, j) => (
                                  <span key={j} className="text-[9px] px-1.5 py-0.5 border border-theme text-theme-muted">{cit.label}</span>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Edit Fields */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div>
                            <label className="text-[10px] text-theme-muted uppercase tracking-wider mb-1 block">Corrected Severity</label>
                            <select
                              value={editSeverity}
                              onChange={(e) => setEditSeverity(e.target.value)}
                              className="w-full bg-theme-secondary border border-theme text-xs p-2 text-theme-primary focus:outline-none focus:border-theme-accent/50"
                            >
                              <option value="">Select severity...</option>
                              {SEVERITY_OPTIONS.map(s => (
                                <option key={s.value} value={s.value}>{s.label}</option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <label className="text-[10px] text-theme-muted uppercase tracking-wider mb-1 block">Evidence Source</label>
                            <input
                              type="text"
                              value={editSource}
                              onChange={(e) => setEditSource(e.target.value)}
                              placeholder="PMID:12345, Clinical..."
                              className="w-full bg-theme-secondary border border-theme text-xs p-2 text-theme-primary placeholder:text-theme-dim focus:outline-none focus:border-theme-accent/50"
                            />
                          </div>
                          <div className="md:col-span-1">
                            <label className="text-[10px] text-theme-muted uppercase tracking-wider mb-1 block">Created</label>
                            <p className="text-xs text-theme-muted p-2">{c.created_at ? new Date(c.created_at).toLocaleString() : 'N/A'}</p>
                          </div>
                        </div>
                        <div>
                          <label className="text-[10px] text-theme-muted uppercase tracking-wider mb-1 block">Evidence / Rationale</label>
                          <textarea
                            value={editEvidence}
                            onChange={(e) => setEditEvidence(e.target.value)}
                            rows={3}
                            placeholder="Clinical evidence or rationale for correction..."
                            className="w-full bg-theme-secondary border border-theme text-xs p-2 text-theme-primary placeholder:text-theme-dim focus:outline-none focus:border-theme-accent/50 resize-none"
                          />
                        </div>

                        {/* Action Buttons */}
                        <div className="flex items-center gap-2 pt-2 border-t border-theme">
                          <button
                            onClick={() => handleAiReview(c)}
                            disabled={aiReviewing[c.id]}
                            className="flex items-center gap-1.5 px-3 py-1.5 border border-cyan-500/40 text-[10px] uppercase tracking-widest text-cyan-400 hover:bg-cyan-500/10 transition-colors disabled:opacity-40"
                          >
                            {aiReviewing[c.id] ? <Loader2 className="w-3 h-3 animate-spin" /> : <Brain className="w-3 h-3" />}
                            AI Review
                          </button>
                          <button
                            onClick={() => handleReview(c.id, 'approved')}
                            disabled={actionLoading === c.id || !editSeverity}
                            className="flex items-center gap-1.5 px-3 py-1.5 border border-emerald-500/40 text-[10px] uppercase tracking-widest text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-40"
                          >
                            <Check className="w-3 h-3" /> Approve
                          </button>
                          <button
                            onClick={() => handleReview(c.id, 'rejected')}
                            disabled={actionLoading === c.id}
                            className="flex items-center gap-1.5 px-3 py-1.5 border border-red-500/40 text-[10px] uppercase tracking-widest text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-40"
                          >
                            <X className="w-3 h-3" /> Reject
                          </button>
                          <button
                            onClick={() => handleDelete(c.id)}
                            disabled={actionLoading === c.id}
                            className="flex items-center gap-1.5 px-3 py-1.5 border border-theme text-[10px] uppercase tracking-widest text-theme-muted hover:text-red-400 hover:border-red-500/40 transition-colors disabled:opacity-40 ml-auto"
                          >
                            <Trash2 className="w-3 h-3" /> Delete
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ))}
          </div>
        )}

        {/* GNN Training Info */}
        <div className="mt-8 border border-theme bg-theme-panel p-6">
          <h3 className="text-[10px] uppercase tracking-widest text-theme-muted mb-3 flex items-center gap-2">
            <Brain className="w-3.5 h-3.5" /> GNN Feedback Loop
          </h3>
          <div className="text-xs text-theme-dim space-y-2">
            <p>Approved corrections with a corrected severity are stored as an <strong className="text-theme-muted">additive overlay</strong> in Neo4j. They do not overwrite original drug data.</p>
            <p>When the same drug pair is queried again, the Research Assistant references both the GNN prediction and the approved correction, noting any discrepancy.</p>
            <p>Use <strong className="text-theme-muted">Export Training Data</strong> to download all approved corrections as JSON for GNN retraining. The dataset includes original predictions, corrected labels, and confidence gaps — suitable for fine-tuning labels and confidence recalibration.</p>
            <p className="text-theme-accent">Virtuous cycle: Better GNN predictions → fewer low-confidence flags → fewer corrections needed.</p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default CorrectionsPage;
