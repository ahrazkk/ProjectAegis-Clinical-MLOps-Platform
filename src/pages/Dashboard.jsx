import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  Beaker,
  FileText,
  X,
  Check,
  AlertCircle,
  TrendingUp,
  GitBranch,
  RefreshCw,
  ExternalLink,
  Microscope,
  Pill,
  Target,
  Layers,
  Box,
  Hexagon
} from 'lucide-react';
import { searchDrugs, predictDDI, analyzePolypharmacy, sendChatMessage, checkHealth } from '../services/api';
import MoleculeViewer from '../components/MoleculeViewer';
import MoleculeViewer2D from '../components/MoleculeViewer2D';
import BodyMapVisualization from '../components/BodyMapVisualization';
import KnowledgeGraphView from '../components/KnowledgeGraphView';
import RiskGauge from '../components/RiskGauge';

// Debounce hook
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const handler = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(handler);
  }, [value, delay]);
  return debouncedValue;
}

export default function Dashboard() {
  const navigate = useNavigate();

  // API State
  const [apiStatus, setApiStatus] = useState('checking');
  const [error, setError] = useState(null);

  // Drug Selection State
  const [selectedDrugs, setSelectedDrugs] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [polypharmacyResult, setPolypharmacyResult] = useState(null);

  // UI State
  const [activeTab, setActiveTab] = useState('molecules2d');
  const [showSearch, setShowSearch] = useState(false);

  // Chat State
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const chatEndRef = useRef(null);

  const debouncedSearch = useDebounce(searchQuery, 300);

  // Check API health on mount
  useEffect(() => {
    const checkApi = async () => {
      try {
        await checkHealth();
        setApiStatus('online');
      } catch (err) {
        console.error('API check failed:', err);
        setApiStatus('offline');
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
      try {
        const response = await searchDrugs(debouncedSearch);
        // Filter out already selected drugs
        const filtered = (response.results || []).filter(
          drug => !selectedDrugs.some(s => s.drugbank_id === drug.drugbank_id || s.name === drug.name)
        );
        setSearchResults(filtered);
      } catch (err) {
        console.error('Search failed:', err);
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    };

    performSearch();
  }, [debouncedSearch, apiStatus, selectedDrugs]);

  // Scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addDrug = (drug) => {
    setSelectedDrugs(prev => [...prev, drug]);
    setSearchQuery('');
    setSearchResults([]);
    setShowSearch(false);
    setResult(null);
    setPolypharmacyResult(null);
  };

  const removeDrug = (drugId) => {
    setSelectedDrugs(prev => prev.filter(d => d.drugbank_id !== drugId && d.name !== drugId));
    setResult(null);
    setPolypharmacyResult(null);
  };

  const runAnalysis = async () => {
    if (selectedDrugs.length < 2 || apiStatus !== 'online') return;

    setIsAnalyzing(true);
    setError(null);

    try {
      if (selectedDrugs.length === 2) {
        // Two-drug prediction
        const response = await predictDDI(
          { name: selectedDrugs[0].name, smiles: selectedDrugs[0].smiles },
          { name: selectedDrugs[1].name, smiles: selectedDrugs[1].smiles }
        );
        setResult(response);
        setPolypharmacyResult(null);
      } else {
        // Polypharmacy analysis
        const drugs = selectedDrugs.map(d => ({ name: d.name, smiles: d.smiles }));
        const response = await analyzePolypharmacy(drugs);
        setPolypharmacyResult(response);

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
      setError('Failed to analyze interactions. Please check your connection and try again.');
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

    try {
      const contextDrugs = selectedDrugs.map(d => d.name);
      const response = await sendChatMessage(userMessage, contextDrugs, sessionId);
      setSessionId(response.session_id);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.response,
        sources: response.sources
      }]);
    } catch (err) {
      console.error('Chat failed:', err);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        isError: true
      }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'critical': return 'from-red-500 to-rose-600';
      case 'high': return 'from-orange-500 to-red-500';
      case 'medium': return 'from-yellow-500 to-orange-500';
      default: return 'from-emerald-500 to-teal-500';
    }
  };

  const getRiskBgColor = (riskLevel) => {
    switch (riskLevel) {
      case 'critical': return 'bg-red-500/10 border-red-500/20 text-red-400';
      case 'high': return 'bg-orange-500/10 border-orange-500/20 text-orange-400';
      case 'medium': return 'bg-yellow-500/10 border-yellow-500/20 text-yellow-400';
      default: return 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400';
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

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-white relative">
      {/* Top Navigation */}
      <header className="h-16 border-b border-white/5 glass-panel sticky top-0 z-50">
        <div className="h-full px-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 rounded-lg hover:bg-white/5 transition-colors"
            >
              <ChevronLeft className="w-5 h-5 text-slate-400" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[var(--accent-cyan)] to-[var(--accent-blue)] flex items-center justify-center shadow-lg shadow-blue-500/10">
                <GitBranch className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-semibold tracking-tight">Drug Interaction Analysis</h1>
                <p className="text-xs text-slate-500 font-medium">Project Aegis v2.0</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* API Status */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${apiStatus === 'online'
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : apiStatus === 'checking'
                ? 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20'
                : 'bg-red-500/10 text-red-400 border border-red-500/20'
              }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${apiStatus === 'online' ? 'bg-emerald-400 animate-pulse' :
                apiStatus === 'checking' ? 'bg-yellow-400 animate-pulse' : 'bg-red-400'
                }`} />
              {apiStatus === 'online' ? 'Connected' : apiStatus === 'checking' ? 'Connecting...' : 'Offline'}
            </div>

            <button className="p-2 rounded-lg hover:bg-white/5 transition-colors text-slate-400">
              <Bell className="w-5 h-5" />
            </button>
            <button className="p-2 rounded-lg hover:bg-white/5 transition-colors text-slate-400">
              <Settings className="w-5 h-5" />
            </button>
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-purple-500" />
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Left Panel - Drug Selection */}
        <aside className="w-80 border-r border-white/5 flex flex-col glass-panel">
          <div className="p-4 border-b border-white/5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-slate-300">Drug Regimen</h2>
              <span className="text-xs text-slate-500">{selectedDrugs.length} selected</span>
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => setShowSearch(true)}
                placeholder="Search drugs..."
                className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 pl-10 pr-4 text-sm placeholder:text-slate-600 focus:outline-none focus:border-[var(--accent-cyan)]/50 focus:ring-2 focus:ring-[var(--accent-cyan)]/20 transition-all"
                disabled={apiStatus !== 'online'}
              />
              {isSearching && (
                <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-blue-400 animate-spin" />
              )}
            </div>

            {/* Search Results Dropdown */}
            <AnimatePresence>
              {showSearch && searchResults.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute left-4 right-4 mt-2 bg-[#1a1a24] border border-white/10 rounded-xl shadow-2xl overflow-hidden z-50 max-h-64 overflow-y-auto"
                >
                  {searchResults.map((drug, i) => (
                    <button
                      key={drug.drugbank_id || i}
                      onClick={() => addDrug(drug)}
                      className="w-full flex items-center justify-between p-3 hover:bg-white/5 transition-colors border-b border-white/5 last:border-0"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                          <Pill className="w-4 h-4 text-blue-400" />
                        </div>
                        <div className="text-left">
                          <div className="text-sm font-medium">{drug.name}</div>
                          <div className="text-xs text-slate-500">{drug.drugbank_id || 'Unknown ID'}</div>
                        </div>
                      </div>
                      <Plus className="w-4 h-4 text-blue-400" />
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>

            {/* No results message */}
            {showSearch && searchQuery.length >= 2 && !isSearching && searchResults.length === 0 && apiStatus === 'online' && (
              <div className="mt-2 p-3 text-center text-xs text-slate-500">
                No drugs found for "{searchQuery}"
              </div>
            )}

            {apiStatus !== 'online' && (
              <div className="mt-2 p-3 text-center text-xs text-orange-400 bg-orange-500/10 rounded-lg border border-orange-500/20">
                API offline - search unavailable
              </div>
            )}
          </div>

          {/* Selected Drugs */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {selectedDrugs.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-6">
                <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mb-4">
                  <Beaker className="w-8 h-8 text-slate-600" />
                </div>
                <p className="text-sm text-slate-500 mb-2">No drugs selected</p>
                <p className="text-xs text-slate-600">Search and add drugs to begin analysis</p>
              </div>
            ) : (
              selectedDrugs.map((drug, i) => (
                <motion.div
                  key={drug.drugbank_id || drug.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="group p-3 bg-white/5 rounded-xl transition-all hover:bg-white/10"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold shadow-lg ${i === 0 ? 'bg-blue-500/10 text-blue-400 ring-1 ring-blue-500/30' :
                        i === 1 ? 'bg-purple-500/10 text-purple-400 ring-1 ring-purple-500/30' :
                          'bg-cyan-500/10 text-cyan-400 ring-1 ring-cyan-500/30'
                        }`}>
                        {drug.name.substring(0, 2).toUpperCase()}
                      </div>
                      <div>
                        <div className="text-sm font-medium text-slate-200">{drug.name}</div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">{drug.category || 'Drug'}</div>
                      </div>
                    </div>
                    <button
                      onClick={() => removeDrug(drug.drugbank_id || drug.name)}
                      className="p-2 rounded-lg opacity-0 group-hover:opacity-100 hover:bg-red-500/10 text-slate-500 hover:text-red-400 transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </motion.div>
              ))
            )}
          </div>

          {/* Run Analysis Button */}
          <div className="p-4 border-t border-white/5">
            <button
              onClick={runAnalysis}
              disabled={selectedDrugs.length < 2 || isAnalyzing || apiStatus !== 'online'}
              className={`w-full py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${selectedDrugs.length < 2 || apiStatus !== 'online'
                ? 'bg-white/5 text-slate-500 cursor-not-allowed border border-white/5'
                : isAnalyzing
                  ? 'bg-blue-600/20 text-blue-300 cursor-wait border border-blue-500/20 animate-pulse'
                  : 'btn-primary-glow'
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
              <p className="text-xs text-center text-slate-500 mt-2">
                Add {2 - selectedDrugs.length} more drug{2 - selectedDrugs.length > 1 ? 's' : ''} to analyze
              </p>
            )}
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Visualization Tabs */}
          <div className="p-4 border-b border-white/5 flex items-center gap-2">
            {[
              { id: 'molecules2d', label: '2D Structure', icon: Hexagon },
              { id: 'molecules', label: '3D Molecules', icon: Box },
              { id: 'graph', label: 'Knowledge Graph', icon: Network },
              { id: 'body', label: 'Body Map', icon: Heart },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab.id
                  ? 'bg-white/10 text-white'
                  : 'text-slate-500 hover:text-white hover:bg-white/5'
                  }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Visualization Area */}
          <div className="flex-1 relative overflow-hidden bg-gradient-to-br from-[#0d0d14] to-[#0a0a10]">
            {selectedDrugs.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center animate-fade-in">
                <div className="w-20 h-20 rounded-2xl bg-blue-500/10 flex items-center justify-center mb-6 shadow-xl shadow-blue-500/5 ring-1 ring-blue-500/20">
                  <Microscope className="w-10 h-10 text-blue-400" />
                </div>
                <h2 className="text-xl font-semibold text-white mb-2">Ready for Analysis</h2>
                <p className="text-slate-500 max-w-sm text-center text-sm leading-relaxed">
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
                {activeTab === 'molecules' && (
                  <MoleculeViewer
                    drugs={selectedDrugs}
                    result={result}
                  />
                )}
                {activeTab === 'graph' && (
                  <KnowledgeGraphView
                    drugs={selectedDrugs}
                    result={result}
                    polypharmacyResult={polypharmacyResult}
                  />
                )}
                {activeTab === 'body' && (
                  <BodyMapVisualization
                    affectedSystems={getBodyMapData()}
                    result={result}
                  />
                )}
              </>
            )}


          </div>
        </main>

        {/* Right Panel - Results & Chat */}
        <aside className="w-96 border-l border-white/5 flex flex-col bg-[#0d0d14]">
          {/* Results Section */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-4 border-b border-white/5">
              <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-blue-400" />
                Analysis Results
              </h2>
            </div>

            <div className="p-4">
              {error && (
                <div className="mb-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm text-red-400 font-medium">Analysis Error</p>
                      <p className="text-xs text-red-400/70 mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {result ? (
                <div className="space-y-4">
                  {/* Risk Card */}
                  <div className={`p-4 rounded-xl border ${getRiskBgColor(result.risk_level)}`}>
                    <div className="flex items-start gap-3">
                      {result.severity === 'no_interaction' ? (
                        <Shield className="w-6 h-6 text-emerald-400" />
                      ) : (
                        <AlertTriangle className="w-6 h-6" />
                      )}
                      <div>
                        <p className="font-semibold capitalize">
                          {result.severity === 'no_interaction'
                            ? 'No Significant Interaction'
                            : `${result.risk_level || result.severity} Risk`}
                        </p>
                        <p className="text-xs opacity-70 mt-1">
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

                  {/* Mechanism */}
                  {result.mechanism_hypothesis && (
                    <div className="p-4 bg-white/5 rounded-xl border border-white/5">
                      <div className="flex items-center gap-2 mb-3">
                        <Brain className="w-4 h-4 text-blue-400" />
                        <span className="text-xs text-slate-500 uppercase tracking-wider">Mechanism</span>
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed">
                        {result.mechanism_hypothesis}
                      </p>
                    </div>
                  )}

                  {/* Affected Systems */}
                  {result.affected_systems && result.affected_systems.length > 0 && (
                    <div className="p-4 bg-white/5 rounded-xl border border-white/5">
                      <div className="flex items-center gap-2 mb-3">
                        <Target className="w-4 h-4 text-red-400" />
                        <span className="text-xs text-slate-500 uppercase tracking-wider">Affected Systems</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {result.affected_systems.map((sys, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 bg-red-500/10 border border-red-500/20 rounded-lg text-xs text-red-400 capitalize"
                          >
                            {sys.system || sys}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Confidence */}
                  {result.confidence && (
                    <div className="p-4 bg-white/5 rounded-xl border border-white/5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-cyan-400" />
                          <span className="text-xs text-slate-500 uppercase tracking-wider">Model Confidence</span>
                        </div>
                        <span className="text-sm font-semibold text-cyan-400">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mb-4">
                    <Activity className="w-8 h-8 text-slate-600" />
                  </div>
                  <p className="text-sm text-slate-500 mb-2">No Analysis Yet</p>
                  <p className="text-xs text-slate-600">Select drugs and run analysis to see results</p>
                </div>
              )}
            </div>
          </div>

          {/* Chat Section */}
          <div className="h-80 border-t border-white/5 flex flex-col">
            <div className="p-3 border-b border-white/5 flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Research Assistant</h3>
              {messages.length > 0 && (
                <button
                  onClick={() => setMessages([])}
                  className="text-xs text-slate-500 hover:text-white transition-colors"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <Sparkles className="w-6 h-6 text-slate-600 mb-2" />
                  <p className="text-xs text-slate-500">Ask about drug interactions, mechanisms, or alternatives</p>
                </div>
              ) : (
                messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] p-3 rounded-xl text-xs leading-relaxed ${msg.role === 'user'
                        ? 'bg-blue-600 text-white rounded-br-none'
                        : msg.isError
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20 rounded-bl-none'
                          : 'bg-white/5 text-slate-300 border border-white/5 rounded-bl-none'
                        }`}
                    >
                      {msg.content}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-white/10">
                          <p className="text-[10px] text-slate-500 mb-1">Sources:</p>
                          {msg.sources.slice(0, 2).map((s, j) => (
                            <p key={j} className="text-[10px] text-blue-400 truncate">{s}</p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              {isChatLoading && (
                <div className="flex justify-start">
                  <div className="bg-white/5 border border-white/5 rounded-xl rounded-bl-none p-3">
                    <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input */}
            <form onSubmit={handleChatSubmit} className="p-3 border-t border-white/5">
              <div className="relative">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder={apiStatus === 'online' ? "Ask about this interaction..." : "Chat unavailable offline"}
                  disabled={apiStatus !== 'online' || isChatLoading}
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 pl-4 pr-12 text-sm placeholder:text-slate-600 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 transition-all disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!chatInput.trim() || apiStatus !== 'online' || isChatLoading}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-blue-600 rounded-lg text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-500 transition-colors"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </form>
          </div>
        </aside>
      </div>
    </div>
  );
}
