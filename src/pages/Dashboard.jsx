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
import { useSystemLogs } from '../hooks/useSystemLogs';
import { searchDrugs, predictDDI, analyzePolypharmacy, sendChatMessage, checkHealth, getDrugInfo, getInteractionInfo } from '../services/api';
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
  const { addLog } = useSystemLogs();

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

  // Enhanced Data State
  const [drugInfoCache, setDrugInfoCache] = useState({});
  const [interactionEvidence, setInteractionEvidence] = useState(null);
  const [isLoadingDrugInfo, setIsLoadingDrugInfo] = useState(false);

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
      addLog('Initiating system health check...', 'info', 'SYSTEM');
      try {
        await checkHealth();
        setApiStatus('online');
        addLog('Backend services online', 'success', 'API');
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

  // Scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addDrug = async (drug) => {
    setSelectedDrugs(prev => [...prev, drug]);
    setSearchQuery('');
    setSearchResults([]);
    setShowSearch(false);
    setResult(null);
    setPolypharmacyResult(null);
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

  const removeDrug = (drugId) => {
    setSelectedDrugs(prev => prev.filter(d => d.drugbank_id !== drugId && d.name !== drugId));
    setResult(null);
    setPolypharmacyResult(null);
    setInteractionEvidence(null);
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
        addLog('Querying PubMedBERT model...', 'info', 'AI');
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
      case 'critical': return 'text-fui-accent-critical';
      case 'high': return 'text-fui-accent-red';
      case 'medium': return 'text-fui-accent-orange';
      default: return 'text-fui-accent-green';
    }
  };

  const getRiskBgColor = (riskLevel) => {
    switch (riskLevel) {
      case 'critical': return 'border-fui-accent-critical/50 text-fui-accent-critical';
      case 'high': return 'border-fui-accent-red/50 text-fui-accent-red';
      case 'medium': return 'border-fui-accent-orange/50 text-fui-accent-orange';
      default: return 'border-fui-accent-green/50 text-fui-accent-green';
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
    <div className="min-h-screen bg-black text-fui-gray-100 font-mono relative">
      {/* Top Navigation */}
      <header className="h-14 border-b border-fui-gray-500/30 bg-black/95 sticky top-0 z-50">
        <div className="h-full px-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 border border-fui-gray-500/30 hover:border-fui-gray-400 transition-colors"
            >
              <ChevronLeft className="w-4 h-4 text-fui-gray-400" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 border border-fui-gray-500 flex items-center justify-center">
                <GitBranch className="w-4 h-4 text-fui-gray-400" />
              </div>
              <div>
                <h1 className="text-sm font-normal tracking-widest uppercase">Drug Interaction Analysis</h1>
                <p className="text-[10px] text-fui-gray-500 uppercase tracking-widest">Project Aegis v2.0</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* API Status */}
            <div className={`flex items-center gap-2 px-3 py-1.5 text-[10px] font-normal uppercase tracking-widest border ${apiStatus === 'online'
              ? 'border-fui-accent-green/50 text-fui-accent-green'
              : apiStatus === 'checking'
                ? 'border-fui-accent-orange/50 text-fui-accent-orange'
                : 'border-fui-accent-red/50 text-fui-accent-red'
              }`}>
              <span className={`w-1.5 h-1.5 ${apiStatus === 'online' ? 'bg-fui-accent-green' :
                apiStatus === 'checking' ? 'bg-fui-accent-orange animate-pulse' : 'bg-fui-accent-red'
                }`} />
              {apiStatus === 'online' ? 'Connected' : apiStatus === 'checking' ? 'Connecting' : 'Offline'}
            </div>

            <button className="p-2 border border-fui-gray-500/30 hover:border-fui-gray-400 transition-colors text-fui-gray-500 hover:text-fui-gray-300">
              <Bell className="w-4 h-4" />
            </button>
            <button className="p-2 border border-fui-gray-500/30 hover:border-fui-gray-400 transition-colors text-fui-gray-500 hover:text-fui-gray-300">
              <Settings className="w-4 h-4" />
            </button>
            <div className="w-8 h-8 border border-fui-gray-500 flex items-center justify-center">
              <User className="w-4 h-4 text-fui-gray-500" />
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-3.5rem)]">
        {/* Left Panel - Drug Selection */}
        <aside className="w-80 border-r border-fui-gray-500/30 flex flex-col bg-black/50">
          <div className="p-4 border-b border-fui-gray-500/30">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[10px] font-normal text-fui-gray-400 uppercase tracking-widest">// Drug Regimen</h2>
              <span className="text-[10px] text-fui-gray-500">{selectedDrugs.length} selected</span>
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-fui-gray-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onFocus={() => setShowSearch(true)}
                placeholder="Search drugs..."
                className="w-full bg-transparent border border-fui-gray-500/30 py-2.5 pl-10 pr-4 text-sm font-mono placeholder:text-fui-gray-600 focus:outline-none focus:border-fui-accent-cyan/50 transition-all"
                disabled={apiStatus !== 'online'}
              />
              {isSearching && (
                <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-fui-accent-cyan animate-spin" />
              )}
            </div>

            {/* Search Results Dropdown */}
            <AnimatePresence>
              {showSearch && searchResults.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute left-4 right-4 mt-2 bg-black border border-fui-gray-500/30 shadow-2xl overflow-hidden z-50 max-h-64 overflow-y-auto"
                >
                  {searchResults.map((drug, i) => {
                    const hasSmiles = drug.has_smiles || (drug.smiles && drug.smiles.length > 5);
                    return (
                    <button
                      key={drug.drugbank_id || i}
                      onClick={() => addDrug(drug)}
                      className={`w-full flex items-center justify-between p-3 hover:bg-fui-gray-500/10 transition-colors border-b border-fui-gray-500/20 last:border-0 ${
                        !hasSmiles ? 'opacity-70' : ''
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 border flex items-center justify-center ${
                          hasSmiles 
                            ? 'border-fui-accent-green/50 bg-fui-accent-green/10' 
                            : 'border-fui-accent-red/50 bg-fui-accent-red/10'
                        }`}>
                          <Pill className={`w-4 h-4 ${
                            hasSmiles ? 'text-fui-accent-green' : 'text-fui-accent-red'
                          }`} />
                        </div>
                        <div className="text-left">
                          <div className="flex items-center gap-2">
                            <span className={`text-sm font-normal ${hasSmiles ? 'text-fui-gray-200' : 'text-fui-gray-400'}`}>
                              {drug.name}
                            </span>
                            {!hasSmiles && (
                              <span className="text-[8px] px-1.5 py-0.5 border border-fui-accent-orange/30 text-fui-accent-orange uppercase tracking-wider">
                                No Structure
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-fui-gray-500 uppercase tracking-wider">
                            {drug.drugbank_id || drug.category || 'Unknown'}
                          </div>
                        </div>
                      </div>
                      <Plus className={`w-4 h-4 ${hasSmiles ? 'text-fui-accent-cyan' : 'text-fui-gray-500'}`} />
                    </button>
                  )})}
                </motion.div>
              )}
            </AnimatePresence>

            {/* No results message */}
            {showSearch && searchQuery.length >= 2 && !isSearching && searchResults.length === 0 && apiStatus === 'online' && (
              <div className="mt-2 p-3 text-center text-[10px] text-fui-gray-500 border border-fui-gray-500/20 uppercase tracking-wider">
                No drugs found for "{searchQuery}"
              </div>
            )}

            {apiStatus !== 'online' && (
              <div className="mt-2 p-3 text-center text-[10px] text-fui-accent-orange border border-fui-accent-orange/30 uppercase tracking-wider">
                API offline - search unavailable
              </div>
            )}
          </div>

          {/* Selected Drugs */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {selectedDrugs.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-6">
                <div className="w-14 h-14 border border-fui-gray-500/30 flex items-center justify-center mb-4">
                  <Beaker className="w-6 h-6 text-fui-gray-600" />
                </div>
                <p className="text-xs text-fui-gray-500 mb-2 uppercase tracking-wider">No drugs selected</p>
                <p className="text-[10px] text-fui-gray-600">Search and add drugs to begin analysis</p>
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
                  className={`group p-3 border transition-all hover:border-fui-gray-500/40 relative ${
                    !hasSmiles ? 'border-fui-accent-orange/30 bg-fui-accent-orange/5' : 'border-fui-gray-500/20'
                  }`}
                >
                  <div className={`absolute -top-px -left-px w-2 h-2 border-t border-l ${!hasSmiles ? 'border-fui-accent-orange' : 'border-fui-gray-500'}`}></div>
                  <div className={`absolute -bottom-px -right-px w-2 h-2 border-b border-r ${!hasSmiles ? 'border-fui-accent-orange' : 'border-fui-gray-500'}`}></div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 border flex items-center justify-center text-[10px] font-normal uppercase tracking-wider ${
                        !hasSmiles ? 'border-fui-accent-red/50 text-fui-accent-red bg-fui-accent-red/10' :
                        i === 0 ? 'border-fui-accent-cyan/50 text-fui-accent-cyan' :
                        i === 1 ? 'border-fui-accent-blue/50 text-fui-accent-blue' :
                          'border-fui-gray-500/50 text-fui-gray-400'
                        }`}>
                        {hasSmiles ? drug.name.substring(0, 2).toUpperCase() : '⚠'}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-normal text-fui-gray-200">{drug.name}</span>
                          {!hasSmiles && (
                            <span className="text-[7px] px-1 py-0.5 border border-fui-accent-red/30 text-fui-accent-red uppercase">
                              No 3D
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-fui-gray-500 uppercase tracking-widest">{drug.category || 'Drug'}</div>
                      </div>
                    </div>
                    <button
                      onClick={() => removeDrug(drug.drugbank_id || drug.name)}
                      className="p-2 border border-transparent opacity-0 group-hover:opacity-100 hover:border-fui-accent-red/30 text-fui-gray-500 hover:text-fui-accent-red transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                  
                  {/* No Structure Warning */}
                  {!hasSmiles && (
                    <div className="mt-2 p-2 border border-fui-accent-orange/20 bg-fui-accent-orange/5">
                      <p className="text-[9px] text-fui-accent-orange">
                        ⚠ No molecular structure available. 3D visualization limited.
                      </p>
                    </div>
                  )}
                  
                  {/* Side Effects Preview */}
                  {sideEffects.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-fui-gray-500/10">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className="w-3 h-3 text-fui-accent-orange" />
                        <span className="text-[9px] text-fui-gray-500 uppercase tracking-wider">Known Side Effects</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {sideEffects.map((effect, j) => (
                          <span
                            key={j}
                            className="px-1.5 py-0.5 text-[8px] border border-fui-accent-orange/20 text-fui-accent-orange/70 uppercase tracking-wide"
                          >
                            {effect}
                          </span>
                        ))}
                        {drugInfo?.side_effects?.length > 5 && (
                          <span className="px-1.5 py-0.5 text-[8px] text-fui-gray-500">
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
                        <span key={j} className="text-[7px] text-fui-gray-600 uppercase tracking-wider">
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
          <div className="p-4 border-t border-fui-gray-500/30">
            <button
              onClick={runAnalysis}
              disabled={selectedDrugs.length < 2 || isAnalyzing || apiStatus !== 'online'}
              className={`w-full py-3 text-xs uppercase tracking-widest font-normal flex items-center justify-center gap-2 transition-all border ${selectedDrugs.length < 2 || apiStatus !== 'online'
                ? 'border-fui-gray-500/30 text-fui-gray-600 cursor-not-allowed'
                : isAnalyzing
                  ? 'border-fui-accent-cyan/50 text-fui-accent-cyan animate-pulse cursor-wait'
                  : 'border-fui-accent-cyan text-fui-accent-cyan hover:bg-fui-accent-cyan/10'
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
              <p className="text-[10px] text-center text-fui-gray-500 mt-2 uppercase tracking-wider">
                Add {2 - selectedDrugs.length} more drug{2 - selectedDrugs.length > 1 ? 's' : ''} to analyze
              </p>
            )}
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Visualization Tabs */}
          <div className="p-4 border-b border-fui-gray-500/30 flex items-center gap-1">
            {[
              { id: 'molecules2d', label: '2D Structure', icon: Hexagon },
              { id: 'molecules', label: '3D Molecules', icon: Box },
              { id: 'graph', label: 'Knowledge Graph', icon: Network },
              { id: 'body', label: 'Body Map', icon: Heart },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-[10px] font-normal uppercase tracking-widest transition-all border ${activeTab === tab.id
                  ? 'border-fui-accent-cyan/50 text-fui-accent-cyan bg-fui-accent-cyan/5'
                  : 'border-transparent text-fui-gray-500 hover:text-fui-gray-300 hover:border-fui-gray-500/30'
                  }`}
              >
                <tab.icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Visualization Area */}
          <div className="flex-1 relative overflow-hidden bg-black">
            {selectedDrugs.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center animate-fade-in">
                <div className="w-16 h-16 border border-fui-gray-500/30 flex items-center justify-center mb-6 relative">
                  <div className="absolute -top-px -left-px w-3 h-3 border-t border-l border-fui-gray-500"></div>
                  <div className="absolute -bottom-px -right-px w-3 h-3 border-b border-r border-fui-gray-500"></div>
                  <Microscope className="w-8 h-8 text-fui-gray-500" />
                </div>
                <h2 className="text-sm font-normal text-fui-gray-200 mb-2 uppercase tracking-widest">Ready for Analysis</h2>
                <p className="text-fui-gray-500 max-w-sm text-center text-xs leading-relaxed">
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
        <aside className="w-96 border-l border-fui-gray-500/30 flex flex-col bg-black/50">
          {/* Results Section */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-4 border-b border-fui-gray-500/30">
              <h2 className="text-[10px] font-normal text-fui-gray-400 flex items-center gap-2 uppercase tracking-widest">
                <Sparkles className="w-3.5 h-3.5 text-fui-accent-cyan" />
                // Analysis Results
              </h2>
            </div>

            <div className="p-4">
              {error && (
                <div className="mb-4 p-4 border border-fui-accent-red/30 relative">
                  <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-accent-red"></div>
                  <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-accent-red"></div>
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-4 h-4 text-fui-accent-red flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs text-fui-accent-red font-normal uppercase tracking-wider">Analysis Error</p>
                      <p className="text-[10px] text-fui-accent-red/70 mt-1">{error}</p>
                    </div>
                  </div>
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
                        <Shield className="w-5 h-5 text-fui-accent-green" />
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

                  {/* Mechanism */}
                  {result.mechanism_hypothesis && (
                    <div className="p-4 border border-fui-gray-500/20 relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-gray-500"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-gray-500"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <Brain className="w-3.5 h-3.5 text-fui-accent-cyan" />
                        <span className="text-[10px] text-fui-gray-500 uppercase tracking-widest">Mechanism</span>
                        {/* Data Source Badge */}
                        <span className={`ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border ${
                          result.source === 'knowledge_graph'
                            ? 'border-fui-accent-cyan/50 text-fui-accent-cyan bg-fui-accent-cyan/10'
                            : result.source === 'pubmedbert'
                            ? 'border-fui-accent-yellow/50 text-fui-accent-yellow bg-fui-accent-yellow/10'
                            : 'border-fui-gray-500/30 text-fui-gray-500'
                        }`}>
                          {result.source === 'knowledge_graph' ? '⚡ Knowledge Graph' :
                           result.source === 'pubmedbert' ? '🧠 PubMedBERT AI' : 
                           result.source || 'AI Model'}
                        </span>
                      </div>
                      <p className="text-xs text-fui-gray-300 leading-relaxed">
                        {result.mechanism_hypothesis}
                      </p>
                    </div>
                  )}

                  {/* Affected Systems */}
                  {result.affected_systems && result.affected_systems.length > 0 && (
                    <div className="p-4 border border-fui-gray-500/20 relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-gray-500"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-gray-500"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <Target className="w-3.5 h-3.5 text-fui-accent-red" />
                        <span className="text-[10px] text-fui-gray-500 uppercase tracking-widest">Affected Systems</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {result.affected_systems.map((sys, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 border border-fui-accent-red/30 text-[10px] text-fui-accent-red uppercase tracking-wider"
                          >
                            {sys.system || sys}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Confidence */}
                  {result.confidence && (
                    <div className="p-4 border border-fui-gray-500/20 relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-gray-500"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-gray-500"></div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <TrendingUp className="w-3.5 h-3.5 text-fui-accent-cyan" />
                          <span className="text-[10px] text-fui-gray-500 uppercase tracking-widest">Model Confidence</span>
                        </div>
                        <span className="text-sm font-normal text-fui-accent-cyan" style={{ textShadow: '0 0 10px rgba(0,255,255,0.3)' }}>
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Context Sentence - Shows what the model analyzed */}
                  {result.context_sentence && (
                    <div className="p-4 border border-fui-gray-500/20 relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-gray-500"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-gray-500"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <FileText className="w-3.5 h-3.5 text-fui-accent-yellow" />
                        <span className="text-[10px] text-fui-gray-500 uppercase tracking-widest">Analysis Context</span>
                        <span className={`ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border ${
                          result.context_source?.includes('ddi_corpus')
                            ? 'border-fui-accent-green/50 text-fui-accent-green bg-fui-accent-green/10'
                            : result.context_source === 'template' 
                            ? 'border-fui-accent-yellow/30 text-fui-accent-yellow/80' 
                            : result.context_source === 'rag'
                            ? 'border-fui-accent-green/30 text-fui-accent-green/80'
                            : result.context_source === 'user_provided'
                            ? 'border-fui-accent-cyan/30 text-fui-accent-cyan/80'
                            : 'border-fui-gray-500/30 text-fui-gray-500'
                        }`}>
                          {result.context_source?.includes('ddi_corpus') ? '✓ Clinical Literature' :
                           result.context_source === 'template' ? 'Template' : 
                           result.context_source === 'rag' ? 'PubMed' : 
                           result.context_source === 'user_provided' ? 'Custom' : 
                           result.context_source || 'Unknown'}
                        </span>
                      </div>
                      <p className="text-[11px] text-fui-gray-300 leading-relaxed italic">
                        "{result.context_sentence}"
                      </p>
                      {result.template_category && (
                        <p className="text-[9px] text-fui-gray-500 mt-2 uppercase tracking-wider">
                          Category: {result.template_category}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Real-World Evidence from FDA FAERS */}
                  {interactionEvidence?.faers_data && (
                    <div className="p-4 border border-fui-accent-blue/30 relative bg-fui-accent-blue/5">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-accent-blue"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-accent-blue"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <Activity className="w-3.5 h-3.5 text-fui-accent-blue" />
                        <span className="text-[10px] text-fui-gray-400 uppercase tracking-widest">FDA Real-World Evidence</span>
                        <span className="ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border border-fui-accent-blue/50 text-fui-accent-blue">
                          OpenFDA FAERS
                        </span>
                      </div>
                      
                      {/* Total Reports */}
                      <div className="flex items-center justify-between mb-3 pb-3 border-b border-fui-gray-500/10">
                        <span className="text-[10px] text-fui-gray-500 uppercase tracking-wider">Total Adverse Event Reports</span>
                        <span className="text-lg font-mono text-fui-accent-blue" style={{ textShadow: '0 0 10px rgba(59,130,246,0.3)' }}>
                          {interactionEvidence.faers_data.total_reports?.toLocaleString() || '0'}
                        </span>
                      </div>
                      
                      {/* Top Reactions Bar Chart */}
                      {interactionEvidence.faers_data.top_reactions?.length > 0 && (
                        <div>
                          <span className="text-[9px] text-fui-gray-500 uppercase tracking-wider">Top Reported Reactions</span>
                          <div className="mt-2 space-y-1.5">
                            {interactionEvidence.faers_data.top_reactions.slice(0, 5).map((reaction, i) => {
                              const maxCount = interactionEvidence.faers_data.top_reactions[0]?.count || 1;
                              const width = Math.max((reaction.count / maxCount) * 100, 5);
                              return (
                                <div key={i} className="flex items-center gap-2">
                                  <div className="flex-1">
                                    <div className="flex items-center justify-between mb-0.5">
                                      <span className="text-[9px] text-fui-gray-300 uppercase tracking-wider truncate max-w-[140px]">
                                        {reaction.reaction}
                                      </span>
                                      <span className="text-[9px] text-fui-accent-blue font-mono">
                                        {reaction.count?.toLocaleString()}
                                      </span>
                                    </div>
                                    <div className="h-1.5 bg-fui-gray-800 overflow-hidden">
                                      <div 
                                        className="h-full bg-gradient-to-r from-fui-accent-blue to-fui-accent-cyan transition-all duration-500"
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
                        <div className="mt-4 pt-3 border-t border-fui-gray-500/10">
                          <span className="text-[9px] text-fui-gray-500 uppercase tracking-wider">Serious Outcomes</span>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {Object.entries(interactionEvidence.faers_data.serious_outcomes)
                              .filter(([_, count]) => count > 0)
                              .slice(0, 4)
                              .map(([outcome, count], i) => (
                                <div key={i} className="px-2 py-1 border border-fui-accent-red/30 bg-fui-accent-red/10">
                                  <span className="text-[8px] text-fui-accent-red uppercase tracking-wider block">
                                    {outcome.replace(/_/g, ' ')}
                                  </span>
                                  <span className="text-[10px] text-fui-accent-red font-mono">
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
                    <div className="p-4 border border-fui-accent-orange/30 relative">
                      <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-accent-orange"></div>
                      <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-accent-orange"></div>
                      <div className="flex items-center gap-2 mb-3">
                        <AlertTriangle className="w-3.5 h-3.5 text-fui-accent-orange" />
                        <span className="text-[10px] text-fui-gray-400 uppercase tracking-widest">Side Effects Comparison</span>
                        <span className="ml-auto px-2 py-0.5 text-[8px] uppercase tracking-wider border border-fui-accent-orange/30 text-fui-accent-orange">
                          SIDER
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-3">
                        {selectedDrugs.slice(0, 2).map((drug, i) => {
                          const info = drugInfoCache[drug.name.toLowerCase()];
                          const effects = info?.side_effects?.slice(0, 6) || [];
                          return (
                            <div key={i}>
                              <span className={`text-[9px] uppercase tracking-wider ${
                                i === 0 ? 'text-fui-accent-cyan' : 'text-fui-accent-blue'
                              }`}>
                                {drug.name}
                              </span>
                              <div className="mt-1.5 space-y-1">
                                {effects.length > 0 ? effects.map((effect, j) => (
                                  <div key={j} className="text-[8px] text-fui-gray-400 py-0.5 px-1.5 border border-fui-gray-500/20">
                                    {effect}
                                  </div>
                                )) : (
                                  <span className="text-[8px] text-fui-gray-600">No data</span>
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
                            <span className="text-[9px] text-fui-accent-red uppercase tracking-wider">
                              ⚠️ Shared Side Effects ({common.length})
                            </span>
                            <div className="mt-1.5 flex flex-wrap gap-1">
                              {common.slice(0, 5).map((effect, i) => (
                                <span key={i} className="text-[8px] text-fui-accent-red px-1.5 py-0.5 border border-fui-accent-red/30 bg-fui-accent-red/10">
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
                  <div className="p-3 border border-fui-gray-500/10 bg-fui-gray-900/50">
                    <span className="text-[9px] text-fui-gray-500 uppercase tracking-wider">Data Sources</span>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {result.source && (
                        <span className="px-2 py-0.5 text-[7px] border border-fui-accent-cyan/30 text-fui-accent-cyan uppercase tracking-wider">
                          {result.source === 'knowledge_graph' ? 'Neo4j KG' : result.source === 'pubmedbert' ? 'PubMedBERT' : result.source}
                        </span>
                      )}
                      {result.context_source?.includes('ddi_corpus') && (
                        <span className="px-2 py-0.5 text-[7px] border border-fui-accent-green/30 text-fui-accent-green uppercase tracking-wider">
                          DDI Corpus 2013
                        </span>
                      )}
                      {interactionEvidence?.faers_data && (
                        <span className="px-2 py-0.5 text-[7px] border border-fui-accent-blue/30 text-fui-accent-blue uppercase tracking-wider">
                          OpenFDA FAERS
                        </span>
                      )}
                      {(drugInfoCache[selectedDrugs[0]?.name?.toLowerCase()]?.side_effects?.length > 0 ||
                        drugInfoCache[selectedDrugs[1]?.name?.toLowerCase()]?.side_effects?.length > 0) && (
                        <span className="px-2 py-0.5 text-[7px] border border-fui-accent-orange/30 text-fui-accent-orange uppercase tracking-wider">
                          SIDER
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-14 h-14 border border-fui-gray-500/30 flex items-center justify-center mb-4">
                    <Activity className="w-6 h-6 text-fui-gray-600" />
                  </div>
                  <p className="text-xs text-fui-gray-500 mb-2 uppercase tracking-wider">No Analysis Yet</p>
                  <p className="text-[10px] text-fui-gray-600">Select drugs and run analysis to see results</p>
                </div>
              )}
            </div>
          </div>

          {/* Chat Section */}
          <div className="h-80 border-t border-fui-gray-500/30 flex flex-col">
            <div className="p-3 border-b border-fui-gray-500/30 flex items-center justify-between">
              <h3 className="text-[10px] font-normal text-fui-gray-400 uppercase tracking-widest">// Research Assistant</h3>
              {messages.length > 0 && (
                <button
                  onClick={() => setMessages([])}
                  className="text-[10px] text-fui-gray-500 hover:text-fui-accent-cyan transition-colors uppercase tracking-wider"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <Sparkles className="w-5 h-5 text-fui-gray-600 mb-2" />
                  <p className="text-[10px] text-fui-gray-500">Ask about drug interactions, mechanisms, or alternatives</p>
                </div>
              ) : (
                messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] p-3 text-xs leading-relaxed ${msg.role === 'user'
                        ? 'border border-fui-accent-cyan/50 text-fui-gray-200 bg-fui-accent-cyan/5'
                        : msg.isError
                          ? 'border border-fui-accent-red/30 text-fui-accent-red'
                          : 'border border-fui-gray-500/30 text-fui-gray-300'
                        }`}
                    >
                      {msg.content}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-fui-gray-500/20">
                          <p className="text-[10px] text-fui-gray-500 mb-1 uppercase tracking-wider">Sources:</p>
                          {msg.sources.slice(0, 2).map((s, j) => (
                            <p key={j} className="text-[10px] text-fui-accent-cyan truncate">{s}</p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              {isChatLoading && (
                <div className="flex justify-start">
                  <div className="border border-fui-gray-500/30 p-3">
                    <Loader2 className="w-4 h-4 text-fui-accent-cyan animate-spin" />
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input */}
            <form onSubmit={handleChatSubmit} className="p-3 border-t border-fui-gray-500/30">
              <div className="relative">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder={apiStatus === 'online' ? "Ask about this interaction..." : "Chat unavailable offline"}
                  disabled={apiStatus !== 'online' || isChatLoading}
                  className="w-full bg-transparent border border-fui-gray-500/30 py-2.5 pl-4 pr-12 text-sm font-mono placeholder:text-fui-gray-600 focus:outline-none focus:border-fui-accent-cyan/50 transition-all disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!chatInput.trim() || apiStatus !== 'online' || isChatLoading}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 border border-fui-accent-cyan/50 text-fui-accent-cyan disabled:opacity-30 disabled:cursor-not-allowed hover:bg-fui-accent-cyan/10 transition-colors"
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
