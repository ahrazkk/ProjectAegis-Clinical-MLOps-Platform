import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitCompare,
  Plus,
  X,
  Search,
  Loader2,
  AlertTriangle,
  CheckCircle,
  AlertCircle,
  Pill,
  Atom,
  FileText,
  Activity,
  ChevronDown,
  ChevronUp,
  Layers,
  Beaker,
  Shield,
  Info,
  Sparkles,
  ArrowRight,
  Target,
  Zap
} from 'lucide-react';
import { searchDrugs, compareDrugs } from '../services/api';

// Severity color schemes - FUI Style
const severityConfig = {
  severe: {
    bg: 'bg-fui-accent-red/10',
    border: 'border-fui-accent-red/40',
    text: 'text-fui-accent-red',
    label: 'Severe',
    icon: AlertTriangle,
    color: '#EF4444'
  },
  moderate: {
    bg: 'bg-fui-accent-orange/10',
    border: 'border-fui-accent-orange/40',
    text: 'text-fui-accent-orange',
    label: 'Moderate',
    icon: AlertCircle,
    color: '#F59E0B'
  },
  minor: {
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/40',
    text: 'text-yellow-400',
    label: 'Minor',
    icon: Info,
    color: '#EAB308'
  },
  no_interaction: {
    bg: 'bg-fui-accent-green/10',
    border: 'border-fui-accent-green/40',
    text: 'text-fui-accent-green',
    label: 'Safe',
    icon: CheckCircle,
    color: '#10B981'
  },
  unknown: {
    bg: 'bg-fui-gray-500/10',
    border: 'border-fui-gray-500/40',
    text: 'text-fui-gray-400',
    label: 'Unknown',
    icon: AlertCircle,
    color: '#6B7280'
  },
  self: {
    bg: 'bg-fui-gray-800/50',
    border: 'border-fui-gray-700/50',
    text: 'text-fui-gray-600',
    label: '-',
    icon: null,
    color: '#374151'
  }
};

// Risk Matrix Cell Component - FUI Style
function RiskCell({ severity, drug1, drug2, onClick }) {
  const config = severityConfig[severity] || severityConfig.unknown;
  const Icon = config.icon;
  
  if (severity === 'self') {
    return (
      <div className="w-full h-full min-h-[50px] flex items-center justify-center bg-black/30 border border-fui-gray-700/30">
        <span className="text-fui-gray-700 text-sm">—</span>
      </div>
    );
  }

  return (
    <button
      onClick={onClick}
      className={`w-full h-full min-h-[50px] flex flex-col items-center justify-center gap-1 border ${config.border} ${config.bg} transition-all cursor-pointer hover:opacity-80`}
    >
      {Icon && <Icon size={14} className={config.text} />}
      <span className={`text-[10px] font-medium uppercase tracking-wide ${config.text}`}>{config.label}</span>
    </button>
  );
}

// Drug Chip Component - FUI Style
function DrugChip({ name, onRemove, index }) {
  const colors = [
    { border: 'border-fui-accent-cyan/50', text: 'text-fui-accent-cyan' },
    { border: 'border-fui-accent-purple/50', text: 'text-fui-accent-purple' },
    { border: 'border-fui-accent-green/50', text: 'text-fui-accent-green' },
    { border: 'border-fui-accent-orange/50', text: 'text-fui-accent-orange' },
    { border: 'border-pink-500/50', text: 'text-pink-400' },
  ];
  const color = colors[index % colors.length];

  return (
    <motion.span
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0, opacity: 0 }}
      className={`inline-flex items-center gap-2 px-3 py-1.5 border ${color.border} ${color.text} text-[10px] font-normal uppercase tracking-widest`}
    >
      <Pill size={12} />
      {name}
      <button 
        onClick={() => onRemove(name)}
        className="ml-1 hover:text-fui-accent-red transition-colors"
      >
        <X size={12} />
      </button>
    </motion.span>
  );
}

// Drug Info Card Component
function DrugInfoCard({ drug, onRemove, index }) {
  const [expanded, setExpanded] = useState(false);
  
  const colors = [
    { accent: 'bg-fui-accent-cyan', border: 'border-fui-accent-cyan/30', text: 'text-fui-accent-cyan' },
    { accent: 'bg-fui-accent-purple', border: 'border-fui-accent-purple/30', text: 'text-fui-accent-purple' },
    { accent: 'bg-fui-accent-green', border: 'border-fui-accent-green/30', text: 'text-fui-accent-green' },
    { accent: 'bg-fui-accent-orange', border: 'border-fui-accent-orange/30', text: 'text-fui-accent-orange' },
    { accent: 'bg-fui-accent-red', border: 'border-fui-accent-red/30', text: 'text-fui-accent-red' },
  ];
  const color = colors[index % colors.length];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      className={`relative overflow-hidden bg-black/50 border ${color.border}`}
    >
      {/* Accent line top */}
      <div className={`h-0.5 ${color.accent}`} />
      
      {/* Header */}
      <div className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 border ${color.border} ${color.text}`}>
              <Pill size={18} />
            </div>
            <div>
              <h3 className="font-medium text-white text-sm">{drug.name}</h3>
              <p className="text-[10px] text-fui-gray-400 font-mono uppercase tracking-widest">{drug.drugbank_id || 'N/A'}</p>
            </div>
          </div>
          <button
            onClick={() => onRemove(drug.name)}
            className="p-1.5 border border-fui-gray-500/30 text-fui-gray-400 hover:text-fui-accent-red hover:border-fui-accent-red/30 transition-colors"
          >
            <X size={14} />
          </button>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-2 mt-3">
          <div className="p-2 bg-black/30 border border-fui-gray-500/20">
            <p className="text-[10px] text-fui-gray-400 uppercase tracking-widest mb-0.5">Class</p>
            <p className="text-xs font-medium text-white truncate">{drug.therapeutic_class || 'Unknown'}</p>
          </div>
          <div className="p-2 bg-black/30 border border-fui-gray-500/20">
            <p className="text-[10px] text-fui-gray-400 uppercase tracking-widest mb-0.5">Interactions</p>
            <p className="text-xs font-bold text-fui-accent-cyan">{drug.interaction_count || 0}</p>
          </div>
          <div className="p-2 bg-black/30 border border-fui-gray-500/20">
            <p className="text-[10px] text-fui-gray-400 uppercase tracking-widest mb-0.5">Mol. Weight</p>
            <p className="text-xs font-medium text-white">{drug.molecular_weight ? parseFloat(drug.molecular_weight).toFixed(1) : '—'}</p>
          </div>
          <div className="p-2 bg-black/30 border border-fui-gray-500/20">
            <p className="text-[10px] text-fui-gray-400 uppercase tracking-widest mb-0.5">Formula</p>
            <p className="text-xs font-mono text-white">{drug.molecular_formula || '—'}</p>
          </div>
        </div>

        {/* SMILES Indicator */}
        <div className="mt-3 flex items-center gap-4">
          {drug.smiles ? (
            <span className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-fui-accent-green">
              <CheckCircle size={10} />
              Has SMILES
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-fui-gray-500">
              <X size={10} />
              No SMILES
            </span>
          )}
        </div>

        {/* Expandable Description */}
        {drug.description && (
          <div className="mt-3 pt-3 border-t border-fui-gray-500/20">
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-fui-gray-400 hover:text-white transition-colors w-full"
            >
              <FileText size={10} />
              Description
              <ChevronDown size={10} className={`ml-auto transition-transform ${expanded ? 'rotate-180' : ''}`} />
            </button>
            <AnimatePresence>
              {expanded && (
                <motion.p
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="text-[11px] text-fui-gray-300 mt-2 leading-relaxed overflow-hidden"
                >
                  {drug.description}
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// Interaction Detail Card
function InteractionDetail({ interaction, index }) {
  const config = severityConfig[interaction.severity] || severityConfig.unknown;
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className={`p-3 bg-black/40 border ${config.border}`}
    >
      <div className="flex items-center gap-3 mb-2">
        <div className="flex items-center gap-2 flex-1">
          <span className="text-sm font-medium text-white">{interaction.drug1}</span>
          <ArrowRight size={12} className="text-fui-gray-500" />
          <span className="text-sm font-medium text-white">{interaction.drug2}</span>
        </div>
        <span className={`flex items-center gap-1.5 px-2 py-1 border ${config.border} ${config.text} text-[10px] uppercase tracking-widest font-medium`}>
          {Icon && <Icon size={10} />}
          {config.label}
        </span>
      </div>
      {interaction.mechanism && (
        <p className="text-xs text-fui-gray-300 leading-relaxed">{interaction.mechanism}</p>
      )}
    </motion.div>
  );
}

// Search Input Component
function SearchInput({ value, onChange, onSelect, results, isSearching, onClose }) {
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div className="relative">
      <div className="flex items-center gap-2 bg-black/50 border border-fui-gray-500/30 px-3 py-2 focus-within:border-fui-accent-cyan/50 transition-colors">
        <Search size={14} className="text-fui-gray-400" />
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Search drugs..."
          className="bg-transparent text-white text-xs focus:outline-none flex-1 placeholder-fui-gray-500"
        />
        {isSearching && <Loader2 size={12} className="text-fui-accent-cyan animate-spin" />}
        <button onClick={onClose} className="p-1 hover:bg-fui-gray-500/20 transition-colors">
          <X size={12} className="text-fui-gray-400" />
        </button>
      </div>

      {/* Results Dropdown */}
      <AnimatePresence>
        {results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full mt-1 left-0 right-0 max-h-60 overflow-y-auto bg-black/95 border border-fui-gray-500/30 z-50"
          >
            {results.map((drug, i) => (
              <button
                key={drug.drugbank_id || drug.name}
                onClick={() => onSelect(drug)}
                className="w-full px-3 py-2 text-left hover:bg-fui-gray-500/20 transition-colors border-b border-fui-gray-500/20 last:border-0 flex items-center justify-between group"
              >
                <div>
                  <span className="text-xs font-medium text-white group-hover:text-fui-accent-cyan transition-colors">
                    {drug.name}
                  </span>
                  {drug.therapeutic_class && (
                    <span className="text-[10px] text-fui-gray-500 ml-2">{drug.therapeutic_class}</span>
                  )}
                </div>
                {drug.smiles && (
                  <Atom size={12} className="text-fui-accent-green opacity-0 group-hover:opacity-100 transition-opacity" />
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Main Drug Comparison Component
export default function DrugComparison({ initialDrugs = [], onClose }) {
  const [selectedDrugs, setSelectedDrugs] = useState(initialDrugs.map(d => d.name || d));
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showSearch, setShowSearch] = useState(false);
  const [selectedInteraction, setSelectedInteraction] = useState(null);

  // Search drugs
  useEffect(() => {
    const search = async () => {
      if (!searchQuery || searchQuery.length < 2) {
        setSearchResults([]);
        return;
      }
      setIsSearching(true);
      try {
        const results = await searchDrugs(searchQuery);
        setSearchResults(
          (results.results || []).filter(d => !selectedDrugs.includes(d.name))
        );
      } catch (err) {
        console.error('Search failed:', err);
      } finally {
        setIsSearching(false);
      }
    };
    
    const timeout = setTimeout(search, 300);
    return () => clearTimeout(timeout);
  }, [searchQuery, selectedDrugs]);

  // Load comparison when drugs change
  useEffect(() => {
    const loadComparison = async () => {
      if (selectedDrugs.length < 2) {
        setComparison(null);
        return;
      }

      setLoading(true);
      setError(null);
      try {
        const data = await compareDrugs(selectedDrugs);
        setComparison(data);
      } catch (err) {
        console.error('Comparison failed:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    loadComparison();
  }, [selectedDrugs]);

  const addDrug = (drug) => {
    if (selectedDrugs.length >= 5) return;
    setSelectedDrugs([...selectedDrugs, drug.name]);
    setSearchQuery('');
    setSearchResults([]);
    setShowSearch(false);
  };

  const removeDrug = (drugName) => {
    setSelectedDrugs(selectedDrugs.filter(d => d !== drugName));
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6 p-2"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 border border-fui-accent-purple/30 text-fui-accent-purple">
            <GitCompare size={20} strokeWidth={1.5} />
          </div>
          <div>
            <h1 className="text-sm font-medium text-white uppercase tracking-widest">// Drug Comparison</h1>
            <p className="text-[10px] text-fui-gray-400 uppercase tracking-widest mt-0.5">Compare 2-5 drugs side-by-side</p>
          </div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-2 border border-fui-gray-500/30 text-fui-gray-400 hover:text-white hover:border-fui-gray-400 transition-colors"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* Drug Selector */}
      <div className="p-4 bg-black/50 border border-fui-gray-500/30">
        <div className="flex items-center gap-2 flex-wrap">
          <AnimatePresence mode="popLayout">
            {selectedDrugs.map((drug, i) => (
              <DrugChip key={drug} name={drug} onRemove={removeDrug} index={i} />
            ))}
          </AnimatePresence>
          
          {selectedDrugs.length < 5 && (
            showSearch ? (
              <div className="w-64">
                <SearchInput
                  value={searchQuery}
                  onChange={setSearchQuery}
                  onSelect={addDrug}
                  results={searchResults}
                  isSearching={isSearching}
                  onClose={() => { setShowSearch(false); setSearchQuery(''); }}
                />
              </div>
            ) : (
              <button
                onClick={() => setShowSearch(true)}
                className="inline-flex items-center gap-2 px-3 py-1.5 border border-dashed border-fui-gray-500/50 text-fui-gray-400 hover:border-fui-accent-cyan/50 hover:text-fui-accent-cyan transition-colors text-xs uppercase tracking-widest"
              >
                <Plus size={14} />
                Add Drug
              </button>
            )
          )}
        </div>

        {selectedDrugs.length === 5 && (
          <p className="text-[10px] text-fui-accent-orange mt-3 flex items-center gap-1 uppercase tracking-widest">
            <AlertCircle size={10} />
            Maximum of 5 drugs reached
          </p>
        )}
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="flex flex-col items-center gap-3">
            <div className="relative">
              <div className="w-12 h-12 border border-fui-accent-purple/30" />
              <div className="absolute inset-0 w-12 h-12 border border-transparent border-t-fui-accent-purple animate-spin" />
            </div>
            <p className="text-fui-gray-400 text-[10px] uppercase tracking-widest">Analyzing interactions...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 bg-fui-accent-red/10 border border-fui-accent-red/30">
          <div className="flex items-center gap-3">
            <AlertCircle className="text-fui-accent-red shrink-0" size={16} />
            <div>
              <p className="text-fui-accent-red text-xs font-medium uppercase tracking-widest">Comparison failed</p>
              <p className="text-fui-accent-red/70 text-[11px] mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Results */}
      {comparison && !loading && (
        <div className="space-y-6">
          {/* Drug Cards Grid */}
          <div className={`grid gap-5 ${
            comparison.drugs.length === 2 ? 'grid-cols-1 md:grid-cols-2' :
            comparison.drugs.length === 3 ? 'grid-cols-1 md:grid-cols-3' :
            'grid-cols-1 md:grid-cols-2 lg:grid-cols-4'
          }`}>
            <AnimatePresence mode="popLayout">
              {comparison.drugs.map((drug, i) => (
                <DrugInfoCard
                  key={drug.name}
                  drug={drug}
                  index={i}
                  onRemove={removeDrug}
                />
              ))}
            </AnimatePresence>
          </div>

          {/* Interaction Matrix */}
          {comparison.drug_names?.length >= 2 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="p-4 bg-black/50 border border-fui-gray-500/30"
            >
              <div className="flex items-center gap-2 mb-4">
                <Layers size={14} className="text-fui-accent-purple" />
                <h3 className="text-[10px] font-medium text-fui-gray-300 uppercase tracking-widest">// Interaction Matrix</h3>
              </div>
              
              <div className="overflow-x-auto flex justify-center">
                <table className="border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2"></th>
                      {comparison.drug_names.map((name, i) => (
                        <th key={name} className="p-2 text-center">
                          <span className="text-xs font-medium text-gray-400">
                            {name}
                          </span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.drug_names.map((rowName, i) => (
                      <tr key={rowName}>
                        <td className="p-2 text-right">
                          <span className="text-xs font-medium text-gray-400">
                            {rowName}
                          </span>
                        </td>
                        {comparison.risk_matrix[i].map((cell, j) => (
                          <td key={j} className="p-1">
                            <div className="w-[80px] h-[50px]">
                              <RiskCell 
                                severity={cell.severity} 
                                drug1={rowName}
                                drug2={comparison.drug_names[j]}
                                onClick={() => cell.severity !== 'self' && setSelectedInteraction({
                                  drug1: rowName,
                                  drug2: comparison.drug_names[j],
                                  ...cell
                                })}
                              />
                            </div>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Legend */}
              <div className="mt-4 pt-3 border-t border-fui-gray-500/20">
                <div className="flex flex-wrap justify-center gap-4">
                  {['no_interaction', 'minor', 'moderate', 'severe'].map((severity) => {
                    const config = severityConfig[severity];
                    return (
                      <div key={severity} className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3"
                          style={{ backgroundColor: config.color }}
                        />
                        <span className="text-[10px] text-fui-gray-400 uppercase tracking-widest">{config.label}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}

          {/* Detected Interactions */}
          {comparison.pairwise_interactions?.filter(i => i.severity !== 'no_interaction').length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="p-4 bg-black/50 border border-fui-gray-500/30"
            >
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle size={14} className="text-fui-accent-orange" />
                <h3 className="text-[10px] font-medium text-fui-gray-300 uppercase tracking-widest">// Detected Interactions</h3>
                <span className="ml-auto px-2 py-0.5 border border-fui-accent-orange/30 text-fui-accent-orange text-[10px] uppercase tracking-widest font-medium">
                  {comparison.pairwise_interactions.filter(i => i.severity !== 'no_interaction').length} found
                </span>
              </div>
              
              <div className="space-y-2">
                {comparison.pairwise_interactions
                  .filter(i => i.severity !== 'no_interaction')
                  .map((interaction, idx) => (
                    <InteractionDetail key={idx} interaction={interaction} index={idx} />
                  ))}
              </div>
            </motion.div>
          )}

          {/* All Safe Message */}
          {comparison.pairwise_interactions?.filter(i => i.severity !== 'no_interaction').length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-6 bg-fui-accent-green/10 border border-fui-accent-green/30 text-center"
            >
              <CheckCircle className="text-fui-accent-green mx-auto mb-3" size={36} strokeWidth={1.5} />
              <h3 className="text-sm font-medium text-white mb-1 uppercase tracking-widest">No Interactions Detected</h3>
              <p className="text-[11px] text-fui-gray-300">
                Based on our database, these drugs appear to be safe to use together.
              </p>
            </motion.div>
          )}
        </div>
      )}

      {/* Empty State */}
      {selectedDrugs.length < 2 && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <div className="w-16 h-16 mx-auto mb-4 border border-fui-gray-500/30 flex items-center justify-center">
            <GitCompare size={32} className="text-fui-gray-600" strokeWidth={1.5} />
          </div>
          <h3 className="text-sm font-medium text-fui-gray-300 mb-1 uppercase tracking-widest">Select drugs to compare</h3>
          <p className="text-[11px] text-fui-gray-500 max-w-md mx-auto">
            Add at least 2 drugs to see their interactions, properties, and potential risks side-by-side.
          </p>
          <button
            onClick={() => setShowSearch(true)}
            className="mt-4 inline-flex items-center gap-2 px-4 py-2 border border-fui-accent-cyan/50 text-fui-accent-cyan hover:bg-fui-accent-cyan/10 transition-colors text-xs uppercase tracking-widest"
          >
            <Plus size={14} />
            Add Your First Drug
          </button>
        </motion.div>
      )}
    </motion.div>
  );
}
