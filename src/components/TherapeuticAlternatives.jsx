import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Lightbulb,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  AlertCircle,
  Pill,
  Shield,
  Loader2,
  ChevronRight,
  Sparkles,
  TrendingUp,
  Atom,
  XCircle
} from 'lucide-react';
import { getTherapeuticAlternatives } from '../services/api';

// Safety score badge
function SafetyBadge({ score }) {
  let color, icon, label;
  
  if (score >= 75) {
    color = 'emerald';
    icon = <CheckCircle size={14} />;
    label = 'Safe';
  } else if (score >= 50) {
    color = 'yellow';
    icon = <AlertCircle size={14} />;
    label = 'Caution';
  } else {
    color = 'red';
    icon = <AlertTriangle size={14} />;
    label = 'Risk';
  }

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs
      bg-${color}-500/20 text-${color}-400 border border-${color}-500/30`}>
      {icon}
      {score}%
    </span>
  );
}

// Alternative drug card
function AlternativeCard({ alternative, rank, originalDrug, interactingDrug }) {
  const [expanded, setExpanded] = useState(false);
  
  const severityColors = {
    no_interaction: 'emerald',
    minor: 'yellow',
    moderate: 'orange',
    severe: 'red',
    unknown: 'gray'
  };

  const color = severityColors[alternative.interaction_severity] || 'gray';

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: rank * 0.1 }}
      className={`p-4 rounded-xl border transition-all cursor-pointer
        ${alternative.is_safer 
          ? 'bg-gradient-to-br from-emerald-500/10 to-green-500/5 border-emerald-500/30 hover:border-emerald-400'
          : `bg-gradient-to-br from-${color}-500/10 to-${color}-500/5 border-${color}-500/30`
        }`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          {/* Rank badge */}
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm
            ${rank === 0 ? 'bg-emerald-500 text-white' :
              rank === 1 ? 'bg-emerald-400/80 text-white' :
              rank === 2 ? 'bg-emerald-400/60 text-white' :
              'bg-gray-700 text-gray-300'}`}>
            {rank + 1}
          </div>
          
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-white">{alternative.name}</h4>
              {alternative.is_safer && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 text-xs">
                  <Sparkles size={10} />
                  Safer
                </span>
              )}
            </div>
            {alternative.drugbank_id && (
              <p className="text-xs text-gray-500">{alternative.drugbank_id}</p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <SafetyBadge score={alternative.safety_score} />
          <ChevronRight 
            size={16} 
            className={`text-gray-500 transition-transform ${expanded ? 'rotate-90' : ''}`}
          />
        </div>
      </div>

      {/* Expanded details */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mt-4 pt-4 border-t border-white/10 space-y-3"
          >
            {/* Interaction with the problematic drug */}
            {interactingDrug && (
              <div className={`p-3 rounded-lg bg-${color}-500/10 border border-${color}-500/20`}>
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-gray-400">Interaction with {interactingDrug}:</span>
                  <span className={`font-medium text-${color}-400 capitalize`}>
                    {alternative.interaction_severity?.replace('_', ' ') || 'Unknown'}
                  </span>
                </div>
                {alternative.mechanism && (
                  <p className="text-xs text-gray-500 mt-1">{alternative.mechanism}</p>
                )}
              </div>
            )}

            {/* SMILES indicator */}
            {alternative.smiles && (
              <div className="flex items-center gap-1 text-xs text-emerald-400">
                <Atom size={12} />
                Molecular structure available
              </div>
            )}

            {/* Comparison summary */}
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className="text-gray-500">{originalDrug}</span>
              <ArrowRight size={12} />
              <span className="text-cyan-400">{alternative.name}</span>
              {alternative.is_safer && (
                <span className="text-emerald-400 ml-2">
                  ↓ Lower risk with {interactingDrug}
                </span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Main Therapeutic Alternatives Component
export default function TherapeuticAlternatives({ 
  drugName, 
  interactingWith = null,
  severity = null,
  onSelectAlternative,
  compact = false 
}) {
  const [alternatives, setAlternatives] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAlternatives = async () => {
      if (!drugName) return;
      
      setLoading(true);
      setError(null);
      try {
        const data = await getTherapeuticAlternatives(drugName, interactingWith);
        setAlternatives(data);
      } catch (err) {
        console.error('Failed to fetch alternatives:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchAlternatives();
  }, [drugName, interactingWith]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-6">
        <Loader2 className="animate-spin text-cyan-400" size={20} />
        <span className="ml-2 text-sm text-gray-400">Finding alternatives...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
        <p className="text-sm text-red-400">Failed to find alternatives: {error}</p>
      </div>
    );
  }

  if (!alternatives) return null;

  // No therapeutic class found
  if (!alternatives.therapeutic_class) {
    return (
      <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
        <div className="flex items-center gap-2 text-gray-400">
          <AlertCircle size={16} />
          <span className="text-sm">No therapeutic class found for {drugName}</span>
        </div>
        <p className="text-xs text-gray-500 mt-1">
          Unable to suggest alternatives without knowing the drug's class.
        </p>
      </div>
    );
  }

  // No alternatives found
  if (!alternatives.alternatives?.length) {
    return (
      <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
        <div className="flex items-center gap-2 text-gray-400">
          <AlertCircle size={16} />
          <span className="text-sm">No alternatives found in {alternatives.therapeutic_class} class</span>
        </div>
      </div>
    );
  }

  const saferCount = alternatives.alternatives.filter(a => a.is_safer).length;

  if (compact) {
    // Compact inline version
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-3 rounded-lg bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border border-emerald-500/30"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Lightbulb className="text-emerald-400" size={16} />
            <span className="text-sm text-gray-300">
              {saferCount} safer alternatives found in 
              <span className="text-emerald-400 ml-1">{alternatives.therapeutic_class}</span>
            </span>
          </div>
          {onSelectAlternative && alternatives.alternatives[0] && (
            <button
              onClick={() => onSelectAlternative(alternatives.alternatives[0])}
              className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
            >
              Try {alternatives.alternatives[0].name}
              <ChevronRight size={12} />
            </button>
          )}
        </div>
      </motion.div>
    );
  }

  // Full panel version
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-4"
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30">
            <Lightbulb className="text-emerald-400" size={20} />
          </div>
          <div>
            <h3 className="font-semibold text-white">Therapeutic Alternatives</h3>
            <p className="text-xs text-gray-500">
              Same class as {alternatives.drug}: 
              <span className="text-cyan-400 ml-1">{alternatives.therapeutic_class}</span>
            </p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-lg font-bold text-emerald-400">{saferCount}</span>
          <span className="text-sm text-gray-500 ml-1">safer options</span>
        </div>
      </div>

      {/* Context banner */}
      {interactingWith && severity && (
        <div className="p-3 rounded-lg bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/30">
          <div className="flex items-center gap-2">
            <XCircle className="text-red-400" size={16} />
            <span className="text-sm text-gray-300">
              <span className="text-white font-medium">{alternatives.drug}</span>
              {' + '}
              <span className="text-white font-medium">{interactingWith}</span>
              {' has a '}
              <span className="text-red-400 font-medium capitalize">{severity}</span>
              {' interaction'}
            </span>
          </div>
        </div>
      )}

      {/* Alternatives list */}
      <div className="space-y-3">
        {alternatives.alternatives.map((alt, i) => (
          <AlternativeCard
            key={alt.name}
            alternative={alt}
            rank={i}
            originalDrug={alternatives.drug}
            interactingDrug={alternatives.interacting_with}
          />
        ))}
      </div>

      {/* Footer note */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
        <div className="flex items-start gap-2">
          <Shield className="text-cyan-400 flex-shrink-0 mt-0.5" size={14} />
          <p className="text-xs text-gray-400">
            These alternatives are drugs in the same therapeutic class ({alternatives.therapeutic_class}).
            Always consult a healthcare provider before switching medications.
            Safety scores are based on known interactions in our database.
          </p>
        </div>
      </div>
    </motion.div>
  );
}
