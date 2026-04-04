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

  // Use risk colors mapping to match the tactical theme
  const borderClass = score >= 75 ? 'border-risk-low' : score >= 50 ? 'border-risk-medium' : 'border-risk-high';
  const textClass = score >= 75 ? 'text-risk-low' : score >= 50 ? 'text-risk-medium' : 'text-risk-high';
  const bgClass = score >= 75 ? 'bg-risk-low/10' : score >= 50 ? 'bg-risk-medium/10' : 'bg-risk-high/10';

  return (
    <span className={`inline-flex shrink-0 items-center justify-center gap-1.5 px-2 py-1 text-[10px] uppercase font-mono tracking-widest
      ${bgClass} ${textClass} border ${borderClass} shadow-sm min-w-[70px]`}>
      {icon}
      {score}%
    </span>
  );
}

// Alternative drug card
function AlternativeCard({ alternative, rank, originalDrug, interactingDrug }) {
  const [expanded, setExpanded] = useState(false);
  
  const severityColors = {
    no_interaction: 'risk-low',
    minor: 'risk-low',
    moderate: 'risk-medium',
    severe: 'risk-high',
    unknown: 'theme-dim'
  };

  const themeColor = severityColors[alternative.interaction_severity] || 'theme-dim';

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: rank * 0.1 }}
      className={`p-4 border transition-all cursor-pointer bg-theme-panel shadow-sm
        ${alternative.is_safer 
          ? 'border-risk-low hover:border-risk-low/70'
          : `border-${themeColor} hover:opacity-80`
        }`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3 min-w-0">
          {/* Rank badge */}
          <div className={`w-8 h-8 shrink-0 flex items-center justify-center font-mono text-[10px] border
            ${rank === 0 ? 'bg-risk-low text-theme-base border-risk-low' :
              rank === 1 ? 'bg-risk-low/80 text-theme-base border-risk-low' :
              rank === 2 ? 'bg-risk-low/60 text-theme-base border-risk-low' :
              'bg-theme-panel text-theme-muted border-theme'}`}>
            {rank + 1}
          </div>
          
          <div className="min-w-0">
            <div className="flex flex-col items-start gap-1.5 mt-0.5">
              <h4 className="font-mono text-[11px] uppercase tracking-widest text-theme-primary leading-snug">{alternative.name}</h4>
              <div className="flex items-center gap-2 flex-wrap">
                {alternative.is_safer && (
                  <span className="flex shrink-0 items-center gap-1 px-2 py-0.5 border border-risk-low bg-risk-low/10 text-risk-low text-[9px] uppercase font-mono tracking-widest">
                    <Sparkles size={10} />
                    Safer
                  </span>
                )}
                {alternative.drugbank_id && (
                  <span className="text-[9px] uppercase font-mono tracking-widest text-theme-dim">
                    {alternative.drugbank_id}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0 mt-1">
          <SafetyBadge score={alternative.safety_score} />
          <ChevronRight 
            size={16} 
            className={`text-theme-dim transition-transform shrink-0 ${expanded ? 'rotate-90' : ''}`}
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
            className="mt-4 pt-4 border-t border-theme space-y-3"
          >
            {/* Interaction with the problematic drug */}
            {interactingDrug && (
              <div className={`p-3 bg-${themeColor}/10 border border-${themeColor}/30`}>
                <div className="flex items-center gap-2 text-[10px] uppercase font-mono tracking-widest">
                  <span className="text-theme-dim">Interaction with {interactingDrug}:</span>
                  <span className={`font-semibold text-${themeColor}`}>
                    {alternative.interaction_severity?.replace('_', ' ') || 'Unknown'}
                  </span>
                </div>
                {alternative.mechanism && (
                  <p className="text-[9px] font-mono tracking-widest text-theme-muted mt-2 leading-relaxed">{alternative.mechanism}</p>
                )}
              </div>
            )}

            {/* SMILES indicator */}
            {alternative.smiles && (
               <div className="flex items-center gap-1 text-[9px] uppercase font-mono tracking-widest text-theme-accent">
                <Atom size={12} />
                Molecular structure available
              </div>
            )}

            {/* Comparison summary */}
            <div className="flex items-center gap-2 text-[9px] uppercase font-mono tracking-widest text-theme-dim">
              <span className="text-theme-muted">{originalDrug}</span>
              <ArrowRight size={12} className="text-theme-accent" />
              <span className="text-theme-accent">{alternative.name}</span>
              {alternative.is_safer && (
                <span className="text-risk-low ml-2">
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
      <div className="flex items-center justify-center p-6 bg-theme-panel border border-theme">
        <Loader2 className="animate-spin text-theme-accent" size={16} />
        <span className="ml-2 text-[10px] uppercase font-mono tracking-widest text-theme-muted">Finding alternatives...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-risk-high/10 border border-risk-high/50">
        <p className="text-[10px] uppercase font-mono tracking-widest text-risk-high">Failed to find alternatives: {error}</p>
      </div>
    );
  }

  if (!alternatives) return null;

  // No therapeutic class found
  if (!alternatives.therapeutic_class) {
    return (
      <div className="p-4 bg-theme-panel border border-theme">
        <div className="flex items-center gap-2 text-theme-muted">
          <AlertCircle size={14} />
          <span className="text-[10px] uppercase font-mono tracking-widest text-theme-muted">No therapeutic class found for {drugName}</span>
        </div>
        <p className="text-[9px] uppercase tracking-widest text-theme-dim font-mono mt-2">
          Unable to suggest alternatives without knowing the drug's class.
        </p>
      </div>
    );
  }

  // No alternatives found
  if (!alternatives.alternatives?.length) {
    return (
      <div className="p-4 bg-theme-panel border border-theme">
        <div className="flex items-center gap-2 text-theme-muted flex-wrap">
          <AlertCircle size={14} />
          <span className="text-[10px] uppercase font-mono tracking-widest text-theme-muted leading-relaxed">No alternatives found in <br className="hidden md:block"/>{alternatives.therapeutic_class} class</span>
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
        className="p-3 bg-risk-low/5 border border-risk-low/30"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Lightbulb className="text-risk-low" size={14} />
            <span className="text-[10px] font-mono tracking-widest uppercase text-theme-muted">
              {saferCount} safer alternatives found in 
              <span className="text-risk-low ml-1">{alternatives.therapeutic_class}</span>
            </span>
          </div>
          {onSelectAlternative && alternatives.alternatives[0] && (
            <button
              onClick={() => onSelectAlternative(alternatives.alternatives[0])}
              className="text-[10px] uppercase font-mono tracking-widest text-theme-accent hover:text-theme-secondary transition-colors flex items-center gap-1"
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
          <div className="p-2 bg-theme-panel border border-theme">
            <Lightbulb className="text-theme-accent" size={20} />
          </div>
          <div>
            <h3 className="font-mono text-[11px] uppercase tracking-widest text-theme-primary">Therapeutic Alternatives</h3>
            <p className="text-[10px] uppercase font-mono tracking-widest text-theme-dim mt-1">
              Same class as <span className="text-theme-accent">{alternatives.drug}</span>: 
              <br className="hidden md:block"/><span className="text-theme-muted">{alternatives.therapeutic_class}</span>
            </p>
          </div>
        </div>

        <div className="text-right flex flex-col items-end">
          <span className="text-lg font-mono font-bold text-risk-low">{saferCount}</span>
          <span className="text-[10px] uppercase font-mono tracking-widest text-theme-dim ml-1">safer<br/>options</span>
        </div>
      </div>

      {/* Context banner */}
      {interactingWith && severity && (
        <div className="p-3 bg-risk-high/10 border border-risk-high/30">
          <div className="flex items-center gap-2">
            <XCircle className="text-risk-high" size={14} />
            <span className="text-[10px] uppercase font-mono tracking-widest text-theme-muted leading-relaxed">
              <span className="font-semibold text-theme-primary">{alternatives.drug}</span>
              {' + '}
              <span className="font-semibold text-theme-primary">{interactingWith}</span>
              {' has a '}
              <span className="text-risk-high font-semibold capitalize">{severity}</span>
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
      <div className="p-3 bg-theme-panel border border-theme">
        <div className="flex items-start gap-2">
          <Shield className="text-theme-accent flex-shrink-0 mt-0.5" size={14} />
          <p className="text-[9px] font-mono tracking-widest uppercase text-theme-dim leading-relaxed">
            These alternatives are drugs in the same therapeutic class ({alternatives.therapeutic_class}).
            Always consult a healthcare provider before switching medications.
          </p>
        </div>
      </div>
    </motion.div>
  );
}
