import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Info } from 'lucide-react';

// Anatomical SVG body with proper organ systems
const organData = {
  brain: {
    name: 'Central Nervous System',
    description: 'Neurological effects including confusion, drowsiness, seizures',
    symptoms: ['Dizziness', 'Headache', 'Confusion', 'Drowsiness', 'Seizures'],
    path: 'M 95,25 C 70,25 60,45 60,60 C 60,80 80,90 100,90 C 120,90 140,80 140,60 C 140,45 130,25 105,25 Z'
  },
  heart: {
    name: 'Cardiovascular System',
    description: 'Cardiac effects including arrhythmias, blood pressure changes',
    symptoms: ['Arrhythmia', 'Hypotension', 'Hypertension', 'Tachycardia', 'Bradycardia'],
    path: 'M 100,140 C 85,125 70,130 70,145 C 70,160 100,185 100,185 C 100,185 130,160 130,145 C 130,130 115,125 100,140 Z'
  },
  lungs: {
    name: 'Respiratory System',
    description: 'Respiratory effects including bronchospasm, breathing difficulty',
    symptoms: ['Bronchospasm', 'Dyspnea', 'Respiratory depression', 'Cough'],
    leftPath: 'M 65,120 C 50,125 45,150 50,175 C 55,195 70,200 80,195 C 90,190 90,170 88,150 C 86,130 80,115 65,120 Z',
    rightPath: 'M 135,120 C 150,125 155,150 150,175 C 145,195 130,200 120,195 C 110,190 110,170 112,150 C 114,130 120,115 135,120 Z'
  },
  liver: {
    name: 'Hepatic System',
    description: 'Liver effects including hepatotoxicity, enzyme elevation',
    symptoms: ['Hepatotoxicity', 'Elevated enzymes', 'Jaundice', 'Liver failure'],
    path: 'M 70,200 C 60,205 55,220 60,235 C 65,250 90,255 110,250 C 130,245 145,235 140,215 C 135,200 120,195 100,195 C 85,195 75,195 70,200 Z'
  },
  kidney: {
    name: 'Renal System',
    description: 'Kidney effects including nephrotoxicity, electrolyte imbalance',
    symptoms: ['Nephrotoxicity', 'Electrolyte imbalance', 'Acute kidney injury'],
    leftPath: 'M 65,245 C 55,250 50,265 55,280 C 60,290 75,290 80,280 C 85,270 80,250 70,245 Z',
    rightPath: 'M 135,245 C 145,250 150,265 145,280 C 140,290 125,290 120,280 C 115,270 120,250 130,245 Z'
  },
  gi: {
    name: 'Gastrointestinal System',
    description: 'GI effects including nausea, bleeding, ulceration',
    symptoms: ['Nausea', 'Vomiting', 'GI bleeding', 'Ulceration', 'Diarrhea'],
    path: 'M 80,260 C 70,265 65,290 70,320 C 75,350 85,360 100,360 C 115,360 125,350 130,320 C 135,290 130,265 120,260 C 110,255 90,255 80,260 Z'
  },
  blood: {
    name: 'Hematological System',
    description: 'Blood effects including bleeding risk, clotting disorders',
    symptoms: ['Bleeding', 'Thrombocytopenia', 'Anemia', 'Coagulation disorder'],
    // Blood is represented as a full body overlay
    isOverlay: true
  },
  skin: {
    name: 'Dermatological',
    description: 'Skin effects including rash, photosensitivity',
    symptoms: ['Rash', 'Pruritus', 'Photosensitivity', 'Stevens-Johnson syndrome'],
    isOverlay: true
  }
};

function OrganPath({ system, severity, isHovered, onHover }) {
  const data = organData[system];
  if (!data || data.isOverlay) return null;

  const getColor = (sev) => {
    if (sev > 0.7) return { fill: '#EF4444', glow: 'rgba(239, 68, 68, 0.8)' };
    if (sev > 0.4) return { fill: '#F97316', glow: 'rgba(249, 115, 22, 0.6)' };
    if (sev > 0) return { fill: '#EAB308', glow: 'rgba(234, 179, 8, 0.5)' };
    return { fill: '#334155', glow: 'none' };
  };

  const colors = getColor(severity);
  const isAffected = severity > 0;

  const renderPath = (path, key = 'main') => (
    <motion.path
      key={key}
      d={path}
      fill={colors.fill}
      stroke={isHovered ? '#ffffff' : isAffected ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.1)'}
      strokeWidth={isHovered ? 2 : 1}
      initial={{ opacity: 0.6 }}
      animate={{
        opacity: isAffected ? [0.6, 1, 0.6] : 0.4,
        filter: isAffected ? `drop-shadow(0 0 ${severity * 15}px ${colors.glow})` : 'none'
      }}
      transition={{
        opacity: { duration: 1.5, repeat: isAffected ? Infinity : 0 },
        filter: { duration: 0 }
      }}
      onMouseEnter={() => onHover(system)}
      onMouseLeave={() => onHover(null)}
      style={{ cursor: 'pointer' }}
    />
  );

  if (data.leftPath && data.rightPath) {
    return (
      <>
        {renderPath(data.leftPath, 'left')}
        {renderPath(data.rightPath, 'right')}
      </>
    );
  }

  return renderPath(data.path);
}

function BodyOutline() {
  const commonProps = {
    fill: "none",
    stroke: "#475569",
    strokeWidth: "0.5",
    vectorEffect: "non-scaling-stroke",
    strokeDasharray: "4 2"
  };

  return (
    <>
      <defs>
        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
        </pattern>
      </defs>

      {/* Head outline */}
      <ellipse cx="100" cy="55" rx="35" ry="40" {...commonProps} />

      {/* Neck */}
      <rect x="90" y="90" width="20" height="20" {...commonProps} />

      {/* Torso */}
      <path
        d="M 60,110 L 60,270 Q 60,290 75,300 L 90,300 L 90,380 L 75,380 Q 60,380 60,395 L 60,480 Q 60,490 70,490 L 85,490 Q 95,490 95,480 L 95,380 L 105,380 L 105,480 Q 105,490 115,490 L 130,490 Q 140,490 140,480 L 140,395 Q 140,380 125,380 L 110,380 L 110,300 L 125,300 Q 140,290 140,270 L 140,110 Z"
        {...commonProps}
        fill="url(#grid)"
      />

      {/* Left arm */}
      <path
        d="M 60,115 L 35,130 Q 20,140 15,160 L 10,220 Q 8,235 15,240 L 25,235 Q 30,230 28,215 L 35,160 Q 40,145 55,135 L 60,130 Z"
        {...commonProps}
      />

      {/* Right arm */}
      <path
        d="M 140,115 L 165,130 Q 180,140 185,160 L 190,220 Q 192,235 185,240 L 175,235 Q 170,230 172,215 L 165,160 Q 160,145 145,135 L 140,130 Z"
        {...commonProps}
      />
    </>
  );
}

function InfoPanel({ system, data, severity }) {
  if (!system || !data) return null;

  const getSeverityLabel = (sev) => {
    if (sev > 0.7) return { text: 'Severe', color: 'text-red-400 bg-red-500/10 border-red-500/20' };
    if (sev > 0.4) return { text: 'Moderate', color: 'text-orange-400 bg-orange-500/10 border-orange-500/20' };
    if (sev > 0) return { text: 'Mild', color: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20' };
    return { text: 'Not Affected', color: 'text-theme-muted bg-theme-secondary border-theme' };
  };

  const severityInfo = getSeverityLabel(severity);

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="absolute right-6 top-1/2 -translate-y-1/2 w-72 bg-theme-panel backdrop-blur-xl border border-theme rounded-2xl p-5 shadow-2xl"
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-semibold text-theme-primary">{data.name}</h3>
          <span className={`inline-block mt-1 px-2 py-0.5 text-xs rounded-full border ${severityInfo.color}`}>
            {severityInfo.text}
          </span>
        </div>
        {severity > 0.5 && (
          <AlertTriangle className="w-5 h-5 text-orange-400" />
        )}
      </div>

      <p className="text-sm text-theme-secondary mb-4">{data.description}</p>

      {severity > 0 && (
        <div>
          <h4 className="text-xs text-theme-muted uppercase tracking-wider mb-2">Potential Symptoms</h4>
          <div className="flex flex-wrap gap-1.5">
            {data.symptoms.slice(0, 4).map((symptom, i) => (
              <span
                key={i}
                className="px-2 py-1 bg-theme-secondary rounded-lg text-xs text-theme-primary border border-theme"
              >
                {symptom}
              </span>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default function BodyMapVisualization({ affectedSystems = {}, result }) {
  const [hoveredSystem, setHoveredSystem] = useState(null);

  const hasAnyAffectedSystem = Object.values(affectedSystems).some(v => v > 0);

  // Blood overlay for hematological effects
  const bloodSeverity = affectedSystems.blood || 0;

  return (
    <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
      {/* Main visualization */}
      <div className="relative">
        <svg
          viewBox="0 0 200 500"
          className="w-auto h-[70vh] max-h-[600px]"
          style={{ filter: 'drop-shadow(0 0 20px rgba(0,0,0,0.5))' }}
        >
          {/* Definitions for gradients and filters */}
          <defs>
            <radialGradient id="bloodOverlay" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="rgba(239, 68, 68, 0.3)" />
              <stop offset="100%" stopColor="rgba(239, 68, 68, 0)" />
            </radialGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Body outline */}
          <BodyOutline />

          {/* Blood system overlay */}
          {bloodSeverity > 0 && (
            <motion.ellipse
              cx="100"
              cy="200"
              rx="60"
              ry="120"
              fill="url(#bloodOverlay)"
              animate={{ opacity: [0.3, 0.6, 0.3] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          )}

          {/* Organ systems */}
          {Object.keys(organData).map((system) => (
            <OrganPath
              key={system}
              system={system}
              severity={affectedSystems[system] || 0}
              isHovered={hoveredSystem === system}
              onHover={setHoveredSystem}
            />
          ))}
        </svg>

        {/* Info panel */}
        <AnimatePresence>
          {hoveredSystem && organData[hoveredSystem] && (
            <InfoPanel
              system={hoveredSystem}
              data={organData[hoveredSystem]}
              severity={affectedSystems[hoveredSystem] || 0}
            />
          )}
        </AnimatePresence>
      </div>

      {/* Legend */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-6">
        <div className="flex items-center gap-4 px-4 py-2 bg-theme-panel backdrop-blur-sm rounded-xl border border-theme">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-slate-500" />
            <span className="text-xs text-theme-muted">Normal</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <span className="text-xs text-theme-muted">Mild</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-orange-500" />
            <span className="text-xs text-theme-muted">Moderate</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs text-theme-muted">Severe</span>
          </div>
        </div>
      </div>

      {/* Empty state */}
      {!hasAnyAffectedSystem && !result && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <Info className="w-12 h-12 text-theme-dim mx-auto mb-4" />
            <p className="text-sm text-theme-muted">Run analysis to see affected body systems</p>
          </div>
        </div>
      )}

      {/* No interaction overlay */}
      {result?.severity === 'no_interaction' && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center px-6 py-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl">
            <div className="w-12 h-12 rounded-full bg-emerald-500/20 flex items-center justify-center mx-auto mb-3">
              <svg className="w-6 h-6 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-sm text-emerald-600 font-medium">No Significant Effects</p>
            <p className="text-xs text-theme-muted mt-1">No organ systems are expected to be affected</p>
          </div>
        </div>
      )}

      {/* Hover instruction */}
      {hasAnyAffectedSystem && (
        <div className="absolute top-6 left-1/2 -translate-x-1/2 px-4 py-2 bg-theme-panel backdrop-blur-sm rounded-lg border border-theme text-xs text-theme-muted">
          Hover over highlighted organs for details
        </div>
      )}
    </div>
  );
}
