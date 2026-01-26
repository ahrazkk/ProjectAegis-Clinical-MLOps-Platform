import React, { useRef, useEffect, useState, useId } from 'react';
import SmilesDrawer from 'smiles-drawer';

// 2D Organic Chemistry Skeletal Formula Viewer
// Renders molecules as they appear in chemistry textbooks

const MoleculeCanvas = ({ smiles, name, width = 300, height = 250, theme = 'dark', isHighlighted = false }) => {
  const canvasRef = useRef(null);
  const [error, setError] = useState(null);
  // Create unique ID for canvas (needed by SmilesDrawer)
  const uniqueId = useId().replace(/:/g, 'mol');
  const canvasId = `smiles-canvas-${uniqueId}`;

  useEffect(() => {
    if (!smiles || typeof smiles !== 'string' || !canvasRef.current) {
      console.warn('MoleculeCanvas: Invalid or missing smiles data:', smiles);
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Clear canvas first
    ctx.clearRect(0, 0, width, height);

    // Configure SmilesDrawer options for organic chemistry look
    const options = {
      width: width,
      height: height - 40, // Leave room for label
      bondThickness: 1.5,
      bondLength: 25,
      shortBondLength: 0.85,
      bondSpacing: 4,
      atomVisualization: 'default', // Shows heteroatoms, hides carbons
      isomeric: true,
      debug: false,
      terminalCarbons: false,
      explicitHydrogens: false,
      overlapSensitivity: 0.42,
      overlapResolutionIterations: 1,
      compactDrawing: true,
      fontSizeLarge: 12,
      fontSizeSmall: 8,
      padding: 20,
      // Dark theme colors
      themes: {
        dark: {
          C: '#E2E8F0',
          O: '#F87171',
          N: '#60A5FA',
          S: '#FBBF24',
          P: '#FB923C',
          F: '#4ADE80',
          Cl: '#34D399',
          Br: '#DC2626',
          I: '#A855F7',
          H: '#94A3B8',
          BACKGROUND: 'transparent',
          BOND: '#94A3B8'
        }
      }
    };

    // Helper function to draw the label after molecule
    const drawLabel = () => {
      ctx.font = 'bold 14px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = isHighlighted ? '#60A5FA' : '#E2E8F0';
      ctx.fillText(name, width / 2, height - 12);
    };

    // Helper function to draw error state
    const drawError = () => {
      ctx.font = '12px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = '#64748B';
      ctx.fillText('Structure unavailable', width / 2, height / 2);
      drawLabel();
    };

    try {
      // SmiDrawer handles both parsing and drawing in one step
      // It accepts: (smiles, canvasSelector, theme, callback) - selector must be a CSS selector string
      const drawer = new SmilesDrawer.SmiDrawer(options);

      // draw() needs CSS selector string (e.g., '#myCanvas'), not DOM element
      drawer.draw(smiles, '#' + canvasId, theme, () => {
        // After structure drawn, add the label
        drawLabel();
        setError(null);
      });

    } catch (err) {
      console.error('SmilesDrawer error:', err);
      setError('Could not render structure');
      drawError();
    }
  }, [smiles, name, width, height, theme, isHighlighted, canvasId]);

  return (
    <div className={`relative transition-all duration-300 ${isHighlighted ? 'scale-105' : ''}`}>
      <canvas
        id={canvasId}
        ref={canvasRef}
        width={width}
        height={height}
        className="rounded-xl"
        style={{ background: 'transparent' }}
      />
      {isHighlighted && (
        <div
          className="absolute inset-0 rounded-xl pointer-events-none"
          style={{
            boxShadow: '0 0 30px rgba(96, 165, 250, 0.3)',
            border: '1px solid rgba(96, 165, 250, 0.2)'
          }}
        />
      )}
    </div>
  );
};


// Interaction arrow component
const InteractionArrow = ({ riskLevel, className = '' }) => {
  const riskColors = {
    critical: { primary: '#EF4444', glow: 'rgba(239, 68, 68, 0.4)' },
    high: { primary: '#F97316', glow: 'rgba(249, 115, 22, 0.4)' },
    medium: { primary: '#EAB308', glow: 'rgba(234, 179, 8, 0.4)' },
    low: { primary: '#22C55E', glow: 'rgba(34, 197, 94, 0.4)' }
  };

  const colors = riskColors[riskLevel] || riskColors.low;

  return (
    <div className={`flex flex-col items-center justify-center ${className}`}>
      <svg width="120" height="60" viewBox="0 0 120 60" className="overflow-visible">
        {/* Glow effect */}
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <linearGradient id={`arrowGradient-${riskLevel}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={colors.primary} stopOpacity="0.3" />
            <stop offset="50%" stopColor={colors.primary} stopOpacity="1" />
            <stop offset="100%" stopColor={colors.primary} stopOpacity="0.3" />
          </linearGradient>
        </defs>

        {/* Double-headed arrow */}
        <g filter="url(#glow)">
          {/* Left arrow head */}
          <path
            d="M 10 30 L 25 20 L 25 26 L 95 26 L 95 20 L 110 30 L 95 40 L 95 34 L 25 34 L 25 40 Z"
            fill={`url(#arrowGradient-${riskLevel})`}
            stroke={colors.primary}
            strokeWidth="1"
          />
        </g>

        {/* Animated particles */}
        <circle r="2" fill={colors.primary}>
          <animate
            attributeName="cx"
            values="20;100"
            dur="1.5s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="cy"
            values="30;30"
            dur="1.5s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0;1;0"
            dur="1.5s"
            repeatCount="indefinite"
          />
        </circle>
        <circle r="2" fill={colors.primary}>
          <animate
            attributeName="cx"
            values="100;20"
            dur="1.5s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="cy"
            values="30;30"
            dur="1.5s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0;1;0"
            dur="1.5s"
            repeatCount="indefinite"
          />
        </circle>
      </svg>

      <span
        className="text-xs font-semibold mt-1 px-2 py-0.5 rounded-full"
        style={{
          backgroundColor: `${colors.primary}20`,
          color: colors.primary,
          border: `1px solid ${colors.primary}40`
        }}
      >
        INTERACTION
      </span>
    </div>
  );
};

// Main 2D Viewer Component
export default function MoleculeViewer2D({ drugs = [], result, isMobile = false }) {
  const hasResult = result && result.severity !== 'no_interaction';
  const riskLevel = result?.risk_level || 'low';

  if (drugs.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-24 h-24 mx-auto mb-6">
            <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 animate-pulse" />
            <div className="absolute inset-2 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
              <svg className="w-12 h-12 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                />
              </svg>
            </div>
          </div>
          <p className="text-slate-400 font-medium mb-2">No molecules to display</p>
          <p className="text-sm text-slate-600">Search and add drugs to view their chemical structures</p>
        </div>
      </div>
    );
  }

  // Mobile-responsive canvas sizes
  const canvasWidth = isMobile ? (drugs.length === 1 ? 280 : 160) : (drugs.length === 1 ? 400 : 320);
  const canvasHeight = isMobile ? 200 : 280;

  return (
    <div className="w-full h-full relative flex flex-col overflow-hidden">
      {/* Main visualization area */}
      <div className={`flex-1 flex items-center justify-center ${isMobile ? 'p-2' : 'p-8'} overflow-hidden`}>
        <div className={`flex items-center justify-center ${isMobile ? 'gap-1 flex-wrap' : 'gap-4'} ${drugs.length > 2 ? 'flex-wrap' : ''}`}>
          {drugs.map((drug, index) => (
            <React.Fragment key={drug.drugbank_id || drug.name || index}>
              <MoleculeCanvas
                smiles={drug.smiles}
                name={drug.name}
                width={canvasWidth}
                height={canvasHeight}
                theme="dark"
                isHighlighted={hasResult}
              />

              {/* Show interaction arrow between molecules */}
              {index < drugs.length - 1 && drugs.length === 2 && (
                <InteractionArrow
                  riskLevel={hasResult ? riskLevel : 'low'}
                  className={`${hasResult ? 'opacity-100' : 'opacity-30'} ${isMobile ? 'scale-75' : ''}`}
                />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Polypharmacy grid for 3+ drugs */}
      {drugs.length > 2 && hasResult && !isMobile && (
        <div className="absolute bottom-20 left-1/2 -translate-x-1/2">
          <div className="px-4 py-2 bg-slate-800/80 backdrop-blur-sm rounded-xl border border-white/10">
            <p className="text-sm text-slate-300 text-center">
              <span className="text-amber-400 font-semibold">{drugs.length} drugs</span> —
              Check Knowledge Graph tab for interaction network
            </p>
          </div>
        </div>
      )}

      {/* Legend - hidden on mobile */}
      {!isMobile && (
        <div className="absolute bottom-4 left-4 flex items-center gap-2.5">
          {[
            { color: '#F87171', label: 'Oxygen (O)' },
            { color: '#60A5FA', label: 'Nitrogen (N)' },
            { color: '#FBBF24', label: 'Sulfur (S)' },
          ].map(({ color, label }) => (
            <div
              key={label}
              className="flex items-center gap-2 px-3 py-2 bg-slate-900/80 backdrop-blur-sm rounded-xl border border-white/5"
            >
              <span className="text-sm font-bold" style={{ color }}>{label.split(' ')[1]}</span>
              <span className="text-xs text-slate-500">{label.split(' ')[0]}</span>
            </div>
          ))}
          <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/80 backdrop-blur-sm rounded-xl border border-white/5">
            <span className="text-sm text-slate-400">—</span>
            <span className="text-xs text-slate-500">C-C bonds (carbons hidden)</span>
          </div>
        </div>
      )}

      {/* View mode indicator */}
      <div className={`absolute ${isMobile ? 'top-2 right-2' : 'top-4 right-4'} px-2 py-1 bg-blue-500/10 backdrop-blur-sm rounded-lg border border-blue-500/20`}>
        <p className={`${isMobile ? 'text-[10px]' : 'text-xs'} text-blue-400 font-medium flex items-center gap-1`}>
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
          2D Skeletal Formula
        </p>
      </div>

      {/* Controls hint - simplified for mobile */}
      <div className={`absolute ${isMobile ? 'bottom-2 right-2 left-2' : 'bottom-4 right-4'} px-3 py-1.5 bg-slate-900/80 backdrop-blur-sm rounded-xl border border-white/5`}>
        <p className={`${isMobile ? 'text-[9px] text-center' : 'text-xs'} text-slate-500`}>
          {isMobile ? 'Organic chemistry notation' : 'Standard organic chemistry notation • Hydrogens implicit'}
        </p>
      </div>
    </div>
  );
}
