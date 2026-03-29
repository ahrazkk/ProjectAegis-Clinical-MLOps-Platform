const fs = require('fs');
const path = require('path');

const targetPath = path.join(__dirname, 'src', 'components', 'KnowledgeGraphView.jsx');

const code = `import React, { useMemo, useState, useRef, useEffect } from 'react';
import { MoleculeCanvas } from './MoleculeViewer2D';

// Math for a Pointy-Topped Hexagon Grid
function hexToPixel(q, r, size) {
  const x = size * Math.sqrt(3) * (q + r / 2);
  const y = size * (3 / 2) * r;
  return { x, y };
}

function getHexPolygon(cx, cy, size) {
  const points = [];
  for (let i = 0; i < 6; i++) {
    const angle_deg = 60 * i - 30; // Pointy topped offset
    const angle_rad = Math.PI / 180 * angle_deg;
    points.push(\`\${cx + size * Math.cos(angle_rad)},\${cy + size * Math.sin(angle_rad)}\`);
  }
  return points.join(" ");
}

export default function KnowledgeGraphView({ drugs = [], result, polypharmacyResult, isMobile = false }) {
  const containerRef = useRef(null);
  const [dim, setDim] = useState({ w: 1000, h: 600 });
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [hoveredNode, setHoveredNode] = useState(null);
  const dragInfo = useRef({ isDragging: false, startX: 0, startY: 0, startPanX: 0, startPanY: 0 });

  useEffect(() => {
    if (containerRef.current) {
      setDim({ w: containerRef.current.clientWidth || 1000, h: containerRef.current.clientHeight || 600 });
    }
    const handleResize = () => {
      if (containerRef.current) {
        setDim({ w: containerRef.current.clientWidth || 1000, h: containerRef.current.clientHeight || 600 });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Handle Zoom Input (Mouse wheel & pinch-to-zoom)
  const handleWheel = (e) => {
    e.preventDefault();
    const zoomSensitivity = 0.002;
    const delta = -e.deltaY;
    setZoom((prevZoom) => {
      const newZoom = prevZoom + (delta * zoomSensitivity);
      return Math.min(Math.max(0.2, newZoom), 4);
    });
  };

  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.addEventListener("wheel", handleWheel, { passive: false });
    }
    return () => {
      if (el) {
        el.removeEventListener("wheel", handleWheel);
      }
    };
  }, []);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.2, 4));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.2, 0.2));

  // Interaction handlers for smooth infinite panning
  const handlePointerDown = (e) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    dragInfo.current = {
      isDragging: true,
      startX: e.clientX,
      startY: e.clientY,
      startPanX: pan.x,
      startPanY: pan.y
    };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  
  const handlePointerMove = (e) => {
    if (!dragInfo.current.isDragging) return;
    setPan({
      x: dragInfo.current.startPanX + (e.clientX - dragInfo.current.startX),
      y: dragInfo.current.startPanY + (e.clientY - dragInfo.current.startY)
    });
  };
  
  const handlePointerUp = (e) => {
    dragInfo.current.isDragging = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const HEX_SIZE = 95;

  const safeDrugsList = Array.isArray(drugs) ? drugs : [];

  const { nodes, edges, gridTiles, directPaths, hasAlert } = useMemo(() => {
    const gTiles = [];
    const GRID_RAD = 12;
    for (let q = -GRID_RAD; q <= GRID_RAD; q++) {
      let r1 = Math.max(-GRID_RAD, -q - GRID_RAD);
      let r2 = Math.min(GRID_RAD, -q + GRID_RAD);
      for (let r = r1; r <= r2; r++) {
        gTiles.push({ q, r, ...hexToPixel(q, r, HEX_SIZE) });
      }
    }

    const nList = [];
    const eList = [];
    const dPaths = [];

    // Fixed axial layouts to ensure space between multiple drugs (maintain 1-hex gap where possible)
    const POSITIONS = [
      {q: -2, r: 0},   // Node 0
      {q: 2, r: 0},    // Node 1
      {q: 0, r: 3},    // Node 2
      {q: 0, r: -3},   // Node 3
      {q: -3, r: 3},   // Node 4
      {q: 3, r: -3},   // Node 5
      {q: 3, r: 3},    // Node 6
      {q: -3, r: -3}   // Node 7
    ];

    const COLORS = ['#00d4ff', '#f97316', '#a855f7', '#fbbf24', '#3b82f6', '#10b981'];

    const addNode = (id, q, r, drugData, color) => {
      const pos = hexToPixel(q, r, HEX_SIZE);
      nList.push({ id, q, r, x: pos.x, y: pos.y, color, ...drugData });
    };

    // Plot REAL user selected drugs (safely)
    if (safeDrugsList && safeDrugsList.length > 0) {
      safeDrugsList.forEach((d, idx) => {
        if (!d) return; // Skip nulls
        const pos = POSITIONS[idx % POSITIONS.length];
        const color = COLORS[idx % COLORS.length];
        const safeName = d.name || \`Drug \${idx}\`;
        addNode(safeName, pos.q, pos.r, d, color);
      });
    }

    // Map REAL Backend Interactions
    let interactions = [];
    let isAlertActive = false;

    try {
      if (polypharmacyResult && Array.isArray(polypharmacyResult.interactions)) {
        interactions = polypharmacyResult.interactions;
      } else if (result && result.risk_level && !['Safe', 'Tolerable', 'No Interaction'].includes(result.risk_level)) {
        interactions = [{
          drug_a: result.drug_a || safeDrugsList[0]?.name || '',
          drug_b: result.drug_b || safeDrugsList[1]?.name || '',
          risk_score: result.risk_score,
          severity: result.severity,
          mechanism_hypothesis: result.mechanism_hypothesis
        }];
      }
    } catch (err) {
      console.error("Error parsing interactions payload:", err);
    }

    // Process drawn interactions between nodes smoothly
    interactions.forEach(intx => {
      if (!intx) return;
      
      const srcName = String(intx.drug_a || "").toLowerCase().trim();
      const tgtName = String(intx.drug_b || "").toLowerCase().trim();
      
      const sourceNode = nList.find(n => n.name && String(n.name).toLowerCase().trim() === srcName);
      const targetNode = nList.find(n => n.name && String(n.name).toLowerCase().trim() === tgtName);

      if (sourceNode && targetNode) {
        isAlertActive = true;
        // Aesthetic check: if it's the exact [-2,0] to [2,0] path, draw the cool visual bypass route
        if (
          (sourceNode.q === -2 && sourceNode.r === 0 && targetNode.q === 2 && targetNode.r === 0) ||
          (sourceNode.q === 2 && sourceNode.r === 0 && targetNode.q === -2 && targetNode.r === 0)
        ) {
           const sq = sourceNode.q === -2 ? -2 : 2;
           const tq = targetNode.q === 2 ? 2 : -2;

           const directPathTiles = [
             { q: sq === -2 ? -1 : 1, r: -1 },
             { q: 0, r: -1 },
             { q: tq === 2 ? 1 : -1, r: -1 },
             { q: tq === 2 ? 1 : -1, r: 0 }
           ];
           
           const polylineCoords = [
             { q: sq, r: 0 },
             ...directPathTiles,
             { q: tq, r: 0 }
           ].map(coord => hexToPixel(coord.q, coord.r, HEX_SIZE));

           dPaths.push({
              tiles: directPathTiles.map(coord => hexToPixel(coord.q, coord.r, HEX_SIZE)),
              polyline: polylineCoords.map(p => \`\${p.x},\${p.y}\`).join(" "),
              info: intx
           });
        } else {
           // Direct straight line for arbitrary N-way network layout
           eList.push({
             id: \`\${sourceNode.id}-\${targetNode.id}\`,
             s: sourceNode,
             t: targetNode,
             color: '#ef4444',
             glow: 'rgba(239, 68, 68, 0.15)',
             info: intx
           });
        }
      }
    });

    return { nodes: nList, edges: eList, gridTiles: gTiles, directPaths: dPaths, hasAlert: isAlertActive };
    
  }, [safeDrugsList, result, polypharmacyResult]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full relative bg-[#06060c] overflow-hidden cursor-grab active:cursor-grabbing rounded-xl border border-[rgba(0,212,255,0.2)]"
      style={{ minHeight: '600px', touchAction: 'none' }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onMouseLeave={() => setHoveredNode(null)}
    >
      {/* Zoom UI Controls */}
      <div className="absolute top-4 right-4 z-50 flex flex-col gap-2 bg-[#0a0a0f]/80 p-2 rounded-lg border border-[rgba(255,255,255,0.1)] backdrop-blur-md shadow-lg">
        <button 
          onClick={(e) => { e.stopPropagation(); handleZoomIn(); }}
          className="w-8 h-8 flex items-center justify-center bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.1)] rounded text-white font-bold transition-colors"
          title="Zoom In"
        >
          +
        </button>
        <button 
          onClick={(e) => { e.stopPropagation(); handleZoomOut(); }}
          className="w-8 h-8 flex items-center justify-center bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.1)] rounded text-white font-bold transition-colors"
          title="Zoom Out"
        >
          -
        </button>
        <div className="text-[10px] text-center text-gray-400 mt-1 font-mono">
          {Math.round(zoom * 100)}%
        </div>
      </div>

      <svg width="100%" height="100%" style={{ overflow: 'visible' }}>
        <g transform={\`translate(\${dim.w / 2 + pan.x}, \${dim.h / 2 + pan.y}) scale(\${zoom})\`}>
          
          {/* LAYER 1: Full seamless honeycomb grid */}
          {gridTiles.map(t => (
            <polygon
              key={\`bg-\${t.q}-\${t.r}\`}
              points={getHexPolygon(t.x, t.y, HEX_SIZE)}
              fill="transparent"
              stroke="rgba(255,255,255,0.04)"
              strokeWidth="1.5"
            />
          ))}

          {/* LAYER 2: Polypharmacy straight lines */}
          {edges.map(e => (
            <g key={\`edge-\${e.id}\`} className="pointer-events-none">
              <line
                x1={e.s.x} y1={e.s.y}
                x2={e.t.x} y2={e.t.y}
                stroke={e.color}
                strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray="8 8"
                strokeOpacity="0.8"
                style={{ 
                  filter: \`drop-shadow(0 0 10px \${e.color})\`,
                  animation: 'dashMove 20s linear infinite'
                }}
              />
            </g>
          ))}

          {/* LAYER 3: DIRECT CONNECTION HIGHLIGHT (Bypass route) */}
          {directPaths.map((pathData, pIdx) => (
            <g key={\`direct-path-\${pIdx}\`} className="pointer-events-none">
              {pathData.tiles.map((pos, i) => (
                 <polygon
                   key={\`direct-hex-\${pIdx}-\${i}\`}
                   points={getHexPolygon(pos.x, pos.y, HEX_SIZE)}
                   fill="rgba(239, 68, 68, 0.08)"
                   stroke="#ef4444"
                   strokeWidth="1"
                   strokeDasharray="4 4"
                   strokeOpacity="0.6"
                   style={{
                     animation: \`pulseGlow 2s infinite \${i * 0.2}s\`
                   }}
                 />
              ))}
              
              <polyline 
                points={pathData.polyline}
                fill="none"
                stroke="#ef4444"
                strokeWidth="4"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeDasharray="12 8"
                style={{ 
                  filter: 'drop-shadow(0 0 10px rgba(239, 68, 68, 0.8))',
                  animation: 'dashMove 20s linear infinite'
                }}
              />
            </g>
          ))}

          {/* LAYER 4: The solid node hexes (Only REAL drugs) */}
          {nodes.map(n => (
            <g 
              key={n.id} 
              onMouseEnter={() => setHoveredNode(n)}
              className="cursor-pointer transition-transform duration-300 hover:scale-[1.05]"
              style={{ transformOrigin: \`\${n.x}px \${n.y}px\` }}
            >
              <polygon
                points={getHexPolygon(n.x, n.y, HEX_SIZE)}
                fill="#0a0a0f"
                stroke={n.color}
                strokeWidth="4"
                style={{ filter: \`drop-shadow(0px 0px 10px \${n.color}80)\` }}
              />

              <foreignObject x={n.x - 65} y={n.y - 75} width="130" height="130" className="pointer-events-none">
                <div className="w-full h-full flex flex-col items-center justify-center">
                  {n.smiles && <MoleculeCanvas smiles={n.smiles} width={130} height={130} />}
                </div>
              </foreignObject>

              <text
                x={n.x} y={n.y + 65}
                textAnchor="middle"
                fill="#fff"
                fontSize="13"
                fontWeight="bold"
                letterSpacing="0.05em"
                className="pointer-events-none"
              >
                {String(n.name || '').substring(0, 16)}
              </text>
            </g>
          ))}
        </g>
      </svg>

      {/* DYNAMIC HOVER TOOLTIP OVERLAY */}
      {hoveredNode && (
        <div 
          className="absolute z-50 p-4 rounded-xl shadow-2xl backdrop-blur-xl border pointer-events-none transition-all duration-200 ease-out"
          style={{
            left: (dim.w / 2) + pan.x + (hoveredNode.x * zoom) + (80 * zoom),
            top: (dim.h / 2) + pan.y + (hoveredNode.y * zoom) - (80 * zoom),
            backgroundColor: 'rgba(10,12,20,0.95)',
            borderColor: hoveredNode.color,
            boxShadow: \`0 0 30px \${hoveredNode.color}40\`,
            width: '280px',
            transform: \`scale(\${Math.max(0.8, Math.min(1.2, zoom))})\`,
            transformOrigin: 'top left'
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: hoveredNode.color, boxShadow: \`0 0 8px \${hoveredNode.color}\` }}></div>
            <h3 className="text-lg font-bold text-white leading-none tracking-wide">{String(hoveredNode.name || 'Unknown')}</h3>
          </div>
          
          <div className="text-xs font-mono uppercase tracking-widest text-opacity-80 mb-3" style={{ color: hoveredNode.color }}>
            {hoveredNode.category || hoveredNode.drug_class || hoveredNode.therapeutic_class || "CLINICAL PROFILE"}
          </div>
          
          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            {hoveredNode.description || hoveredNode.pharmacology || "Active substance modeled in the Graph visualization environment."}
          </p>
          
          <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-2">
            <div>
              <span className="text-xs text-gray-500 uppercase tracking-wider block mb-1">Graph Node Status</span>
              <span className="text-sm text-white font-medium">Selected Search Target</span>
            </div>
            {hoveredNode.smiles && (
               <div>
                  <span className="text-xs text-gray-500 uppercase tracking-wider block mt-2 mb-1">SMILES Topology</span>
                  <span className="text-[10px] text-gray-400 font-mono break-all">
                    {String(hoveredNode.smiles).substring(0,80)}{String(hoveredNode.smiles).length > 80 ? '...' : ''}
                  </span>
               </div>
            )}
          </div>
        </div>
      )}

      {/* Floating HUD */}
      <div className="absolute bottom-4 left-4 pointer-events-none bg-[#0a0a0f]/80 px-4 py-3 rounded-lg border border-[rgba(255,255,255,0.1)] backdrop-blur-md shadow-lg">
        <div className="flex flex-col gap-2">
          <div className="text-xs font-mono text-gray-400 mb-1 border-b border-white/10 pb-2">NETWORK LEGEND</div>
          
          {safeDrugsList.map((d, i) => {
             if (!d || !d.name) return null;
             return (
               <div key={i} className="flex items-center gap-2">
                 <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length], boxShadow: \`0 0 8px \${COLORS[i % COLORS.length]}\`}}></div>
                 <span className="text-[10px] text-gray-300 tracking-widest font-bold top-[1px]">{String(d.name).toUpperCase()}</span>
               </div>
             )
          })}

          {/* Dynamic interaction red line toggle */}
          {hasAlert && (
            <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-700 transition-opacity duration-500">
              <div className="w-3 h-2 bg-red-500 shadow-[0_0_8px_#ef4444]"></div>
              <span className="text-[10px] text-red-500 tracking-widest font-bold top-[1px] animate-pulse">DETECTED INTERACTION</span>
            </div>
          )}
        </div>
      </div>

      <style dangerouslySetInnerHTML={{__html: \`
        @keyframes dashMove {
          to { stroke-dashoffset: -400; }
        }
        @keyframes pulseGlow {
          0%, 100% { fill-opacity: 0.08; stroke-opacity: 0.4; }
          50% { fill-opacity: 0.2; stroke-opacity: 0.8; stroke-width: 2px; }
        }
      \`}} />
    </div>
  );
}
`;

fs.writeFileSync(targetPath, code);
console.log('Successfully written defensive V5 of KnowledgeGraphView.jsx. Resolves blank screen crash from null payload variables!');
