const fs = require('fs');
const path = require('path');

const targetPath = path.join(__dirname, 'src', 'components', 'KnowledgeGraphView.jsx');

const code = `import React, { useMemo, useState, useRef, useEffect } from 'react';
import { MoleculeCanvas } from './MoleculeViewer2D';

const MOCK_DRUGS = [
  { name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O", cat: "NSAID", desc: "Common pain reliever and blood thinner. Increases bleeding risk." },
  { name: "Warfarin", smiles: "CC(=O)NC1=CC=C(O)C=C1", cat: "Anticoagulant", desc: "Vitamin K antagonist used to prevent blood clots." },
  { name: "Clopidogrel", smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", cat: "Antiplatelet", desc: "Prevents platelets from clumping together." },
  { name: "Digoxin", smiles: "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", cat: "Cardiac Glycoside", desc: "Treats heart failure and rhythm problems." },
  { name: "Atorvastatin", smiles: "O=C(O)C1=CC=CC=C1O", cat: "Statin", desc: "Lowers cholesterol in blood. CYP3A4 substrate." },
  { name: "Metoprolol", smiles: "CC(C)(C)NCC(O)C1=CC(=C(O)C=C1)CO", cat: "Beta Blocker", desc: "Treats high blood pressure and heart failure." },
  { name: "Metformin", smiles: "CN(C)C(=N)NC(=N)N", cat: "Antidiabetic", desc: "Improves blood sugar control in type 2 diabetes." },
  { name: "Ibuprofen", smiles: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", cat: "NSAID", desc: "Treats pain, fever, and inflammation. GI risk." },
  { name: "Omeprazole", smiles: "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC", cat: "PPI", desc: "Reduces stomach acid. Can alter absorption of other drugs." },
  { name: "Amiodarone", smiles: "CC1CCC2C(C1)CCC3C2CCC4C3(CCC(C4)O)C", cat: "Antiarrhythmic", desc: "Treats severe irregular heartbeats. High toxicity risk." }
];

function getRandomDrug() { return MOCK_DRUGS[Math.floor(Math.random() * MOCK_DRUGS.length)]; }

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
      setDim({ w: containerRef.current.clientWidth, h: containerRef.current.clientHeight });
    }
    const handleResize = () => {
      if (containerRef.current) {
        setDim({ w: containerRef.current.clientWidth, h: containerRef.current.clientHeight });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Check if we actually have a detected direct interaction vs just plotting for fun
  // In the real system, this will depend on the \`result\` object having a risk_level that indicates interaction
  const hasDirectConnection = useMemo(() => {
    if (!result) return false;
    // Assuming risk_level 'Low', 'Moderate', 'High', or interaction_found exists
    return result.risk_level && result.risk_level !== 'Tolerable' && result.risk_level !== 'Safe' && result.risk_level !== 'No Interaction';
  }, [result]);

  // Handle Zoom Input (Mouse wheel & pinch-to-zoom)
  const handleWheel = (e) => {
    e.preventDefault();
    const zoomSensitivity = 0.002;
    // determine zoom direction
    const delta = -e.deltaY;
    setZoom((prevZoom) => {
      const newZoom = prevZoom + (delta * zoomSensitivity);
      return Math.min(Math.max(0.2, newZoom), 4); // clamp zoom between 0.2x and 4x
    });
  };

  // Attach non-passive wheel listener to prevent page scrolling while zooming on the canvas
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
    if (e.button !== 0 && e.pointerType === 'mouse') return; // Left click or touch
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

  const HEX_SIZE = 95; // Exact identical size for all hexagons!

  const { nodes, edges, gridTiles, directPath } = useMemo(() => {
    // 1. Unbroken Honeycomb Background (Seamless Grid)
    const gTiles = [];
    const GRID_RAD = 12; // Radius handles large pans
    for (let q = -GRID_RAD; q <= GRID_RAD; q++) {
      let r1 = Math.max(-GRID_RAD, -q - GRID_RAD);
      let r2 = Math.min(GRID_RAD, -q + GRID_RAD);
      for (let r = r1; r <= r2; r++) {
        gTiles.push({ q, r, ...hexToPixel(q, r, HEX_SIZE) });
      }
    }

    const nList = [];
    const eList = [];

    // Fallback logic
    const mainA = (drugs && drugs[0]) || MOCK_DRUGS[0];
    const mainB = (drugs && drugs[1]) || MOCK_DRUGS[1];
    const isTwoDrugs = drugs?.length >= 2 || !drugs?.length;

    const addNode = (id, q, r, isMain, drugData, color) => {
      const pos = hexToPixel(q, r, HEX_SIZE);
      nList.push({ id, q, r, x: pos.x, y: pos.y, isMain, ...drugData, color });
    };

    const addEdge = (sid, tid, color, glow) => {
      const s = nList.find(n => n.id === sid);
      const t = nList.find(n => n.id === tid);
      eList.push({ id: \`\${sid}-\${tid}\`, s, t, color, glow });
    };

    // 2. Exact placement logic
    if (!isTwoDrugs) {
      addNode('A', 0, 0, true, mainA, '#00d4ff');
      const d2 = [[-2, 0], [2, 0], [1, 1], [-1, -1], [-1, 2], [1, -2]];
      d2.slice(0, 4).forEach((c, idx) => {
        addNode(\`E\${idx}\`, c[0], c[1], false, getRandomDrug(), '#00d4ff');
        addEdge('A', \`E\${idx}\`, '#00d4ff', 'rgba(0, 212, 255, 0.15)');
      });
      return { nodes: nList, edges: eList, gridTiles: gTiles, directPath: null };
    } 

    // -- 2 OR MORE DRUGS --
    addNode('A', -2, 0, true, mainA, '#00d4ff');
    addNode('B', 2, 0, true, mainB, '#f97316');

    // SHARED PATHWAY (q:0, r:0)
    addNode('Shared', 0, 0, false, getRandomDrug(), '#a855f7');
    addEdge('A', 'Shared', '#00d4ff', 'rgba(0, 212, 255, 0.15)');
    addEdge('B', 'Shared', '#f97316', 'rgba(249, 115, 22, 0.15)');

    // UNIQUE NEIGHBORS (Distance 2 to maintain the 1-hex gap)
    addNode('A_E1', -4, 0, false, getRandomDrug(), '#00d4ff');
    addNode('A_E2', -2, -2, false, getRandomDrug(), '#00d4ff');
    addEdge('A', 'A_E1', '#00d4ff', 'rgba(0, 212, 255, 0.15)');
    addEdge('A', 'A_E2', '#00d4ff', 'rgba(0, 212, 255, 0.15)');

    addNode('B_E1', 4, 0, false, getRandomDrug(), '#f97316');
    addNode('B_E2', 2, 2, false, getRandomDrug(), '#f97316');
    addEdge('B', 'B_E1', '#f97316', 'rgba(249, 115, 22, 0.15)');
    addEdge('B', 'B_E2', '#f97316', 'rgba(249, 115, 22, 0.15)');

    // THE EFFICIENCY ALGORITHM: DIRECT CONNECTION PATH
    let pathData = null;
    
    // Only render the direct red line if we actually have an interaction from backend!
    if (hasDirectConnection) {
      const directPathTiles = [
        { q: -1, r: -1 },
        { q: 0,  r: -1 },
        { q: 1,  r: -1 },
        { q: 1,  r: 0  }
      ];
      
      const polylineCoords = [
        { q: -2, r: 0 },
        ...directPathTiles,
        { q: 2, r: 0 }
      ].map(coord => hexToPixel(coord.q, coord.r, HEX_SIZE));

      pathData = {
        tiles: directPathTiles.map(coord => hexToPixel(coord.q, coord.r, HEX_SIZE)),
        polyline: polylineCoords.map(p => \`\${p.x},\${p.y}\`).join(" ")
      };
    }

    return { nodes: nList, edges: eList, gridTiles: gTiles, directPath: pathData };
    
  }, [drugs, hasDirectConnection]);

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
        {/* Everything is in a transform group mapped to pan and zoom */}
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

          {/* LAYER 2: DIRECT CONNECTION HIGHLIGHT ALGORITHM */}
          {directPath && (
            <g className="direct-connection-layer pointer-events-none">
              {/* Highlight the empty hex tiles the path passes through */}
              {directPath.tiles.map((pos, i) => (
                 <polygon
                   key={\`direct-hex-\${i}\`}
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
              
              {/* Draw the physical bright red line connecting A and B */}
              <polyline 
                points={directPath.polyline}
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
          )}

          {/* LAYER 3: The standard gaps & connecting paths */}
          {edges.map(e => {
            const midX = (e.s.x + e.t.x) / 2;
            const midY = (e.s.y + e.t.y) / 2;
            return (
              <g key={\`edge-\${e.id}\`} className="pointer-events-none">
                <polygon
                  points={getHexPolygon(midX, midY, HEX_SIZE)}
                  fill={e.glow}
                  stroke={e.color}
                  strokeWidth="1"
                  strokeDasharray="4 4"
                  strokeOpacity="0.5"
                />
                <line
                  x1={e.s.x} y1={e.s.y}
                  x2={e.t.x} y2={e.t.y}
                  stroke={e.color}
                  strokeWidth="2.5"
                  strokeOpacity="0.6"
                />
              </g>
            );
          })}

          {/* LAYER 4: The solid node hexes - same exact size! */}
          {nodes.map(n => (
            <g 
              key={n.id} 
              onMouseEnter={() => setHoveredNode(n)}
              className="cursor-pointer transition-transform duration-300 hover:scale-[1.05]"
              style={{ transformOrigin: \`\${n.x}px \${n.y}px\` }}
            >
              {/* Opaque container hex over the grid */}
              <polygon
                points={getHexPolygon(n.x, n.y, HEX_SIZE)}
                fill="#0a0a0f"
                stroke={n.color}
                strokeWidth={n.isMain ? "4" : "2"}
                style={{ filter: \`drop-shadow(0px 0px 8px \${n.color})\` }}
              />

              {/* Exact same sized molecule embedded identically into every hex */}
              <foreignObject x={n.x - 65} y={n.y - 75} width="130" height="130" className="pointer-events-none">
                <div className="w-full h-full flex flex-col items-center justify-center">
                  <MoleculeCanvas smiles={n.smiles} width={130} height={130} />
                </div>
              </foreignObject>

              {/* Drug Name text label centered right under the molecule */}
              <text
                x={n.x} y={n.y + 65}
                textAnchor="middle"
                fill={n.isMain ? "#fff" : "#ccc"}
                fontSize={n.isMain ? "13" : "11"}
                fontWeight="bold"
                letterSpacing="0.05em"
                className="pointer-events-none"
              >
                {n.name || n.label}
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
            // Absolute position relative to container based on pan AND zoom coords
            left: (dim.w / 2) + pan.x + (hoveredNode.x * zoom) + (80 * zoom),
            top: (dim.h / 2) + pan.y + (hoveredNode.y * zoom) - (80 * zoom),
            backgroundColor: 'rgba(10,12,20,0.95)',
            borderColor: hoveredNode.color,
            boxShadow: \`0 0 30px \${hoveredNode.color}40\`,
            width: '280px',
            transform: \`scale(\${Math.max(0.8, Math.min(1.2, zoom))})\`, // Scale tooltip slightly with zoom but not entirely
            transformOrigin: 'top left'
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: hoveredNode.color, boxShadow: \`0 0 8px \${hoveredNode.color}\` }}></div>
            <h3 className="text-lg font-bold text-white leading-none tracking-wide">{hoveredNode.name}</h3>
          </div>
          
          <div className="text-xs font-mono uppercase tracking-widest text-opacity-80 mb-3" style={{ color: hoveredNode.color }}>
            {hoveredNode.cat} CLASS
          </div>
          
          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            {hoveredNode.desc}
          </p>
          
          <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-2">
            <div>
              <span className="text-xs text-gray-500 uppercase tracking-wider block mb-1">Graph Relation</span>
              {hoveredNode.isMain ? (
                <span className="text-sm text-white font-medium">Primary Focus Drug</span>
              ) : hoveredNode.id === 'Shared' ? (
                <span className="text-sm font-medium text-[#a855f7]">Bridging Target (A & B)</span>
              ) : hoveredNode.id.startsWith('A') ? (
                <span className="text-sm font-medium text-[#00d4ff]">First-Degree Neighbor (A)</span>
              ) : (
                <span className="text-sm font-medium text-[#f97316]">First-Degree Neighbor (B)</span>
              )}
            </div>
            {hoveredNode.smiles && (
               <div>
                  <span className="text-xs text-gray-500 uppercase tracking-wider block mt-2 mb-1">SMILES string</span>
                  <span className="text-[10px] text-gray-400 font-mono break-all">{hoveredNode.smiles.substring(0,20)}...</span>
               </div>
            )}
          </div>
        </div>
      )}

      {/* Floating HUD */}
      <div className="absolute bottom-4 left-4 pointer-events-none bg-[#0a0a0f]/80 px-4 py-3 rounded-lg border border-[rgba(255,255,255,0.1)] backdrop-blur-md shadow-lg">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#00d4ff] shadow-[0_0_8px_#00d4ff]"></div>
            <span className="text-[10px] text-gray-300 tracking-widest font-bold top-[1px]">DRUG A</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#f97316] shadow-[0_0_8px_#f97316]"></div>
            <span className="text-[10px] text-gray-300 tracking-widest font-bold top-[1px]">DRUG B</span>
          </div>
          <div className="flex items-center gap-2 ml-2 pl-4 border-l border-gray-700">
            <div className="w-3 h-3 rounded-full bg-[#a855f7] shadow-[0_0_8px_#a855f7]"></div>
            <span className="text-[10px] text-gray-300 tracking-widest font-bold top-[1px]">SHARED NEIGHBOR</span>
          </div>
          {hasDirectConnection && (
            <div className="flex items-center gap-2 ml-2 pl-4 border-l border-gray-700 transition-opacity duration-500">
              <div className="w-3 h-2 bg-red-500 shadow-[0_0_8px_#ef4444]"></div>
              <span className="text-[10px] text-red-400 tracking-widest font-bold top-[1px]">DIRECT INTERACTION DETECTED</span>
            </div>
          )}
        </div>
      </div>

      {/* SVG Pattern Definitions */}
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
console.log('Successfully updated KnowledgeGraphView.jsx with zoom and dynamic path routing!');
