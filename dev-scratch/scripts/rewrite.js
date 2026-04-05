const fs = require('fs');

const code = `import React, { useMemo, useState, useRef, useEffect } from 'react';
import { MoleculeCanvas } from './MoleculeViewer2D';

const MOCK_SMILES = [
  "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1",
  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
  "O=C(O)C1=CC=CC=C1O", "CC(C)(C)NCC(O)C1=CC(=C(O)C=C1)CO",
  "CN(C)C(=N)NC(=N)N", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
];
function getRandomSmiles() { return MOCK_SMILES[Math.floor(Math.random() * MOCK_SMILES.length)]; }

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

  // Interaction handlers for smooth infinite panning
  const handlePointerDown = (e) => {
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

  const { nodes, edges, gridTiles } = useMemo(() => {
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

    // Safety fallback
    const safeDrugs = drugs && drugs.length > 0 ? drugs : [{name: 'Aspirin', smiles: MOCK_SMILES[0]}, {name: 'Warfarin', smiles: MOCK_SMILES[1]}];

    const addNode = (id, q, r, isMain, label, smiles, color) => {
      const pos = hexToPixel(q, r, HEX_SIZE);
      nList.push({ id, q, r, x: pos.x, y: pos.y, isMain, label, smiles, color });
    };

    const addEdge = (sid, tid, color, glow) => {
      const s = nList.find(n => n.id === sid);
      const t = nList.find(n => n.id === tid);
      eList.push({ id: \`\${sid}-\${tid}\`, s, t, color, glow });
    };

    // 2. Exact placement logic (All edges represent a gap of exactly 1 empty hexagon grid!)
    if (safeDrugs.length === 1) {
      addNode('A', 0, 0, true, safeDrugs[0].name, safeDrugs[0].smiles || MOCK_SMILES[0], '#00d4ff');
      // Neighbors exactly at distance 2 (leaves a rigid gap of 1 perfectly empty hexagon)
      const d2 = [[-2, 0], [2, 0], [1, 1], [-1, -1], [-1, 2], [1, -2]];
      d2.slice(0, 4).forEach((c, idx) => {
        addNode(\`E\${idx}\`, c[0], c[1], false, 'Secondary Interaction', getRandomSmiles(), '#00d4ff');
        addEdge('A', \`E\${idx}\`, '#00d4ff', 'rgba(0, 212, 255, 0.15)');
      });
    } else {
      // 2 or more drugs => Highlight the "Shared Neighbor" Concept
      addNode('A', -2, 0, true, safeDrugs[0].name, safeDrugs[0].smiles || MOCK_SMILES[0], '#00d4ff');
      addNode('B', 2, 0, true, safeDrugs[1].name, safeDrugs[1].smiles || MOCK_SMILES[1], '#f97316');

      // The core requirement: 2 drugs connected ONLY through neighbours!
      // Center (0,0) is exactly distance 2 away from both A(-2,0) and B(2,0)!
      addNode('Shared', 0, 0, false, 'Shared Pathway', getRandomSmiles(), '#a855f7');
      addEdge('A', 'Shared', '#00d4ff', 'rgba(0, 212, 255, 0.15)');
      addEdge('B', 'Shared', '#f97316', 'rgba(249, 115, 22, 0.15)');

      // Add a couple unique neighbors for A (distance 2 away from A)
      addNode('A_E1', -4, 0, false, 'A-Specific Neighbor', getRandomSmiles(), '#00d4ff');
      addNode('A_E2', -2, -2, false, 'A-Specific Neighbor', getRandomSmiles(), '#00d4ff');
      addEdge('A', 'A_E1', '#00d4ff', 'rgba(0, 212, 255, 0.15)');
      addEdge('A', 'A_E2', '#00d4ff', 'rgba(0, 212, 255, 0.15)');

      // Add a couple unique neighbors for B (distance 2 away from B)
      addNode('B_E1', 4, 0, false, 'B-Specific Neighbor', getRandomSmiles(), '#f97316');
      addNode('B_E2', 2, 2, false, 'B-Specific Neighbor', getRandomSmiles(), '#f97316');
      addEdge('B', 'B_E1', '#f97316', 'rgba(249, 115, 22, 0.15)');
      addEdge('B', 'B_E2', '#f97316', 'rgba(249, 115, 22, 0.15)');
    }

    return { nodes: nList, edges: eList, gridTiles: gTiles };
  }, [drugs]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full relative bg-[#06060c] overflow-hidden cursor-grab active:cursor-grabbing rounded-xl border border-[rgba(0,212,255,0.2)]"
      style={{ minHeight: '600px', touchAction: 'none' }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      <svg width="100%" height="100%" style={{ overflow: 'visible' }}>
        {/* Everything is in a transform group mapped to center plus panning state */}
        <g transform={\`translate(\${dim.w / 2 + pan.x}, \${dim.h / 2 + pan.y})\`}>
          
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

          {/* LAYER 2: The gaps & connecting paths */}
          {edges.map(e => {
            // The "Gap of 1" Hexagon perfectly bisects the connection line
            const midX = (e.s.x + e.t.x) / 2;
            const midY = (e.s.y + e.t.y) / 2;
            return (
              <g key={\`edge-\${e.id}\`}>
                {/* Highlight the empty gap hex to visibly show the literal step path */}
                <polygon
                  points={getHexPolygon(midX, midY, HEX_SIZE)}
                  fill={e.glow}
                  stroke={e.color}
                  strokeWidth="1"
                  strokeDasharray="4 4"
                  strokeOpacity="0.5"
                />
                {/* The connective line physically threading the gap */}
                <line
                  x1={e.s.x} y1={e.s.y}
                  x2={e.t.x} y2={e.t.y}
                  stroke={e.color}
                  strokeWidth="3"
                  strokeOpacity="0.7"
                />
              </g>
            );
          })}

          {/* LAYER 3: The solid node hexes - same exact size! */}
          {nodes.map(n => (
            <g key={n.id}>
              {/* Opaque container hex over the grid */}
              <polygon
                points={getHexPolygon(n.x, n.y, HEX_SIZE)}
                fill="#0a0a0f"
                stroke={n.color}
                strokeWidth={n.isMain ? "4" : "2"}
                style={{ filter: \`drop-shadow(0px 0px 8px \${n.color})\` }}
              />

              {/* Exact same sized molecule embedded identically into every hex */}
              <foreignObject x={n.x - 65} y={n.y - 75} width="130" height="130">
                <div className="w-full h-full flex flex-col items-center justify-center pointer-events-none">
                  {/* Notice every smile gets 130x130, no more math-based shrinking! */}
                  <MoleculeCanvas smiles={n.smiles} width={130} height={130} />
                </div>
              </foreignObject>

              {/* High-def SVG Text label centered right under the molecule */}
              <text
                x={n.x} y={n.y + 65}
                textAnchor="middle"
                fill={n.isMain ? "#fff" : "#ccc"}
                fontSize={n.isMain ? "13" : "11"}
                fontWeight="bold"
                letterSpacing="0.05em"
              >
                {n.label}
              </text>
            </g>
          ))}
        </g>
      </svg>

      {/* Floating HUD */}
      <div className="absolute bottom-4 left-4 pointer-events-none bg-[#0a0a0f]/80 px-4 py-3 rounded-lg border border-[rgba(255,255,255,0.1)] backdrop-blur-md shadow-lg">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#00d4ff] shadow-[0_0_8px_#00d4ff]"></div>
            <span className="text-[10px] text-gray-300 tracking-widest font-bold top-[1px]">INPUT DRUG A</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#f97316] shadow-[0_0_8px_#f97316]"></div>
            <span className="text-[10px] text-gray-300 tracking-widest font-bold top-[1px]">INPUT DRUG B</span>
          </div>
          <div className="flex items-center gap-2 ml-2 pl-4 border-l border-gray-700">
            <div className="w-3 h-3 rounded-full bg-[#a855f7] shadow-[0_0_8px_#a855f7]"></div>
            <span className="text-[10px] text-gray-300 tracking-widest font-bold top-[1px]">SHARED NEIGHBOR</span>
          </div>
        </div>
      </div>
    </div>
  );
}
`;

fs.writeFileSync('c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/components/KnowledgeGraphView.jsx', code);
console.log('File successfully written via node script.');
