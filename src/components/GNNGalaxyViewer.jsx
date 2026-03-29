import React, { useMemo, useState, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Html, Line } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

import rawGnnData from '../assets/gnn_real_data.json';

const DrugNode = ({ pos, size, color, name, isMain = false, isHovered, onPointerOver, onPointerOut, opacity = 1 }) => {
  const meshRef = useRef();

  useFrame(({ clock }) => {
    if (meshRef.current && isMain) {
      meshRef.current.position.y = pos[1] + Math.sin(clock.getElapsedTime() * 2) * (size * 0.2);
    }
  });

  return (
    <group position={pos} ref={meshRef}>
      <mesh onPointerOver={onPointerOver} onPointerOut={onPointerOut}>
         <sphereGeometry args={[size, 16, 16]} />
         <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
      </mesh>
      
      {/* Halo for main drugs */}
      {isMain && (
        <mesh>
          <sphereGeometry args={[size * 1.8, 32, 32]} />
          <meshBasicMaterial color={color} transparent opacity={opacity * 0.3} depthWrite={false}/>
        </mesh>
      )}

      {(isMain || isHovered) && (
        <Html position={[0, -size - (isMain ? 0.8 : 0.3), 0]} center zIndexRange={[100, 0]} className="pointer-events-none">
          <div className={isMain ? 
            "bg-[#050814]/90 border border-current px-3 py-1 rounded-sm text-[11px] shadow-[0_0_15px_currentColor] backdrop-blur-md whitespace-nowrap font-bold tracking-widest uppercase transition-all" : 
            "bg-[#050814]/80 border border-slate-500/50 px-2 py-0.5 rounded shadow-lg text-[9px] text-slate-200 backdrop-blur-sm whitespace-nowrap font-mono z-50"}
            style={{ color: isMain ? color : 'white', borderColor: isMain ? color : undefined }}>
            {name}
          </div>
        </Html>
      )}
    </group>
  );
};

const TrueLatentSpace = ({ selectedDrugs }) => {
  const [hoveredNode, setHoveredNode] = useState(null);

  const { nodes, edges } = useMemo(() => {
    // 1. Data sanity check
    if (!rawGnnData || !rawGnnData.nodes) return { nodes: [], edges: [] };

    // Scale T-SNE positions to fit well in our camera
    const scale = 0.25;
    
    // Get valid names from props
    let nameA = selectedDrugs && selectedDrugs[0] && selectedDrugs[0].name ? selectedDrugs[0].name.toLowerCase() : null;
    let nameB = selectedDrugs && selectedDrugs[1] && selectedDrugs[1].name ? selectedDrugs[1].name.toLowerCase() : null;

    let nodeA_id = null;
    let nodeB_id = null;
    
    const nodeDict = {};
    rawGnnData.nodes.forEach(n => {
       nodeDict[n.id] = { 
           ...n, 
           pos: [n.pos[0]*scale, n.pos[1]*scale, n.pos[2]*scale],
           isA: false,
           isB: false,
           hopA: Infinity,
           hopB: Infinity
       };
       // Find exact drug matches
       if (nameA && n.name.toLowerCase() === nameA) nodeA_id = n.id;
       if (nameB && n.name.toLowerCase() === nameB) nodeB_id = n.id;
    });

    // If exact match fails, try partial match
    if (!nodeA_id && nameA) {
        let partial = rawGnnData.nodes.find(n => n.name.toLowerCase().includes(nameA));
        if (partial) nodeA_id = partial.id;
    }
    if (!nodeB_id && nameB) {
        let partial = rawGnnData.nodes.find(n => n.name.toLowerCase().includes(nameB));
        if (partial) nodeB_id = partial.id;
    }

    if (nodeA_id !== null) nodeDict[nodeA_id].isA = true;
    if (nodeB_id !== null) nodeDict[nodeB_id].isB = true;

    // 2. BFS to find 3-Hop distances
    const getHops = (startId, hopField) => {
        if (startId === null || !rawGnnData.adj) return;
        nodeDict[startId][hopField] = 0;
        let queue = [{id: startId, d: 0}];
        let visited = new Set([startId]);
        
        while(queue.length > 0) {
            let curr = queue.shift();
            if (curr.d >= 3) continue; // max 3 hops
            
            let neighbors = rawGnnData.adj[curr.id] || [];
            neighbors.forEach(nxt => {
                if (!visited.has(nxt)) {
                    visited.add(nxt);
                    nodeDict[nxt][hopField] = curr.d + 1;
                    queue.push({id: nxt, d: curr.d + 1});
                }
            });
        }
    };

    getHops(nodeA_id, 'hopA');
    getHops(nodeB_id, 'hopB');

    // Default global drawing settings
    let limitGlobalDraw = (nodeA_id !== null || nodeB_id !== null); // If scanning drugs, hide irrelevant edges

    // 3. Generate visual node attributes based on distances
    const finalNodes = Object.values(nodeDict);
    
    // 4. Generate edges (bonds) selectively to avoid lag
    const finalEdges = [];
    if (rawGnnData.adj) {
        Object.keys(rawGnnData.adj).forEach(u => {
            let uId = parseInt(u);
            let uInfo = nodeDict[uId];
            if(!uInfo) return;

            rawGnnData.adj[u].forEach(v => {
                let vId = parseInt(v);
                if (vId > uId) { // Undirected, only process once
                    let vInfo = nodeDict[vId];
                    if(!vInfo) return;

                    let inHopA = (uInfo.hopA <= 3 && vInfo.hopA <= 3);
                    let inHopB = (uInfo.hopB <= 3 && vInfo.hopB <= 3);
                    
                    let isDirect = false;
                    let color = "#475569"; // Slate for far background edges
                    let opacity = 0.05;
                    
                    // Check direct DDI connection prediction line
                    if ((uInfo.isA && vInfo.isB) || (uInfo.isB && vInfo.isA)) {
                        isDirect = true;
                        color = "#ef4444"; // Red direct link!
                        opacity = 1.0;
                    } else if (inHopA && inHopB) {
                        // Bridge between both
                        color = "#a855f7"; 
                        opacity = 0.5;
                    } else if (inHopA) {
                        color = "#00d2ff"; 
                        opacity = Math.max(0.15, 0.4 - (Math.max(uInfo.hopA, vInfo.hopA) * 0.1));
                    } else if (inHopB) {
                        color = "#ff8c00"; 
                        opacity = Math.max(0.15, 0.4 - (Math.max(uInfo.hopB, vInfo.hopB) * 0.1));
                    }

                    // Only push edge if it belongs to a local cluster OR we are viewing total galaxy (no drugs selected)
                    if (!limitGlobalDraw || inHopA || inHopB || isDirect) {
                        finalEdges.push({
                            start: uInfo.pos,
                            end: vInfo.pos,
                            color,
                            opacity,
                            isDirect
                        });
                    }
                }
            });
        });
    }

    return { nodes: finalNodes, edges: finalEdges };
  }, [selectedDrugs]);

  return (
    <group>
      {/* 1. Edges / Lines between nodes */}
      {edges.map((e, i) => (
        <Line 
           key={`edge-${i}`} 
           points={[e.start, e.end]} 
           color={e.color} 
           transparent 
           opacity={e.opacity} 
           lineWidth={e.isDirect ? 4 : 1}
           depthWrite={false}
        />
      ))}
      
      {/* 2. Vector space Points (Molecules & Genes) */}
      {nodes.map((n, i) => {
          let isMain = n.isA || n.isB;
          let color = "#1e293b"; // faint slate for background noise
          let opacity = 0.2;
          let size = 0.12; 
          
          if (n.isA) {
              color = "#00d2ff";
              opacity = 1;
              size = 0.8;
          } else if (n.isB) {
              color = "#ff8c00";
              opacity = 1;
              size = 0.8;
          } else if (n.hopA <= 3 && n.hopB <= 3) {
              // Bridge nodes (purple)
              color = "#a855f7";
              opacity = 0.8 - (Math.max(n.hopA, n.hopB) * 0.15);
              size = 0.4;
          } else if (n.hopA <= 3) {
              // In Drug A cluster
              color = "#00d2ff";
              opacity = 0.7 - (n.hopA * 0.18);
              size = 0.35 - (n.hopA * 0.05);
          } else if (n.hopB <= 3) {
              // In Drug B cluster
              color = "#ff8c00";
              opacity = 0.7 - (n.hopB * 0.18);
              size = 0.35 - (n.hopB * 0.05);
          }

          let isHovered = (hoveredNode === n.id);
          if (isHovered && !isMain) {
              opacity = 1.0;
              size = Math.max(size, 0.4);
              color = "#ffffff";
          }

          return (
            <DrugNode 
              key={`node-${n.id}`} 
              pos={n.pos} 
              size={size} 
              color={color} 
              opacity={opacity}
              name={n.name}
              isMain={isMain}
              isHovered={isHovered}
              onPointerOver={(e) => { e.stopPropagation(); setHoveredNode(n.id); document.body.style.cursor = 'pointer'; }}
              onPointerOut={(e) => { setHoveredNode(null); document.body.style.cursor = 'auto'; }}
            />
          );
      })}
    </group>
  );
};

export default function GNNGalaxyViewer({ drugs, isMobile }) {
  return (
    <div className="w-full h-full relative overflow-hidden rounded-xl border border-white/5" style={{ minHeight: isMobile ? '400px' : '650px', background: '#03050a' }}>
      <Canvas camera={{ position: [0, 8, 30], fov: 60 }}>
        <color attach="background" args={['#03050a']} />
        
        <ambientLight intensity={0.5} />
        <pointLight position={[15, 15, 15]} intensity={1} color="#ffffff" />
        <pointLight position={[-15, -15, -15]} intensity={0.5} color="#00d2ff" />
        <pointLight position={[15, -15, 15]} intensity={0.5} color="#ff8c00" />
        
        <Stars radius={150} depth={50} count={9000} factor={6} saturation={1} fade speed={1.5} />

        <TrueLatentSpace selectedDrugs={drugs} />

        <OrbitControls 
          enablePan={true} 
          enableZoom={true} 
          enableRotate={true}
          autoRotate={true}
          autoRotateSpeed={0.3}
          maxDistance={150}
          minDistance={2}
        />

        <EffectComposer>
          <Bloom luminanceThreshold={0.10} luminanceSmoothing={0.9} intensity={1.5} mipmapBlur />
        </EffectComposer>
      </Canvas>
      
      {/* Legend & UI Overlays */}
      <div className="absolute bottom-4 left-4 pointer-events-none bg-black/60 border border-white/5 p-4 flex flex-col gap-3 rounded-lg text-[11px] text-slate-300 font-mono backdrop-blur-md shadow-xl">
        <div className="text-white/80 font-bold border-b border-white/10 pb-1 mb-1">TRUE 3D LATENT EMBEDDINGS</div>
        <div className="flex items-center gap-3"><div className="w-3 h-3 rounded-full bg-[#00d2ff] border shadow-[0_0_10px_#00d2ff]"></div> Source Hub (Drug A) & Hops</div>
        <div className="flex items-center gap-3"><div className="w-3 h-3 rounded-full bg-[#ff8c00] border shadow-[0_0_10px_#ff8c00]"></div> Target Hub (Drug B) & Hops</div>
        <div className="flex items-center gap-3"><div className="w-3 h-3 rounded-full border-2 border-[#a855f7] bg-[#a855f7]/30 shadow-[0_0_10px_#a855f7]"></div> Biological Bridge (Shared Hops)</div>
        <div className="flex items-center gap-3"><div className="w-6 h-0.5 bg-[#ef4444] shadow-[0_0_10px_#ef4444]"></div> Known Neo4j Link (Direct)</div>
        <div className="flex items-center gap-3"><div className="w-1 h-1 rounded-full bg-[#1e293b]"></div> Entire Vector Database (Zoom Out)</div>
      </div>
      
      <div className="absolute top-4 right-4 pointer-events-none flex flex-col items-end gap-2">
         <div className="bg-black/60 border border-cyan-500/40 text-cyan-300 px-3 py-1.5 rounded text-[10px] uppercase tracking-[0.25em] shadow-[0_0_15px_rgba(0,255,255,0.15)] backdrop-blur">
            GNN T-SNE Space Active
         </div>
      </div>
    </div>
  );
}
