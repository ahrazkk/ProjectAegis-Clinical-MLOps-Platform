const fs = require('fs');
const path = 'src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

// 1. Fix the \n\n bug
content = content.replace(/\{\/\* ═══════════ ARCHITECTURE SCROLL PARALLAX ═══════════ \*\/}\\n\\n/, '{/* ═══════════ ARCHITECTURE SCROLL PARALLAX ═══════════ */}');

// 2. Replace the messy SVG/absolute node graph with a pristine Flexbox Flow Diagram
const startMarker = '{/* Right Block: Interactive Node Mock (Col span 7) */}';
const endMarkerMatch = 'Benign / Unknown</div>';

const startIndex = content.indexOf(startMarker);
const endIndex = content.indexOf(endMarkerMatch, startIndex);

if (startIndex !== -1 && endIndex !== -1) {
  // We need to find the ending `</div>` for the legend and the wrapper `</div>`
  // so we'll just slice dynamically up to the next closing tags.
  // Actually, we can just replace everything between startMarker and the closing of the legend.
  
  const endSlice = content.indexOf('</div>', endIndex + endMarkerMatch.length) + 6; // closes the Benign div
  const legendEnd = content.indexOf('</div>', endSlice) + 6; // closes the legend wrapper div
  const wrapperEnd = content.indexOf('</div>', legendEnd) + 6; // closes the col-span-7 div

  const newGraph = `{/* Right Block: Simplified Flex Flow Diagram (Col span 7) */}
              <div className="col-span-1 xl:col-span-7 relative bg-black/40 min-h-[500px] lg:min-h-[700px] flex items-center justify-center p-8 overflow-hidden">
                  
                  {/* Background Grid & Blurs */}
                  <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:30px_30px] [mask-image:radial-gradient(ellipse_70%_70%_at_50%_50%,#000_100%,transparent_100%)] pointer-events-none" />
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[40rem] h-[40rem] bg-purple-500/10 rounded-full blur-[120px] mix-blend-screen pointer-events-none" />

                  {/* HUD Elements */}
                  <div className="absolute top-8 left-8 z-30 pointer-events-none hidden sm:block">
                    <div className="flex items-center gap-2 mb-2">
                       <span className="relative flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-pink-400 opacity-75" />
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-pink-500" />
                       </span>
                       <span className="text-[10px] font-mono text-white/50 tracking-widest uppercase">GNN Live Flow</span>
                    </div>
                  </div>

                  {/* The Flow Diagram */}
                  <div className="relative z-10 w-full max-w-2xl flex flex-col md:flex-row items-center gap-8 lg:gap-16">
                      
                      {/* Left: Hub Node */}
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.8 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        className="relative flex flex-col items-center shrink-0"
                      >
                         <div className="absolute -top-8 text-[10px] font-mono text-[#a78bfa] bg-black/60 px-2 py-0.5 rounded border border-purple-500/20">Target Input</div>
                         <div className="w-32 h-32 rounded-full bg-black/80 backdrop-blur-xl border border-white/20 flex flex-col items-center justify-center shadow-[0_0_50px_rgba(168,85,247,0.3)] z-20">
                            <span className="font-heading font-medium text-xl tracking-wide text-white">Aspirin</span>
                            <span className="text-[10px] text-white/40 font-mono mt-1">CHEMBL123</span>
                         </div>
                         
                         {/* Visual Spines connecting from central node -> Right nodes (Desktop) */}
                         <div className="hidden md:block absolute top-1/2 left-full w-12 border-t border-white/20 -translate-y-1/2 z-0" />
                         <div className="hidden md:block absolute top-[20%] left-[calc(100%+2.9rem)] bottom-[20%] w-px bg-white/20 z-0" />
                      </motion.div>

                      {/* Right: Interaction Cards */}
                      <div className="flex flex-col gap-4 relative z-10 w-full">
                          
                          {/* Warfarin Card */}
                          <motion.div 
                             initial={{ opacity: 0, x: 20 }}
                             whileInView={{ opacity: 1, x: 0 }}
                             transition={{ delay: 0.2 }}
                             className="flex items-center group relative overflow-hidden rounded-xl border border-white/10 bg-black/60 backdrop-blur-md hover:bg-white/[0.02] transition-colors"
                          >
                             {/* Connector (Desktop) */}
                             <div className="hidden md:block absolute -left-8 top-1/2 w-8 border-t border-white/20 -translate-y-1/2 z-0" />
                             
                             {/* Indicator Line */}
                             <div className="w-1.5 self-stretch bg-pink-500 rounded-l-xl shadow-[0_0_10px_rgba(236,72,153,0.5)]" />
                             
                             <div className="flex-1 p-4">
                                <div className="flex justify-between items-center mb-1">
                                   <div className="flex items-center gap-3">
                                      <span className="text-white font-heading font-medium tracking-wide">Warfarin</span>
                                      <span className="text-[9px] font-mono text-white/40 uppercase">Anticoagulant</span>
                                   </div>
                                   <div className="flex items-center gap-2">
                                      <span className="text-[10px] font-mono text-pink-400 bg-pink-500/10 px-2 py-0.5 rounded">0.984</span>
                                   </div>
                                </div>
                                <div className="text-xs text-white/50 font-body">CYP2C9 Inhibition (Severe Bleeding)</div>
                             </div>
                             
                             {/* Highlight glow effect on hover */}
                             <div className="absolute inset-0 bg-gradient-to-r from-pink-500/0 via-pink-500/0 to-pink-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                          </motion.div>

                          {/* Ibuprofen Card */}
                          <motion.div 
                             initial={{ opacity: 0, x: 20 }}
                             whileInView={{ opacity: 1, x: 0 }}
                             transition={{ delay: 0.4 }}
                             className="flex items-center group relative overflow-hidden rounded-xl border border-white/10 bg-black/60 backdrop-blur-md hover:bg-white/[0.02] transition-colors"
                          >
                             {/* Connector (Desktop) */}
                             <div className="hidden md:block absolute -left-8 top-1/2 w-8 border-t border-white/20 -translate-y-1/2 z-0" />
                             
                             {/* Indicator Line */}
                             <div className="w-1.5 self-stretch bg-yellow-500 rounded-l-xl shadow-[0_0_10px_rgba(234,179,8,0.3)]" />
                             
                             <div className="flex-1 p-4">
                                <div className="flex justify-between items-center mb-1">
                                   <div className="flex items-center gap-3">
                                      <span className="text-white font-heading font-medium tracking-wide">Ibuprofen</span>
                                      <span className="text-[9px] font-mono text-white/40 uppercase">NSAID</span>
                                   </div>
                                   <div className="flex items-center gap-2">
                                      <span className="text-[10px] font-mono text-yellow-400 bg-yellow-500/10 px-2 py-0.5 rounded">0.835</span>
                                   </div>
                                </div>
                                <div className="text-xs text-white/50 font-body">Decreased Renal Clearance</div>
                             </div>
                             
                             <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/0 via-yellow-500/0 to-yellow-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                          </motion.div>

                          {/* Clopidogrel Card */}
                          <motion.div 
                             initial={{ opacity: 0, x: 20 }}
                             whileInView={{ opacity: 1, x: 0 }}
                             transition={{ delay: 0.6 }}
                             className="flex items-center group relative overflow-hidden rounded-xl border border-white/10 bg-black/60 backdrop-blur-md hover:bg-white/[0.02] transition-colors"
                          >
                             {/* Connector (Desktop) */}
                             <div className="hidden md:block absolute -left-8 top-1/2 w-8 border-t border-white/20 -translate-y-1/2 z-0" />
                             
                             {/* Indicator Line */}
                             <div className="w-1.5 self-stretch bg-white/30 rounded-l-xl" />
                             
                             <div className="flex-1 p-4">
                                <div className="flex justify-between items-center mb-1">
                                   <div className="flex items-center gap-3">
                                      <span className="text-white font-heading font-medium tracking-wide">Clopidogrel</span>
                                      <span className="text-[9px] font-mono text-white/40 uppercase">Antiplatelet</span>
                                   </div>
                                   <div className="flex items-center gap-2">
                                      <span className="text-[10px] font-mono text-white/60 bg-white/10 px-2 py-0.5 rounded">0.121</span>
                                   </div>
                                </div>
                                <div className="text-xs text-white/50 font-body">Benign / No Structural Topology Match</div>
                             </div>
                          </motion.div>

                      </div>
                  </div>

                  {/* Legend */}
                  <div className="absolute bottom-6 left-6 flex flex-wrap items-center gap-6 text-[10px] uppercase font-mono tracking-widest text-white/40 z-30 border border-white/5 bg-black/40 px-4 py-2 rounded-full backdrop-blur-md">
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-pink-500" /> Severe</div>
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-yellow-500" /> Moderate</div>
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-white/30" /> Benign</div>
                  </div>
              </div>`;

  const before = content.slice(0, startIndex);
  const after = content.slice(wrapperEnd);
  
  content = before + newGraph + after;
  fs.writeFileSync(path, content, 'utf8');
  console.log("Successfully replaced Graph with Flow Map.");
} else {
  console.log("Could not find start/end marks.");
}
