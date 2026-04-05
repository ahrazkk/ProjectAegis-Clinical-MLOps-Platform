const fs = require('fs');
const path = 'src/pages/ResearchPage.jsx';

let content = fs.readFileSync(path, 'utf8');

const tMarker1 = "{/* Phase 3 */}";
const endBlockIndex = content.indexOf("</Box>", content.indexOf(tMarker1)) + 6;

const startSplit = content.indexOf("                  <div className=\"grid lg:grid-cols-3 gap-6\">");

if (startSplit !== -1 && endBlockIndex !== -1) {

const newEvolution = `                  <div className="space-y-6 relative before:absolute before:inset-y-0 before:left-8 before:w-px before:bg-fui-gray-500/20 before:-z-10 ml-2">
                    
                    {/* Phase 1 */}
                    <Box className="!border-fui-gray-500/20 opacity-60 hover:opacity-100 transition-opacity relative overflow-hidden group ml-12">
                      <div className="absolute top-1/2 -left-12 w-12 h-px bg-fui-gray-500/20 hidden md:block"></div>
                      <div className="absolute top-0 right-0 p-4 flex gap-2">
                        <span className="text-[9px] bg-red-500/10 text-red-500 px-2 py-1 rounded border border-red-500/20 uppercase tracking-widest">Deprecated</span>
                      </div>
                      <div className="flex flex-col xl:flex-row gap-6 lg:gap-12 items-start xl:items-center">
                        <div className="flex-1">
                           <div className="text-[10px] text-fui-gray-500 uppercase tracking-widest mb-2 font-bold">V1.0 • PUBMEDBERT NLP</div>
                           <h3 className="text-xl text-fui-gray-300 line-through decoration-red-500/30 mb-4">Text-Based Inference</h3>
                           <p className="text-xs text-fui-gray-500 leading-relaxed mb-4">
                             Initial heuristic system utilizing a HuggingFace transformer to blindly parse medical literature. Failed to capture actual chemical synergy. Suffered from massive hallucination loops, API rate-limiting, and severe computational bottlenecks when analyzing polypharmacy scenarios.
                           </p>
                        </div>
                        <div className="w-full xl:w-64 bg-black/50 rounded p-4 border border-white/5 space-y-4 shrink-0">
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-400 mb-1"><span>Accuracy</span> <span className="text-red-400 font-mono">~52.0%</span></div>
                              <div className="h-1 bg-white/5 rounded overflow-hidden"><div className="h-full bg-red-500/50 w-[52%]"></div></div>
                           </div>
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-400 mb-1"><span>F1 Score</span> <span className="text-red-400 font-mono">0.312</span></div>
                              <div className="h-1 bg-white/5 rounded overflow-hidden"><div className="h-full bg-red-500/50 w-[31%]"></div></div>
                           </div>
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-400 mb-1"><span>Latency</span> <span className="text-red-400 font-mono">8.40 s</span></div>
                           </div>
                        </div>
                      </div>
                    </Box>

                    {/* Phase 2 */}
                    <Box className="!border-fui-gray-500/40 opacity-80 hover:opacity-100 transition-opacity relative overflow-hidden group ml-12">
                      <div className="absolute top-1/2 -left-12 w-12 h-px bg-fui-gray-500/40 hidden md:block"></div>
                      <div className="absolute top-0 right-0 p-4 flex gap-2">
                        <span className="text-[9px] bg-yellow-500/10 text-yellow-500 px-2 py-1 rounded border border-yellow-500/20 uppercase tracking-widest">Archived</span>
                      </div>
                      <div className="flex flex-col xl:flex-row gap-6 lg:gap-12 items-start xl:items-center">
                        <div className="flex-1">
                           <div className="text-[10px] text-fui-gray-400 uppercase tracking-widest mb-2 font-bold">V2.0 • MICROSCOPIC GIN</div>
                           <h3 className="text-xl text-fui-gray-200 mb-4">Chemical Graph Isomorphism</h3>
                           <p className="text-xs text-fui-gray-400 leading-relaxed mb-4">
                             Pivoted to parsing raw molecular structures (SMILES) directly into PyTorch graph tensors. Used RDKit to map atoms as nodes and chemical bonds as edges. While chemically accurate, we reached hardware limits instantly (OOM) without uncovering broader clinical pathways.
                           </p>
                           <div className="flex gap-4">
                             <div className="flex items-center gap-1.5 text-[10px] text-fui-gray-400 bg-white/5 py-1 px-2 rounded-full border border-white/5">
                               <span className="text-red-400 font-bold">×</span> VRAM OOM @ Ep. 30
                             </div>
                             <div className="flex items-center gap-1.5 text-[10px] text-fui-gray-400 bg-white/5 py-1 px-2 rounded-full border border-white/5">
                               <span className="text-red-400 font-bold">×</span> Missing EHR Context
                             </div>
                           </div>
                        </div>
                        <div className="w-full xl:w-64 bg-black/50 rounded p-4 border border-white/5 space-y-4 shrink-0">
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-400 mb-1"><span>Accuracy</span> <span className="text-yellow-400 font-mono">65.7%</span></div>
                              <div className="h-1 bg-white/5 rounded overflow-hidden"><div className="h-full bg-yellow-500/60 w-[65.7%]"></div></div>
                           </div>
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-400 mb-1"><span>F1 Score</span> <span className="text-yellow-400 font-mono">0.584</span></div>
                              <div className="h-1 bg-white/5 rounded overflow-hidden"><div className="h-full bg-yellow-500/60 w-[58.4%]"></div></div>
                           </div>
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-400 mb-1"><span>Latency</span> <span className="text-yellow-400 font-mono">3.20 s</span></div>
                           </div>
                        </div>
                      </div>
                    </Box>

                    {/* Phase 3 */}
                    <Box glow="rgba(0,212,255,0.05)" className="!border-cyan-500/50 relative overflow-hidden group ml-12">
                      <div className="absolute top-1/2 -left-12 w-12 h-px bg-cyan-500/50 shadow-[0_0_10px_#00d4ff] hidden md:block"></div>
                      <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-transparent pointer-events-none" />
                      <div className="absolute top-0 right-0 p-4 flex gap-2">
                        <span className="text-[9px] bg-cyan-500/10 text-cyan-400 px-2 py-1 rounded border border-cyan-500/30 shadow-[0_0_10px_rgba(0,212,255,0.2)] animate-pulse uppercase tracking-widest">Active Core</span>
                      </div>
                      <div className="flex flex-col xl:flex-row gap-6 lg:gap-12 items-start xl:items-center relative z-10">
                        <div className="flex-1">
                           <div className="text-[10px] text-cyan-400 uppercase tracking-widest mb-2 font-bold flex items-center gap-2">
                             <Activity className="w-3 h-3" /> V3.0 • MACROSCOPIC GRAPHSAGE
                           </div>
                           <h3 className="text-xl text-white mb-4">Global Interactome Matrix</h3>
                           <p className="text-xs text-fui-gray-300 leading-relaxed mb-4">
                             Abandoned single-molecule scanning. Transformed the entire FDA database into a unified, massive graph network. Drugs are nodes. Clinical pathways and known interactions are edges. Features are embedded via high-dimensional vectors and passed dynamically through GraphSAGE message layers.
                           </p>
                           <div className="grid grid-cols-2 gap-4">
                              <div className="bg-black/40 border border-cyan-500/20 p-3 rounded">
                                 <div className="text-[9px] text-fui-gray-500 uppercase mb-1">Density Scaling</div>
                                 <div className="text-xs text-cyan-400 font-mono">1.08 → 79.25 edges</div>
                              </div>
                              <div className="bg-black/40 border border-cyan-500/20 p-3 rounded">
                                 <div className="text-[9px] text-fui-gray-500 uppercase mb-1">Vector Breadth</div>
                                 <div className="text-xs text-cyan-400 font-mono">1,343 latent feats</div>
                              </div>
                           </div>
                        </div>
                        <div className="w-full xl:w-64 bg-cyan-500/5 rounded p-4 border border-cyan-500/30 space-y-4 shrink-0 shadow-[0_0_20px_rgba(0,212,255,0.05)]">
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-200 mb-1"><span>ROC-AUC</span> <span className="text-cyan-400 font-bold font-mono">98.27%</span></div>
                              <div className="h-1 bg-cyan-500/20 rounded overflow-hidden"><div className="h-full bg-cyan-400 shadow-[0_0_10px_#00d4ff] w-[98.27%]"></div></div>
                           </div>
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-200 mb-1"><span>Global F1</span> <span className="text-cyan-400 font-bold font-mono">0.811</span></div>
                              <div className="h-1 bg-cyan-500/20 rounded overflow-hidden"><div className="h-full bg-cyan-400 shadow-[0_0_10px_#00d4ff] w-[81.1%]"></div></div>
                           </div>
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-200 mb-1"><span>Recall</span> <span className="text-cyan-400 font-bold font-mono">99.4%</span></div>
                              <div className="h-1 bg-cyan-500/20 rounded overflow-hidden"><div className="h-full bg-cyan-400 shadow-[0_0_10px_#00d4ff] w-[99.4%]"></div></div>
                           </div>
                           <div>
                              <div className="flex justify-between text-[10px] text-fui-gray-200 mb-1"><span>Latency</span> <span className="text-cyan-400 font-bold font-mono">24 ms</span></div>
                           </div>
                        </div>
                      </div>
                    </Box>
                  </div>`;
                 
  const pre = content.substring(0, startSplit);
  const post = content.substring(endBlockIndex);
  content = pre + newEvolution + post;
  fs.writeFileSync(path, content, 'utf8');
  console.log("Rewrote Phase array effectively!");
} else {
  console.log("Could not find startSplit or endBlockIndex");
}
