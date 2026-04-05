# -*- coding: utf-8 -*-
import io

file_path = "src/pages/ResearchPage.jsx"
with io.open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# We look for where activeTab === 'evolution' starts.
idx_start = text.find("activeTab === 'evolution' && (")
if idx_start == -1:
    print("Cannot find evolution start")
    exit(1)

# We look for where the next tab starts (gnn)
idx_end = text.find("activeTab === 'gnn' && (")
if idx_end == -1:
    print("Cannot find gnn start")
    exit(1)

# Retreat idx_start to the comment block above it
pre_comment = text.rfind("{/*", 0, idx_start)
if pre_comment != -1:
    idx_start = pre_comment

# Retreat idx_end to the comment block above it
pre_end_comment = text.rfind("{/*", 0, idx_end)
if pre_end_comment != -1:
    idx_end = pre_end_comment

pre_code = text[:idx_start]
post_code = text[idx_end:]

new_evolution_ui = """{/* START MODEL EVOLUTION TAB */}
            {activeTab === 'evolution' && (
              <motion.div key="evolution" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -30 }} className="space-y-16">

                {/* Cyberpunk Header */}
                <div className="relative overflow-hidden group rounded-xl border border-cyan-500/30 bg-black/60 backdrop-blur-md p-10 md:p-14 shadow-[0_0_80px_-15px_rgba(0,212,255,0.2)]">
                  <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-gradient-to-bl from-cyan-500/10 via-purple-500/5 to-transparent rounded-full blur-[100px] pointer-events-none -translate-y-1/2 translate-x-1/3" />
                  
                  <div className="relative z-10 flex flex-col md:flex-row gap-8 items-center justify-between">
                    <div className="max-w-2xl">
                      <div className="inline-flex items-center gap-2 border border-cyan-500/50 px-4 py-1.5 bg-cyan-500/10 text-cyan-400 text-[11px] tracking-[0.2em] mb-8 rounded-full">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                        AEGIS CORE ARCHITECTURE
                      </div>
                      <h2 className="text-4xl md:text-5xl font-extralight text-white mb-6 tracking-tight">
                        Algorithmic <span className="font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">Evolution</span>
                      </h2>
                      <p className="text-fui-gray-400 text-sm md:text-base leading-relaxed">
                        Project Aegis has undergone three distinct evolutionary phases. We started with heuristic NLP baselines (Phase 1), advanced into experimental microscopic graph structures (Phase 2), and arrived at our current state-of-the-art Macroscopic GraphSAGE topology—achieving <span className="text-cyan-400">98.67% testing accuracy</span> across massive multi-modal Datasets.
                      </p>
                    </div>
                    <div className="relative w-48 h-48 shrink-0">
                      <Activity className="absolute inset-0 w-full h-full text-cyan-500/20 animate-[spin_10s_linear_infinite]" />
                      <Atom className="absolute inset-8 w-32 h-32 text-blue-400/40 animate-[spin_15s_linear_infinite_reverse]" />
                      <Microscope className="absolute inset-16 w-16 h-16 text-white drop-shadow-[0_0_15px_rgba(255,255,255,0.8)]" />
                    </div>
                  </div>
                </div>

                <div className="grid md:grid-cols-1 gap-12 relative before:absolute before:inset-0 before:ml-[23px] md:before:ml-1/2 before:-translate-x-px md:before:translate-x-0 before:w-0.5 before:bg-gradient-to-b before:from-purple-500/50 before:via-blue-500/50 before:to-cyan-400/50">
                  
                  {/* Gen 1: NLP Baseline */}
                  <motion.div initial={{ x: -50, opacity: 0 }} whileInView={{ x: 0, opacity: 1 }} viewport={{ once: true }} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group border border-purple-500/20 bg-black/40 p-8 rounded-2xl hover:border-purple-500/50 transition-colors ml-12 md:ml-0 md:w-[calc(50%-3rem)] left-0">
                    <div className="absolute top-1/2 -translate-y-1/2 -left-14 md:-left-16 w-6 h-6 rounded-full bg-black border-4 border-purple-500 shadow-[0_0_15px_rgba(168,85,247,0.6)] z-10" />
                    <div className="w-full">
                      <div className="text-[10px] text-purple-400 tracking-[0.2em] mb-2 font-bold">V1.0 • PUBMEDBERT NLP</div>
                      <h3 className="text-2xl text-white mb-4">Text-Based Inference</h3>
                      <p className="text-fui-gray-400 text-xs leading-relaxed mb-6">
                        Initial text-based inference system utilizing a HuggingFace transformer to blindly parse medical literature. Failed to capture actual chemical synergy. Suffered from massive hallucination loops, API rate-limiting, and severe computational bottlenecks when analyzing polypharmacy scenarios.
                      </p>
                      <div className="flex gap-6">
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Accuracy</div>
                          <div className="text-xl text-purple-400 font-bold">~52.0%</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Latency</div>
                          <div className="text-xl text-fui-gray-300 font-bold">8.4s</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Status</div>
                          <div className="text-xl text-red-500 font-bold line-through">DEPRECATED</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Gen 2: Micro GNN */}
                  <motion.div initial={{ x: 50, opacity: 0 }} whileInView={{ x: 0, opacity: 1 }} viewport={{ once: true }} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group border border-blue-500/20 bg-black/40 p-8 rounded-2xl hover:border-blue-500/50 transition-colors ml-12 md:ml-0 md:w-[calc(50%-3rem)] md:left-1/2 md:translate-x-[3rem]">
                    <div className="absolute top-1/2 -translate-y-1/2 -left-14 md:-left-16 w-6 h-6 rounded-full bg-black border-4 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.6)] z-10" />
                    <div className="w-full">
                      <div className="text-[10px] text-blue-400 tracking-[0.2em] mb-2 font-bold">V2.0 • MICROSCOPIC GIN</div>
                      <h3 className="text-2xl text-white mb-4">Chemical Graph Isomorphism</h3>
                      <p className="text-fui-gray-400 text-xs leading-relaxed mb-6">
                        Pivoted to parsing raw molecular structures (SMILES) directly into PyTorch graph tensors. Used RDKit to map atoms as nodes and chemical bonds as edges. Reached hardware limits instantly.
                      </p>
                      <ul className="space-y-2 mb-6">
                        <li className="flex items-center gap-2 text-xs text-fui-gray-300"><span className="text-red-400 font-bold">X</span> GPU Memory Overflows (OOM) at Epoch 30</li>
                        <li className="flex items-center gap-2 text-xs text-fui-gray-300"><span className="text-red-400 font-bold">X</span> Insufficient representation of clinical pathways</li>
                      </ul>
                      <div className="flex gap-6">
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Accuracy</div>
                          <div className="text-xl text-blue-400 font-bold">65.7%</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Status</div>
                          <div className="text-xl text-fui-gray-500 font-bold">ARCHIVED</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Gen 3: Macro GNN */}
                  <motion.div initial={{ y: 50, opacity: 0 }} whileInView={{ y: 0, opacity: 1 }} viewport={{ once: true }} className="relative flex items-center justify-between md:justify-normal group border border-cyan-400/50 bg-cyan-950/10 p-8 md:p-12 rounded-2xl shadow-[0_0_40px_-10px_rgba(34,211,238,0.15)] ml-12 md:ml-0 md:w-[calc(50%-3rem)] left-0 border-l-4">
                    <div className="absolute top-1/2 -translate-y-1/2 -left-[60px] md:-left-[68px] w-8 h-8 rounded-full bg-cyan-400 border-4 border-black shadow-[0_0_20px_rgba(34,211,238,1)] z-10 flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full animate-ping" />
                    </div>
                    <div className="w-full">
                      <div className="flex justify-between items-start mb-4">
                        <div className="text-[10px] text-cyan-400 tracking-[0.2em] font-bold">V3.0 • MACROSCOPIC GRAPHSAGE</div>
                        <div className="px-3 py-1 bg-green-500/20 text-green-400 text-[10px] font-bold rounded shadow-[0_0_10px_rgba(34,197,94,0.2)]">CURRENT PROD</div>
                      </div>
                      <h3 className="text-3xl text-white mb-4 font-light">Global Interactome Matrix</h3>
                      <p className="text-fui-gray-300 text-sm leading-relaxed mb-8">
                        Completely abandoned single-molecule scanning. Transformed the entire FDA database into a unified, massive graph network. Drugs are nodes. Clinical pathways, adverse effects, and known interactions are edges. Features are embedded via high-dimensional PCA vectors and passed dynamically through GraphSAGE message layers.
                      </p>
                      
                      <div className="grid grid-cols-2 gap-4 mb-8">
                        <div className="bg-black/50 border border-fui-gray-800 p-4 rounded-lg">
                          <Globe className="w-5 h-5 text-cyan-400 mb-2" />
                          <div className="text-[10px] text-fui-gray-500 uppercase">Density Scaling</div>
                          <div className="text-sm text-fui-gray-200 mt-1">Nodes increased from 1.08 to 79.25 edges</div>
                        </div>
                        <div className="bg-black/50 border border-fui-gray-800 p-4 rounded-lg">
                          <Layers className="w-5 h-5 text-cyan-400 mb-2" />
                          <div className="text-[10px] text-fui-gray-500 uppercase">Vector Breadth</div>
                          <div className="text-sm text-fui-gray-200 mt-1">1,343 latent features per chemical node</div>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-8 items-center bg-cyan-950/30 p-6 rounded-xl border border-cyan-500/20">
                        <div>
                          <div className="text-[10px] text-cyan-500 mb-1 uppercase tracking-widest">Test AUC</div>
                          <div className="text-4xl text-white font-bold drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]">98.67%</div>
                        </div>
                        <div className="w-px h-12 bg-cyan-500/30 hidden md:block" />
                        <div>
                          <div className="text-[10px] text-cyan-500 mb-1 uppercase tracking-widest">F1 Score</div>
                          <div className="text-2xl text-cyan-100 font-bold">94.34%</div>
                        </div>
                        <div className="w-px h-12 bg-cyan-500/30 hidden md:block" />
                        <div>
                          <div className="text-[10px] text-cyan-500 mb-1 uppercase tracking-widest">Endpoint Latency</div>
                          <div className="text-2xl text-cyan-100 font-bold">&lt; 20ms</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                </div>
              </motion.div>
            )}
            """

with io.open(file_path, "w", encoding="utf-8") as f:
    f.write(pre_code + new_evolution_ui + post_code)
print("Inject complete without messing up jsx syntaxes!")
