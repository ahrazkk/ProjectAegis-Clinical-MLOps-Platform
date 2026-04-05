const fs = require('fs');
const path = require('path');

const targetPath = path.join(__dirname, 'src/pages/LandingPageV2.jsx');
let fileContent = fs.readFileSync(targetPath, 'utf8');

// Find the start of STATS
const statsMarker = " {/* ═══════════ STATS";
const startIndex = fileContent.indexOf(statsMarker);

if (startIndex === -1) {
  console.error("Could not find STATS marker");
  process.exit(1);
}

// Keep the top part
const topHalf = fileContent.substring(0, startIndex);

const newDesign = `
      {/* ═══════════ BENTO GRID FEATURES ═══════════ */}
      <section className="relative py-32 z-10" id="features">
        <div className="absolute inset-0 bg-radial-gradient from-purple-500/[0.02] to-transparent pointer-events-none" />
        <div className="max-w-7xl mx-auto px-6 lg:px-8 relative">
          <Reveal>
            <div className="text-center max-w-2xl mx-auto mb-20 lg:mb-32">
              <h2 className="text-sm font-mono tracking-[0.3em] font-medium text-pink-400/80 mb-6 uppercase">Platform Capabilities</h2>
              <p className="mt-2 text-4xl lg:text-6xl font-heading font-semibold tracking-tight text-white mb-6 leading-tight">
                Designed for precision. <br/>
                <span className="text-white/40">Built for scale.</span>
              </p>
              <p className="text-lg text-white/50 font-body max-w-xl mx-auto leading-relaxed">
                Experience unparalleled predictive accuracy powered by specialized Graph Neural Networks and real-time inference.
              </p>
            </div>
          </Reveal>

          <div className="grid grid-cols-1 md:grid-cols-6 lg:grid-cols-12 gap-6 relative">
            <div className="absolute top-[20%] left-[-10%] w-[40rem] h-[40rem] bg-pink-500/10 rounded-full blur-[120px] pointer-events-none mix-blend-screen" />
            <div className="absolute bottom-[-10%] right-[-10%] w-[40rem] h-[40rem] bg-purple-500/10 rounded-full blur-[120px] pointer-events-none mix-blend-screen" />
            
            {/* Feature 1 - Large spanning across 8 cols */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-100px" }} transition={{ duration: 0.8, ease: "easeOut" }}
              className="md:col-span-6 lg:col-span-8 group relative flex flex-col justify-between overflow-hidden border border-white/5 bg-white/[0.01] backdrop-blur-xl p-8 lg:p-12 min-h-[400px]"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-pink-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
              <div className="relative z-10">
                <div className="w-12 h-12 rounded bg-pink-500/10 flex items-center justify-center text-pink-400 mb-8 border border-pink-500/20">
                  <Activity className="w-6 h-6" />
                </div>
                <h3 className="text-2xl font-heading font-medium text-white mb-4">GNN Engine</h3>
                <p className="text-white/50 font-body leading-relaxed max-w-md">
                  Our proprietary Graph Neural Network models analyze billions of distinct molecular interaction pathways with staggering 97% accuracy. We map the unseen topology of complex therapies.
                </p>
              </div>
              <div className="relative z-10 mt-12 flex space-x-4">
                <div className="h-1 flex-1 bg-white/5 overflow-hidden"><motion.div className="h-full bg-pink-400" initial={{ width: 0 }} whileInView={{ width: "97%" }} transition={{ delay: 0.5, duration: 1.5 }} /></div>
              </div>
            </motion.div>

            {/* Feature 2 - Small box 4 cols */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-100px" }} transition={{ duration: 0.8, ease: "easeOut", delay: 0.1 }}
              className="md:col-span-6 lg:col-span-4 group relative flex flex-col justify-between overflow-hidden border border-white/5 bg-white/[0.01] backdrop-blur-xl p-8 lg:p-12 min-h-[400px]"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
              <div className="relative z-10">
                <div className="w-12 h-12 rounded bg-purple-500/10 flex items-center justify-center text-purple-400 mb-8 border border-purple-500/20">
                  <Cpu className="w-6 h-6" />
                </div>
                <h3 className="text-2xl font-heading font-medium text-white mb-4">Microsecond Inference</h3>
                <p className="text-white/50 font-body leading-relaxed">
                  Engineered with an ultra-low latency architecture executing multi-modal predictions concurrently across edge contexts.
                </p>
              </div>
            </motion.div>

            {/* Feature 3 - Small box 4 cols */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-100px" }} transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
              className="md:col-span-6 lg:col-span-4 group relative flex flex-col justify-between overflow-hidden border border-white/5 bg-white/[0.01] backdrop-blur-xl p-8 lg:p-12 min-h-[400px]"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
              <div className="relative z-10">
                <div className="w-12 h-12 rounded bg-cyan-500/10 flex items-center justify-center text-cyan-400 mb-8 border border-cyan-500/20">
                  <ShieldAlert className="w-6 h-6" />
                </div>
                <h3 className="text-2xl font-heading font-medium text-white mb-4">FDA & EMA Aligned</h3>
                <p className="text-white/50 font-body leading-relaxed">
                  Synchronizing with global pharmacological directives via our automated regulatory NLP compliance pipelines.
                </p>
              </div>
            </motion.div>

            {/* Feature 4 - Large spanning across 8 cols */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-100px" }} transition={{ duration: 0.8, ease: "easeOut", delay: 0.3 }}
              className="md:col-span-6 lg:col-span-8 group relative flex flex-col justify-between overflow-hidden border border-white/5 bg-white/[0.01] backdrop-blur-xl p-8 lg:p-12 min-h-[400px]"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-pink-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
           
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 relative z-10 h-full">
                <div className="flex flex-col justify-center">
                  <div className="w-12 h-12 rounded bg-pink-500/10 flex items-center justify-center text-pink-400 mb-8 border border-pink-500/20">
                    <Database className="w-6 h-6" />
                  </div>
                  <h3 className="text-2xl font-heading font-medium text-white mb-4">Unified Patient Vectors</h3>
                  <p className="text-white/50 font-body leading-relaxed">
                    Aggregate EHR, multi-omic profiles, and continuous telemetry strings into highly-dimensional patient vector embeddings.
                  </p>
                </div>
                <div className="flex items-center justify-center">
                   <div className="relative w-full h-[150px] border border-white/10 bg-black/50 p-6 flex flex-col gap-3">
                      <motion.div className="h-2 bg-gradient-to-r from-pink-500/50 to-purple-500/50 rounded-full" initial={{ width: "20%" }} whileInView={{ width: "85%" }} transition={{ delay: 0.8, duration: 1.5 }} />
                      <motion.div className="h-2 bg-gradient-to-r from-purple-500/50 to-cyan-500/50 rounded-full" initial={{ width: "10%" }} whileInView={{ width: "65%" }} transition={{ delay: 0.9, duration: 1.5 }} />
                      <motion.div className="h-2 bg-gradient-to-r from-cyan-500/50 to-blue-500/50 rounded-full" initial={{ width: "5%" }} whileInView={{ width: "45%" }} transition={{ delay: 1.0, duration: 1.5 }} />
                   </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ═══════════ ARCHITECTURE SCROLL PARALLAX ═══════════ */}
      <section className="relative py-40 z-10 border-t border-white/[0.03] overflow-hidden bg-black/50" id="technology">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
            
            <div className="max-w-xl">
              <Reveal>
                <div className="flex items-center gap-3 mb-8">
                  <div className="h-[1px] w-8 bg-purple-500"></div>
                  <span className="text-sm font-mono tracking-[0.3em] font-medium text-purple-400 uppercase">Architecture</span>
                </div>
                <h2 className="text-4xl lg:text-6xl font-heading font-semibold tracking-tight text-white mb-8 leading-tight">
                  A modern technical foundation.
                </h2>
                <p className="text-lg text-white/50 font-body leading-relaxed mb-12">
                  Built to scale dynamically under massive load. By leveraging asynchronous Python workers, Neo4j graph databases, and an end-to-end multi-layered ML workflow, we achieve sub-millisecond query responses for critical clinical interactions.
                </p>
              </Reveal>

              <div className="space-y-10">
                {[
                  { title: "React & Vite Front-End", desc: "Ultra-fast module reloading, Tailwind CSS for deterministic styling, and Framer Motion for performant 60fps WebGL-like animations." },
                  { title: "FastAPI Backend", desc: "A concurrent Python layer validating requests with Pydantic and orchestrating heavily-distributed Celery workers." },
                  { title: "Neo4j Knowledge Graph", desc: "Graph representations of 20M+ known drug-protein-gene pathways to fuel the GNN embedding layer." }
                ].map((item, i) => (
                  <Reveal key={i} delay={i * 0.1}>
                    <div className="relative pl-8 before:absolute before:left-0 before:top-2 before:w-1.5 before:h-1.5 before:bg-pink-500 before:rounded-sm group">
                      <h4 className="text-xl font-heading font-medium text-white mb-2 group-hover:text-pink-400 transition-colors">{item.title}</h4>
                      <p className="text-white/40 font-body leading-relaxed text-sm">{item.desc}</p>
                    </div>
                  </Reveal>
                ))}
              </div>
            </div>

            {/* Visual Abstract Representation */}
            <div className="relative h-[600px] w-full border border-white/5 bg-white/[0.01] backdrop-blur-xl flex items-center justify-center overflow-hidden">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-purple-500/10 via-transparent to-transparent opacity-50" />
                
                <motion.div 
                  className="absolute w-64 h-64 border border-pink-500/20 rounded-full"
                  animate={{ rotate: 360, scale: [1, 1.05, 1] }} 
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                />
                <motion.div 
                  className="absolute w-96 h-96 border border-purple-500/20 rounded-full"
                  animate={{ rotate: -360, scale: [1, 1.1, 1] }} 
                  transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
                />
                
                {/* Floating code block representing logic */}
                <motion.div 
                  initial={{ y: 0 }}
                  animate={{ y: [-15, 15, -15] }}
                  transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                  className="relative z-10 bg-black/80 border border-white/10 p-6 shadow-2xl backdrop-blur-md font-mono text-xs text-pink-400/80 w-[80%]"
                >
                  <div className="flex gap-2 mb-4">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/50"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/50"></div>
                  </div>
                  <p>const inference = await model.predict(</p>
                  <p className="pl-4">patientVector,</p>
                  <p className="pl-4">drugGraphContext;</p>
                  <p className="pl-4">temperature=0.1</p>
                  <p>);</p>
                  <br />
                  <p className="text-purple-400">return NextResponse.json(inference)</p>
                </motion.div>
            </div>

          </div>
        </div>
      </section>

      {/* ═══════════ CTA SECTION ═══════════ */}
      <section className="relative py-40 z-10 border-t border-white/[0.03] overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-pink-500/[0.02]" />
        
        <div className="max-w-4xl mx-auto px-6 text-center relative z-10">
          <Reveal>
            <h2 className="text-4xl lg:text-7xl font-heading font-semibold tracking-tight text-white mb-8">
              Begin modernizing your <br />clinical workflows.
            </h2>
            <p className="text-lg text-white/40 mb-12 max-w-2xl mx-auto">
              Access the most powerful GNN-based DDI prediction platform ever built. Enter the dashboard and revolutionize discovery.
            </p>
          </Reveal>
          
          <Reveal delay={0.2}>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
              <motion.button 
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate('/dashboard')}
                className="group relative px-12 py-5 border border-pink-400/80 bg-white/5 backdrop-blur-md text-white font-heading font-semibold tracking-widest overflow-hidden rounded-none hover:bg-white/10 transition-all w-full sm:w-auto"
              >
                 <span className="relative flex items-center justify-center gap-3 text-sm">
                   Launch Platform <ExternalLink className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                 </span>
              </motion.button>
              
              <button 
                className="text-white/50 hover:text-white font-mono tracking-widest text-sm uppercase transition-colors"
                onClick={() => {/* Implement contact or secondary */}}
              >
                 Contact Team
              </button>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══════════ FOOTER ═══════════ */}
      <footer className="py-12 border-t border-white/[0.03] bg-black relative z-10">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
             <div className="flex flex-col">
               <span className="font-heading font-bold text-lg tracking-wider text-white">Project <span className="font-cursive italic font-normal">Aegis</span></span>
               <span className="text-xs text-white/30 mt-2 font-mono">T. Ghori, A. Kibria, R. Nazimuddin, K. Chaudhari</span>
             </div>
             
             <div className="flex items-center gap-6">
                <a href="https://github.com/ahrazkk/ProjectAegis-Clinical-MLOps-Platform" target="_blank" rel="noopener noreferrer" className="text-white/20 hover:text-pink-400 transition-colors">
                  <Github className="w-5 h-5" />
                </a>
                <a href="mailto:1kibriaahr@gmail.com" className="text-white/20 hover:text-pink-400 transition-colors">
                  <Mail className="w-5 h-5" />
                </a>
                <a href="#" className="text-white/20 hover:text-pink-400 transition-colors">
                  <Linkedin className="w-5 h-5" />
                </a>
             </div>
          </div>
          <div className="mt-12 pt-8 border-t border-white/[0.03] flex justify-between items-center text-[10px] text-white/20 font-mono uppercase tracking-widest">
            <span>&copy; 2026 Aegis Predictive AI</span>
            <span>All Systems Operational</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
`;

const newFileContent = topHalf + newDesign;
fs.writeFileSync(targetPath, newFileContent, 'utf8');
console.log("Successfully rebuilt the LandingPageV2 bottom section!");
