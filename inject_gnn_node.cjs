const fs = require('fs');
const path = 'c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

// The new component to be added outside LandingPage
const gnnNodeComponent = `
// Interactive node for the Graph Data section
function GNNNode({ x, y, label, effect, color, glow, delay }) {
  return (
    <motion.div 
      className="absolute flex flex-col items-center justify-center group"
      style={{ left: x, top: y, x: '-50%', y: '-50%' }}
      initial={{ scale: 0, opacity: 0 }}
      whileInView={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 100, damping: 15, delay }}
      whileHover={{ scale: 1.1, zIndex: 50 }}
    >
      <div className={\`relative w-16 h-16 rounded-full border-2 \${color} bg-black/80 flex items-center justify-center text-[10px] font-mono text-white/80 cursor-pointer shadow-lg z-10 transition-all duration-300 group-hover:bg-white/5\`}>
        {label}
        <div className={\`absolute inset-0 rounded-full \${glow} blur-md -z-10 opacity-50 group-hover:opacity-100 transition-opacity duration-300\`} />
      </div>
      
      {/* Tooltip that appears on hover */}
      <div className="absolute top-full mt-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none w-max z-50">
         <div className="bg-black/90 border border-white/10 px-4 py-2 rounded-md backdrop-blur-xl shadow-2xl">
            <div className="text-[10px] uppercase font-mono tracking-widest text-[#a78bfa] mb-1">Inference</div>
            <div className="text-xs text-white font-medium">{effect}</div>
         </div>
      </div>
    </motion.div>
  );
}
`;

// Insert the component right before 'export default function LandingPage'
content = content.replace('export default function LandingPage', gnnNodeComponent + '\nexport default function LandingPage');

// The new section JSX
const newSection = `
      {/* ═══════════ GRAPH STATS & MOCK ═══════════ */}
      <section className="relative py-32 z-10">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 relative">
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-8 border border-white/5 bg-white/[0.01] backdrop-blur-3xl overflow-hidden min-h-[600px]">
            
            {/* Left Side: Stats */}
            <div className="col-span-1 lg:col-span-4 p-10 lg:p-14 flex flex-col justify-center relative bg-gradient-to-br from-black/60 to-black/20 z-10 border-r border-white/5">
               <Reveal>
                  <h3 className="text-xl font-heading font-semibold tracking-tight text-white mb-10 border-b border-white/[0.03] pb-6">Model Performance & Scale</h3>
                  <div className="space-y-10">
                     {[
                       { label: 'Macro F1 Score', value: '0.968', desc: 'Predictive precision across 86 side-effect classes' },
                       { label: 'Model ROC-AUC', value: '98.67%', desc: 'Exceptionally high area under the curve classifying missing edges' },
                       { label: 'Embedded Nodes', value: '14,200+', desc: 'Unique drugs and molecular entities mapped into the latent space' },
                       { label: 'Computed Edges', value: '2.4M+', desc: 'Known interactions and metabolic pathways forming the graph topology' }
                     ].map((stat, i) => (
                        <div key={i} className="group cursor-default">
                           <div className="text-[10px] font-mono uppercase tracking-widest text-pink-400 mb-2">{stat.label}</div>
                           <div className="text-4xl font-heading font-normal text-white mb-2 group-hover:text-purple-400 transition-colors">{stat.value}</div>
                           <div className="text-xs text-white/40 font-body leading-relaxed">{stat.desc}</div>
                        </div>
                     ))}
                  </div>
               </Reveal>
            </div>

            {/* Right Side: Interactive Node Mock */}
            <div className="col-span-1 lg:col-span-8 relative overflow-hidden bg-black/40 min-h-[500px]">
                {/* Background Grid & Blurs */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f10_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f10_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[30rem] h-[30rem] bg-purple-500/10 rounded-full blur-[100px] mix-blend-screen pointer-events-none" />
                
                {/* Label overlay */}
                <div className="absolute top-8 right-8 text-right z-30 pointer-events-none hidden sm:block">
                  <div className="text-[10px] font-mono tracking-widest text-white/30 uppercase mb-1">Visualization</div>
                  <div className="text-sm font-heading text-white/60">GNN Node Connectivity</div>
                </div>

                {/* Graph Visualization Container */}
                <div className="relative w-full h-full flex items-center justify-center">
                    
                    {/* SVG Connecting Lines */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none">
                       {/* Center to Top-Left */}
                       <motion.line x1="50%" y1="50%" x2="25%" y2="25%" stroke="rgba(236,72,153,0.3)" strokeWidth="2" strokeDasharray="4 4"
                          initial={{ pathLength: 0, opacity: 0 }} whileInView={{ pathLength: 1, opacity: 1 }} transition={{ duration: 1.5, delay: 0.5 }} />
                       {/* Center to Top-Right */}
                       <motion.line x1="50%" y1="50%" x2="75%" y2="30%" stroke="rgba(255,255,255,0.1)" strokeWidth="1.5"
                          initial={{ pathLength: 0, opacity: 0 }} whileInView={{ pathLength: 1, opacity: 1 }} transition={{ duration: 1.5, delay: 0.7 }} />
                       {/* Center to Bottom-Right */}
                       <motion.line x1="50%" y1="50%" x2="80%" y2="70%" stroke="rgba(234,179,8,0.3)" strokeWidth="2"
                          initial={{ pathLength: 0, opacity: 0 }} whileInView={{ pathLength: 1, opacity: 1 }} transition={{ duration: 1.5, delay: 0.9 }} />
                       {/* Center to Bottom-Left */}
                       <motion.line x1="50%" y1="50%" x2="30%" y2="80%" stroke="rgba(168,85,247,0.3)" strokeWidth="2" strokeDasharray="2 6"
                          initial={{ pathLength: 0, opacity: 0 }} whileInView={{ pathLength: 1, opacity: 1 }} transition={{ duration: 1.5, delay: 1.1 }} />
                    </svg>

                    {/* Surrounding Nodes */}
                    <div className="absolute inset-0 pointer-events-none">
                       <div className="pointer-events-auto w-full h-full relative">
                         <GNNNode x="25%" y="25%" label="Warfarin" effect="Severe Bleeding (0.98)" color="border-pink-500/80" glow="bg-pink-500/20" delay={0.6} />
                         <GNNNode x="75%" y="30%" label="Clopidogrel" effect="Unknown Interaction (0.12)" color="border-white/20" glow="bg-white/5" delay={0.8} />
                         <GNNNode x="80%" y="70%" label="Ibuprofen" effect="Mild GI Risk (0.83)" color="border-yellow-500/50" glow="bg-yellow-500/10" delay={1.0} />
                         <GNNNode x="30%" y="80%" label="Apixaban" effect="Bleeding Risk (0.92)" color="border-purple-500" glow="bg-purple-500/20" delay={1.2} />
                       </div>
                    </div>

                    {/* Center Node */}
                    <motion.div 
                      className="absolute z-20 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 cursor-crosshair group"
                      initial={{ scale: 0, opacity: 0 }}
                      whileInView={{ scale: 1, opacity: 1 }}
                      transition={{ type: "spring", stiffness: 100, damping: 15 }}
                    >
                      <div className="w-28 h-28 rounded-full bg-black/50 backdrop-blur-md border border-white text-white flex items-center justify-center font-heading font-medium tracking-wide shadow-[0_0_40px_rgba(255,255,255,0.15)] group-hover:scale-105 transition-transform duration-500">
                        Aspirin
                      </div>
                      <div className="absolute -inset-6 bg-white/5 rounded-full blur-[15px] -z-10 animate-pulse pointer-events-none" />
                    </motion.div>
                </div>

                {/* Legend */}
                <div className="absolute bottom-6 sm:bottom-10 left-6 sm:left-10 flex flex-wrap items-center gap-6 text-[10px] uppercase font-mono tracking-widest text-white/40 z-30">
                   <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-pink-500/80 shadow-[0_0_10px_rgba(236,72,153,0.5)]" /> Severe</div>
                   <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-yellow-500/80 shadow-[0_0_10px_rgba(234,179,8,0.5)]" /> Moderate</div>
                   <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-white/20" /> Benign</div>
                </div>

            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ ARCHITECTURE SCROLL PARALLAX`;

// Insert the new section
content = content.replace('{/* ═══════════ ARCHITECTURE SCROLL PARALLAX', newSection);

fs.writeFileSync(path, content, 'utf8');
console.log("Successfully added GNN Graph Stats & Mock!");
