const fs = require('fs');
const path = 'c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

// 1. REWRITE GNNNode TO INCLUDE DRUG CATEGORY / DETAILS ON HOVER
const newGNNNode = `function GNNNode({ x, y, label, color, glow, delay, drugClass, targetMechanism }) {
  return (
    <motion.div
      className="absolute flex flex-col items-center justify-center pointer-events-none group z-20"
      style={{ left: x, top: y, x: '-50%', y: '-50%' }}
      initial={{ scale: 0, opacity: 0 }}
      whileInView={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 100, damping: 15, delay }}
    >
      <div className={\`relative w-16 h-16 rounded-full border-2 \${color} bg-black/80 flex items-center justify-center text-[10px] font-mono text-white/80 shadow-lg pointer-events-auto transition-transform duration-300 group-hover:scale-110 group-hover:bg-white/5\`}>
        {label}
        <div className={\`absolute inset-0 rounded-full \${glow} blur-md -z-10 opacity-50 group-hover:opacity-100 transition-opacity duration-300\`} />
      </div>
      
      {/* Node Tooltip */}
      <div className="absolute top-full mt-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none w-max z-50">
         <div className="bg-black/95 border border-white/10 p-3 rounded-md backdrop-blur-xl shadow-2xl flex flex-col gap-1.5 min-w-[140px]">
            <div className="text-[10px] uppercase font-mono tracking-widest text-purple-400 mb-1 border-b border-white/[0.05] pb-1">{label}</div>
            {drugClass && <div className="text-[10px] text-white/60"><span className="text-white/30">Class:</span> {drugClass}</div>}
            {targetMechanism && <div className="text-[10px] text-white/60"><span className="text-white/30">Target:</span> {targetMechanism}</div>}
         </div>
      </div>
    </motion.div>
  );
}`;

content = content.replace(/function GNNNode.*?return \([\s\S]*?\}\s*\);?\s*\}/, newGNNNode);


// 2. REWRITE ASPIRIN CENTER TO FIX ALIGNMENT
const oldAspirin = `<motion.div 
                      className="absolute z-30 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none flex flex-col items-center"
                      initial={{ scale: 0, opacity: 0 }}
                      whileInView={{ scale: 1, opacity: 1 }}
                      transition={{ type: "spring", stiffness: 100, damping: 15 }}
                    >
                      <div className="text-[10px] font-mono text-[#a78bfa] mb-3 bg-black/60 px-2 py-0.5 rounded border border-purple-500/20">Target Input</div>
                      <div className="w-28 h-28 rounded-full bg-black/80 backdrop-blur-md border border-white flex flex-col items-center justify-center shadow-[0_0_40px_rgba(255,255,255,0.15)] bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-white/10 to-transparent">
                        <span className="font-heading font-medium tracking-wide text-white">Aspirin</span>
                        <span className="text-[9px] text-white/40 font-mono mt-1">CHEMBL123</span>
                      </div>
                      <div className="absolute top-[32px] bottom-0 -inset-8 bg-white/5 rounded-full blur-[20px] -z-10 animate-pulse pointer-events-none" />
                    </motion.div>`;

const newAspirin = `<motion.div 
                      className="absolute z-30 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                      initial={{ scale: 0, opacity: 0 }}
                      whileInView={{ scale: 1, opacity: 1 }}
                      transition={{ type: "spring", stiffness: 100, damping: 15 }}
                    >
                      <div className="absolute -top-10 left-1/2 -translate-x-1/2 text-[10px] font-mono text-[#a78bfa] bg-black/60 px-2 py-0.5 rounded border border-purple-500/20 whitespace-nowrap">Target Input</div>
                      
                      <div className="w-28 h-28 rounded-full bg-black/80 backdrop-blur-md border border-white flex flex-col items-center justify-center shadow-[0_0_40px_rgba(255,255,255,0.15)] bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-white/10 to-transparent relative">
                        <span className="font-heading font-medium tracking-wide text-white">Aspirin</span>
                        <span className="text-[9px] text-white/40 font-mono mt-1">CHEMBL123</span>
                        <div className="absolute inset-0 -m-8 bg-white/5 rounded-full blur-[20px] -z-10 animate-pulse pointer-events-none" />
                      </div>
                    </motion.div>`;

content = content.replace(oldAspirin, newAspirin);


// 3. UPDATE SURROUNDING NODES WITH NEW PROPS
const oldNodes = `<GNNNode x="25%" y="20%" label="Warfarin" color="border-pink-500/80" glow="bg-pink-500/20" delay={0.6} />
                         <GNNNode x="75%" y="25%" label="Clopidogrel" color="border-white/20" glow="bg-white/5" delay={0.8} />
                         <GNNNode x="80%" y="70%" label="Ibuprofen" color="border-yellow-500/50" glow="bg-yellow-500/10" delay={1.0} />
                         <GNNNode x="30%" y="75%" label="Apixaban" color="border-purple-500" glow="bg-purple-500/20" delay={1.2} />`;

const newNodes = `<GNNNode x="25%" y="20%" label="Warfarin" drugClass="Anticoagulant" targetMechanism="VKORC1 Inhibitor" color="border-pink-500/80" glow="bg-pink-500/20" delay={0.6} />
                         <GNNNode x="75%" y="25%" label="Clopidogrel" drugClass="Antiplatelet" targetMechanism="P2Y12 Antagonist" color="border-white/20" glow="bg-white/5" delay={0.8} />
                         <GNNNode x="80%" y="70%" label="Ibuprofen" drugClass="NSAID" targetMechanism="COX Inhibitor" color="border-yellow-500/50" glow="bg-yellow-500/10" delay={1.0} />
                         <GNNNode x="30%" y="75%" label="Apixaban" drugClass="Anticoagulant" targetMechanism="Factor Xa Inhibitor" color="border-purple-500" glow="bg-purple-500/20" delay={1.2} />`;

content = content.replace(oldNodes, newNodes);

// 4. ADD SENTENCE UNDER PROBABILITY DENSITY
const oldProb = `<div className="h-[200px] w-full relative">`;
const newProb = `<p className="text-[10px] text-white/40 mb-4 font-body leading-relaxed">
                        The GNN embedding space maps high-dimensional topological interactions. The 0.56 threshold boundary confidently segments clinically severe interactions from benign relationships based on edge-weight gradients.
                     </p>
                     <div className="h-[200px] w-full relative">`;

content = content.replace(oldProb, newProb);

fs.writeFileSync(path, content, 'utf8');
console.log("Done updates!");
