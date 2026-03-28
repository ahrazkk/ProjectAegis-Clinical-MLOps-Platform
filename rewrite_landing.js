const fs = require('fs');

const path = 'c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

// 1. Add SpotlightCard and ScrollPipeline component
const componentsToAdd = 
// Spotlight card effect for feature grid
function SpotlightCard({ children, hoverColor = 'rgba(139,92,246,0.15)', className = '' }) {
  const divRef = React.useRef(null);
  const [position, setPosition] = React.useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = React.useState(0);

  const handleMouseMove = (e) => {
    if (!divRef.current) return;
    const rect = divRef.current.getBoundingClientRect();
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  return (
    <div
      ref={divRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setOpacity(1)}
      onMouseLeave={() => setOpacity(0)}
      className={\\\elative overflow-hidden rounded-xl border border-white/[0.04] bg-white/[0.015] \\\\\\}
    >
      <div
        className="pointer-events-none absolute -inset-px z-0 transition-opacity duration-500"
        style={{
          opacity,
          background: \\\adial-gradient(600px circle at \\\px \\\px, \\\, transparent 40%)\\\,
        }}
      />
      <div className="relative z-10 h-full w-full">
        {children}
      </div>
    </div>
  );
}

// Scroll timeline for architecture
function ScrollPipeline({ steps }) {
  const containerRef = React.useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start center', 'end center']
  });

  return (
    <div ref={containerRef} className="relative py-10 w-full max-w-5xl mx-auto px-4">
      {/* Central Glow Line */}
      <div className="absolute top-0 bottom-0 left-8 md:left-1/2 w-[2px] bg-white/[0.05]">
        <motion.div 
          className="absolute top-0 w-full bg-gradient-to-b from-purple-500 via-pink-500 to-indigo-500 shadow-[0_0_15px_rgba(236,72,153,0.6)]" 
          style={{ height: useTransform(scrollYProgress, [0, 1], ['0%', '100%']) }} 
        />
      </div>

      <div className="space-y-24">
        {steps.map((step, i) => {
          const isEven = i % 2 === 0;
          return (
            <div key={i} className={\\\elative flex flex-col md:flex-row items-center gap-8 \\\\\\}>
               {/* Content Block */}
               <motion.div 
                 initial={{ opacity: 0, x: isEven ? 50 : -50 }}
                 whileInView={{ opacity: 1, x: 0 }}
                 viewport={{ once: true, margin: '-100px' }}
                 transition={{ duration: 0.7, delay: 0.2 }}
                 className="w-full md:w-1/2 flex flex-col pl-20 md:pl-0"
               >
                  <SpotlightCard hoverColor={\\\\\\22\\\} className="p-6 sm:p-8 backdrop-blur-md">
                    <div className="flex items-center gap-4 mb-4">
                      <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.05]">
                        <step.icon className="w-6 h-6" style={{ color: step.accent }} />
                      </div>
                      <h3 className="text-lg sm:text-xl font-heading font-bold text-white/90">{step.title}</h3>
                    </div>
                    <p className="text-sm sm:text-base text-white/40 leading-relaxed font-body">{step.desc}</p>
                  </SpotlightCard>
               </motion.div>
               
               {/* Node/Marker */}
               <div className="absolute left-[24px] md:left-1/2 -translate-x-[45%] md:-translate-x-1/2 w-4 h-4 rounded-full bg-[#040405] border-[3px] border-white/20 z-10 flex items-center justify-center">
                 <motion.div 
                   className="absolute w-8 h-8 rounded-full border border-current opacity-0"
                   style={{ 
                     color: step.accent, 
                     scale: useTransform(scrollYProgress, [i / steps.length, (i + 0.5) / steps.length], [0.5, 1.5]),
                     opacity: useTransform(scrollYProgress, [i / steps.length, (i + 0.2) / steps.length], [0, 1])
                   }}
                 />
                 <motion.div 
                   className="w-full h-full rounded-full bg-current blur-[2px]"
                   style={{ 
                     color: step.accent, 
                     opacity: useTransform(scrollYProgress, [i / steps.length, (i + 0.2) / steps.length], [0, 1]) 
                   }}
                 />
               </div>

               {/* Empty Space for layout */}
               <div className="hidden md:block w-1/2" />
            </div>
          )
        })}
      </div>
    </div>
  )
}
\;

content = content.replace(\"function GradientDivider({ className = '' }) {\", componentsToAdd + '\\nfunction GradientDivider({ className = \\"\\" }) {');

// 2. Replace techStack data
const tcRegex = /const techStack = \\[\\s\\S]*?\\];/;
const newTechStack = \const techStack = [
    { icon: Database, title: 'Data Ingestion & Graph', desc: 'Heterogeneous data from DrugBank, FDA, and PubChem is parsed into a Neo4j Knowledge Graph, creating a macroscopic map of 2k+ drugs and 53k+ clinical pathways.', accent: '#A78BFA' },
    { icon: Brain, title: 'Biomedical Entity Extraction', desc: 'Fine-tuned PubMedBERT extracts precise pharmacological contexts from 30M+ literature abstracts, encoding semantic meaning into dense vector embeddings.', accent: '#8B5CF6' },
    { icon: Network, title: 'GraphSAGE Neural Inference', desc: 'Message-passing neural networks aggregate structural topologies and relational geometries to predict novel drug interactions with 98.67% accuracy.', accent: '#EC4899' },
    { icon: Sparkles, title: 'RAG Explanations', desc: 'The large language model layer cross-references predictive logits against clinical literature, grounding black-box AI within human-readable citations.', accent: '#818CF8' }
  ];\;
content = content.replace(tcRegex, newTechStack);

// 3. Replace Features block
const featuresRegex = /<section id=\"features\"[\\s\\S]*?<\\/section>/;
const newFeatures = \<section id="features" className="py-20 sm:py-36 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <Reveal className="text-center mb-16 sm:mb-24">
            <span className="inline-block py-1.5 px-4 rounded-full bg-white/[0.02] border border-white/[0.05] font-mono text-[10px] text-purple-400 uppercase tracking-[0.2em] mb-6">Platform Capabilities</span>
            <h2 className="mb-6">
              <span className="font-heading text-4xl sm:text-5xl md:text-6xl font-bold">Enterprise-Grade </span>
              <span className="font-cursive italic text-4xl sm:text-5xl md:text-6xl bg-gradient-to-r from-purple-400 via-pink-400 to-indigo-400 bg-clip-text text-transparent">DDI Analysis</span>
            </h2>
            <p className="font-body text-base text-white/40 max-w-2xl mx-auto leading-relaxed">
              Built for clinical decision support with explainable predictions, realtime inference, and comprehensive therapeutic drug coverage.
            </p>
          </Reveal>

          <StaggerReveal className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6" staggerDelay={0.1}>
            {features.map((feature, i) => (
              <StaggerChild key={i} className="h-[250px] sm:h-[300px]">
                <SpotlightCard hoverColor={\\\gba(139,92,246,\\\)\\\} className="h-full group">
                  <div className="p-8 h-full flex flex-col justify-center">
                    <div className="w-12 h-12 rounded-xl bg-white/[0.02] border border-white/[0.05] flex items-center justify-center mb-6 group-hover:scale-110 group-hover:bg-purple-500/10 transition-all duration-500">
                      <feature.icon className="w-5 h-5 text-purple-300/50 group-hover:text-purple-300 transition-colors" />
                    </div>
                    <h3 className="font-heading text-lg font-semibold mb-3 text-white/90">{feature.title}</h3>
                    <p className="font-body text-sm text-white/30 leading-relaxed flex-grow">{feature.description}</p>
                    
                    {/* Glowing structural lines */}
                    <div className="absolute top-0 right-0 w-24 h-24 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none">
                      <div className="absolute top-4 right-4 w-12 h-[1px] bg-gradient-to-l from-purple-500/40 to-transparent" />
                      <div className="absolute top-4 right-4 w-[1px] h-12 bg-gradient-to-b from-purple-500/40 to-transparent" />
                    </div>
                  </div>
                </SpotlightCard>
              </StaggerChild>
            ))}
          </StaggerReveal>
        </div>
      </section>\;
content = content.replace(featuresRegex, newFeatures);

// 4. Replace Technology Block
const techRegex = /<section id=\"technology\"[\\s\\S]*?<\\/section>/;
const newTech = \<section id="technology" className="py-24 sm:py-40 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#080512] to-[#040405]" />
        <div className="absolute inset-0" style={{ background: 'radial-gradient(circle at 50% 30%, rgba(139,92,246,0.03) 0%, transparent 60%)' }} />
        <NoiseOverlay opacity={0.015} />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 relative z-10 text-center mb-16 sm:mb-24">
          <Reveal>
            <span className="inline-block py-1.5 px-4 rounded-full bg-white/[0.02] border border-white/[0.05] font-mono text-[10px] text-pink-400 uppercase tracking-[0.2em] mb-6">Pipeline Architecture</span>
            <h2 className="mb-6">
              <span className="font-heading text-4xl sm:text-5xl md:text-6xl font-bold">Production </span>
              <span className="font-cursive italic text-4xl sm:text-5xl md:text-6xl bg-gradient-to-r from-pink-400 to-indigo-400 bg-clip-text text-transparent">Ecosystem</span>
            </h2>
            <p className="font-body text-base text-white/40 max-w-2xl mx-auto leading-relaxed">
              Trace the lifecycle of a predictive request from massive heterogeneous datasets through our deeply optimized PyTorch infrastructure to realtime clinical outputs.
            </p>
          </Reveal>
        </div>

        <ScrollPipeline steps={techStack} />

      </section>\;
content = content.replace(techRegex, newTech);

fs.writeFileSync(path, content, 'utf8');
console.log('Update script finished successfully.');
