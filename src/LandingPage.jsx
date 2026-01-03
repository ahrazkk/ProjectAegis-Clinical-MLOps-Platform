import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { Hexagon, ArrowRight, X, FileText } from 'lucide-react';
import ParticleSystem from './components/MolecularParticles';

export default function LandingPage({ onEnter }) {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [showReport, setShowReport] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => {
      if (!containerRef.current) return;
      const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
      const maxScroll = scrollHeight - clientHeight;
      const progress = Math.min(Math.max(scrollTop / maxScroll, 0), 1);
      setScrollProgress(progress);
    };

    const container = containerRef.current;
    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div
      className="relative w-full h-screen bg-[#0B0F19] overflow-hidden text-white font-sans selection:bg-blue-500/30"
      style={{ backgroundColor: '#0B0F19', color: 'white', minHeight: '100vh', width: '100%' }}
    >
      <div className="absolute inset-0 z-0" style={{ pointerEvents: 'none' }}>
        <Canvas
          camera={{ position: [0, 0, 12], fov: 45 }}
          gl={{ alpha: true }}
          style={{ background: 'transparent' }}
        >
          <ParticleSystem scrollProgress={scrollProgress} />
          <ambientLight intensity={0.5} />
          <EffectComposer>
            <Bloom
              intensity={0.1}
              luminanceThreshold={1.2}
              luminanceSmoothing={0.025}
              mipmapBlur
            />
          </EffectComposer>
        </Canvas>
      </div>

      <div
        ref={containerRef}
        className="absolute inset-0 z-10 overflow-y-auto custom-scrollbar scroll-smooth"
      >
        <header className="fixed top-0 left-0 w-full px-8 py-6 flex justify-between items-center z-50 mix-blend-difference">
          <div className="flex items-center gap-3">
            <Hexagon className="w-8 h-8 text-white" strokeWidth={1.5} />
            <span className="text-xl font-bold tracking-wider">MoleculeAI</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowReport(true)}
              className="text-xs font-bold tracking-widest uppercase opacity-80 hover:opacity-100 hover:text-blue-400 transition-colors flex items-center gap-2"
            >
              <FileText className="w-4 h-4" /> Research
            </button>
            <div className="w-8 h-8 border border-white/30 rotate-45 flex items-center justify-center">
              <div className="w-1 h-1 bg-white rounded-full" />
            </div>
          </div>
        </header>

        <section className="h-screen flex flex-col justify-center px-8 lg:px-24 max-w-7xl mx-auto relative">
          <div className="max-w-4xl">
            <h1 className="text-5xl lg:text-7xl font-light leading-tight mb-8">
              Advanced <span className="font-serif italic text-blue-300">Geometric Deep Learning</span> for Drug Interaction Prediction
            </h1>
            <p className="text-xl text-slate-400 mb-10 max-w-2xl leading-relaxed">
              Leveraging Graph Neural Networks (GNNs) and heterogeneous knowledge graphs to predict adverse drug events with explainable AI.
            </p>
            <div className="flex gap-4">
              <button
                onClick={onEnter}
                className="group flex items-center gap-3 px-8 py-4 bg-white text-slate-900 rounded-full font-bold tracking-wide hover:bg-blue-50 transition-all duration-300"
              >
                Launch Platform <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
              <button
                onClick={() => setShowReport(true)}
                className="px-8 py-4 border border-white/20 rounded-full font-bold tracking-wide hover:bg-white/10 transition-all duration-300"
              >
                Read the Report
              </button>
            </div>
          </div>

          <div className="absolute bottom-10 right-10 flex flex-col items-center gap-2 opacity-50 animate-bounce">
            <div className="w-px h-12 bg-gradient-to-b from-transparent to-white" />
            <span className="text-[10px] uppercase tracking-widest writing-vertical-rl">Scroll</span>
          </div>
        </section>

        <section className="h-screen flex items-center px-8 lg:px-24 max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
            <div className="space-y-8">
              <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center border border-blue-500/20">
                <Hexagon className="w-6 h-6 text-blue-400" />
              </div>
              <h2 className="text-4xl font-light">
                Heterogeneous Graphs & <br />
                <span className="font-serif italic text-purple-300">Explainable AI</span>
              </h2>
              <p className="text-slate-400 leading-relaxed text-lg">
                Our architecture integrates GraphRAG for retrieval-augmented generation, allowing researchers to trace predictions back to source literature and molecular substructures.
              </p>

              <div className="grid grid-cols-2 gap-8 pt-8 border-t border-white/10">
                <div>
                  <div className="text-3xl font-bold text-white mb-1">180+</div>
                  <div className="text-sm text-slate-500 uppercase tracking-wider">Prediction Methods</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-white mb-1">ISO 25010</div>
                  <div className="text-sm text-slate-500 uppercase tracking-wider">Quality Standard</div>
                </div>
              </div>
            </div>
            <div className="hidden lg:block" />
          </div>
        </section>

        <section className="h-[50vh] flex flex-col items-center justify-center text-center px-8">
          <h2 className="text-3xl lg:text-5xl font-light mb-8">Ready to explore the future?</h2>
          <button
            onClick={onEnter}
            className="px-10 py-5 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full font-bold text-lg shadow-lg shadow-blue-900/20 hover:scale-105 transition-transform"
          >
            Enter Dashboard
          </button>
        </section>

        <footer className="py-12 border-t border-white/10 text-center text-slate-500 text-sm">
          <p> 2025 MoleculeAI. All rights reserved.</p>
        </footer>
      </div>

      {showReport && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-[#131B2C] w-full max-w-4xl h-[80vh] rounded-3xl border border-white/10 shadow-2xl flex flex-col overflow-hidden relative animate-in fade-in zoom-in-95 duration-300">
            <button
              onClick={() => setShowReport(false)}
              className="absolute top-6 right-6 p-2 bg-white/5 rounded-full hover:bg-white/10 transition-colors z-10"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>

            <div className="p-8 border-b border-white/5 bg-[#0B0F19]">
              <div className="flex items-center gap-3 mb-2">
                <span className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs font-bold uppercase tracking-wider border border-blue-500/20">Technical Report</span>
                <span className="text-slate-500 text-xs">November 2024</span>
              </div>
              <h2 className="text-2xl font-bold text-white">Advanced Computational Frameworks for DDI Prediction</h2>
            </div>

            <div className="flex-1 overflow-y-auto p-8 custom-scrollbar prose prose-invert prose-lg max-w-none">
              <p className="lead text-xl text-slate-300">
                This report reviews the engineering architecture of a modern Drug-Drug Interaction (DDI) prediction system leveraging Geometric Deep Learning.
              </p>

              <h3 className="text-blue-400">1. Introduction</h3>
              <p>
                Adverse drug events remain a critical challenge in pharmacology. Traditional methods often fail to capture complex non-linear interactions.
                Our approach utilizes <strong>Graph Neural Networks (GNNs)</strong> to model molecules as graphs, where atoms are nodes and bonds are edges.
              </p>

              <h3 className="text-blue-400">2. System Architecture</h3>
              <ul className="list-disc pl-5 space-y-2 text-slate-300">
                <li><strong>Data Ingestion:</strong> Aggregates data from DrugBank, PubChem, and TwoSides.</li>
                <li><strong>Graph Construction:</strong> Builds heterogeneous graphs representing drugs, proteins, and side effects.</li>
                <li><strong>GNN Encoder:</strong> Uses Message Passing Neural Networks (MPNNs) to generate molecular embeddings.</li>
                <li><strong>Link Prediction:</strong> A decoder network predicts the likelihood of an edge (interaction) between two drug nodes.</li>
              </ul>

              <h3 className="text-blue-400">3. Explainability (XAI)</h3>
              <p>
                To ensure clinical trust, we implement <strong>GNNExplainer</strong>, which identifies the specific subgraphs (functional groups) responsible for a predicted interaction.
                This allows clinicians to understand <em>why</em> an interaction is flagged.
              </p>

              <h3 className="text-blue-400">4. Future Directions</h3>
              <p>
                We are currently integrating <strong>Large Language Models (LLMs)</strong> via GraphRAG to synthesize textual evidence from medical literature,
                enabling a dual-modality prediction (Structure + Text).
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
