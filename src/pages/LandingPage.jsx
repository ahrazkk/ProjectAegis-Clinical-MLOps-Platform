import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowRight, 
  Shield, 
  Brain, 
  Network, 
  Sparkles, 
  ChevronDown,
  Zap,
  Activity,
  Database,
  GitBranch,
  Lock,
  BarChart3,
  X,
  Github,
  Linkedin
} from 'lucide-react';
import ParticleSystem from '../components/MolecularParticles';

const features = [
  {
    icon: Brain,
    title: 'Graph Neural Networks',
    description: 'Advanced GNN architecture processes molecular graphs to predict interactions with 94% accuracy'
  },
  {
    icon: Network,
    title: 'Knowledge Graph Integration',
    description: 'Neo4j-powered knowledge graph connects drugs, proteins, pathways, and literature'
  },
  {
    icon: Shield,
    title: 'Explainable AI',
    description: 'GNNExplainer identifies substructures responsible for predicted interactions'
  },
  {
    icon: Zap,
    title: 'Real-time Analysis',
    description: 'Sub-second predictions powered by optimized PyTorch models'
  },
  {
    icon: Activity,
    title: 'Multi-drug Analysis',
    description: 'Analyze polypharmacy scenarios with N-way interaction detection'
  },
  {
    icon: Database,
    title: 'Comprehensive Database',
    description: 'Integrated DrugBank, PubChem, and TwoSides datasets'
  }
];

const stats = [
  { value: '500K+', label: 'Drug Pairs Analyzed' },
  { value: '94.2%', label: 'Model Accuracy' },
  { value: '<100ms', label: 'Prediction Time' },
  { value: '86+', label: 'Interaction Types' }
];

export default function LandingPage() {
  const navigate = useNavigate();
  const [scrollY, setScrollY] = useState(0);
  const [showModal, setShowModal] = useState(false);
  const heroRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollProgress = Math.min(scrollY / 500, 1);

  return (
    <div className="min-h-screen bg-[#030712] text-white overflow-x-hidden">
      {/* Navigation */}
      <motion.nav 
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-[#030712]/80 border-b border-white/5"
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
              <GitBranch className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold tracking-tight">
              Project<span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Aegis</span>
            </span>
          </div>
          
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-sm text-slate-400 hover:text-white transition-colors">Features</a>
            <a href="#technology" className="text-sm text-slate-400 hover:text-white transition-colors">Technology</a>
            <button 
              onClick={() => setShowModal(true)}
              className="text-sm text-slate-400 hover:text-white transition-colors"
            >
              Research
            </button>
          </div>

          <button 
            onClick={() => navigate('/dashboard')}
            className="px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg text-sm font-semibold hover:shadow-lg hover:shadow-blue-500/25 transition-all"
          >
            Launch Platform
          </button>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <section ref={heroRef} className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
        {/* 3D Background */}
        <div className="absolute inset-0 z-0">
          <Canvas camera={{ position: [0, 0, 15], fov: 45 }} gl={{ alpha: true }}>
            <ParticleSystem scrollProgress={scrollProgress} />
            <ambientLight intensity={0.4} />
            <EffectComposer>
              <Bloom intensity={0.3} luminanceThreshold={0.8} luminanceSmoothing={0.9} />
            </EffectComposer>
          </Canvas>
        </div>

        {/* Gradient overlays */}
        <div className="absolute inset-0 bg-gradient-to-b from-blue-900/20 via-transparent to-[#030712] pointer-events-none" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_transparent_0%,_#030712_70%)] pointer-events-none" />

        {/* Content */}
        <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-8">
              <Sparkles className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-blue-300">Powered by Graph Neural Networks</span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold leading-tight mb-6">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-white to-slate-400">
                Drug Interaction
              </span>
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-teal-400">
                Intelligence
              </span>
            </h1>

            <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
              Advanced AI-powered platform for predicting drug-drug interactions using 
              geometric deep learning and heterogeneous knowledge graphs.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <motion.button 
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate('/dashboard')}
                className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl text-lg font-semibold shadow-xl shadow-blue-500/20 hover:shadow-blue-500/40 transition-all flex items-center gap-3"
              >
                Enter Dashboard
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </motion.button>
              
              <motion.button 
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowModal(true)}
                className="px-8 py-4 bg-white/5 border border-white/10 rounded-xl text-lg font-semibold hover:bg-white/10 transition-all"
              >
                View Research
              </motion.button>
            </div>
          </motion.div>

          {/* Scroll indicator */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
            className="absolute bottom-10 left-1/2 -translate-x-1/2"
          >
            <motion.div 
              animate={{ y: [0, 8, 0] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="flex flex-col items-center gap-2"
            >
              <span className="text-xs text-slate-500 uppercase tracking-widest">Scroll</span>
              <ChevronDown className="w-5 h-5 text-slate-500" />
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative py-24 border-y border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400 mb-2">
                  {stat.value}
                </div>
                <div className="text-sm text-slate-500 uppercase tracking-wider">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-32">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="text-sm text-cyan-400 uppercase tracking-widest mb-4 block">Capabilities</span>
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Enterprise-Grade DDI Analysis
            </h2>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              Built for clinical decision support with explainable predictions and comprehensive drug coverage
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="group p-8 rounded-2xl bg-gradient-to-b from-slate-900/50 to-transparent border border-white/5 hover:border-blue-500/30 transition-all"
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <feature.icon className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-slate-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section id="technology" className="py-32 bg-gradient-to-b from-transparent via-blue-950/10 to-transparent">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <span className="text-sm text-cyan-400 uppercase tracking-widest mb-4 block">Technology Stack</span>
              <h2 className="text-4xl font-bold mb-6">
                Cutting-Edge Architecture
              </h2>
              <p className="text-lg text-slate-400 mb-8 leading-relaxed">
                Our platform combines state-of-the-art graph neural networks with 
                retrieval-augmented generation for explainable, evidence-based predictions.
              </p>

              <div className="space-y-4">
                {[
                  { icon: Brain, title: 'GNN Encoder', desc: 'Message Passing Neural Networks for molecular embeddings' },
                  { icon: Database, title: 'Knowledge Graph', desc: 'Neo4j heterogeneous graph with 1M+ relationships' },
                  { icon: Lock, title: 'GraphRAG', desc: 'LLM-powered research assistant with citation support' },
                  { icon: BarChart3, title: 'XAI Module', desc: 'GNNExplainer for interpretable predictions' }
                ].map((item, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    viewport={{ once: true }}
                    className="flex items-start gap-4 p-4 rounded-xl bg-white/5 border border-white/5"
                  >
                    <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
                      <item.icon className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                      <h4 className="font-semibold mb-1">{item.title}</h4>
                      <p className="text-sm text-slate-500">{item.desc}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative"
            >
              {/* Architecture diagram placeholder */}
              <div className="aspect-square rounded-3xl bg-gradient-to-br from-slate-900 to-slate-950 border border-white/10 p-8 relative overflow-hidden">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_30%,_rgba(59,130,246,0.1)_0%,_transparent_50%)]" />
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_70%,_rgba(34,211,238,0.1)_0%,_transparent_50%)]" />
                
                {/* Nodes */}
                <div className="absolute top-1/4 left-1/4 w-16 h-16 rounded-2xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
                  <span className="text-xs font-bold text-blue-400">Drug A</span>
                </div>
                <div className="absolute top-1/4 right-1/4 w-16 h-16 rounded-2xl bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
                  <span className="text-xs font-bold text-cyan-400">Drug B</span>
                </div>
                <div className="absolute bottom-1/3 left-1/2 -translate-x-1/2 w-20 h-20 rounded-2xl bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
                  <span className="text-xs font-bold text-purple-400 text-center">GNN<br/>Model</span>
                </div>
                <div className="absolute bottom-1/6 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg bg-emerald-500/20 border border-emerald-500/30">
                  <span className="text-xs font-bold text-emerald-400">Prediction</span>
                </div>

                {/* Connections */}
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
                  <path d="M30 30 L50 50" stroke="rgba(59,130,246,0.3)" strokeWidth="0.5" strokeDasharray="2 2" />
                  <path d="M70 30 L50 50" stroke="rgba(34,211,238,0.3)" strokeWidth="0.5" strokeDasharray="2 2" />
                  <path d="M50 60 L50 75" stroke="rgba(168,85,247,0.3)" strokeWidth="0.5" strokeDasharray="2 2" />
                </svg>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Ready to Analyze Drug Interactions?
            </h2>
            <p className="text-xl text-slate-400 mb-10">
              Start exploring our AI-powered platform for clinical decision support
            </p>
            <motion.button 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate('/dashboard')}
              className="px-10 py-5 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl text-xl font-semibold shadow-xl shadow-blue-500/20 hover:shadow-blue-500/40 transition-all"
            >
              Launch Dashboard
            </motion.button>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
                <GitBranch className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm text-slate-500">Project Aegis © 2025</span>
            </div>
            <div className="flex items-center gap-6">
              <a href="#" className="text-slate-500 hover:text-white transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="#" className="text-slate-500 hover:text-white transition-colors">
                <Linkedin className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </footer>

      {/* Research Modal */}
      <AnimatePresence>
        {showModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
            onClick={() => setShowModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="bg-slate-900 w-full max-w-4xl max-h-[80vh] rounded-2xl border border-white/10 overflow-hidden"
            >
              <div className="p-6 border-b border-white/10 flex items-center justify-between">
                <div>
                  <span className="text-xs text-blue-400 uppercase tracking-wider">Technical Report</span>
                  <h3 className="text-xl font-bold mt-1">Geometric Deep Learning for DDI Prediction</h3>
                </div>
                <button 
                  onClick={() => setShowModal(false)}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="p-6 overflow-y-auto max-h-[60vh] prose prose-invert prose-sm">
                <h4 className="text-blue-400">1. Introduction</h4>
                <p>
                  Adverse drug events from drug-drug interactions (DDIs) represent a significant challenge in 
                  clinical pharmacology. Our system leverages Graph Neural Networks (GNNs) to model molecules 
                  as graphs, enabling accurate prediction of novel interactions.
                </p>
                
                <h4 className="text-blue-400">2. System Architecture</h4>
                <ul>
                  <li><strong>Data Layer:</strong> DrugBank, PubChem, and TwoSides integration</li>
                  <li><strong>Graph Construction:</strong> Heterogeneous knowledge graphs with drugs, proteins, and side effects</li>
                  <li><strong>GNN Encoder:</strong> Message Passing Neural Networks for molecular embeddings</li>
                  <li><strong>Prediction Head:</strong> Link prediction decoder for interaction classification</li>
                </ul>

                <h4 className="text-blue-400">3. Explainability</h4>
                <p>
                  We implement GNNExplainer to identify subgraphs responsible for predictions, enabling 
                  clinicians to understand the molecular basis of flagged interactions.
                </p>

                <h4 className="text-blue-400">4. Performance</h4>
                <p>
                  Our model achieves 94.2% accuracy on the DDI benchmark dataset with sub-100ms inference time, 
                  making it suitable for real-time clinical decision support.
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
