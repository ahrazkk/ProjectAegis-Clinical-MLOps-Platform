import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
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

// Fractured Glass Effect - CSS/SVG based approach that works reliably
const FracturedGlassOverlay = ({ mousePos, isActive }) => {
  const effectSize = 220;
  const halfSize = effectSize / 2;
  
  // Generate shard data once
  const shards = useMemo(() => {
    const pieces = [];
    const numShards = 12;
    for (let i = 0; i < numShards; i++) {
      const angle = (i / numShards) * Math.PI * 2;
      const nextAngle = ((i + 1) / numShards) * Math.PI * 2;
      
      // Random offsets for each shard
      pieces.push({ 
        id: i, 
        angle,
        nextAngle,
        offsetX: (Math.random() - 0.5) * 12,
        offsetY: (Math.random() - 0.5) * 12,
        scale: 0.95 + Math.random() * 0.1,
        hue: (Math.random() - 0.5) * 30,
        delay: Math.random() * 0.1,
      });
    }
    return pieces;
  }, []);

  if (!isActive) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.5 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
      className="fixed pointer-events-none"
      style={{
        left: mousePos.x - halfSize,
        top: mousePos.y - halfSize,
        width: effectSize,
        height: effectSize,
        zIndex: 5,
      }}
    >
      {/* Glass shards with backdrop blur - each shard distorts differently */}
      {shards.map((shard) => {
        // Calculate polygon points for this shard (pie slice)
        const x1 = 50 + Math.cos(shard.angle) * 50;
        const y1 = 50 + Math.sin(shard.angle) * 50;
        const x2 = 50 + Math.cos(shard.nextAngle) * 50;
        const y2 = 50 + Math.sin(shard.nextAngle) * 50;
        
        return (
          <motion.div
            key={shard.id}
            className="absolute inset-0"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ 
              opacity: 1, 
              scale: 1,
              x: [0, shard.offsetX * 0.3, 0],
              y: [0, shard.offsetY * 0.3, 0],
            }}
            transition={{
              opacity: { duration: 0.1, delay: shard.delay },
              x: { duration: 0.8, repeat: Infinity, ease: "easeInOut" },
              y: { duration: 0.8, repeat: Infinity, ease: "easeInOut", delay: 0.1 },
            }}
            style={{
              clipPath: `polygon(50% 50%, ${x1}% ${y1}%, ${x2}% ${y2}%)`,
              backdropFilter: `blur(2px) hue-rotate(${shard.hue}deg) brightness(1.1) contrast(1.05)`,
              WebkitBackdropFilter: `blur(2px) hue-rotate(${shard.hue}deg) brightness(1.1) contrast(1.05)`,
              transform: `translate(${shard.offsetX}px, ${shard.offsetY}px) scale(${shard.scale})`,
              background: 'rgba(0, 212, 255, 0.03)',
            }}
          />
        );
      })}

      {/* Fracture lines SVG */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 220 220">
        <defs>
          <linearGradient id="frac-line-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="rgba(0, 212, 255, 0.9)" />
            <stop offset="50%" stopColor="rgba(255, 255, 255, 1)" />
            <stop offset="100%" stopColor="rgba(168, 85, 247, 0.9)" />
          </linearGradient>
          <filter id="fracture-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        
        {/* Main radial fracture lines */}
        <g stroke="url(#frac-line-grad)" strokeWidth="2" fill="none" filter="url(#fracture-glow)">
          {shards.map((shard) => {
            const x2 = 110 + Math.cos(shard.angle) * 105;
            const y2 = 110 + Math.sin(shard.angle) * 105;
            return (
              <motion.line 
                key={shard.id} 
                x1="110" 
                y1="110" 
                x2={x2} 
                y2={y2} 
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 0.9 }}
                transition={{ duration: 0.2, delay: shard.delay }}
              />
            );
          })}
        </g>
        
        {/* Secondary branching cracks */}
        <g stroke="rgba(255,255,255,0.5)" strokeWidth="1" fill="none">
          {shards.map((shard, i) => {
            const midR = 45 + (i % 4) * 12;
            const x1 = 110 + Math.cos(shard.angle) * midR;
            const y1 = 110 + Math.sin(shard.angle) * midR;
            const branchAngle = shard.angle + ((i % 2) ? 0.5 : -0.5);
            const x2 = x1 + Math.cos(branchAngle) * 20;
            const y2 = y1 + Math.sin(branchAngle) * 20;
            return <line key={`b-${shard.id}`} x1={x1} y1={y1} x2={x2} y2={y2} opacity="0.6" />;
          })}
        </g>
        
        {/* Impact center */}
        <motion.circle 
          cx="110" 
          cy="110" 
          r="5" 
          fill="rgba(255,255,255,0.95)" 
          filter="url(#fracture-glow)"
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 0.5, repeat: Infinity }}
        />
        <circle cx="110" cy="110" r="12" stroke="rgba(0, 212, 255, 0.7)" strokeWidth="1.5" fill="none" />
        <circle cx="110" cy="110" r="25" stroke="rgba(168, 85, 247, 0.4)" strokeWidth="1" fill="none" />
        <circle cx="110" cy="110" r="50" stroke="rgba(0, 212, 255, 0.2)" strokeWidth="0.5" fill="none" />
      </svg>

      {/* Outer glow ring */}
      <div 
        className="absolute inset-0 rounded-full pointer-events-none"
        style={{
          boxShadow: `
            0 0 30px rgba(0, 212, 255, 0.4),
            0 0 60px rgba(0, 212, 255, 0.2),
            inset 0 0 40px rgba(0, 212, 255, 0.15)
          `,
        }}
      />
      
      {/* Chromatic aberration edge */}
      <motion.div
        className="absolute inset-1 rounded-full pointer-events-none"
        animate={{
          boxShadow: [
            'inset 3px 0 12px rgba(255,0,100,0.4), inset -3px 0 12px rgba(0,200,255,0.4)',
            'inset -3px 0 12px rgba(255,0,100,0.4), inset 3px 0 12px rgba(0,200,255,0.4)',
          ]
        }}
        transition={{ duration: 0.15, repeat: Infinity, ease: "linear" }}
      />
    </motion.div>
  );
};

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
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000 });
  const [isMouseActive, setIsMouseActive] = useState(false);
  const heroRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Track mouse for fractured glass overlay
  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePos({ x: e.clientX, y: e.clientY });
      setIsMouseActive(true);
    };
    const handleMouseLeave = () => setIsMouseActive(false);
    
    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  const scrollProgress = Math.min(scrollY / 500, 1);

  return (
    <div className="min-h-screen bg-black text-fui-gray-100 overflow-x-hidden font-mono">
      {/* Navigation */}
      <motion.nav 
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="fixed top-0 left-0 right-0 z-50 bg-black/95 border-b border-fui-gray-500/30"
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 border border-fui-gray-500 flex items-center justify-center relative">
              <GitBranch className="w-5 h-5 text-fui-gray-100" />
              <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-gray-400"></div>
              <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-gray-400"></div>
            </div>
            <span className="text-lg font-normal tracking-widest uppercase">
              Project<span className="text-fui-accent-cyan">Aegis</span>
            </span>
          </div>
          
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-xs text-fui-gray-400 hover:text-fui-accent-cyan transition-colors uppercase tracking-widest">Features</a>
            <a href="#technology" className="text-xs text-fui-gray-400 hover:text-fui-accent-cyan transition-colors uppercase tracking-widest">Technology</a>
            <button 
              onClick={() => setShowModal(true)}
              className="text-xs text-fui-gray-400 hover:text-fui-accent-cyan transition-colors uppercase tracking-widest"
            >
              Research
            </button>
          </div>

          <button 
            onClick={() => navigate('/dashboard')}
            className="btn-blueprint"
          >
            Launch Platform
          </button>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <section ref={heroRef} className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
        {/* 3D Background with Fractured Glass Distortion Shader */}
        <div className="absolute inset-0 z-0">
          <Canvas 
            camera={{ position: [0, 0, 15], fov: 45 }} 
            gl={{ alpha: true, antialias: true }}
          >
            <ParticleSystem scrollProgress={scrollProgress} />
          </Canvas>
        </div>

        {/* Fractured Glass Visual Overlay (SVG lines) */}
        <AnimatePresence>
          <FracturedGlassOverlay mousePos={mousePos} isActive={isMouseActive} />
        </AnimatePresence>

        {/* Subtle vignette overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black pointer-events-none" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_transparent_0%,_rgba(0,0,0,0.6)_80%)] pointer-events-none" />

        {/* Content */}
        <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 border border-fui-gray-500/50 mb-8">
              <Sparkles className="w-4 h-4 text-fui-gray-400" />
              <span className="text-xs text-fui-gray-400 uppercase tracking-widest">Powered by Graph Neural Networks</span>
            </div>

            <h1 className="text-4xl md:text-6xl font-light leading-tight mb-6 tracking-wide">
              <span className="text-fui-gray-100">
                Drug Interaction
              </span>
              <br />
              <span className="text-fui-accent-cyan" style={{ textShadow: '0 0 30px rgba(0, 255, 255, 0.3)' }}>
                Intelligence
              </span>
            </h1>

            <p className="text-sm text-fui-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed tracking-wide">
              Advanced AI-powered platform for predicting drug-drug interactions using 
              geometric deep learning and heterogeneous knowledge graphs.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <motion.button 
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                onClick={() => navigate('/dashboard')}
                className="group px-8 py-4 border border-fui-accent-cyan text-fui-accent-cyan text-sm uppercase tracking-widest hover:bg-fui-accent-cyan/10 transition-all flex items-center gap-3"
                style={{ boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)' }}
              >
                Enter Dashboard
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </motion.button>
              
              <motion.button 
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                onClick={() => setShowModal(true)}
                className="px-8 py-4 border border-fui-gray-500 text-fui-gray-300 text-sm uppercase tracking-widest hover:border-fui-gray-400 hover:text-fui-gray-100 transition-all"
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
              <span className="text-[10px] text-fui-gray-500 uppercase tracking-[0.3em]">Scroll</span>
              <ChevronDown className="w-4 h-4 text-fui-gray-500" />
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative py-24 border-y border-fui-gray-500/20">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="text-center border border-fui-gray-500/20 p-6 relative"
              >
                <div className="absolute -top-px -left-px w-3 h-3 border-t border-l border-fui-gray-500"></div>
                <div className="absolute -bottom-px -right-px w-3 h-3 border-b border-r border-fui-gray-500"></div>
                <div className="text-3xl md:text-4xl font-light text-fui-accent-cyan mb-2" style={{ textShadow: '0 0 20px rgba(0, 255, 255, 0.3)' }}>
                  {stat.value}
                </div>
                <div className="text-[10px] text-fui-gray-500 uppercase tracking-[0.2em]">
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
            <span className="text-[10px] text-fui-accent-cyan uppercase tracking-[0.3em] mb-4 block">// Capabilities</span>
            <h2 className="text-3xl md:text-4xl font-light mb-6 tracking-wide">
              Enterprise-Grade DDI Analysis
            </h2>
            <p className="text-sm text-fui-gray-400 max-w-2xl mx-auto tracking-wide">
              Built for clinical decision support with explainable predictions and comprehensive drug coverage
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="group p-6 border border-fui-gray-500/20 hover:border-fui-gray-500/40 transition-all relative"
              >
                <div className="absolute -top-px -left-px w-2 h-2 border-t border-l border-fui-gray-500"></div>
                <div className="absolute -bottom-px -right-px w-2 h-2 border-b border-r border-fui-gray-500"></div>
                <div className="w-10 h-10 border border-fui-gray-500/50 flex items-center justify-center mb-4 group-hover:border-fui-accent-cyan/50 transition-colors">
                  <feature.icon className="w-5 h-5 text-fui-gray-400 group-hover:text-fui-accent-cyan transition-colors" />
                </div>
                <h3 className="text-sm font-normal mb-2 tracking-wide">{feature.title}</h3>
                <p className="text-xs text-fui-gray-500 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section id="technology" className="py-32 relative">
        {/* Subtle blueprint accent lines */}
        <div className="absolute left-0 top-1/2 w-20 h-px bg-gradient-to-r from-transparent to-fui-gray-500/30"></div>
        <div className="absolute right-0 top-1/2 w-20 h-px bg-gradient-to-l from-transparent to-fui-gray-500/30"></div>
        
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <span className="text-[10px] text-fui-accent-cyan uppercase tracking-[0.3em] mb-4 block">// Technology Stack</span>
              <h2 className="text-3xl font-light mb-6 tracking-wide">
                Cutting-Edge Architecture
              </h2>
              <p className="text-sm text-fui-gray-400 mb-8 leading-relaxed tracking-wide">
                Our platform combines state-of-the-art graph neural networks with 
                retrieval-augmented generation for explainable, evidence-based predictions.
              </p>

              <div className="space-y-3">
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
                    className="flex items-start gap-4 p-4 border border-fui-gray-500/20 hover:border-fui-gray-500/40 transition-colors"
                  >
                    <div className="w-8 h-8 border border-fui-gray-500/50 flex items-center justify-center flex-shrink-0">
                      <item.icon className="w-4 h-4 text-fui-gray-400" />
                    </div>
                    <div>
                      <h4 className="text-sm font-normal mb-1 tracking-wide">{item.title}</h4>
                      <p className="text-xs text-fui-gray-500">{item.desc}</p>
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
              {/* Architecture diagram - Blueprint style */}
              <div className="aspect-square border border-fui-gray-500/30 p-8 relative overflow-hidden bg-black">
                {/* Blueprint grid overlay */}
                <div className="absolute inset-0 opacity-30" style={{ backgroundImage: 'linear-gradient(rgba(102,102,102,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(102,102,102,0.3) 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
                
                {/* Corner markers */}
                <div className="absolute top-2 left-2 w-4 h-4 border-t border-l border-fui-gray-500"></div>
                <div className="absolute top-2 right-2 w-4 h-4 border-t border-r border-fui-gray-500"></div>
                <div className="absolute bottom-2 left-2 w-4 h-4 border-b border-l border-fui-gray-500"></div>
                <div className="absolute bottom-2 right-2 w-4 h-4 border-b border-r border-fui-gray-500"></div>
                
                {/* Nodes */}
                <div className="absolute top-1/4 left-1/4 w-16 h-16 border border-fui-gray-500 flex items-center justify-center bg-black">
                  <span className="text-[10px] font-normal text-fui-gray-400 uppercase tracking-wider">Drug A</span>
                </div>
                <div className="absolute top-1/4 right-1/4 w-16 h-16 border border-fui-gray-500 flex items-center justify-center bg-black">
                  <span className="text-[10px] font-normal text-fui-gray-400 uppercase tracking-wider">Drug B</span>
                </div>
                <div className="absolute bottom-1/3 left-1/2 -translate-x-1/2 w-20 h-20 border border-fui-accent-cyan/50 flex items-center justify-center bg-black" style={{ boxShadow: '0 0 20px rgba(0,255,255,0.1)' }}>
                  <span className="text-[10px] font-normal text-fui-accent-cyan text-center uppercase tracking-wider">GNN<br/>Model</span>
                </div>
                <div className="absolute bottom-[15%] left-1/2 -translate-x-1/2 px-4 py-2 border border-fui-accent-green/50 bg-black" style={{ boxShadow: '0 0 15px rgba(0,255,136,0.1)' }}>
                  <span className="text-[10px] font-normal text-fui-accent-green uppercase tracking-wider">Prediction</span>
                </div>

                {/* Connections - dashed lines */}
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
                  <path d="M30 30 L50 50" stroke="rgba(102,102,102,0.5)" strokeWidth="0.3" strokeDasharray="2 2" />
                  <path d="M70 30 L50 50" stroke="rgba(102,102,102,0.5)" strokeWidth="0.3" strokeDasharray="2 2" />
                  <path d="M50 60 L50 75" stroke="rgba(0,255,255,0.3)" strokeWidth="0.3" strokeDasharray="2 2" />
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
            <h2 className="text-3xl md:text-4xl font-light mb-6 tracking-wide">
              Ready to Analyze Drug Interactions?
            </h2>
            <p className="text-sm text-fui-gray-400 mb-10 tracking-wide">
              Start exploring our AI-powered platform for clinical decision support
            </p>
            <motion.button 
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              onClick={() => navigate('/dashboard')}
              className="px-10 py-4 border border-fui-accent-cyan text-fui-accent-cyan text-sm uppercase tracking-widest hover:bg-fui-accent-cyan/10 transition-all"
              style={{ boxShadow: '0 0 30px rgba(0, 255, 255, 0.2)' }}
            >
              Launch Dashboard
            </motion.button>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-fui-gray-500/20">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 border border-fui-gray-500 flex items-center justify-center">
                <GitBranch className="w-4 h-4 text-fui-gray-400" />
              </div>
              <span className="text-xs text-fui-gray-500 uppercase tracking-widest">Project Aegis © 2025</span>
            </div>
            <div className="flex items-center gap-6">
              <a href="#" className="text-fui-gray-500 hover:text-fui-accent-cyan transition-colors">
                <Github className="w-4 h-4" />
              </a>
              <a href="#" className="text-fui-gray-500 hover:text-fui-accent-cyan transition-colors">
                <Linkedin className="w-4 h-4" />
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
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/90"
            onClick={() => setShowModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="bg-black w-full max-w-4xl max-h-[80vh] border border-fui-gray-500/30 overflow-hidden relative"
            >
              {/* Corner markers */}
              <div className="absolute top-0 left-0 w-4 h-4 border-t border-l border-fui-gray-500"></div>
              <div className="absolute top-0 right-0 w-4 h-4 border-t border-r border-fui-gray-500"></div>
              <div className="absolute bottom-0 left-0 w-4 h-4 border-b border-l border-fui-gray-500"></div>
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b border-r border-fui-gray-500"></div>
              <div className="p-6 border-b border-fui-gray-500/30 flex items-center justify-between">
                <div>
                  <span className="text-[10px] text-fui-accent-cyan uppercase tracking-[0.2em]">// Technical Report</span>
                  <h3 className="text-lg font-normal mt-1 tracking-wide">Geometric Deep Learning for DDI Prediction</h3>
                </div>
                <button 
                  onClick={() => setShowModal(false)}
                  className="p-2 border border-fui-gray-500/30 hover:border-fui-gray-400 transition-colors"
                >
                  <X className="w-4 h-4 text-fui-gray-400" />
                </button>
              </div>
              <div className="p-6 overflow-y-auto max-h-[60vh] prose-invert text-sm">
                <h4 className="text-fui-accent-cyan text-xs uppercase tracking-widest mb-3">1. Introduction</h4>
                <p className="text-fui-gray-400 leading-relaxed mb-6">
                  Adverse drug events from drug-drug interactions (DDIs) represent a significant challenge in 
                  clinical pharmacology. Our system leverages Graph Neural Networks (GNNs) to model molecules 
                  as graphs, enabling accurate prediction of novel interactions.
                </p>
                
                <h4 className="text-fui-accent-cyan text-xs uppercase tracking-widest mb-3">2. System Architecture</h4>
                <ul className="text-fui-gray-400 space-y-2 mb-6 list-none pl-0">
                  <li className="flex items-start gap-2"><span className="text-fui-gray-500">→</span> <span><strong className="text-fui-gray-100">Data Layer:</strong> DrugBank, PubChem, and TwoSides integration</span></li>
                  <li className="flex items-start gap-2"><span className="text-fui-gray-500">→</span> <span><strong className="text-fui-gray-100">Graph Construction:</strong> Heterogeneous knowledge graphs with drugs, proteins, and side effects</span></li>
                  <li className="flex items-start gap-2"><span className="text-fui-gray-500">→</span> <span><strong className="text-fui-gray-100">GNN Encoder:</strong> Message Passing Neural Networks for molecular embeddings</span></li>
                  <li className="flex items-start gap-2"><span className="text-fui-gray-500">→</span> <span><strong className="text-fui-gray-100">Prediction Head:</strong> Link prediction decoder for interaction classification</span></li>
                </ul>

                <h4 className="text-fui-accent-cyan text-xs uppercase tracking-widest mb-3">3. Explainability</h4>
                <p className="text-fui-gray-400 leading-relaxed mb-6">
                  We implement GNNExplainer to identify subgraphs responsible for predictions, enabling 
                  clinicians to understand the molecular basis of flagged interactions.
                </p>

                <h4 className="text-fui-accent-cyan text-xs uppercase tracking-widest mb-3">4. Performance</h4>
                <p className="text-fui-gray-400 leading-relaxed">
                  Our model achieves <span className="text-fui-accent-green">94.2%</span> accuracy on the DDI benchmark dataset with sub-<span className="text-fui-accent-green">100ms</span> inference time, 
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
