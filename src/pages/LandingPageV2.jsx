import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import { motion, AnimatePresence, useMotionValue, useTransform, useSpring } from 'framer-motion';
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
  Linkedin,
  AlertTriangle,
  CheckCircle2,
  Pill,
  Loader2,
  Server
} from 'lucide-react';
import DrugInteractionBackground from '../components/DrugInteractionBackground';
import { checkHealth } from '../services/api';

const features = [
  {
    icon: Brain,
    title: 'PubMedBERT Encoder',
    description: 'Fine-tuned biomedical BERT model processes drug pair contexts with 92.7% AUC accuracy',
    color: 'from-cyan-500 to-blue-500'
  },
  {
    icon: Network,
    title: 'Knowledge Graph Integration',
    description: 'Neo4j-powered graph with 2,000+ drugs and 1,600+ verified interactions from DDI Corpus',
    color: 'from-purple-500 to-pink-500'
  },
  {
    icon: Shield,
    title: 'Evidence-Based Explanations',
    description: 'RAG-powered citations from PubMed literature provide clinical context for predictions',
    color: 'from-emerald-500 to-teal-500'
  },
  {
    icon: Zap,
    title: 'Real-time Analysis',
    description: 'Sub-200ms predictions powered by optimized PyTorch models on Google Cloud Run',
    color: 'from-yellow-500 to-orange-500'
  },
  {
    icon: Activity,
    title: 'Therapeutic Classification',
    description: 'Automatic drug categorization with 60%+ coverage for clinical decision support',
    color: 'from-red-500 to-rose-500'
  },
  {
    icon: Database,
    title: 'Future: Graph Neural Networks',
    description: 'Roadmap includes GNN architecture for molecular graph processing and enhanced accuracy',
    color: 'from-indigo-500 to-violet-500'
  }
];

const stats = [
  { value: '2K+', label: 'Drugs', suffix: '' },
  { value: '92.7', label: 'AUC Score', suffix: '%' },
  { value: '<200', label: 'Latency', suffix: 'ms' },
  { value: '1.6K+', label: 'Interactions', suffix: '' }
];

// Animated counter component
function AnimatedCounter({ value, suffix = '' }) {
  const [displayValue, setDisplayValue] = useState(0);
  const numericValue = parseFloat(value.replace(/[^0-9.]/g, ''));
  const hasK = value.includes('K');
  
  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const stepValue = numericValue / steps;
    let current = 0;
    
    const timer = setInterval(() => {
      current += stepValue;
      if (current >= numericValue) {
        setDisplayValue(numericValue);
        clearInterval(timer);
      } else {
        setDisplayValue(Math.floor(current * 10) / 10);
      }
    }, duration / steps);
    
    return () => clearInterval(timer);
  }, [numericValue]);

  const prefix = value.includes('<') ? '<' : '';
  const kSuffix = hasK ? 'K' : '';
  const plusSuffix = value.includes('+') ? '+' : '';
  
  return (
    <span>
      {prefix}{displayValue % 1 === 0 ? Math.floor(displayValue) : displayValue.toFixed(1)}{kSuffix}{plusSuffix}{suffix}
    </span>
  );
}

// Hexagonal grid background
function HexGrid() {
  return (
    <div className="absolute inset-0 overflow-hidden opacity-[0.03]">
      <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="hexagons" width="50" height="43.4" patternUnits="userSpaceOnUse" patternTransform="scale(2)">
            <polygon 
              points="25,0 50,14.4 50,43.4 25,57.7 0,43.4 0,14.4" 
              fill="none" 
              stroke="white" 
              strokeWidth="0.5"
              transform="translate(0, -7.2)"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#hexagons)" />
      </svg>
    </div>
  );
}

// Animated text reveal
function AnimatedText({ text, className, delay = 0 }) {
  return (
    <motion.span className={className}>
      {text.split('').map((char, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ 
            duration: 0.4, 
            delay: delay + i * 0.03,
            ease: [0.25, 0.46, 0.45, 0.94]
          }}
          className="inline-block"
          style={{ whiteSpace: char === ' ' ? 'pre' : 'normal' }}
        >
          {char}
        </motion.span>
      ))}
    </motion.span>
  );
}

// Glowing orb cursor follower
function GlowingCursor() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  useEffect(() => {
    const handleMouseMove = (e) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <motion.div
      className="fixed w-96 h-96 pointer-events-none z-0"
      animate={{ x: position.x - 192, y: position.y - 192 }}
      transition={{ type: 'spring', damping: 30, stiffness: 200 }}
      style={{
        background: 'radial-gradient(circle, rgba(0,255,255,0.08) 0%, transparent 70%)',
      }}
    />
  );
}

// Interactive drug demo widget
function DrugDemoWidget() {
  const [drugA, setDrugA] = useState('Warfarin');
  const [drugB, setDrugB] = useState('Aspirin');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  
  const drugs = ['Warfarin', 'Aspirin', 'Ibuprofen', 'Metformin', 'Lisinopril', 'Omeprazole'];
  
  const analyze = () => {
    setAnalyzing(true);
    setResult(null);
    setTimeout(() => {
      setAnalyzing(false);
      setResult({
        severity: drugA === 'Warfarin' && drugB === 'Aspirin' ? 'high' : 
                  drugA === 'Metformin' && drugB === 'Ibuprofen' ? 'moderate' : 'low',
        confidence: (85 + Math.random() * 10).toFixed(1)
      });
    }, 1500);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 1, duration: 0.8 }}
      className="relative bg-black/50 backdrop-blur-md border border-white/15 p-4 xl:p-5 rounded-sm"
      style={{ boxShadow: '0 0 60px rgba(0,255,255,0.12), 0 0 100px rgba(139,92,246,0.08)' }}
    >
      {/* Corner accents */}
      <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-cyan-500/50"></div>
      <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-purple-500/50"></div>
      <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-purple-500/50"></div>
      <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-cyan-500/50"></div>
      
      <div className="text-[9px] uppercase tracking-[0.3em] text-cyan-400/80 mb-3 flex items-center gap-2">
        <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse"></div>
        Live Demo
      </div>
      
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[9px] uppercase tracking-wider text-white/40 mb-1 block">Drug A</label>
            <select 
              value={drugA}
              onChange={(e) => { setDrugA(e.target.value); setResult(null); }}
              className="w-full bg-white/5 border border-white/10 text-white text-xs px-2 py-1.5 rounded-sm focus:border-cyan-500/50 focus:outline-none transition-colors"
            >
              {drugs.map(d => <option key={d} value={d} className="bg-black">{d}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[9px] uppercase tracking-wider text-white/40 mb-1 block">Drug B</label>
            <select 
              value={drugB}
              onChange={(e) => { setDrugB(e.target.value); setResult(null); }}
              className="w-full bg-white/5 border border-white/10 text-white text-xs px-2 py-1.5 rounded-sm focus:border-purple-500/50 focus:outline-none transition-colors"
            >
              {drugs.filter(d => d !== drugA).map(d => <option key={d} value={d} className="bg-black">{d}</option>)}
            </select>
          </div>
        </div>
        
        <button
          onClick={analyze}
          disabled={analyzing}
          className="w-full py-2 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-white/20 text-white text-[10px] uppercase tracking-widest hover:from-cyan-500/30 hover:to-purple-500/30 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {analyzing ? (
            <>
              <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              Analyzing...
            </>
          ) : (
            <>
              <Zap className="w-3 h-3" />
              Analyze Interaction
            </>
          )}
        </button>
        
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={`p-3 rounded-sm border ${
                result.severity === 'high' ? 'bg-red-500/10 border-red-500/30' :
                result.severity === 'moderate' ? 'bg-yellow-500/10 border-yellow-500/30' :
                'bg-green-500/10 border-green-500/30'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                {result.severity === 'high' ? (
                  <AlertTriangle className="w-3 h-3 text-red-400" />
                ) : result.severity === 'moderate' ? (
                  <AlertTriangle className="w-3 h-3 text-yellow-400" />
                ) : (
                  <CheckCircle2 className="w-3 h-3 text-green-400" />
                )}
                <span className={`text-[10px] uppercase tracking-wider font-medium ${
                  result.severity === 'high' ? 'text-red-400' :
                  result.severity === 'moderate' ? 'text-yellow-400' :
                  'text-green-400'
                }`}>
                  {result.severity} risk
                </span>
              </div>
              <div className="text-[9px] text-white/50">
                Confidence: {result.confidence}% • PubMedBERT-v2
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

// Floating pills animation
function FloatingPills() {
  const pills = [
    { x: '10%', y: '20%', delay: 0, rotation: 15 },
    { x: '85%', y: '30%', delay: 0.5, rotation: -20 },
    { x: '15%', y: '70%', delay: 1, rotation: 25 },
    { x: '80%', y: '75%', delay: 1.5, rotation: -15 },
  ];

  return (
    <>
      {pills.map((pill, i) => (
        <motion.div
          key={i}
          className="absolute text-white/5"
          style={{ left: pill.x, top: pill.y }}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ 
            opacity: 1, 
            scale: 1,
            y: [0, -20, 0],
            rotate: [pill.rotation, pill.rotation + 10, pill.rotation]
          }}
          transition={{ 
            opacity: { delay: pill.delay, duration: 1 },
            scale: { delay: pill.delay, duration: 1 },
            y: { delay: pill.delay, duration: 4, repeat: Infinity, ease: 'easeInOut' },
            rotate: { delay: pill.delay, duration: 6, repeat: Infinity, ease: 'easeInOut' }
          }}
        >
          <Pill className="w-16 h-16" />
        </motion.div>
      ))}
    </>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const [scrollY, setScrollY] = useState(0);
  const [showModal, setShowModal] = useState(false);
  const heroRef = useRef(null);
  
  // Backend warm-up status
  const [backendStatus, setBackendStatus] = useState('connecting'); // 'connecting' | 'ready' | 'error'
  const [connectionTime, setConnectionTime] = useState(0);

  // Warm up backend on landing page load
  useEffect(() => {
    const startTime = Date.now();
    let timer;
    
    // Update elapsed time every second while connecting
    timer = setInterval(() => {
      if (backendStatus === 'connecting') {
        setConnectionTime(Math.floor((Date.now() - startTime) / 1000));
      }
    }, 1000);
    
    // Silent ping to warm up backend
    checkHealth()
      .then(() => {
        setBackendStatus('ready');
        setConnectionTime(Math.floor((Date.now() - startTime) / 1000));
        clearInterval(timer);
      })
      .catch(() => {
        setBackendStatus('error');
        clearInterval(timer);
      });
    
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollProgress = Math.min(scrollY / 500, 1);

  return (
    <div className="min-h-screen text-white overflow-x-hidden landing-page" style={{ backgroundColor: '#0D1117' }}>
      {/* Cursor glow effect */}
      <GlowingCursor />
      
      {/* Hexagonal grid background */}
      <HexGrid />
      
      {/* Navigation */}
      <motion.nav 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="fixed top-0 left-0 right-0 z-50 bg-[#0D1117]/90 backdrop-blur-xl border-b border-white/10"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
          <motion.div 
            className="flex items-center gap-2 sm:gap-3"
            whileHover={{ scale: 1.02 }}
          >
            <div className="relative w-8 h-8 sm:w-10 sm:h-10 flex items-center justify-center">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-lg opacity-20"></div>
              <div className="absolute inset-[2px] bg-black rounded-lg"></div>
              <GitBranch className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400 relative z-10" />
            </div>
            <span className="text-sm sm:text-lg font-light tracking-widest">
              <span className="text-white/80">PROJECT</span>
              <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent font-normal">AEGIS</span>
            </span>
          </motion.div>
          
          <div className="hidden md:flex items-center gap-8">
            {['Features', 'Technology', 'Research'].map((item, i) => (
              <motion.a
                key={item}
                href={item === 'Research' ? '/research' : `#${item.toLowerCase()}`}
                onClick={item === 'Research' ? (e) => { e.preventDefault(); navigate('/research'); } : undefined}
                className="text-xs text-white/50 hover:text-white transition-colors uppercase tracking-[0.2em] relative group"
                whileHover={{ y: -2 }}
              >
                {item}
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-gradient-to-r from-cyan-400 to-purple-400 group-hover:w-full transition-all duration-300"></span>
              </motion.a>
            ))}
          </div>

          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate('/dashboard')}
            className="px-3 py-1.5 sm:px-6 sm:py-2.5 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 border border-white/20 text-white text-[10px] sm:text-xs uppercase tracking-wider sm:tracking-widest hover:border-white/40 transition-all relative overflow-hidden group"
          >
            <span className="relative z-10 hidden sm:inline">Launch Platform</span>
            <span className="relative z-10 sm:hidden">Launch</span>
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
          </motion.button>
        </div>
      </motion.nav>

      {/* Hero Section - Split Layout */}
      <section ref={heroRef} className="relative min-h-screen flex items-center overflow-hidden pt-16 sm:pt-20">
        {/* 3D Background */}
        <div className="absolute inset-0 z-0">
          <Canvas camera={{ position: [0, 0, 12], fov: 50 }} gl={{ alpha: true, antialias: true }}>
            <DrugInteractionBackground scrollProgress={scrollProgress} />
            <EffectComposer>
              <Bloom 
                intensity={0.5} 
                luminanceThreshold={0.5} 
                luminanceSmoothing={0.9}
                mipmapBlur
              />
              <ChromaticAberration
                blendFunction={BlendFunction.NORMAL}
                offset={[0.0008, 0.0008]}
              />
            </EffectComposer>
          </Canvas>
        </div>

        {/* Gradient overlays - reduced opacity for better 3D visibility */}
        <div className="absolute inset-0 bg-gradient-to-b from-[#0D1117] via-transparent to-[#0D1117] pointer-events-none z-[1]" />
        <div className="absolute inset-0 bg-gradient-to-r from-[#0D1117]/60 via-transparent to-[#0D1117]/60 pointer-events-none z-[1]" />
        
        {/* Floating pills decoration - hidden on mobile */}
        <div className="hidden sm:block">
          <FloatingPills />
        </div>

        {/* Content - Split Layout */}
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 w-full">
          <div className="grid lg:grid-cols-5 gap-8 lg:gap-16 items-center min-h-[70vh] sm:min-h-[80vh] py-8 sm:py-0">
            {/* Left side - Hero text */}
            <div className="lg:col-span-3 space-y-5 sm:space-y-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 sm:py-2 bg-white/10 border border-white/20 backdrop-blur-sm rounded-full"
              >
                <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <span className="text-[10px] sm:text-xs text-white/80 uppercase tracking-widest">AI-Powered Drug Safety</span>
              </motion.div>

              <div className="space-y-2 sm:space-y-4">
                <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-extralight leading-[1.1] tracking-tight hero-text">
                  <span className="text-white block">
                    <AnimatedText text="Drug Interaction" className="" delay={0.4} />
                  </span>
                  <motion.span 
                    className="block mt-2"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                  >
                    <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent font-light"
                      style={{ 
                        textShadow: '0 0 80px rgba(139, 92, 246, 0.5)',
                        filter: 'drop-shadow(0 0 30px rgba(0, 255, 255, 0.3))'
                      }}
                    >
                      Intelligence
                    </span>
                  </motion.span>
                </h1>
              </div>

              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.2 }}
                className="text-sm sm:text-base lg:text-lg text-white/70 max-w-xl leading-relaxed font-light"
              >
                Predict drug-drug interactions with unprecedented accuracy using 
                <span className="text-cyan-400 font-normal"> biomedical language models</span> and 
                <span className="text-purple-400 font-normal"> Neo4j knowledge graphs</span>.
              </motion.p>

              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.4 }}
                className="flex flex-col sm:flex-row gap-3 sm:gap-4"
              >
                <motion.button 
                  whileHover={{ scale: 1.02, boxShadow: '0 0 40px rgba(0, 255, 255, 0.3)' }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => navigate('/dashboard')}
                  className="group px-6 sm:px-8 py-3 sm:py-4 bg-gradient-to-r from-cyan-500 to-cyan-600 text-black text-xs sm:text-sm uppercase tracking-widest font-medium flex items-center justify-center gap-3 rounded-sm"
                >
                  Enter Dashboard
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </motion.button>
                
                <motion.button 
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => navigate('/research')}
                  className="px-6 sm:px-8 py-3 sm:py-4 border border-white/20 text-white/80 text-xs sm:text-sm uppercase tracking-widest hover:bg-white/5 transition-all rounded-sm text-center"
                >
                  View Research
                </motion.button>
              </motion.div>

              {/* Backend Status Indicator */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.6 }}
                className="pb-16 sm:pb-0"
              >
                <AnimatePresence mode="wait">
                  {backendStatus === 'connecting' && (
                    <motion.div
                      key="connecting"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="flex items-start gap-3 p-3 sm:p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-sm max-w-xl"
                    >
                      <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-400 animate-spin flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-xs sm:text-sm text-yellow-400 font-medium">
                          Warming up AI Backend... ({connectionTime}s)
                        </p>
                        <p className="text-[10px] sm:text-xs text-white/50 mt-1">
                          Feel free to explore the page while the ML models load. This may take 15-25 seconds on first visit.
                        </p>
                      </div>
                    </motion.div>
                  )}
                  
                  {backendStatus === 'ready' && (
                    <motion.div
                      key="ready"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="flex items-center gap-3 p-3 sm:p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-sm max-w-xl"
                    >
                      <CheckCircle2 className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-400 flex-shrink-0" />
                      <div>
                        <p className="text-xs sm:text-sm text-emerald-400 font-medium">
                          AI Backend Ready! 
                          <span className="text-white/50 font-normal ml-2">
                            (loaded in {connectionTime}s)
                          </span>
                        </p>
                        <p className="text-[10px] sm:text-xs text-white/50 mt-1">
                          The platform is fully operational. Enter the dashboard to analyze drug interactions.
                        </p>
                      </div>
                    </motion.div>
                  )}
                  
                  {backendStatus === 'error' && (
                    <motion.div
                      key="error"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="flex items-center gap-3 p-3 sm:p-4 bg-red-500/10 border border-red-500/30 rounded-sm max-w-xl"
                    >
                      <AlertTriangle className="w-4 h-4 sm:w-5 sm:h-5 text-red-400 flex-shrink-0" />
                      <div>
                        <p className="text-xs sm:text-sm text-red-400 font-medium">
                          Backend Unavailable
                        </p>
                        <p className="text-[10px] sm:text-xs text-white/50 mt-1">
                          The AI backend is currently offline. Some features may be limited.
                        </p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            </div>

            {/* Right side - Interactive Demo */}
            <div className="hidden lg:flex lg:col-span-2 justify-end">
              <div className="w-full max-w-xs xl:max-w-sm">
                <DrugDemoWidget />
              </div>
            </div>
          </div>

          {/* Scroll indicator - hidden on mobile */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2 }}
            className="hidden sm:block absolute bottom-8 left-1/2 -translate-x-1/2"
          >
            <motion.div 
              animate={{ y: [0, 8, 0] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="flex flex-col items-center gap-2"
            >
              <span className="text-[10px] text-white/50 uppercase tracking-[0.3em]">Explore</span>
              <ChevronDown className="w-4 h-4 text-white/50" />
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section - Animated counters */}
      <section className="relative py-12 sm:py-24 border-y border-white/10">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-950/10 to-transparent"></div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 relative">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-8">
            {stats.map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="text-center group"
              >
                <div className="relative inline-block">
                  <div 
                    className="text-3xl sm:text-5xl md:text-6xl font-extralight bg-gradient-to-b from-white to-white/70 bg-clip-text text-transparent mb-1 sm:mb-2"
                    style={{ textShadow: '0 0 40px rgba(0, 255, 255, 0.3)' }}
                  >
                    <AnimatedCounter value={stat.value} suffix={stat.suffix} />
                  </div>
                  <div className="absolute -inset-4 bg-gradient-to-r from-cyan-500/0 via-cyan-500/10 to-purple-500/0 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                </div>
                <div className="text-[9px] sm:text-[10px] text-white/60 uppercase tracking-[0.2em] sm:tracking-[0.3em]">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section - Bento Grid */}
      <section id="features" className="py-16 sm:py-32 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-8 sm:mb-16"
          >
            <span className="text-[10px] sm:text-xs text-cyan-400/80 uppercase tracking-[0.2em] sm:tracking-[0.3em] mb-3 sm:mb-4 block">Capabilities</span>
            <h2 className="text-2xl sm:text-4xl md:text-5xl font-extralight mb-4 sm:mb-6 px-2">
              Enterprise-Grade <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">DDI Analysis</span>
            </h2>
            <p className="text-sm sm:text-base text-white/60 max-w-2xl mx-auto font-light px-4">
              Built for clinical decision support with explainable predictions and comprehensive drug coverage
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
                className="group relative p-4 sm:p-6 bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all duration-500 rounded-sm overflow-hidden"
              >
                {/* Gradient overlay on hover */}
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity duration-500`}></div>
                
                {/* Icon */}
                <div className={`w-10 h-10 sm:w-12 sm:h-12 rounded-lg bg-gradient-to-br ${feature.color} p-[1px] mb-3 sm:mb-5`}>
                  <div className="w-full h-full bg-black rounded-lg flex items-center justify-center">
                    <feature.icon className="w-4 h-4 sm:w-5 sm:h-5 text-white/90" />
                  </div>
                </div>
                
                <h3 className="text-sm sm:text-base font-normal mb-1 sm:mb-2 text-white">{feature.title}</h3>
                <p className="text-xs sm:text-sm text-white/60 leading-relaxed">{feature.description}</p>
                
                {/* Corner accent */}
                <div className={`absolute top-0 right-0 w-16 h-16 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-10 blur-2xl transition-opacity duration-500`}></div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section id="technology" className="py-16 sm:py-32 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-purple-950/10 to-transparent"></div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 relative">
          <div className="grid lg:grid-cols-2 gap-8 sm:gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <span className="text-[10px] sm:text-xs text-purple-400/80 uppercase tracking-[0.2em] sm:tracking-[0.3em] mb-3 sm:mb-4 block">Technology Stack</span>
              <h2 className="text-2xl sm:text-4xl font-extralight mb-4 sm:mb-6">
                Production <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">Architecture</span>
              </h2>
              <p className="text-sm sm:text-base text-white/60 mb-6 sm:mb-10 leading-relaxed font-light">
                Our platform combines fine-tuned biomedical language models with 
                retrieval-augmented generation for explainable, evidence-based predictions.
              </p>

              <div className="space-y-3 sm:space-y-4">
                {[
                  { icon: Brain, title: 'PubMedBERT', desc: 'Fine-tuned transformer for biomedical text understanding', color: 'cyan' },
                  { icon: Database, title: 'Neo4j Graph', desc: '2,000+ drugs with verified DDI relationships', color: 'purple' },
                  { icon: Lock, title: 'RAG Pipeline', desc: 'PubMed-powered research with literature citations', color: 'pink' },
                  { icon: BarChart3, title: 'Cloud Deploy', desc: 'Google Cloud Run with containerized microservices', color: 'emerald' }
                ].map((item, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    viewport={{ once: true }}
                    className="flex items-start gap-3 sm:gap-4 p-3 sm:p-4 bg-white/[0.02] border border-white/5 hover:border-white/10 transition-all rounded-sm group"
                  >
                    <div className={`w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-${item.color}-500/10 border border-${item.color}-500/20 flex items-center justify-center flex-shrink-0`}>
                      <item.icon className={`w-4 h-4 sm:w-5 sm:h-5 text-${item.color}-400`} />
                    </div>
                    <div>
                      <h4 className="text-xs sm:text-sm font-normal mb-0.5 sm:mb-1 text-white/90">{item.title}</h4>
                      <p className="text-[10px] sm:text-xs text-white/40">{item.desc}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative hidden lg:block"
            >
              {/* Architecture visualization */}
              <div className="aspect-square relative">
                {/* Animated rings */}
                <div className="absolute inset-0 flex items-center justify-center">
                  {[1, 2, 3].map((ring) => (
                    <motion.div
                      key={ring}
                      className="absolute border border-white/5 rounded-full"
                      style={{ 
                        width: `${ring * 30}%`, 
                        height: `${ring * 30}%`,
                      }}
                      animate={{ rotate: 360 }}
                      transition={{ duration: 20 + ring * 10, repeat: Infinity, ease: 'linear' }}
                    />
                  ))}
                </div>
                
                {/* Center node */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <motion.div 
                    className="w-32 h-32 rounded-full bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-white/20 flex items-center justify-center backdrop-blur-sm"
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ duration: 4, repeat: Infinity }}
                  >
                    <Brain className="w-12 h-12 text-white/70" />
                  </motion.div>
                </div>
                
                {/* Orbiting nodes */}
                {['Drug A', 'Drug B', 'Proteins', 'Pathways'].map((label, i) => (
                  <motion.div
                    key={label}
                    className="absolute w-16 h-16 rounded-full bg-black border border-white/20 flex items-center justify-center"
                    style={{
                      top: `${50 + 35 * Math.sin(i * Math.PI / 2)}%`,
                      left: `${50 + 35 * Math.cos(i * Math.PI / 2)}%`,
                      transform: 'translate(-50%, -50%)'
                    }}
                    animate={{ 
                      boxShadow: ['0 0 0 rgba(0,255,255,0)', '0 0 20px rgba(0,255,255,0.3)', '0 0 0 rgba(0,255,255,0)']
                    }}
                    transition={{ duration: 2, delay: i * 0.5, repeat: Infinity }}
                  >
                    <span className="text-[8px] text-white/70 uppercase tracking-wider text-center">{label}</span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 sm:py-32 relative">
        <div className="absolute inset-0 bg-gradient-to-t from-cyan-950/20 via-transparent to-transparent"></div>
        <div className="max-w-4xl mx-auto px-4 sm:px-6 text-center relative">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-2xl sm:text-4xl md:text-5xl font-extralight mb-4 sm:mb-6 px-2">
              Ready to <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">Transform</span> Drug Safety?
            </h2>
            <p className="text-sm sm:text-lg text-white/60 mb-6 sm:mb-10 font-light px-4">
              Start exploring our AI-powered platform for clinical decision support
            </p>
            <motion.button 
              whileHover={{ scale: 1.02, boxShadow: '0 0 60px rgba(0, 255, 255, 0.4)' }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate('/dashboard')}
              className="px-8 sm:px-12 py-3 sm:py-5 bg-gradient-to-r from-cyan-500 to-purple-500 text-black text-xs sm:text-sm uppercase tracking-widest font-medium rounded-sm"
            >
              Launch Dashboard
            </motion.button>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 sm:py-12 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 sm:gap-6">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center">
                <GitBranch className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-white/70" />
              </div>
              <span className="text-[10px] sm:text-xs text-white/50 uppercase tracking-widest">Project Aegis © 2026</span>
            </div>
            <div className="flex items-center gap-4 sm:gap-6">
              <a href="https://github.com/ahrazkk/ProjectAegis-Clinical-MLOps-Platform" target="_blank" rel="noopener noreferrer" className="text-white/50 hover:text-white transition-colors">
                <Github className="w-4 h-4" />
              </a>
              <a href="#" className="text-white/50 hover:text-white transition-colors">
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
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm"
            onClick={() => setShowModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="bg-black/80 backdrop-blur-xl w-full max-w-4xl max-h-[80vh] border border-white/10 overflow-hidden relative rounded-sm"
              style={{ boxShadow: '0 0 100px rgba(0, 255, 255, 0.1)' }}
            >
              <div className="p-6 border-b border-white/10 flex items-center justify-between">
                <div>
                  <span className="text-xs text-cyan-400/80 uppercase tracking-[0.2em]">Technical Report</span>
                  <h3 className="text-xl font-light mt-1">Project Aegis: DDI Prediction System</h3>
                </div>
                <button 
                  onClick={() => setShowModal(false)}
                  className="p-2 hover:bg-white/5 transition-colors rounded-sm"
                >
                  <X className="w-5 h-5 text-white/50" />
                </button>
              </div>
              <div className="p-6 overflow-y-auto max-h-[60vh] text-sm">
                <h4 className="text-cyan-400 text-xs uppercase tracking-widest mb-3">1. Problem Statement</h4>
                <p className="text-white/70 leading-relaxed mb-6">
                  Adverse drug events from drug-drug interactions (DDIs) cause over 195,000 hospitalizations annually
                  in the US alone. Project Aegis provides real-time DDI prediction to support clinical decision-making.
                </p>
                
                <h4 className="text-cyan-400 text-xs uppercase tracking-widest mb-3">2. Model Architecture</h4>
                <ul className="text-white/70 space-y-2 mb-6 list-none pl-0">
                  <li className="flex items-start gap-2"><span className="text-purple-400">→</span> <span><strong className="text-white/90">Encoder:</strong> PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)</span></li>
                  <li className="flex items-start gap-2"><span className="text-purple-400">→</span> <span><strong className="text-white/90">Relation Head:</strong> Binary classification with concatenated drug embeddings</span></li>
                  <li className="flex items-start gap-2"><span className="text-purple-400">→</span> <span><strong className="text-white/90">Training Data:</strong> DDI Corpus 2013 with 1,600+ annotated interactions</span></li>
                  <li className="flex items-start gap-2"><span className="text-purple-400">→</span> <span><strong className="text-white/90">Knowledge Graph:</strong> Neo4j Aura with 2,000+ drugs</span></li>
                </ul>

                <h4 className="text-cyan-400 text-xs uppercase tracking-widest mb-3">3. Performance</h4>
                <p className="text-white/70 leading-relaxed mb-6">
                  Our model achieves <span className="text-emerald-400 font-medium">92.7% AUC</span> on the DDI Corpus 2013 benchmark with sub-<span className="text-emerald-400 font-medium">200ms</span> inference time, 
                  making it suitable for real-time clinical decision support.
                </p>

                <h4 className="text-cyan-400 text-xs uppercase tracking-widest mb-3">4. Future Roadmap</h4>
                <p className="text-white/70 leading-relaxed mb-6">
                  Planned enhancements include Graph Neural Network (GNN) integration for molecular structure analysis
                  and multi-drug polypharmacy predictions.
                </p>
                
                <button
                  onClick={() => { setShowModal(false); navigate('/research'); }}
                  className="w-full py-3 mt-4 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-white/20 text-white text-xs uppercase tracking-widest hover:from-cyan-500/30 hover:to-purple-500/30 transition-all"
                >
                  View Full Research Documentation →
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
