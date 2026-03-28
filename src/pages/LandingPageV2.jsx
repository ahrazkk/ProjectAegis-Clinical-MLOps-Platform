import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import { motion, AnimatePresence, useScroll, useTransform, useInView } from 'framer-motion';
import { Cpu, ShieldAlert, ArrowRight, Shield, Brain, Network, Sparkles, ChevronDown, Zap, Activity,
  Database, GitBranch, Lock, BarChart3, X, Github, Linkedin, AlertTriangle,
  CheckCircle2, Loader2, Mail, ExternalLink, ArrowUpRight,
} from 'lucide-react';
import DrugInteractionBackground from '../components/DrugInteractionBackground';
import YoloTrackingOverlay from '../components/YoloTrackingOverlay';
import { checkHealth } from '../services/api';
import { AreaChart, Area, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import gnnData from '../assets/gnn_real_data.json';

// ─────────────────────────────────────────────────────────────────────────────
// DATA
// ─────────────────────────────────────────────────────────────────────────────
const features = [
  { icon: Brain, title: 'PubMedBERT Encoder', description: 'Fine-tuned biomedical BERT model processes drug pair contexts with 98.67% AUC accuracy', color: 'from-purple-500/20 to-violet-500/20' },
  { icon: Network, title: 'Knowledge Graph', description: 'Neo4j-powered graph with 2,000+ drugs and 1,600+ verified interactions from DDI Corpus', color: 'from-pink-500/20 to-purple-500/20' },
  { icon: Shield, title: 'Evidence-Based', description: 'RAG-powered citations from PubMed literature provide clinical context for predictions', color: 'from-indigo-500/20 to-blue-500/20' },
  { icon: Zap, title: 'Real-time Analysis', description: 'Sub-200ms predictions powered by optimized PyTorch models on Google Cloud Run', color: 'from-violet-500/20 to-purple-500/20' },
  { icon: Activity, title: 'Therapeutic Classification', description: 'Automatic drug categorization with 60%+ coverage for clinical decision support', color: 'from-purple-500/20 to-pink-500/20' },
  { icon: Database, title: 'GraphSAGE Neural Nets', description: 'Macroscopic GraphSAGE architecture processing 53k+ clinical pathways with 98.67% precision', color: 'from-blue-500/20 to-indigo-500/20' },
];

const stats = [
  { value: '2K+', label: 'Drugs Indexed', suffix: '', color: '#A78BFA' },
  { value: '98.6', label: 'AUC Score', suffix: '%', color: '#C084FC' },
  { value: '<200', label: 'Inference', suffix: 'ms', color: '#EC4899' },
  { value: '53.4K+', label: 'Interactions', suffix: '', color: '#818CF8' },
];

const techStack = [
    { icon: Database, title: 'Data Ingestion & Graph', desc: 'Heterogeneous data from DrugBank, FDA, and PubChem is parsed into a Neo4j Knowledge Graph, creating a macroscopic map of 2k+ drugs and 53k+ clinical pathways.', accent: '#A78BFA' },
    { icon: Brain, title: 'Biomedical Entity Extraction', desc: 'Fine-tuned PubMedBERT extracts precise pharmacological contexts from 30M+ literature abstracts, encoding semantic meaning into dense vector embeddings.', accent: '#8B5CF6' },
    { icon: Network, title: 'GraphSAGE Neural Inference', desc: 'Message-passing neural networks aggregate structural topologies and relational geometries to predict novel drug interactions with 98.67% accuracy.', accent: '#EC4899' },
    { icon: Sparkles, title: 'RAG Explanations', desc: 'The large language model layer cross-references predictive logits against clinical literature, grounding black-box AI within human-readable citations.', accent: '#818CF8' }
  ];

// Font cycling options — many entries, fast→fast→slow deceleration to final cursive
const fontCycle = [
  // Rapid-fire phase (~25ms each) — chaotic mix of all fonts
  { family: 'Space Grotesk, sans-serif', weight: 700, style: 'normal' },
  { family: 'JetBrains Mono, monospace', weight: 400, style: 'normal' },
  { family: 'Georgia, serif', weight: 700, style: 'italic' },
  { family: 'Inter, sans-serif', weight: 800, style: 'normal' },
  { family: 'Instrument Serif, serif', weight: 400, style: 'italic' },
  { family: 'Arial, sans-serif', weight: 900, style: 'normal' },
  { family: 'JetBrains Mono, monospace', weight: 700, style: 'normal' },
  { family: 'Space Grotesk, sans-serif', weight: 300, style: 'normal' },
  { family: 'Cormorant Garamond, serif', weight: 700, style: 'normal' },
  { family: 'Inter, sans-serif', weight: 300, style: 'normal' },
  { family: 'Georgia, serif', weight: 400, style: 'normal' },
  { family: 'Instrument Serif, serif', weight: 400, style: 'normal' },
  { family: 'Space Grotesk, sans-serif', weight: 500, style: 'normal' },
  { family: 'JetBrains Mono, monospace', weight: 300, style: 'normal' },
  { family: 'Inter, sans-serif', weight: 600, style: 'normal' },
  // Mid phase — still fast but starting to slow
  { family: 'Instrument Serif, serif', weight: 400, style: 'italic' },
  { family: 'Georgia, serif', weight: 400, style: 'italic' },
  { family: 'Space Grotesk, sans-serif', weight: 700, style: 'normal' },
  { family: 'Cormorant Garamond, serif', weight: 300, style: 'italic' },
  { family: 'JetBrains Mono, monospace', weight: 500, style: 'normal' },
  { family: 'Inter, sans-serif', weight: 700, style: 'normal' },
  { family: 'Instrument Serif, serif', weight: 400, style: 'italic' },
  { family: 'Space Grotesk, sans-serif', weight: 400, style: 'normal' },
  { family: 'Georgia, serif', weight: 700, style: 'italic' },
  // Decelerating — serif/cursive fonts start dominating
  { family: 'Cormorant Garamond, serif', weight: 600, style: 'normal' },
  { family: 'Instrument Serif, serif', weight: 400, style: 'italic' },
  { family: 'Inter, sans-serif', weight: 500, style: 'normal' },
  { family: 'Cormorant Garamond, serif', weight: 500, style: 'italic' },
  { family: 'Instrument Serif, serif', weight: 400, style: 'italic' },
  { family: 'Cormorant Garamond, serif', weight: 600, style: 'italic' },
  // Final slow approach — almost there
  { family: 'Instrument Serif, serif', weight: 400, style: 'italic' },
  { family: 'Cormorant Garamond, serif', weight: 400, style: 'italic' },
  { family: 'Cormorant Garamond, serif', weight: 600, style: 'italic' },
  { family: 'Cormorant Garamond, serif', weight: 500, style: 'italic' }, // final
];

// ─────────────────────────────────────────────────────────────────────────────
// REUSABLE COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function AnimatedCounter({ value, suffix = '', startCounting = true }) {
  const [displayValue, setDisplayValue] = useState(0);
  const numericValue = parseFloat(value.replace(/[^0-9.]/g, ''));
  const hasK = value.includes('K');

  useEffect(() => {
    if (!startCounting) return;
    const duration = 2000;
    const steps = 60;
    const stepValue = numericValue / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += stepValue;
      if (current >= numericValue) { setDisplayValue(numericValue); clearInterval(timer); }
      else setDisplayValue(Math.floor(current * 10) / 10);
    }, duration / steps);
    return () => clearInterval(timer);
  }, [numericValue, startCounting]);

  const prefix = value.includes('<') ? '<' : '';
  const kSuffix = hasK ? 'K' : '';
  const plusSuffix = value.includes('+') ? '+' : '';
  return <span>{prefix}{displayValue % 1 === 0 ? Math.floor(displayValue) : displayValue.toFixed(1)}{kSuffix}{plusSuffix}{suffix}</span>;
}

// InView reveal — uses intersection observer, much smoother than scroll-linked
function Reveal({ children, className = '', delay = 0, direction = 'up', once = true }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once, margin: '-60px' });

  const variants = {
    hidden: {
      opacity: 0,
      y: direction === 'up' ? 40 : direction === 'down' ? -40 : 0,
      x: direction === 'left' ? 50 : direction === 'right' ? -50 : 0,
      scale: 0.97,
    },
    visible: {
      opacity: 1, y: 0, x: 0, scale: 1,
      transition: {
        duration: 0.7,
        delay: delay,
        ease: [0.25, 0.46, 0.45, 0.94],
      },
    },
  };

  return (
    <motion.div
      ref={ref}
      className={className}
      initial="hidden"
      animate={isInView ? 'visible' : 'hidden'}
      variants={variants}
    >
      {children}
    </motion.div>
  );
}

// Staggered children reveal
function StaggerReveal({ children, className = '', staggerDelay = 0.08 }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-40px' });

  return (
    <motion.div
      ref={ref}
      className={className}
      initial="hidden"
      animate={isInView ? 'visible' : 'hidden'}
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: staggerDelay } },
      }}
    >
      {children}
    </motion.div>
  );
}

function StaggerChild({ children, className = '' }) {
  return (
    <motion.div
      className={className}
      variants={{
        hidden: { opacity: 0, y: 30, scale: 0.96 },
        visible: {
          opacity: 1, y: 0, scale: 1,
          transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] },
        },
      }}
    >
      {children}
    </motion.div>
  );
}

// Parallax section — subtle y-shift based on scroll
function ParallaxSection({ children, className = '', speed = 0.1 }) {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] });
  const y = useTransform(scrollYProgress, [0, 1], [speed * 100, -speed * 100]);
  return (
    <motion.div ref={ref} className={className} style={{ y }}>
      {children}
    </motion.div>
  );
}

// Font cycling animation for the accent word
function FontCycleWord({ word }) {
  const [fontIndex, setFontIndex] = useState(0);
  const [settled, setSettled] = useState(false);
  const total = fontCycle.length;

  useEffect(() => {
    if (fontIndex >= total - 1) {
      setSettled(true);
      return;
    }
    // Cubic slowdown: starts at ~25ms, ends at ~400ms for dramatic deceleration
    const progress = fontIndex / (total - 1);
    const delay = 25 + Math.pow(progress, 3) * 375;
    const timer = setTimeout(() => setFontIndex(i => i + 1), delay);
    return () => clearTimeout(timer);
  }, [fontIndex, total]);

  const font = fontCycle[fontIndex];
  return (
    <motion.span
      className="inline-block bg-gradient-to-r from-purple-400 via-pink-400 to-purple-300 bg-clip-text text-transparent"
      style={{
        fontFamily: font.family,
        fontWeight: font.weight,
        fontStyle: font.style,
        filter: settled ? 'drop-shadow(0 0 60px rgba(139,92,246,0.4))' : 'none',
        transition: 'filter 0.6s ease',
      }}
      animate={settled ? { scale: [1, 1.015, 1] } : {}}
      transition={settled ? { duration: 3, repeat: Infinity, ease: 'easeInOut' } : {}}
    >
      {word}
    </motion.span>
  );
}

// Word-by-word entrance
function WordReveal({ text, className = '', delay = 0 }) {
  return (
    <span className={className}>
      {text.split(' ').map((word, i) => (
        <motion.span
          key={i}
          className="inline-block mr-[0.3em]"
          initial={{ opacity: 0, y: 40, filter: 'blur(10px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 0.7, delay: delay + i * 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
          {word}
        </motion.span>
      ))}
    </span>
  );
}

function NoiseOverlay({ opacity = 0.03, className = '' }) {
  return <div className={`absolute inset-0 pointer-events-none yolo-dither animate-grain ${className}`} style={{ opacity }} />;
}


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
      className={`relative overflow-hidden rounded-xl border border-white/[0.04] bg-white/[0.015] ${className}`}
    >
      <div
        className="pointer-events-none absolute -inset-px z-0 transition-opacity duration-500"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, ${hoverColor}, transparent 40%)`,
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
            <div key={i} className={`relative flex flex-col md:flex-row items-center gap-8 ${isEven ? 'md:flex-row-reverse' : ''}`}>
               {/* Content Block */}
               <motion.div 
                 initial={{ opacity: 0, x: isEven ? 50 : -50 }}
                 whileInView={{ opacity: 1, x: 0 }}
                 viewport={{ once: true, margin: '-100px' }}
                 transition={{ duration: 0.7, delay: 0.2 }}
                 className="w-full md:w-1/2 flex flex-col pl-20 md:pl-0"
               >
                  <SpotlightCard hoverColor={`${step.accent}22`} className="p-6 sm:p-8 backdrop-blur-md">
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
               <div className="absolute left-[24px] md:left-1/2 -translate-x-[45%] md:-translate-x-1/2 w-4 h-4 rounded-full bg-[#040405] border-[3px] border-white/20 z-10 flex items-center justify-center box-content">
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

function GradientDivider({ className = '' }) {
  return <div className={`w-full h-px ${className}`} style={{ background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.15) 30%, rgba(236,72,153,0.15) 70%, transparent)' }} />;
}

const CAMERA_CONFIG = { position: [0, 0, 12], fov: 50 };
const GL_CONFIG = { alpha: true, antialias: true };
const CHROMATIC_OFFSET = [0.0008, 0.0008];

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────


// Interactive node for the Graph Data section
function GNNNode({ x, y, label, color, glow, delay }) {
  return (
    <motion.div 
      className="absolute flex flex-col items-center justify-center pointer-events-none group z-20"
      style={{ left: x, top: y, x: '-50%', y: '-50%' }}
      initial={{ scale: 0, opacity: 0 }}
      whileInView={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 100, damping: 15, delay }}
    >
      <div className={`relative w-16 h-16 rounded-full border-2 ${color} bg-black/80 flex items-center justify-center text-[10px] font-mono text-white/80 shadow-lg pointer-events-auto transition-transform duration-300 group-hover:scale-110 group-hover:bg-white/5`}>
        {label}
        <div className={`absolute inset-0 rounded-full ${glow} blur-md -z-10 opacity-50 group-hover:opacity-100 transition-opacity duration-300`} />
      </div>
    </motion.div>
  );
}

// Edge component with exact interaction hover over the line
function GNNEdge({ x1, y1, x2, y2, interaction, probability, severityColor, lineProps, delay }) {
  return (
    <motion.div 
      className="absolute top-0 left-0 w-full h-full z-10"
      initial={{ opacity: 0 }} 
      whileInView={{ opacity: 1 }} 
      transition={{ delay }}
    >
      <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible">
         <motion.line x1={x1} y1={y1} x2={x2} y2={y2} stroke={lineProps.stroke || "rgba(255,255,255,0.2)"} strokeWidth={lineProps.width || 2} strokeDasharray={lineProps.dash}
            initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} transition={{ duration: 1.5, delay }} />
      </svg>
      
      {/* Invisible thick line overlay for easy hovering */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible">
         <line className="pointer-events-auto cursor-crosshair group peer" stroke="transparent" strokeWidth="30" x1={x1} y1={y1} x2={x2} y2={y2} />
         
         <foreignObject x={0} y={0} width="100%" height="100%" className="pointer-events-none opacity-0 peer-hover:opacity-100 transition-opacity duration-300">
           <div className="absolute flex items-center justify-center w-full h-full">
              <div 
                className="bg-black/95 border border-white/10 px-4 py-2 rounded-md backdrop-blur-xl shadow-2xl flex flex-col items-center gap-1"
                style={{
                  position: 'absolute',
                  left: `calc(${x1} + (${x2} - ${x1})/2)` ,
                  top: `calc(${y1} + (${y2} - ${y1})/2)`,
                  transform: 'translate(-50%, -50%)',
                }}
              >
                  <div className="flex items-center gap-2 mb-1 border-b border-white/[0.05] pb-1 w-full justify-between">
                     <span className="text-[9px] uppercase font-mono tracking-widest text-white/50">Inference Edge</span>
                     <div className={`w-1.5 h-1.5 rounded-full shadow-[0_0_8px_currentColor] ${severityColor.bg}`} />
                  </div>
                  <div className={`text-xs font-semibold ${severityColor.text}`}>{interaction}</div>
                  <div className="text-[10px] font-mono text-white/40">Probability: <span className="text-white/80">{probability}</span></div>
              </div>
           </div>
         </foreignObject>
      </svg>
    </motion.div>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const [scrollY, setScrollY] = useState(0);
  const heroRef = useRef(null);
  const canvasContainerRef = useRef(null);
  const moleculePositionsRef = useRef({ drugA: [-3.5,0,0], drugB: [3.5,0,0], center: [0,0,0] });
  const cameraRef = useRef(null);
  const statsRef = useRef(null);
  const statsInView = useInView(statsRef, { once: true, margin: '-80px' });
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [connectionTime, setConnectionTime] = useState(0);

  useEffect(() => {
    const startTime = Date.now();
    let timer = setInterval(() => { if (backendStatus === 'connecting') setConnectionTime(Math.floor((Date.now() - startTime) / 1000)); }, 1000);
    checkHealth()
      .then(() => { setBackendStatus('ready'); setConnectionTime(Math.floor((Date.now() - startTime) / 1000)); clearInterval(timer); })
      .catch(() => { setBackendStatus('error'); clearInterval(timer); });
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const handler = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handler, { passive: true });
    return () => window.removeEventListener('scroll', handler);
  }, []);

  const scrollProgress = Math.min(scrollY / 500, 1);

  return (
    <div className="min-h-screen text-white overflow-x-hidden landing-page" style={{ backgroundColor: '#040405' }}>

      {/* ═══════════ NAVIGATION ═══════════ */}
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="fixed top-0 left-0 right-0 z-50 bg-[#040405]/70 backdrop-blur-2xl border-b border-white/[0.04]"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
          <motion.div className="flex items-center gap-2 sm:gap-3" whileHover={{ scale: 1.02 }}>
            <div className="relative w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center">
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/15 to-pink-500/15 rounded-lg"></div>
              <GitBranch className="w-4 h-4 sm:w-5 sm:h-5 text-white/60 relative z-10" />
            </div>
            <span className="text-sm sm:text-base tracking-wide text-white/90">
              <span className="font-heading font-medium">Project</span>{' '}
              <span className="font-cursive italic text-purple-300/80">Aegis</span>
            </span>
          </motion.div>

          <div className="hidden md:flex items-center gap-8">
            {['Features', 'Technology', 'Research'].map(label => (
              <button key={label}
                onClick={() => label === 'Research' ? navigate('/research') : document.getElementById(label.toLowerCase())?.scrollIntoView({ behavior: 'smooth' })}
                className="font-body text-xs text-white/35 hover:text-white/70 transition-colors tracking-wide"
              >{label}</button>
            ))}
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate('/dashboard')}
            className="border border-pink-400/80 bg-white/5 backdrop-blur-md px-4 py-1.5 sm:px-5 sm:py-2 text-white/80 text-[10px] sm:text-xs font-heading tracking-wider hover:text-white hover:bg-white/10 transition-all rounded-none"
          >
            <span className="hidden sm:inline">Launch Platform</span>
            <span className="sm:hidden">Launch</span>
          </motion.button>
        </div>
      </motion.nav>

      {/* ═══════════ HERO ═══════════ */}
      <section ref={heroRef} className="relative min-h-[105vh] flex items-center overflow-hidden pt-16 sm:pt-20">
<div ref={canvasContainerRef} className="absolute inset-0 z-0">
            <Canvas 
              camera={CAMERA_CONFIG} 
              gl={GL_CONFIG}
              onCreated={({ camera }) => { cameraRef.current = camera; }}
            >
              <DrugInteractionBackground scrollProgress={scrollProgress} moleculePositionsRef={moleculePositionsRef} />
              <EffectComposer>
                <Bloom intensity={0.8} luminanceThreshold={0.35} luminanceSmoothing={0.85} mipmapBlur />
                <ChromaticAberration blendFunction={BlendFunction.NORMAL} offset={CHROMATIC_OFFSET} />
              </EffectComposer>
            </Canvas>
        </div>

        <YoloTrackingOverlay moleculePositionsRef={moleculePositionsRef} cameraRef={cameraRef} canvasContainerRef={canvasContainerRef} />

        {/* Background gradient wash — dimmed */}
        <div className="absolute inset-0 pointer-events-none z-[1]" style={{
          background: `
            radial-gradient(ellipse at 15% 35%, rgba(139,92,246,0.10) 0%, transparent 45%),
            radial-gradient(ellipse at 85% 25%, rgba(236,72,153,0.07) 0%, transparent 45%),
            radial-gradient(ellipse at 50% 85%, rgba(30,58,138,0.12) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(4,4,5,0.5) 0%, transparent 80%)
          `,
        }} />

        <div className="absolute top-0 left-0 right-0 h-40 bg-gradient-to-b from-[#040405] to-transparent pointer-events-none z-[1]" />
        <div className="absolute bottom-0 left-0 right-0 h-56 bg-gradient-to-t from-[#040405] to-transparent pointer-events-none z-[1]" />
        <NoiseOverlay opacity={0.03} className="z-[1]" />

        <div className="relative z-10 flex flex-col items-center justify-center min-h-screen text-center px-4 sm:px-6 w-full">
          {/* Stronger backdrop for readability */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[140%] h-[500px] pointer-events-none -z-[1]"
            style={{ background: 'radial-gradient(ellipse, rgba(4,4,5,0.85) 0%, rgba(4,4,5,0.5) 35%, transparent 65%)' }} />

          {/* Kicker */}
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
            className="flex items-center gap-3 mb-6 sm:mb-10">
            <div className="w-8 h-px bg-gradient-to-r from-transparent to-purple-500/40" />
            <span className="font-mono text-[9px] sm:text-[10px] uppercase tracking-[0.35em] text-purple-400/50">
              AI-Powered Drug Safety
            </span>
            <div className="w-8 h-px bg-gradient-to-l from-transparent to-purple-500/40" />
          </motion.div>

          {/* Main heading — "DDI Intelligence" white + "Reimagined" font-cycling cursive */}
          <h1 className="leading-[0.95] max-w-6xl">
            <span className="block font-heading text-4xl sm:text-6xl md:text-7xl xl:text-8xl font-bold tracking-tight text-white">
              <WordReveal text="DDI Intelligence" delay={0.4} />
            </span>
            <motion.span
              className="block mt-1 sm:mt-2 text-4xl sm:text-6xl md:text-7xl xl:text-8xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.9, duration: 0.8 }}
            >
              <FontCycleWord word="Reimagined" />
            </motion.span>
          </h1>

          {/* Description */}
          <motion.p initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.8, duration: 0.7 }}
            className="font-body text-sm sm:text-base lg:text-lg text-white/40 max-w-lg mt-6 sm:mt-8 leading-relaxed">
            Predicting drug-drug interactions with{' '}
            <span className="text-purple-400/80">biomedical language models</span> and{' '}
            <span className="text-pink-400/70">knowledge graphs</span>
          </motion.p>

          {/* CTAs */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 2.1 }} className="flex flex-col sm:flex-row gap-3 sm:gap-5 mt-8 sm:mt-12">
            <motion.button whileHover={{ scale: 1.03, y: -2 }} whileTap={{ scale: 0.97 }}
              onClick={() => navigate('/dashboard')}
              className="group relative px-8 sm:px-12 py-4 sm:py-5 border border-pink-400/80 bg-white/5 backdrop-blur-md text-white font-heading font-semibold tracking-widest overflow-hidden rounded-none hover:bg-white/10 transition-all"
            >
              <span className="relative flex items-center justify-center gap-3 text-xs sm:text-sm">
                Enter Dashboard <ArrowRight className="w-4 h-4 group-hover:translate-x-1.5 transition-transform duration-300" />
              </span>
            </motion.button>

            <motion.button whileHover={{ scale: 1.03, y: -2 }} whileTap={{ scale: 0.97 }}
              onClick={() => navigate('/research')}
              className="border border-pink-400/80 bg-white/5 backdrop-blur-md px-8 sm:px-12 py-4 sm:py-5 text-white/80 text-xs sm:text-sm font-heading font-medium tracking-widest hover:text-white hover:bg-white/10 transition-all rounded-none">
              <span className="flex items-center justify-center gap-3">View Research <ArrowUpRight className="w-4 h-4" /></span>
            </motion.button>
          </motion.div>

          {/* Backend Status */}
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.4 }} className="mt-6 sm:mt-8 pb-16 sm:pb-0">
            <AnimatePresence mode="wait">
              {backendStatus === 'connecting' && (
                <motion.div key="c" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex items-start gap-3 p-3 sm:p-4 bg-yellow-500/5 border border-yellow-500/15 max-w-xl rounded-lg backdrop-blur-sm">
                  <Loader2 className="w-4 h-4 text-yellow-400/60 animate-spin flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs text-yellow-400/60 font-body">Warming up AI Backend... ({connectionTime}s)</p>
                    <p className="text-[10px] text-white/20 mt-1 font-body">May take 25-35s on first visit</p>
                  </div>
                </motion.div>
              )}
              {backendStatus === 'ready' && (
                <motion.div key="r" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex items-center gap-3 p-3 bg-emerald-500/5 border border-emerald-500/15 max-w-xl rounded-lg">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400/60" />
                  <p className="text-xs text-emerald-400/60 font-body">AI Backend ready ({connectionTime}s)</p>
                </motion.div>
              )}
              {backendStatus === 'error' && (
                <motion.div key="e" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex items-center gap-3 p-3 bg-red-500/5 border border-red-500/15 max-w-xl rounded-lg">
                  <AlertTriangle className="w-4 h-4 text-red-400/60" />
                  <p className="text-xs text-red-400/60 font-body">Backend unavailable. Some features may be limited.</p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Scroll indicator */}
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.8 }}
            className="hidden sm:block absolute bottom-8 left-1/2 -translate-x-1/2">
            <motion.div animate={{ y: [0, 8, 0] }} transition={{ repeat: Infinity, duration: 2 }}
              className="flex flex-col items-center gap-2">
              <span className="font-mono text-[9px] text-white/15 uppercase tracking-[0.3em]">Scroll</span>
              <ChevronDown className="w-4 h-4 text-white/15" />
            </motion.div>
          </motion.div>
        </div>
      </section>

     
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

      
      
                  {/* ═══════════ GRAPH STATS, DISTRIBUTION & MOCK (FULL WIDTH) ═══════════ */}
      {/* Seamless cross-fade gradient pushing up from the dark block */}
      <div className="w-full h-40 bg-gradient-to-b from-transparent to-black pointer-events-none -mt-20 z-0 relative" />
      
      <section className="relative pb-32 pt-16 z-10 bg-black">
        {/* Horizontal seamless layout bleeding edge to edge visually */}
        <div className="w-full border-y border-white/5 bg-[#0a0a0a]/50 backdrop-blur-3xl shadow-[0_0_80px_rgba(0,0,0,0.8)] z-20">
          
          <div className="max-w-[1600px] mx-auto w-full">
            <div className="grid grid-cols-1 xl:grid-cols-12 w-full">
              
              <div className="col-span-1 xl:col-span-5 p-8 lg:p-14 flex flex-col relative bg-gradient-to-b from-black/80 to-black/40 z-10 border-r border-white/5">
               <Reveal>
                  <h3 className="text-xl lg:text-3xl font-heading font-semibold tracking-tight text-white mb-3">Model Accuracy & Scale</h3>
                  <p className="text-sm text-white/40 font-body mb-10 pb-8 border-b border-white/[0.03]">We process massive graph embeddings at 98.6% real-world inference accuracy. *Evaluated on validated DDI benchmarks.</p>
                  
                  <div className="grid grid-cols-2 gap-x-6 gap-y-10 mb-12">
                     {[
                       { label: 'Macro F1', value: '0.968', desc: 'Predictive precision across 86 classes' },
                       { label: 'ROC-AUC', value: '98.67%', desc: 'High true-positive rate bounding' },
                       { label: 'Nodes', value: '14,200+', desc: 'Isolated graph embeddings' },
                       { label: 'Edges', value: '2.4M+', desc: 'Metabolic & interaction pathways' }
                     ].map((stat, i) => (
                        <div key={i} className="group cursor-default">
                           <div className="text-[10px] font-mono uppercase tracking-widest text-[#ec4899] mb-1">{stat.label}</div>
                           <div className="text-3xl font-heading font-light text-white mb-1 group-hover:text-purple-400 transition-colors">{stat.value}</div>
                           <div className="text-[10px] text-white/30 font-body leading-relaxed max-w-[140px]">{stat.desc}</div>
                        </div>
                     ))}
                  </div>

                  {/* Probability Density Chart */}
                  <div className="mt-auto relative">
                     <div className="text-[10px] uppercase font-mono tracking-widest text-white/50 mb-4 flex items-center justify-between">
                       <span>Probability Density</span>
                       <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-[#facc15] rounded-full animate-pulse"/> Optimum: 0.56</span>
                     </div>
                     <p className="text-[10px] text-white/40 mb-4 font-body leading-relaxed">
                        The GNN embedding space maps high-dimensional topological interactions. The 0.56 threshold boundary confidently segments clinically severe interactions from benign relationships based on edge-weight gradients.
                     </p>
                     <div className="h-[200px] w-full relative">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={gnnData?.probability || []} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                            <defs>
                              <linearGradient id="colorNegCard" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#a855f7" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                              </linearGradient>
                              <linearGradient id="colorPosCard" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ec4899" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#ec4899" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <XAxis dataKey="prob" type="number" domain={[0, 1]} stroke="#ffffff15" tick={{ fill: '#ffffff40', fontSize: 10 }} tickCount={6} />
                            <YAxis stroke="#ffffff15" tick={{ fill: '#ffffff40', fontSize: 10 }} opacity={0.5} />
                            <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: '#333', fontSize: '11px', backdropFilter: 'blur(10px)' }} itemStyle={{ color: '#fff' }} />
                            <ReferenceLine x={0.56} stroke="#facc15" strokeDasharray="3 3" strokeWidth={1} opacity={0.6} />
                            <Area type="monotone" dataKey="neg" name="True N" stroke="#a855f7" strokeWidth={2} fill="url(#colorNegCard)" />
                            <Area type="monotone" dataKey="pos" name="True P" stroke="#ec4899" strokeWidth={2} fill="url(#colorPosCard)" />
                          </AreaChart>
                        </ResponsiveContainer>
                     </div>
                  </div>
               </Reveal>
            </div>

            {/* Right Block: Simplified Flex Flow Diagram (Col span 7) */}
              <div className="col-span-1 xl:col-span-7 relative bg-black/40 min-h-[500px] lg:min-h-[700px] flex items-center justify-center p-8 overflow-hidden">
                  
                  {/* Background Grid & Blurs */}
                  <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:30px_30px] [mask-image:radial-gradient(ellipse_70%_70%_at_50%_50%,#000_100%,transparent_100%)] pointer-events-none" />
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[40rem] h-[40rem] bg-purple-500/10 rounded-full blur-[120px] mix-blend-screen pointer-events-none" />

                  {/* HUD Elements */}
                  <div className="absolute top-8 left-8 z-30 pointer-events-none hidden sm:block">
                    <div className="flex items-center gap-2 mb-2">
                       <span className="relative flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-pink-400 opacity-75" />
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-pink-500" />
                       </span>
                       <span className="text-[10px] font-mono text-white/50 tracking-widest uppercase">GNN Live Flow</span>
                    </div>
                  </div>

                  {/* The Flow Diagram */}
                  <div className="relative z-10 w-full max-w-2xl flex flex-col md:flex-row items-center gap-8 lg:gap-16">
                      
                      {/* Left: Hub Node */}
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.8 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        className="relative flex flex-col items-center shrink-0"
                      >
                         <div className="absolute -top-8 text-[10px] font-mono text-[#a78bfa] bg-black/60 px-2 py-0.5 rounded border border-purple-500/20">Target Input</div>
                         <div className="w-32 h-32 rounded-full bg-black/80 backdrop-blur-xl border border-white/20 flex flex-col items-center justify-center shadow-[0_0_50px_rgba(168,85,247,0.3)] z-20">
                            <span className="font-heading font-medium text-xl tracking-wide text-white">Aspirin</span>
                            <span className="text-[10px] text-white/40 font-mono mt-1">CHEMBL123</span>
                         </div>
                         
                         {/* Visual Spines connecting from central node -> Right nodes (Desktop) */}
                         <div className="hidden md:block absolute top-1/2 left-full w-12 border-t border-white/20 -translate-y-1/2 z-0" />
                         <div className="hidden md:block absolute top-[20%] left-[calc(100%+2.9rem)] bottom-[20%] w-px bg-white/20 z-0" />
                      </motion.div>

                      {/* Right: Interaction Cards */}
                      <div className="flex flex-col gap-4 relative z-10 w-full">
                          
                          {/* Warfarin Card */}
                          <motion.div 
                             initial={{ opacity: 0, x: 20 }}
                             whileInView={{ opacity: 1, x: 0 }}
                             transition={{ delay: 0.2 }}
                             className="flex items-center group relative overflow-hidden rounded-xl border border-white/10 bg-black/60 backdrop-blur-md hover:bg-white/[0.02] transition-colors"
                          >
                             {/* Connector (Desktop) */}
                             <div className="hidden md:block absolute -left-8 top-1/2 w-8 border-t border-white/20 -translate-y-1/2 z-0" />
                             
                             {/* Indicator Line */}
                             <div className="w-1.5 self-stretch bg-pink-500 rounded-l-xl shadow-[0_0_10px_rgba(236,72,153,0.5)]" />
                             
                             <div className="flex-1 p-4">
                                <div className="flex justify-between items-center mb-1">
                                   <div className="flex items-center gap-3">
                                      <span className="text-white font-heading font-medium tracking-wide">Warfarin</span>
                                      <span className="text-[9px] font-mono text-white/40 uppercase">Anticoagulant</span>
                                   </div>
                                   <div className="flex items-center gap-2">
                                      <span className="text-[10px] font-mono text-pink-400 bg-pink-500/10 px-2 py-0.5 rounded">0.984</span>
                                   </div>
                                </div>
                                <div className="text-xs text-white/50 font-body">CYP2C9 Inhibition (Severe Bleeding)</div>
                             </div>
                             
                             {/* Highlight glow effect on hover */}
                             <div className="absolute inset-0 bg-gradient-to-r from-pink-500/0 via-pink-500/0 to-pink-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                          </motion.div>

                          {/* Ibuprofen Card */}
                          <motion.div 
                             initial={{ opacity: 0, x: 20 }}
                             whileInView={{ opacity: 1, x: 0 }}
                             transition={{ delay: 0.4 }}
                             className="flex items-center group relative overflow-hidden rounded-xl border border-white/10 bg-black/60 backdrop-blur-md hover:bg-white/[0.02] transition-colors"
                          >
                             {/* Connector (Desktop) */}
                             <div className="hidden md:block absolute -left-8 top-1/2 w-8 border-t border-white/20 -translate-y-1/2 z-0" />
                             
                             {/* Indicator Line */}
                             <div className="w-1.5 self-stretch bg-yellow-500 rounded-l-xl shadow-[0_0_10px_rgba(234,179,8,0.3)]" />
                             
                             <div className="flex-1 p-4">
                                <div className="flex justify-between items-center mb-1">
                                   <div className="flex items-center gap-3">
                                      <span className="text-white font-heading font-medium tracking-wide">Ibuprofen</span>
                                      <span className="text-[9px] font-mono text-white/40 uppercase">NSAID</span>
                                   </div>
                                   <div className="flex items-center gap-2">
                                      <span className="text-[10px] font-mono text-yellow-400 bg-yellow-500/10 px-2 py-0.5 rounded">0.835</span>
                                   </div>
                                </div>
                                <div className="text-xs text-white/50 font-body">Decreased Renal Clearance</div>
                             </div>
                             
                             <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/0 via-yellow-500/0 to-yellow-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                          </motion.div>

                          {/* Clopidogrel Card */}
                          <motion.div 
                             initial={{ opacity: 0, x: 20 }}
                             whileInView={{ opacity: 1, x: 0 }}
                             transition={{ delay: 0.6 }}
                             className="flex items-center group relative overflow-hidden rounded-xl border border-white/10 bg-black/60 backdrop-blur-md hover:bg-white/[0.02] transition-colors"
                          >
                             {/* Connector (Desktop) */}
                             <div className="hidden md:block absolute -left-8 top-1/2 w-8 border-t border-white/20 -translate-y-1/2 z-0" />
                             
                             {/* Indicator Line */}
                             <div className="w-1.5 self-stretch bg-white/30 rounded-l-xl" />
                             
                             <div className="flex-1 p-4">
                                <div className="flex justify-between items-center mb-1">
                                   <div className="flex items-center gap-3">
                                      <span className="text-white font-heading font-medium tracking-wide">Clopidogrel</span>
                                      <span className="text-[9px] font-mono text-white/40 uppercase">Antiplatelet</span>
                                   </div>
                                   <div className="flex items-center gap-2">
                                      <span className="text-[10px] font-mono text-white/60 bg-white/10 px-2 py-0.5 rounded">0.121</span>
                                   </div>
                                </div>
                                <div className="text-xs text-white/50 font-body">Benign / No Structural Topology Match</div>
                             </div>
                          </motion.div>

                      </div>
                  </div>

                  {/* Legend */}
                  <div className="absolute bottom-6 left-6 flex flex-wrap items-center gap-6 text-[10px] uppercase font-mono tracking-widest text-white/40 z-30 border border-white/5 bg-black/40 px-4 py-2 rounded-full backdrop-blur-md">
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-pink-500" /> Severe</div>
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-yellow-500" /> Moderate</div>
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-white/30" /> Benign</div>
                  </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ ARCHITECTURE SCROLL PARALLAX ═══════════ */}      <section className="relative py-40 z-10 border-t border-white/[0.03] overflow-hidden bg-black/50" id="technology">
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


