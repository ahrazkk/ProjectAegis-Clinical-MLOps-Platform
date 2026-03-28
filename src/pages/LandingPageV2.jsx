import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import { motion, AnimatePresence, useScroll, useTransform, useInView } from 'framer-motion';
import {
  ArrowRight, Shield, Brain, Network, Sparkles, ChevronDown, Zap, Activity,
  Database, GitBranch, Lock, BarChart3, X, Github, Linkedin, AlertTriangle,
  CheckCircle2, Loader2, Mail, ExternalLink, ArrowUpRight,
} from 'lucide-react';
import DrugInteractionBackground from '../components/DrugInteractionBackground';
import YoloTrackingOverlay from '../components/YoloTrackingOverlay';
import { checkHealth } from '../services/api';

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
  { icon: Brain, title: 'PubMedBERT', desc: 'Fine-tuned transformer for biomedical text understanding', accent: '#8B5CF6' },
  { icon: Database, title: 'Neo4j Graph', desc: '2,000+ drugs with verified DDI relationships', accent: '#A78BFA' },
  { icon: Lock, title: 'RAG Pipeline', desc: 'PubMed-powered research with literature citations', accent: '#EC4899' },
  { icon: BarChart3, title: 'Cloud Deploy', desc: 'Google Cloud Run with containerized microservices', accent: '#818CF8' },
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

function GradientDivider({ className = '' }) {
  return <div className={`w-full h-px ${className}`} style={{ background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.15) 30%, rgba(236,72,153,0.15) 70%, transparent)' }} />;
}

const CAMERA_CONFIG = { position: [0, 0, 12], fov: 50 };
const GL_CONFIG = { alpha: true, antialias: true };
const CHROMATIC_OFFSET = [0.0008, 0.0008];

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
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
            className="btn-flow-border px-4 py-1.5 sm:px-5 sm:py-2 bg-white/[0.03] text-white/60 text-[10px] sm:text-xs font-heading tracking-wider hover:text-white transition-all rounded-md"
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
              className="group relative px-8 sm:px-12 py-4 sm:py-5 text-white font-heading font-semibold tracking-widest overflow-hidden rounded-lg"
              style={{
                background: 'linear-gradient(135deg, #6D28D9 0%, #BE185D 50%, #4F46E5 100%)',
                boxShadow: '0 0 25px rgba(109,40,217,0.15), 0 8px 32px rgba(0,0,0,0.5)',
              }}>
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/15 to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-700 ease-out" />
              <div className="absolute inset-0 bg-gradient-to-b from-white/[0.08] to-transparent" style={{ borderRadius: 'inherit' }} />
              <span className="relative flex items-center justify-center gap-3 text-xs sm:text-sm">
                Enter Dashboard <ArrowRight className="w-4 h-4 group-hover:translate-x-1.5 transition-transform duration-300" />
              </span>
            </motion.button>

            <motion.button whileHover={{ scale: 1.03, y: -2 }} whileTap={{ scale: 0.97 }}
              onClick={() => navigate('/research')}
              className="btn-flow-border px-8 sm:px-12 py-4 sm:py-5 bg-white/[0.02] text-white/50 text-xs sm:text-sm font-heading font-medium tracking-widest hover:text-white hover:bg-white/[0.05] transition-all rounded-lg backdrop-blur-sm">
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

      {/* ═══════════ STATS — smooth inView reveal ═══════════ */}
      <section ref={statsRef} className="relative py-16 sm:py-28 border-y border-white/[0.03]">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-purple-950/[0.03] to-transparent"></div>
        <NoiseOverlay opacity={0.012} />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 relative">
          <StaggerReveal className="grid grid-cols-2 md:grid-cols-4 gap-6 sm:gap-10" staggerDelay={0.12}>
            {stats.map((stat, i) => (
              <StaggerChild key={i}>
                <div className="text-center group">
                  <div className="relative inline-block">
                    <div className="font-heading text-3xl sm:text-4xl md:text-5xl font-bold mb-1 sm:mb-2 tabular-nums"
                      style={{ color: stat.color, textShadow: `0 0 40px ${stat.color}22` }}>
                      <AnimatedCounter value={stat.value} suffix={stat.suffix} startCounting={statsInView} />
                    </div>
                  </div>
                  <div className="font-mono text-[8px] sm:text-[9px] text-white/25 uppercase tracking-[0.2em] sm:tracking-[0.3em]">
                    {stat.label}
                  </div>
                  <div className="mt-3 mx-auto w-8 h-[1px]" style={{ background: `linear-gradient(90deg, transparent, ${stat.color}44, transparent)` }} />
                </div>
              </StaggerChild>
            ))}
          </StaggerReveal>
        </div>
      </section>

      {/* ═══════════ FEATURES — overlapping card grid with stagger ═══════════ */}
      <section id="features" className="py-20 sm:py-36 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <Reveal className="text-center mb-10 sm:mb-20">
            <span className="font-mono text-[9px] sm:text-[10px] text-purple-400/40 uppercase tracking-[0.3em] mb-3 block">Capabilities</span>
            <h2 className="mb-4 sm:mb-6 px-2">
              <span className="font-heading text-3xl sm:text-5xl md:text-6xl font-bold">Enterprise-Grade </span>
              <span className="font-cursive italic text-3xl sm:text-5xl md:text-6xl bg-gradient-to-r from-purple-400/90 to-pink-400/80 bg-clip-text text-transparent">
                DDI Analysis
              </span>
            </h2>
            <p className="font-body text-sm sm:text-base text-white/30 max-w-2xl mx-auto px-4">
              Built for clinical decision support with explainable predictions and comprehensive drug coverage
            </p>
          </Reveal>

          <StaggerReveal className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4" staggerDelay={0.1}>
            {features.map((feature, i) => (
              <StaggerChild key={i}>
                <motion.div
                  whileHover={{ y: -6, transition: { duration: 0.25 } }}
                  className="group relative p-5 sm:p-7 bg-white/[0.015] backdrop-blur-sm border border-white/[0.04] hover:border-purple-500/20 transition-all duration-400 overflow-hidden rounded-lg"
                >
                  <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                  <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/[0.02] to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />

                  <div className="relative z-10">
                    <div className="w-10 h-10 rounded-lg bg-white/[0.03] border border-white/[0.05] flex items-center justify-center mb-4 group-hover:border-purple-500/15 transition-colors">
                      <feature.icon className="w-4 h-4 text-purple-400/50 group-hover:text-purple-400/80 transition-colors" />
                    </div>
                    <h3 className="font-heading text-sm sm:text-base font-semibold mb-2 text-white/85">{feature.title}</h3>
                    <p className="font-body text-xs sm:text-sm text-white/30 leading-relaxed">{feature.description}</p>
                  </div>

                  <div className="absolute top-0 right-0 w-16 h-16 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                    <div className="absolute top-3 right-3 w-6 h-[1px] bg-gradient-to-l from-purple-400/30 to-transparent" />
                    <div className="absolute top-3 right-3 w-[1px] h-6 bg-gradient-to-t from-purple-400/30 to-transparent" />
                  </div>
                </motion.div>
              </StaggerChild>
            ))}
          </StaggerReveal>
        </div>
      </section>

      <GradientDivider />

      {/* ═══════════ TECHNOLOGY — with parallax offset ═══════════ */}
      <section id="technology" className="py-20 sm:py-36 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-purple-950/[0.03] to-transparent" />
        <NoiseOverlay opacity={0.01} />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 relative">
          <div className="grid lg:grid-cols-2 gap-8 sm:gap-16 items-center">
            <Reveal direction="left">
              <span className="font-mono text-[9px] sm:text-[10px] text-pink-400/35 uppercase tracking-[0.3em] mb-3 block">Technology Stack</span>
              <h2 className="mb-4 sm:mb-6">
                <span className="font-heading text-3xl sm:text-5xl font-bold">Production </span>
                <span className="font-cursive italic text-3xl sm:text-5xl bg-gradient-to-r from-purple-400/90 to-pink-400/80 bg-clip-text text-transparent">Architecture</span>
              </h2>
              <p className="font-body text-sm sm:text-base text-white/30 mb-6 sm:mb-10 leading-relaxed">
                Our platform combines fine-tuned biomedical language models with retrieval-augmented generation for explainable, evidence-based predictions.
              </p>

              <StaggerReveal className="space-y-3 sm:space-y-4" staggerDelay={0.1}>
                {techStack.map((item, i) => (
                  <StaggerChild key={i}>
                    <motion.div whileHover={{ x: 4, transition: { duration: 0.2 } }}
                      className="flex items-start gap-3 sm:gap-4 p-3 sm:p-4 bg-white/[0.015] border border-white/[0.03] hover:border-purple-500/15 transition-all group rounded-lg">
                      <div className="w-[2px] h-8 sm:h-10 flex-shrink-0 mt-1 rounded-full"
                        style={{ background: `linear-gradient(to bottom, ${item.accent}, transparent)` }} />
                      <div>
                        <h4 className="font-heading text-xs sm:text-sm font-semibold mb-0.5 text-white/70 group-hover:text-white/90 transition-colors">{item.title}</h4>
                        <p className="font-body text-[10px] sm:text-xs text-white/25">{item.desc}</p>
                      </div>
                    </motion.div>
                  </StaggerChild>
                ))}
              </StaggerReveal>
            </Reveal>

            {/* Architecture visualization — parallax */}
            <ParallaxSection className="relative hidden lg:block" speed={0.08}>
              <Reveal direction="right" delay={0.2}>
                <div className="aspect-square relative">
                  {[1, 2, 3].map(ring => (
                    <motion.div key={ring}
                      className="absolute border border-white/[0.03] rounded-full"
                      style={{ width: `${ring * 30}%`, height: `${ring * 30}%`, top: `${50 - ring * 15}%`, left: `${50 - ring * 15}%` }}
                      animate={{ rotate: ring % 2 === 0 ? 360 : -360 }}
                      transition={{ duration: 25 + ring * 12, repeat: Infinity, ease: 'linear' }}>
                      <div className="absolute w-2 h-2 rounded-full top-0 left-1/2 -translate-x-1/2 -translate-y-1/2"
                        style={{ background: ring === 1 ? '#8B5CF6' : ring === 2 ? '#EC4899' : '#818CF8', boxShadow: `0 0 10px ${ring === 1 ? '#8B5CF6' : ring === 2 ? '#EC4899' : '#818CF8'}44` }} />
                    </motion.div>
                  ))}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <motion.div className="w-24 h-24 rounded-full bg-gradient-to-br from-purple-500/[0.06] to-pink-500/[0.06] border border-white/[0.04] flex items-center justify-center"
                      animate={{ scale: [1, 1.03, 1] }} transition={{ duration: 4, repeat: Infinity }}>
                      <Brain className="w-9 h-9 text-purple-400/40" />
                    </motion.div>
                  </div>
                  {['Drug A', 'Drug B', 'Proteins', 'Pathways'].map((label, i) => (
                    <motion.div key={label}
                      className="absolute w-14 h-14 rounded-full bg-[#080809] border border-white/[0.04] flex items-center justify-center"
                      style={{ top: `${50 + 35 * Math.sin(i * Math.PI / 2)}%`, left: `${50 + 35 * Math.cos(i * Math.PI / 2)}%`, transform: 'translate(-50%, -50%)' }}
                      animate={{ boxShadow: ['0 0 0 rgba(139,92,246,0)', '0 0 20px rgba(139,92,246,0.1)', '0 0 0 rgba(139,92,246,0)'] }}
                      transition={{ duration: 3, delay: i * 0.7, repeat: Infinity }}>
                      <span className="font-mono text-[7px] text-white/30 uppercase tracking-wider text-center">{label}</span>
                    </motion.div>
                  ))}
                </div>
              </Reveal>
            </ParallaxSection>
          </div>
        </div>
      </section>

      <GradientDivider />

      {/* ═══════════ CTA ═══════════ */}
      <section className="py-20 sm:py-36 relative">
        <div className="absolute inset-0" style={{ background: 'radial-gradient(ellipse at 50% 80%, rgba(139,92,246,0.06) 0%, transparent 60%)' }} />
        <NoiseOverlay opacity={0.012} />
        <div className="max-w-4xl mx-auto px-4 sm:px-6 text-center relative">
          <Reveal>
            <h2 className="mb-4 sm:mb-6 px-2">
              <span className="font-heading text-3xl sm:text-5xl md:text-6xl font-bold">Ready to </span>
              <span className="font-cursive italic text-3xl sm:text-5xl md:text-6xl bg-gradient-to-r from-purple-400/90 to-pink-400/80 bg-clip-text text-transparent">Transform</span>
              <br className="sm:hidden" />
              <span className="font-heading text-3xl sm:text-5xl md:text-6xl font-bold"> Drug Safety?</span>
            </h2>
            <p className="font-body text-sm sm:text-base text-white/30 mb-8 sm:mb-12 px-4">
              Start exploring our AI-powered platform for clinical decision support
            </p>
            <motion.button whileHover={{ scale: 1.03, y: -2 }} whileTap={{ scale: 0.97 }}
              onClick={() => navigate('/dashboard')}
              className="group relative px-10 sm:px-14 py-4 sm:py-5 text-white font-heading font-semibold tracking-widest overflow-hidden rounded-lg"
              style={{ background: 'linear-gradient(135deg, #6D28D9 0%, #BE185D 50%, #4F46E5 100%)', boxShadow: '0 0 25px rgba(109,40,217,0.15), 0 8px 32px rgba(0,0,0,0.5)' }}>
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/15 to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-b from-white/[0.08] to-transparent" style={{ borderRadius: 'inherit' }} />
              <span className="relative flex items-center justify-center gap-3 text-xs sm:text-sm">
                Launch Dashboard <ExternalLink className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </span>
            </motion.button>
          </Reveal>
        </div>
      </section>

      {/* ═══════════ FOOTER ═══════════ */}
      <footer className="py-12 sm:py-16 border-t border-white/[0.03]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <Reveal>
            <div className="mb-10 pb-10 border-b border-white/[0.03]">
              <span className="font-mono text-[9px] text-purple-400/30 uppercase tracking-[0.3em] mb-4 block">Research Team</span>
              <p className="font-cursive italic text-base sm:text-lg text-white/40">
                Taha Ghori, Ahraz Kibria, Raidah Nazimuddin, Kartike Chaudhari
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="mb-10 pb-10 border-b border-white/[0.03]">
              <span className="font-mono text-[9px] text-pink-400/30 uppercase tracking-[0.3em] mb-3 block">Contact</span>
              <a href="mailto:1kibriaahr@gmail.com" className="inline-flex items-center gap-2 font-body text-sm text-purple-400/50 hover:text-purple-400/80 transition-colors group">
                <span>1kibriaahr@gmail.com</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
              </a>
            </div>
          </Reveal>
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 sm:gap-6">
            <div className="flex items-center gap-2">
              <span className="text-xs text-white/25 tracking-wide">
                <span className="font-heading font-medium">Project</span>{' '}
                <span className="font-cursive italic">Aegis</span>
              </span>
              <span className="text-white/10 font-body text-[10px]">&copy; 2026</span>
            </div>
            <div className="flex items-center gap-5">
              <a href="https://github.com/ahrazkk/ProjectAegis-Clinical-MLOps-Platform" target="_blank" rel="noopener noreferrer" className="text-white/15 hover:text-purple-400/50 transition-colors"><Github className="w-4 h-4" /></a>
              <a href="mailto:1kibriaahr@gmail.com" className="text-white/15 hover:text-purple-400/50 transition-colors"><Mail className="w-4 h-4" /></a>
              <a href="#" className="text-white/15 hover:text-purple-400/50 transition-colors"><Linkedin className="w-4 h-4" /></a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
