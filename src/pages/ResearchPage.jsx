import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import {
  ArrowLeft,
  Brain,
  Database,
  Server,
  Cloud,
  GitBranch,
  Activity,
  Zap,
  CheckCircle,
  ArrowRight,
  ArrowDown,
  BookOpen,
  Layers,
  Cpu,
  Globe,
  Search,
  FileText,
  MessageSquare,
  Sparkles,
  CircleDot,
  Camera,
  FlaskConical,
  Microscope,
  Atom,
  Eye,
  Shield,
  Target,
  ScanLine
} from 'lucide-react';

// ── Animated counter ───────────────────────────────────────────────
const AnimatedNumber = ({ value, suffix = '', prefix = '', decimals = 0, duration = 2 }) => {
  const [display, setDisplay] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  useEffect(() => {
    if (!isInView) return;
    const num = parseFloat(value);
    if (isNaN(num)) { setDisplay(value); return; }
    let start = 0;
    const step = num / (duration * 60);
    const id = setInterval(() => {
      start += step;
      if (start >= num) { setDisplay(num); clearInterval(id); }
      else setDisplay(start);
    }, 1000 / 60);
    return () => clearInterval(id);
  }, [isInView, value, duration]);

  return (
    <span ref={ref}>
      {prefix}{typeof display === 'number' ? display.toFixed(decimals) : display}{suffix}
    </span>
  );
};

// ── Pulsing status dot ─────────────────────────────────────────────
const StatusDot = ({ color = 'green', label }) => (
  <span className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest">
    <span className="relative flex h-2 w-2">
      <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${color === 'green' ? 'bg-green-400' : color === 'cyan' ? 'bg-cyan-400' : 'bg-purple-400'}`} />
      <span className={`relative inline-flex rounded-full h-2 w-2 ${color === 'green' ? 'bg-green-400' : color === 'cyan' ? 'bg-cyan-400' : 'bg-purple-400'}`} />
    </span>
    <span className={color === 'green' ? 'text-green-400' : color === 'cyan' ? 'text-cyan-400' : 'text-purple-400'}>{label}</span>
  </span>
);

// ── Box container ──────────────────────────────────────────────────
const Box = ({ children, className = '', glow }) => (
  <div
    className={`bg-black/80 backdrop-blur-sm border border-fui-gray-500/30 p-6 md:p-8 ${className}`}
    style={glow ? { boxShadow: `0 0 30px ${glow}` } : undefined}
  >
    {children}
  </div>
);

// ── SVG progress ring ──────────────────────────────────────────────
const ProgressRing = ({ value, max = 100, size = 100, stroke = 6, color = '#00d4ff', label, sublabel }) => {
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-40px' });

  return (
    <div ref={ref} className="flex flex-col items-center gap-2">
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={stroke} />
        <motion.circle
          cx={size / 2} cy={size / 2} r={radius} fill="none"
          stroke={color} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={isInView ? { strokeDashoffset: circumference - (value / max) * circumference } : {}}
          transition={{ duration: 1.8, ease: 'easeOut' }}
        />
      </svg>
      <div className="text-center -mt-[68px] mb-6">
        <div className="text-lg font-light" style={{ color }}>
          {isInView ? <AnimatedNumber value={value} suffix="%" decimals={1} /> : '0%'}
        </div>
      </div>
      <div className="text-[10px] text-fui-gray-400 uppercase tracking-widest text-center">{label}</div>
      {sublabel && <div className="text-[9px] text-fui-gray-500 text-center -mt-1">{sublabel}</div>}
    </div>
  );
};

// ── Animated bar metric ────────────────────────────────────────────
const MetricBar = ({ label, value, max = 100, color = '#00d4ff', suffix = '%', index = 0 }) => (
  <motion.div
    initial={{ opacity: 0, x: -20 }}
    whileInView={{ opacity: 1, x: 0 }}
    transition={{ delay: index * 0.08 }}
    viewport={{ once: true }}
    className="space-y-1.5"
  >
    <div className="flex justify-between text-xs">
      <span className="text-fui-gray-400 uppercase tracking-widest">{label}</span>
      <span className="text-fui-gray-100">{value}{suffix}</span>
    </div>
    <div className="h-2.5 bg-fui-gray-500/20 overflow-hidden relative">
      <motion.div
        initial={{ width: 0 }}
        whileInView={{ width: `${(value / max) * 100}%` }}
        transition={{ duration: 1.2, delay: index * 0.08, ease: 'easeOut' }}
        viewport={{ once: true }}
        className="h-full relative"
        style={{ backgroundColor: color }}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-white/20" />
      </motion.div>
    </div>
  </motion.div>
);

// ── Timeline item ──────────────────────────────────────────────────
const TimelineItem = ({ title, description, status, icon: Icon, index }) => {
  const colorMap = {
    done: { dot: 'border-green-500/50', text: 'text-green-400' },
    active: { dot: 'border-cyan-500/50', text: 'text-cyan-400' },
    upcoming: { dot: 'border-fui-gray-500/30', text: 'text-fui-gray-500' },
  };
  const c = colorMap[status] || colorMap.upcoming;
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }} viewport={{ once: true }}
      className="flex gap-4"
    >
      <div className="flex flex-col items-center">
        <div className={`w-10 h-10 border ${c.dot} flex items-center justify-center bg-black`}>
          {status === 'done'
            ? <CheckCircle className="w-4 h-4 text-green-400" />
            : <Icon className={`w-4 h-4 ${c.text}`} />}
        </div>
        <div className="w-px flex-1 bg-fui-gray-500/20 mt-1" />
      </div>
      <div className="pb-8 flex-1">
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <span className="text-sm text-fui-gray-100">{title}</span>
          {status === 'active' && <StatusDot color="cyan" label="In Progress" />}
          {status === 'done' && <span className="text-[9px] text-green-400 uppercase tracking-widest">Complete</span>}
          {status === 'upcoming' && <span className="text-[9px] text-fui-gray-500 uppercase tracking-widest">Planned</span>}
        </div>
        <p className="text-xs text-fui-gray-400 leading-relaxed">{description}</p>
      </div>
    </motion.div>
  );
};

// ════════════════════════════════════════════════════════════════════
//  MAIN RESEARCH PAGE
// ════════════════════════════════════════════════════════════════════
export default function ResearchPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Microscope },
    { id: 'evolution', label: 'Model Evolution', icon: Activity },
    { id: 'gnn', label: 'GNN Research', icon: Atom },
    { id: 'scanner', label: 'Pill Scanner', icon: Camera },
    { id: 'pipeline', label: 'System Pipeline', icon: GitBranch },
    { id: 'deployment', label: 'Infrastructure', icon: Cloud },
  ];

  return (
    <div className="min-h-screen bg-black text-fui-gray-100 font-mono relative">
      {/* Background grid */}
      <div className="fixed inset-0 opacity-[0.03] pointer-events-none" style={{
        backgroundImage: 'linear-gradient(rgba(0,212,255,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.4) 1px, transparent 1px)',
        backgroundSize: '60px 60px'
      }} />

      {/* ── Top Nav ── */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/95 backdrop-blur-sm border-b border-fui-gray-500/30">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex items-center justify-between">
          <button onClick={() => navigate('/')} className="flex items-center gap-2 text-fui-gray-400 hover:text-fui-gray-100 transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span className="text-xs uppercase tracking-widest hidden sm:inline">Back to Home</span>
          </button>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 border border-fui-gray-500 flex items-center justify-center">
              <FlaskConical className="w-4 h-4 text-cyan-400" />
            </div>
            <span className="text-sm tracking-widest uppercase">
              Project<span className="text-cyan-400">Aegis</span>
              <span className="text-fui-gray-500 ml-2 text-[10px]">Research Lab</span>
            </span>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="pt-24 md:pt-32 pb-12 border-b border-fui-gray-500/20">
        <div className="max-w-7xl mx-auto px-4 md:px-6">
          <Box>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              <span className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-4 block">// Active Research &amp; Development</span>
              <h1 className="text-2xl md:text-4xl lg:text-5xl font-light mb-4 tracking-wide">
                What We&apos;re <span className="text-cyan-400">Building</span>
              </h1>
              <p className="text-sm text-fui-gray-400 max-w-2xl leading-relaxed mb-8">
                From molecular graph neural networks to real-time pill camera detection &mdash; here&apos;s what&apos;s
                actively being researched, trained, and deployed in Project Aegis.
              </p>

              {/* Quick stat banner */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { value: '3', label: 'AI Models', sub: 'PubMedBERT + GNN + Pill CNN', color: '#00d4ff' },
                  { value: '2,080', label: 'Drugs in Graph', sub: 'Neo4j Aura Knowledge Base', color: '#00ff88' },
                  { value: '56', label: 'Pill Classes', sub: 'Camera Detection Model', color: '#a855f7' },
                  { value: '5', label: 'API Endpoints', sub: 'Scanner Pipeline', color: '#f59e0b' },
                ].map((s, i) => (
                  <motion.div
                    key={s.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 + i * 0.1 }}
                    className="border border-fui-gray-500/20 bg-black/40 p-4 text-center"
                  >
                    <div className="text-2xl md:text-3xl font-light" style={{ color: s.color }}>{s.value}</div>
                    <div className="text-[10px] text-fui-gray-400 uppercase tracking-widest mt-1">{s.label}</div>
                    <div className="text-[8px] text-fui-gray-500 mt-0.5">{s.sub}</div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </Box>
        </div>
      </section>

      {/* ── Tab Nav ── */}
      <section className="border-b border-fui-gray-500/20 sticky top-[73px] bg-black/95 backdrop-blur-sm z-40">
        <div className="max-w-7xl mx-auto px-4 md:px-6">
          <div className="flex gap-0 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 md:px-6 py-4 text-[10px] md:text-xs uppercase tracking-widest border-b-2 transition-colors whitespace-nowrap flex items-center gap-2 ${
                  activeTab === tab.id
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-fui-gray-500 hover:text-fui-gray-300'
                }`}
              >
                <tab.icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Content ── */}
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-12 md:py-16 relative z-10">
        <AnimatePresence mode="wait">

          {/* ════════════════ OVERVIEW TAB ════════════════ */}
          {activeTab === 'overview' && (
            <motion.div key="overview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-12">

              {/* Research Timeline */}
              <Box>
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-8">// Development Timeline &mdash; March 2026</div>
                <div className="grid lg:grid-cols-2 gap-8">
                  <div>
                    {[
                      { title: 'PubMedBERT NLP Model', description: 'Fine-tuned on DDI Corpus 2013 (27,792 pairs). Achieves 92.7% AUC for relation extraction from biomedical text.', status: 'done', icon: Brain },
                      { title: 'Neo4j Knowledge Graph', description: '2,080 drugs with 1,693 verified interactions. SMILES coverage for 1,350 drugs. Therapeutic classifications via RxNorm.', status: 'done', icon: Database },
                      { title: 'Graph Neural Network (GNN)', description: 'Macroscopic GraphSAGE topology leveraging FDA clinical pathways. 1,343 latent features per node. Achieves 98.67% AUC and 98.65% Average Precision.', status: 'done', icon: Atom },
                      { title: 'Pill Camera Detection', description: 'MobileNetV2 transfer learning — 56 drug classes. TF.js inference in-browser + backend CV pipeline (shape, color, imprint).', status: 'done', icon: Camera },
                      { title: 'Cloud Deployment', description: 'Full stack on GCP Cloud Run. Frontend + Backend + Scanner API. Custom domain aegishealth.dev with managed SSL.', status: 'active', icon: Cloud },
                      { title: 'RAG Pipeline Enhancement', description: 'Real-time PubMed literature retrieval to augment predictions with latest research. Auto-enrichment of Neo4j graph.', status: 'upcoming', icon: Sparkles },
                    ].map((item, i) => <TimelineItem key={item.title} {...item} index={i} />)}
                  </div>

                  {/* Model Performance Rings */}
                  <div className="space-y-8">
                    <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Model Performance Metrics</div>
                    <div className="grid grid-cols-2 gap-6">
                      <ProgressRing value={92.7} color="#00d4ff" label="PubMedBERT" sublabel="AUC-ROC" />
                      <ProgressRing value={98.67} color="#00ff88" label="GNN" sublabel="Test AUC" />
                      <ProgressRing value={89.2} color="#a855f7" label="Precision" sublabel="NLP Model" />
                      <ProgressRing value={87.5} color="#f59e0b" label="Recall" sublabel="NLP Model" />
                    </div>

                    {/* Ensemble bar */}
                    <Box className="!p-4 mt-4">
                      <div className="text-[10px] text-fui-gray-500 uppercase tracking-widest mb-3">Ensemble (NLP + GNN + KG)</div>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-5 bg-fui-gray-500/10 overflow-hidden relative rounded-sm">
                          <motion.div initial={{ width: 0 }} whileInView={{ width: '70%' }} transition={{ duration: 1.2 }} viewport={{ once: true }}
                            className="h-full bg-gradient-to-r from-cyan-500/80 via-purple-500/80 to-green-500/80 relative">
                            <div className="absolute inset-0 bg-[linear-gradient(90deg,transparent_0%,rgba(255,255,255,0.15)_50%,transparent_100%)] animate-pulse" />
                          </motion.div>
                        </div>
                        <span className="text-xs text-fui-gray-100 w-12 text-right">70%</span>
                      </div>
                      <div className="text-[9px] text-fui-gray-500 mt-2">NLP 45% &middot; GNN 15% &middot; Knowledge Graph 40% &mdash; weights being optimized</div>
                    </Box>
                  </div>
                </div>
              </Box>

              {/* Problem / Solution */}
              <div className="grid lg:grid-cols-2 gap-8">
                <Box>
                  <h2 className="text-xl font-light mb-6 tracking-wide">The Problem</h2>
                  <p className="text-sm text-fui-gray-400 leading-relaxed mb-6">
                    Drug-drug interactions cause real harm daily. FDA FAERS records
                    thousands of adverse events that could have been prevented.
                  </p>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { v: '195K+', l: 'Hospitalizations/yr' },
                      { v: '$3.5B', l: 'Annual Cost' },
                      { v: '4-5%', l: 'Preventable ADEs' },
                    ].map((s) => (
                      <div key={s.l} className="border border-red-500/20 bg-red-500/5 p-3 text-center">
                        <div className="text-lg text-red-400 font-light">{s.v}</div>
                        <div className="text-[9px] text-fui-gray-500 uppercase tracking-widest mt-1">{s.l}</div>
                      </div>
                    ))}
                  </div>
                </Box>
                <Box>
                  <h2 className="text-xl font-light mb-6 tracking-wide">Our Approach</h2>
                  <p className="text-sm text-fui-gray-400 leading-relaxed mb-6">
                    Multi-modal AI combining NLP, graph learning, and computer vision
                    for comprehensive drug safety analysis.
                  </p>
                  <ul className="space-y-3 text-sm text-fui-gray-400">
                    {[
                      'PubMedBERT for clinical text understanding',
                      'GNN for molecular structure analysis',
                      'Camera-based pill identification (56 classes)',
                      'Knowledge graph with 2,080+ drug profiles',
                    ].map((t) => (
                      <li key={t} className="flex items-start gap-3">
                        <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                        <span>{t}</span>
                      </li>
                    ))}
                  </ul>
                </Box>
              </div>
            </motion.div>
          )}

          {/* ════════════════ GNN RESEARCH TAB ════════════════ */}
          {/* START MODEL EVOLUTION TAB */}
            tf-8 -*-
import io

file_path = "src/pages/ResearchPage.jsx"
with io.open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# We look for where activeTab === 'evolution' starts.
idx_start = text.find("activeTab === 'evolution' && (")
if idx_start == -1:
    print("Cannot find evolution start")
    exit(1)

# We look for where the next tab starts (gnn)
idx_end = text.find("activeTab === 'gnn' && (")
if idx_end == -1:
    print("Cannot find gnn start")
    exit(1)

# Retreat idx_start to the comment block above it
pre_comment = text.rfind("{/*", 0, idx_start)
if pre_comment != -1:
    idx_start = pre_comment

# Retreat idx_end to the comment block above it
pre_end_comment = text.rfind("{/*", 0, idx_end)
if pre_end_comment != -1:
    idx_end = pre_end_comment

pre_code = text[:idx_start]
post_code = text[idx_end:]

new_evolution_ui = """{/* START MODEL EVOLUTION TAB */}
            {activeTab === 'evolution' && (
              <motion.div key="evolution" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -30 }} className="space-y-16">

                {/* Cyberpunk Header */}
                <div className="relative overflow-hidden group rounded-xl border border-cyan-500/30 bg-black/60 backdrop-blur-md p-10 md:p-14 shadow-[0_0_80px_-15px_rgba(0,212,255,0.2)]">
                  <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-gradient-to-bl from-cyan-500/10 via-purple-500/5 to-transparent rounded-full blur-[100px] pointer-events-none -translate-y-1/2 translate-x-1/3" />
                  
                  <div className="relative z-10 flex flex-col md:flex-row gap-8 items-center justify-between">
                    <div className="max-w-2xl">
                      <div className="inline-flex items-center gap-2 border border-cyan-500/50 px-4 py-1.5 bg-cyan-500/10 text-cyan-400 text-[11px] tracking-[0.2em] mb-8 rounded-full">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                        AEGIS CORE ARCHITECTURE
                      </div>
                      <h2 className="text-4xl md:text-5xl font-extralight text-white mb-6 tracking-tight">
                        Algorithmic <span className="font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">Evolution</span>
                      </h2>
                      <p className="text-fui-gray-400 text-sm md:text-base leading-relaxed">
                        Project Aegis has undergone three distinct evolutionary phases. We started with heuristic NLP baselines (Phase 1), advanced into experimental microscopic graph structures (Phase 2), and arrived at our current state-of-the-art Macroscopic GraphSAGE topology—achieving <span className="text-cyan-400">98.67% testing accuracy</span> across massive multi-modal Datasets.
                      </p>
                    </div>
                    <div className="relative w-48 h-48 shrink-0">
                      <Activity className="absolute inset-0 w-full h-full text-cyan-500/20 animate-[spin_10s_linear_infinite]" />
                      <Atom className="absolute inset-8 w-32 h-32 text-blue-400/40 animate-[spin_15s_linear_infinite_reverse]" />
                      <Microscope className="absolute inset-16 w-16 h-16 text-white drop-shadow-[0_0_15px_rgba(255,255,255,0.8)]" />
                    </div>
                  </div>
                </div>

                <div className="grid md:grid-cols-1 gap-12 relative before:absolute before:inset-0 before:ml-[23px] before:w-0.5 before:bg-gradient-to-b before:from-purple-500/50 before:via-blue-500/50 before:to-cyan-400/50">
                  
                  {/* Gen 1: NLP Baseline */}
                  <motion.div initial={{ x: -50, opacity: 0 }} whileInView={{ x: 0, opacity: 1 }} viewport={{ once: true }} className="relative flex items-center group border border-purple-500/20 bg-black/40 p-8 rounded-2xl hover:border-purple-500/50 transition-colors ml-12 lg:ml-16 mr-0 w-full lg:w-[calc(100%-4rem)]">
                    <div className="absolute top-1/2 -translate-y-1/2 -left-14 md:-left-16 w-6 h-6 rounded-full bg-black border-4 border-purple-500 shadow-[0_0_15px_rgba(168,85,247,0.6)] z-10" />
                    <div className="w-full">
                      <div className="text-[10px] text-purple-400 tracking-[0.2em] mb-2 font-bold">V1.0 • PUBMEDBERT NLP</div>
                      <h3 className="text-2xl text-white mb-4">Text-Based Inference</h3>
                      <p className="text-fui-gray-400 text-xs leading-relaxed mb-6">
                        Initial text-based inference system utilizing a HuggingFace transformer to blindly parse medical literature. Failed to capture actual chemical synergy. Suffered from massive hallucination loops, API rate-limiting, and severe computational bottlenecks when analyzing polypharmacy scenarios.
                      </p>
                      <div className="flex gap-6">
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Accuracy</div>
                          <div className="text-xl text-purple-400 font-bold">~52.0%</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Latency</div>
                          <div className="text-xl text-fui-gray-300 font-bold">8.4s</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Status</div>
                          <div className="text-xl text-red-500 font-bold line-through">DEPRECATED</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Gen 2: Micro GNN */}
                  <motion.div initial={{ x: 50, opacity: 0 }} whileInView={{ x: 0, opacity: 1 }} viewport={{ once: true }} className="relative flex items-center group border border-blue-500/20 bg-black/40 p-8 rounded-2xl hover:border-blue-500/50 transition-colors ml-12 lg:ml-16 mr-0 w-full lg:w-[calc(100%-4rem)]">
                    <div className="absolute top-1/2 -translate-y-1/2 -left-14 md:-left-16 w-6 h-6 rounded-full bg-black border-4 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.6)] z-10" />
                    <div className="w-full">
                      <div className="text-[10px] text-blue-400 tracking-[0.2em] mb-2 font-bold">V2.0 • MICROSCOPIC GIN</div>
                      <h3 className="text-2xl text-white mb-4">Chemical Graph Isomorphism</h3>
                      <p className="text-fui-gray-400 text-xs leading-relaxed mb-6">
                        Pivoted to parsing raw molecular structures (SMILES) directly into PyTorch graph tensors. Used RDKit to map atoms as nodes and chemical bonds as edges. Reached hardware limits instantly.
                      </p>
                      <ul className="space-y-2 mb-6">
                        <li className="flex items-center gap-2 text-xs text-fui-gray-300"><span className="text-red-400 font-bold">X</span> GPU Memory Overflows (OOM) at Epoch 30</li>
                        <li className="flex items-center gap-2 text-xs text-fui-gray-300"><span className="text-red-400 font-bold">X</span> Insufficient representation of clinical pathways</li>
                      </ul>
                      <div className="flex gap-6">
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Accuracy</div>
                          <div className="text-xl text-blue-400 font-bold">65.7%</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-fui-gray-500 mb-1 uppercase">Status</div>
                          <div className="text-xl text-fui-gray-500 font-bold">ARCHIVED</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Gen 3: Macro GNN */}
                  <motion.div initial={{ y: 50, opacity: 0 }} whileInView={{ y: 0, opacity: 1 }} viewport={{ once: true }} className="relative flex items-center group border border-cyan-400/50 bg-cyan-950/10 p-8 md:p-12 rounded-2xl shadow-[0_0_40px_-10px_rgba(34,211,238,0.15)] ml-12 lg:ml-16 border-l-4 mr-0 w-full lg:w-[calc(100%-4rem)]">
                    <div className="absolute top-1/2 -translate-y-1/2 -left-[60px] md:-left-[68px] w-8 h-8 rounded-full bg-cyan-400 border-4 border-black shadow-[0_0_20px_rgba(34,211,238,1)] z-10 flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full animate-ping" />
                    </div>
                    <div className="w-full">
                      <div className="flex justify-between items-start mb-4">
                        <div className="text-[10px] text-cyan-400 tracking-[0.2em] font-bold">V3.0 • MACROSCOPIC GRAPHSAGE</div>
                        <div className="px-3 py-1 bg-green-500/20 text-green-400 text-[10px] font-bold rounded shadow-[0_0_10px_rgba(34,197,94,0.2)]">CURRENT PROD</div>
                      </div>
                      <h3 className="text-3xl text-white mb-4 font-light">Global Interactome Matrix</h3>
                      <p className="text-fui-gray-300 text-sm leading-relaxed mb-8">
                        Completely abandoned single-molecule scanning. Transformed the entire FDA database into a unified, massive graph network. Drugs are nodes. Clinical pathways, adverse effects, and known interactions are edges. Features are embedded via high-dimensional PCA vectors and passed dynamically through GraphSAGE message layers.
                      </p>
                      
                      <div className="grid grid-cols-2 gap-4 mb-8">
                        <div className="bg-black/50 border border-fui-gray-800 p-4 rounded-lg">
                          <Globe className="w-5 h-5 text-cyan-400 mb-2" />
                          <div className="text-[10px] text-fui-gray-500 uppercase">Density Scaling</div>
                          <div className="text-sm text-fui-gray-200 mt-1">Nodes increased from 1.08 to 79.25 edges</div>
                        </div>
                        <div className="bg-black/50 border border-fui-gray-800 p-4 rounded-lg">
                          <Layers className="w-5 h-5 text-cyan-400 mb-2" />
                          <div className="text-[10px] text-fui-gray-500 uppercase">Vector Breadth</div>
                          <div className="text-sm text-fui-gray-200 mt-1">1,343 latent features per chemical node</div>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-8 items-center bg-cyan-950/30 p-6 rounded-xl border border-cyan-500/20">
                        <div>
                          <div className="text-[10px] text-cyan-500 mb-1 uppercase tracking-widest">Test AUC</div>
                          <div className="text-4xl text-white font-bold drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]">98.67%</div>
                        </div>
                        <div className="w-px h-12 bg-cyan-500/30 hidden md:block" />
                        <div>
                          <div className="text-[10px] text-cyan-500 mb-1 uppercase tracking-widest">F1 Score</div>
                          <div className="text-2xl text-cyan-100 font-bold">94.34%</div>
                        </div>
                        <div className="w-px h-12 bg-cyan-500/30 hidden md:block" />
                        <div>
                          <div className="text-[10px] text-cyan-500 mb-1 uppercase tracking-widest">Endpoint Latency</div>
                          <div className="text-2xl text-cyan-100 font-bold">&lt; 20ms</div>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                </div>
              </motion.div>
            )}
            
            {/* END MODEL EVOLUTION TAB */}
            {activeTab === 'gnn' && (
            <motion.div key="gnn" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-12">

              {/* GNN Header */}
              <Box glow="rgba(0,212,255,0.06)">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 border border-cyan-500/50 flex items-center justify-center bg-cyan-500/10">
                    <Atom className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-light tracking-wide">Graph Neural Network &mdash; DDI Prediction</h2>
                    <div className="flex items-center gap-3 mt-1">
                      <StatusDot color="green" label="Trained" />
                      <span className="text-[10px] text-fui-gray-500">Edge-Conditioned GIN &middot; 241K Parameters</span>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-fui-gray-400 leading-relaxed max-w-3xl">
                  Predicts drug-drug interactions directly from molecular graphs (SMILES &rarr; atom/bond graphs).
                  Each drug is represented as a graph where atoms are nodes and chemical bonds are edges,
                  enabling the model to learn structural patterns associated with interactions.
                </p>
              </Box>

              {/* Architecture */}
              <Box>
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// GNN Architecture</div>
                <div className="hidden md:flex items-center justify-between gap-2">
                  {[
                    { label: 'SMILES Input', sub: 'Drug A + Drug B', highlight: false },
                    { label: 'Graph Construction', sub: 'Drugs \u2192 Nodes\nInteractions \u2192 Edges', highlight: false },
                    { label: 'GraphSAGE Layers', sub: 'Neighbor Aggregation\n+ Clinical Vectors', highlight: true, hcolor: 'cyan' },
                    { label: 'Graph Pooling', sub: 'Global Mean + Max', highlight: false },
                    { label: 'Classifier', sub: 'MLP + Sigmoid', highlight: false },
                    { label: 'Platt Scaling', sub: '\u03c3(0.23\u00b7z + 0.07)', highlight: true, hcolor: 'green' },
                  ].map((step, i, arr) => (
                    <React.Fragment key={step.label}>
                      <motion.div
                        initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }} viewport={{ once: true }}
                        className={`flex-1 min-w-[100px] border p-4 text-center ${
                          step.highlight
                            ? step.hcolor === 'cyan' ? 'border-cyan-500/50 bg-cyan-500/5' : 'border-green-500/50 bg-green-500/5'
                            : 'border-fui-gray-500/30 bg-black/50'
                        }`}
                      >
                        <div className="text-[10px] text-cyan-400 uppercase tracking-widest mb-2">{step.label}</div>
                        <div className="text-[9px] text-fui-gray-400 whitespace-pre-line">{step.sub}</div>
                      </motion.div>
                      {i < arr.length - 1 && <ArrowRight className="w-4 h-4 text-fui-gray-500 flex-shrink-0" />}
                    </React.Fragment>
                  ))}
                </div>
                <div className="md:hidden space-y-3">
                  {[
                    'Global Interactome \u2192 PCA Embeddings \u2192 GraphSAGE Layers \u00d73',
                    'Relational Aggregation \u2192 Feed Forward Net \u2192 Probability Output',
                  ].map((line) => (
                    <div key={line} className="border border-cyan-500/30 bg-cyan-500/5 p-4 text-center">
                      <div className="text-[10px] text-cyan-400">{line}</div>
                    </div>
                  ))}
                </div>
              </Box>

              {/* Training metrics */}
              <div className="grid lg:grid-cols-2 gap-8">
                <Box>
                  <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Training Results</div>
                  <div className="space-y-4">
                    <MetricBar label="Test AUC" value={98.67} color="#00ff88" index={0} />
                    <MetricBar label="Testing Accuracy" value={97.10} color="#00d4ff" index={1} />
                    <MetricBar label="Average Precision" value={98.65} color="#a855f7" index={2} />
                    <MetricBar label="F1 Score" value={94.34} max={100} color="#00ff88" suffix="%" index={3} />
                    <MetricBar label="Calibrated (Non-DDI)" value={44.6} max={100} color="#f59e0b" suffix="% avg" index={4} />
                  </div>
                  <div className="mt-6 text-[9px] text-fui-gray-500 border-t border-fui-gray-500/20 pt-4 space-y-1">
                    <div>Platt scaling: <span className="text-cyan-400">a=0.2347, b=0.0744</span></div>
                    <div>Warfarin + Aspirin: raw 0.985 &rarr; calibrated <span className="text-green-400">0.992</span></div>
                    <div>Amoxicillin + Metformin: raw 0.223 &rarr; calibrated <span className="text-green-400">0.446</span></div>
                  </div>
                </Box>

                <Box>
                  <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Training Configuration</div>
                  <div className="space-y-3">
                    {[
                      ['Architecture', 'Edge-Conditioned GIN'],
                      ['Parameters', '241,089'],
                      ['Hidden Dim', '128'],
                      ['GNN Layers', '3 + Jumping Knowledge'],
                      ['Max Atoms', '50 (covers 93% of drugs)'],
                      ['Training Samples', '2,002 (filtered)'],
                      ['Validation', '368 samples'],
                      ['Data Source', 'Neo4j Aura (688 DDI pairs)'],
                      ['Negatives Ratio', '1:1 balanced'],
                      ['Dropout', '0.2'],
                      ['Optimizer', 'AdamW (lr=5e-4)'],
                      ['Training Time', '5 min 56 sec (GPU)'],
                      ['GPU', 'NVIDIA RTX 3060 Laptop'],
                    ].map(([k, v], i) => (
                      <motion.div key={k} initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
                        transition={{ delay: i * 0.03 }} viewport={{ once: true }}
                        className="flex justify-between text-xs border-b border-fui-gray-500/10 pb-2"
                      >
                        <span className="text-fui-gray-400">{k}</span>
                        <span className="text-fui-gray-100 text-right">{v}</span>
                      </motion.div>
                    ))}
                  </div>
                </Box>
              </div>

              {/* Optimizations */}
              <Box>
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Key Optimizations</div>
                <div className="grid md:grid-cols-3 gap-6">
                  {[
                    {
                      icon: Zap, color: '#00ff88',
                      title: '43\u00d7 Featurization Speedup',
                      desc: 'Removed unnecessary 3D coordinate embedding (AllChem.EmbedMolecule). The GNN only needs 2D topology \u2014 cutting processing from 0.34s to 0.008s per molecule.',
                    },
                    {
                      icon: Target, color: '#00d4ff',
                      title: 'Platt Scaling Calibration',
                      desc: 'Raw model outputs were all >0.9 for every pair. Learned \u03c3(0.23\u00b7logit + 0.07) on validation set so probabilities actually distinguish DDI from non-DDI.',
                    },
                    {
                      icon: Shield, color: '#a855f7',
                      title: 'Metal Compound Filtering',
                      desc: 'Removed 258 metallic/ionic compounds (Pt, Au, Zn, Fe, Ca) that caused RDKit to hang during featurization. Training set: 2,340 \u2192 2,002 clean samples.',
                    },
                  ].map((opt, i) => (
                    <motion.div key={opt.title}
                      initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.15 }} viewport={{ once: true }}
                      className="border border-fui-gray-500/20 bg-black/40 p-5"
                    >
                      <opt.icon className="w-6 h-6 mb-3" style={{ color: opt.color }} />
                      <h3 className="text-sm text-fui-gray-100 mb-2">{opt.title}</h3>
                      <p className="text-xs text-fui-gray-400 leading-relaxed">{opt.desc}</p>
                    </motion.div>
                  ))}
                </div>
              </Box>
            </motion.div>
          )}

          {/* ════════════════ PILL SCANNER TAB ════════════════ */}
          {activeTab === 'scanner' && (
            <motion.div key="scanner" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-12">

              <Box glow="rgba(168,85,247,0.06)">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 border border-purple-500/50 flex items-center justify-center bg-purple-500/10">
                    <Camera className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-light tracking-wide">Camera-Based Drug Identification</h2>
                    <div className="flex items-center gap-3 mt-1">
                      <StatusDot color="green" label="Deployed" />
                      <span className="text-[10px] text-fui-gray-500">MobileNetV2 &middot; TensorFlow.js &middot; 56 Drug Classes</span>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-fui-gray-400 leading-relaxed max-w-3xl">
                  Point your camera at a pill to identify it instantly. The system combines a MobileNetV2 CNN
                  running in your browser with a backend computer vision pipeline for shape, color, and imprint
                  analysis &mdash; then cross-references with your medication list to check for interactions.
                </p>
              </Box>

              {/* Detection modes */}
              <div className="grid md:grid-cols-4 gap-6">
                {[
                  { icon: ScanLine, color: '#00d4ff', title: 'Barcode Scan', desc: 'UPC/EAN/NDC barcode decoding with QuaggaJS. Instant lookup via OpenFDA API.' },
                  { icon: FileText, color: '#00ff88', title: 'OCR Text', desc: 'Tesseract.js reads drug labels. Fuzzy matching handles camera blur and partial text.' },
                  { icon: Eye, color: '#a855f7', title: 'Pill Vision', desc: 'CNN classifies pill image (56 classes). CV pipeline extracts shape, color, and imprint.' },
                  { icon: Sparkles, color: '#f59e0b', title: 'Auto Mode', desc: 'Tries all three sequentially \u2014 barcode \u2192 OCR \u2192 pill vision \u2014 for best results.' },
                ].map((mode, i) => (
                  <motion.div key={mode.title}
                    initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.1 }} viewport={{ once: true }}
                  >
                    <Box className="h-full">
                      <mode.icon className="w-8 h-8 mb-4" style={{ color: mode.color }} />
                      <h3 className="text-sm text-fui-gray-100 mb-2">{mode.title}</h3>
                      <p className="text-xs text-fui-gray-400 leading-relaxed">{mode.desc}</p>
                    </Box>
                  </motion.div>
                ))}
              </div>

              {/* CV Pipeline */}
              <Box>
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Computer Vision Pipeline</div>
                <div className="hidden md:flex items-center justify-between gap-1">
                  {[
                    { label: 'Camera Frame', sub: 'WebRTC capture' },
                    { label: 'Segmentation', sub: 'Otsu threshold + morphology' },
                    { label: 'Color Analysis', sub: '14 colors, 2-tone detect' },
                    { label: 'Shape Detect', sub: '10 shape classes' },
                    { label: 'Imprint OCR', sub: 'Enhanced contrast crop' },
                    { label: 'CNN Classify', sub: 'MobileNetV2, 56 classes' },
                    { label: 'Drug Match', sub: 'Backend API + DailyMed' },
                  ].map((step, i, arr) => (
                    <React.Fragment key={step.label}>
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95 }} whileInView={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.08 }} viewport={{ once: true }}
                        className="flex-1 min-w-0 border border-purple-500/20 bg-purple-500/5 p-3 text-center"
                      >
                        <div className="text-[9px] text-purple-400 uppercase tracking-wider mb-1">{step.label}</div>
                        <div className="text-[8px] text-fui-gray-500">{step.sub}</div>
                      </motion.div>
                      {i < arr.length - 1 && <ArrowRight className="w-3 h-3 text-fui-gray-500 flex-shrink-0" />}
                    </React.Fragment>
                  ))}
                </div>
                <div className="md:hidden space-y-2">
                  {[
                    'Camera Frame \u2192 Segmentation \u2192 Color Analysis',
                    'Shape Detect \u2192 Imprint OCR \u2192 CNN Classify \u2192 Drug Match',
                  ].map((line) => (
                    <div key={line} className="border border-purple-500/20 bg-purple-500/5 p-3 text-center">
                      <div className="text-[10px] text-purple-400">{line}</div>
                    </div>
                  ))}
                </div>
              </Box>

              {/* Model + API details */}
              <div className="grid lg:grid-cols-2 gap-8">
                <Box>
                  <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// CNN Model Details</div>
                  <div className="space-y-3">
                    {[
                      ['Base Architecture', 'MobileNetV2 (ImageNet)'],
                      ['Training', '2-Phase Transfer Learning'],
                      ['Phase 1', 'Frozen backbone, train head'],
                      ['Phase 2', 'Fine-tune top layers'],
                      ['Output Classes', '56 drug types'],
                      ['Inference', 'TensorFlow.js (in-browser)'],
                      ['Model Size', '~12 MB (3 weight shards)'],
                      ['Fallback', 'Pure CV if model fails to load'],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between text-xs border-b border-fui-gray-500/10 pb-2">
                        <span className="text-fui-gray-400">{k}</span>
                        <span className="text-fui-gray-100 text-right">{v}</span>
                      </div>
                    ))}
                  </div>
                </Box>

                <Box>
                  <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Scanner API Endpoints</div>
                  <div className="space-y-3">
                    {[
                      { method: 'GET', path: '/api/v1/drugs/ndc/<code>/', desc: 'NDC barcode lookup' },
                      { method: 'GET', path: '/api/v1/drugs/pill-search/', desc: 'Color/shape/imprint search' },
                      { method: 'GET', path: '/api/v1/drugs/enhanced-search/', desc: 'Fuzzy OCR text search' },
                      { method: 'POST', path: '/api/v1/scanner/validate-barcode/', desc: 'Barcode validation' },
                      { method: 'POST', path: '/api/v1/scanner/analyze-pill/', desc: 'Pill image analysis' },
                    ].map((ep, i) => (
                      <motion.div key={ep.path}
                        initial={{ opacity: 0, x: -10 }} whileInView={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.08 }} viewport={{ once: true }}
                        className="border border-fui-gray-500/20 bg-black/40 p-3 flex items-center gap-3"
                      >
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 ${ep.method === 'GET' ? 'text-green-400 bg-green-500/10' : 'text-yellow-400 bg-yellow-500/10'}`}>
                          {ep.method}
                        </span>
                        <div className="flex-1 min-w-0">
                          <code className="text-[10px] text-fui-gray-300 block truncate">{ep.path}</code>
                          <span className="text-[9px] text-fui-gray-500">{ep.desc}</span>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                  <div className="mt-4 flex items-center gap-2">
                    <StatusDot color="green" label="All endpoints live" />
                    <span className="text-[9px] text-fui-gray-500">External: OpenFDA &middot; NIH DailyMed</span>
                  </div>
                </Box>
              </div>

              {/* Drug class tags */}
              <Box>
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-4">// Recognized Drug Classes (56 total)</div>
                <div className="flex flex-wrap gap-2">
                  {['Acetaminophen', 'Alprazolam', 'Amlodipine', 'Amoxicillin', 'Aspirin', 'Atorvastatin',
                    'Azithromycin', 'Carvedilol', 'Ciprofloxacin', 'Citalopram', 'Clonazepam',
                    'Diazepam', 'Gabapentin', 'Hydrocodone', 'Ibuprofen', 'Levothyroxine',
                    'Lisinopril', 'Losartan', 'Metformin', 'Metoprolol', 'Morphine',
                    'Omeprazole', 'Oxycodone', 'Prednisone', 'Sertraline', 'Simvastatin',
                    'Tramadol', 'Warfarin', 'Zolpidem'].map((drug) => (
                    <span key={drug} className="text-[9px] text-fui-gray-400 border border-fui-gray-500/20 px-2 py-1 bg-black/40">
                      {drug}
                    </span>
                  ))}
                  <span className="text-[9px] text-fui-gray-500 border border-fui-gray-500/10 px-2 py-1">+27 more</span>
                </div>
              </Box>
            </motion.div>
          )}

          {/* ════════════════ PIPELINE TAB ════════════════ */}
          {activeTab === 'pipeline' && (
            <motion.div key="pipeline" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-12">

              {/* System Flow */}
              <Box>
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Prediction Pipeline &mdash; End to End</div>
                <div className="hidden lg:flex items-start justify-between gap-2">
                  {[
                    { label: 'User Query', sub: '\u201cAspirin + Warfarin\u201d', icon: Search, color: 'cyan' },
                    { label: 'Neo4j Lookup', sub: 'Knowledge Graph', icon: Database, color: 'green' },
                    { label: 'Context Retrieval', sub: 'DDI Corpus + PubMed', icon: FileText, color: 'purple' },
                    { label: 'PubMedBERT', sub: 'NLP Analysis', icon: Brain, color: 'cyan' },
                    { label: 'GNN Inference', sub: 'Molecular Graphs', icon: Atom, color: 'green' },
                    { label: 'Risk Scoring', sub: 'Ensemble Output', icon: Activity, color: 'cyan' },
                  ].map((step, i, arr) => (
                    <React.Fragment key={step.label}>
                      <motion.div
                        initial={{ opacity: 0, y: 15 }} whileInView={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }} viewport={{ once: true }}
                        className={`flex-1 min-w-0 border p-4 text-center ${
                          step.color === 'cyan' ? 'border-cyan-500/40 bg-cyan-500/5' :
                          step.color === 'green' ? 'border-green-500/40 bg-green-500/5' :
                          'border-purple-500/40 bg-purple-500/5'
                        }`}
                      >
                        <step.icon className={`w-6 h-6 mx-auto mb-2 ${
                          step.color === 'cyan' ? 'text-cyan-400' : step.color === 'green' ? 'text-green-400' : 'text-purple-400'
                        }`} />
                        <div className="text-[10px] text-fui-gray-100 uppercase tracking-widest mb-1">{step.label}</div>
                        <div className="text-[9px] text-fui-gray-500">{step.sub}</div>
                      </motion.div>
                      {i < arr.length - 1 && <div className="flex-shrink-0 pt-8"><ArrowRight className="w-4 h-4 text-fui-gray-500" /></div>}
                    </React.Fragment>
                  ))}
                </div>
                <div className="lg:hidden space-y-3">
                  {[
                    { label: 'User Query', icon: Search },
                    { label: 'Neo4j Lookup', icon: Database },
                    { label: 'PubMedBERT + GNN', icon: Brain },
                    { label: 'Risk Score Output', icon: Activity },
                  ].map((step, i, arr) => (
                    <React.Fragment key={step.label}>
                      <div className="border border-cyan-500/30 bg-cyan-500/5 p-4 flex items-center gap-4">
                        <step.icon className="w-5 h-5 text-cyan-400" />
                        <span className="text-xs text-fui-gray-100 uppercase tracking-widest">{step.label}</span>
                      </div>
                      {i < arr.length - 1 && <div className="flex justify-center"><ArrowDown className="w-4 h-4 text-fui-gray-500" /></div>}
                    </React.Fragment>
                  ))}
                </div>
              </Box>

              {/* 3-tier Architecture */}
              <Box className="relative overflow-hidden">
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// System Architecture</div>
                <div className="absolute inset-0 opacity-[0.04] pointer-events-none" style={{
                  backgroundImage: 'linear-gradient(rgba(102,102,102,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(102,102,102,0.5) 1px, transparent 1px)',
                  backgroundSize: '30px 30px'
                }} />
                <div className="space-y-6 relative">
                  {[
                    { label: 'Presentation Layer', color: 'cyan', items: ['React 19 + Vite', 'TensorFlow.js (Pill CNN)', 'Three.js / Framer Motion', 'DrugScanner Component'] },
                    { label: 'Application Layer', color: 'purple', items: ['Django REST Framework', 'PubMedBERT (NLP)', 'GNN Predictor (PyTorch)', 'Scanner API (CV Pipeline)'] },
                    { label: 'Data Layer', color: 'green', items: ['Neo4j Aura (2,080 drugs)', 'SQLite (sessions/logs)', 'PubChem + RxNorm APIs', 'OpenFDA + DailyMed'] },
                  ].map((tier, i) => (
                    <React.Fragment key={tier.label}>
                      <motion.div
                        initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.15 }} viewport={{ once: true }}
                        className={`border p-4 relative ${
                          tier.color === 'cyan' ? 'border-cyan-500/30 bg-cyan-500/5' :
                          tier.color === 'purple' ? 'border-purple-500/30 bg-purple-500/5' :
                          'border-green-500/30 bg-green-500/5'
                        }`}
                      >
                        <div className={`absolute -top-3 left-4 px-2 bg-black text-[10px] uppercase tracking-widest ${
                          tier.color === 'cyan' ? 'text-cyan-400' : tier.color === 'purple' ? 'text-purple-400' : 'text-green-400'
                        }`}>
                          {tier.label}
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-2">
                          {tier.items.map((item) => (
                            <div key={item} className="border border-fui-gray-500/20 p-3 text-center bg-black/50">
                              <div className="text-[10px] text-fui-gray-400">{item}</div>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                      {i < 2 && <div className="flex justify-center"><div className="w-px h-6 bg-fui-gray-500/30" /></div>}
                    </React.Fragment>
                  ))}
                </div>
              </Box>

              {/* Data Sources + Schema */}
              <div className="grid lg:grid-cols-2 gap-8">
                <Box>
                  <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Data Sources</div>
                  <div className="space-y-4">
                    {[
                      { name: 'DDI Corpus 2013', stat: '27,792 text pairs', desc: 'Annotated biomedical literature for NLP' },
                      { name: 'Neo4j Aura', stat: '1,693 interactions', desc: 'Verified knowledge graph (SMILES + classifications)' },
                      { name: 'PubChem API', stat: '1,350 SMILES', desc: 'Molecular structures for GNN featurization' },
                      { name: 'OpenFDA / DailyMed', stat: 'Live API', desc: 'NDC lookup and pill identification fallback' },
                    ].map((source, i) => (
                      <motion.div key={source.name}
                        initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }} viewport={{ once: true }}
                        className="border border-fui-gray-500/20 bg-black/40 p-4"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-fui-gray-100">{source.name}</span>
                          <span className="text-[10px] text-cyan-400">{source.stat}</span>
                        </div>
                        <p className="text-xs text-fui-gray-400">{source.desc}</p>
                      </motion.div>
                    ))}
                  </div>
                </Box>

                <Box>
                  <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Neo4j Schema</div>
                  <div className="font-mono text-xs text-fui-gray-400 space-y-2">
                    <div><span className="text-purple-400">(:Drug)</span> {'{'}</div>
                    <div className="pl-4">name: <span className="text-green-400">String</span>,</div>
                    <div className="pl-4">smiles: <span className="text-green-400">String?</span>,</div>
                    <div className="pl-4">drugbank_id: <span className="text-green-400">String?</span>,</div>
                    <div className="pl-4">therapeutic_class: <span className="text-green-400">String?</span>,</div>
                    <div className="pl-4">category: <span className="text-green-400">String?</span></div>
                    <div>{'}'}</div>
                    <div className="mt-3"><span className="text-purple-400">[:INTERACTS_WITH]</span> {'{'}</div>
                    <div className="pl-4">severity: <span className="text-green-400">String</span>,</div>
                    <div className="pl-4">description: <span className="text-green-400">String?</span></div>
                    <div>{'}'}</div>
                  </div>
                  <div className="mt-6 grid grid-cols-3 gap-3">
                    {[
                      { v: '2,080', l: 'Drug Nodes' },
                      { v: '1,693', l: 'Interactions' },
                      { v: '1,350', l: 'With SMILES' },
                    ].map((s) => (
                      <div key={s.l} className="border border-green-500/20 bg-green-500/5 p-3 text-center">
                        <div className="text-lg text-green-400 font-light">{s.v}</div>
                        <div className="text-[9px] text-fui-gray-500 uppercase tracking-widest">{s.l}</div>
                      </div>
                    ))}
                  </div>
                </Box>
              </div>
            </motion.div>
          )}

          {/* ════════════════ DEPLOYMENT TAB ════════════════ */}
          {activeTab === 'deployment' && (
            <motion.div key="deployment" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-12">

              {/* Cloud overview */}
              <Box glow="rgba(0,255,136,0.04)">
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Google Cloud Platform &mdash; Cloud Run</div>
                <div className="grid md:grid-cols-3 gap-6">
                  {[
                    {
                      icon: Globe, color: '#00d4ff', title: 'Frontend',
                      lines: ['React 19 + Nginx', 'TF.js pill model served', '512 MB / 1 vCPU', 'Scale-to-zero'],
                    },
                    {
                      icon: Server, color: '#a855f7', title: 'Backend',
                      lines: ['Django + Gunicorn', 'PubMedBERT + GNN', '4 GB / 2 vCPU', 'Scanner API included'],
                    },
                    {
                      icon: Database, color: '#00ff88', title: 'Data Services',
                      lines: ['Neo4j Aura (Graph DB)', 'SQLite (session data)', 'PubChem / OpenFDA API', 'RxNorm / DailyMed'],
                    },
                  ].map((svc, i) => (
                    <motion.div key={svc.title}
                      initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.1 }} viewport={{ once: true }}
                      className="border border-fui-gray-500/20 bg-black/40 p-5"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                          <svc.icon className="w-5 h-5" style={{ color: svc.color }} />
                          <span className="text-xs text-fui-gray-100 uppercase tracking-widest">{svc.title}</span>
                        </div>
                        <span className="text-[8px] px-2 py-0.5 border border-green-500/50 text-green-400">LIVE</span>
                      </div>
                      <ul className="space-y-1.5">
                        {svc.lines.map((l) => (
                          <li key={l} className="text-xs text-fui-gray-400 flex items-center gap-2">
                            <span className="w-1 h-1 bg-fui-gray-500 rounded-full" />
                            {l}
                          </li>
                        ))}
                      </ul>
                    </motion.div>
                  ))}
                </div>
              </Box>

              {/* Service URLs + Tech Stack */}
              <div className="grid lg:grid-cols-2 gap-8">
                <Box>
                  <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Service Endpoints</div>
                  <div className="space-y-4">
                    {[
                      { label: 'Production Domain', url: 'aegishealth.dev', badge: 'HTTPS' },
                      { label: 'Frontend', url: 'aegis-frontend-*.us-central1.run.app', badge: 'LIVE' },
                      { label: 'Backend API', url: 'aegis-backend-*.us-central1.run.app/api/v1', badge: 'LIVE' },
                      { label: 'Neo4j Aura', url: 'neo4j+s://*.databases.neo4j.io', badge: 'CONNECTED' },
                    ].map((ep) => (
                      <div key={ep.label} className="border border-fui-gray-500/20 bg-black/40 p-4">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-fui-gray-100">{ep.label}</span>
                          <span className="text-[8px] px-2 py-0.5 border border-green-500/50 text-green-400">{ep.badge}</span>
                        </div>
                        <code className="text-[10px] text-cyan-400 break-all">{ep.url}</code>
                      </div>
                    ))}
                  </div>
                </Box>

                <Box>
                  <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Technology Stack</div>
                  <div className="grid grid-cols-2 gap-4">
                    {[
                      { category: 'Frontend', items: ['React 19', 'Vite 7', 'TailwindCSS', 'Three.js', 'TensorFlow.js', 'Framer Motion'] },
                      { category: 'Backend', items: ['Django 4.2', 'PyTorch 2.0', 'Transformers', 'RDKit', 'Gunicorn'] },
                      { category: 'AI Models', items: ['PubMedBERT', 'GNN (GIN)', 'MobileNetV2', 'Tesseract.js', 'QuaggaJS'] },
                      { category: 'DevOps', items: ['Docker', 'Cloud Run', 'Artifact Registry', 'Cloud DNS', 'GitHub'] },
                    ].map((stack) => (
                      <div key={stack.category} className="border border-fui-gray-500/20 bg-black/40 p-4">
                        <h3 className="text-[10px] text-cyan-400 uppercase tracking-widest mb-3">{stack.category}</h3>
                        <ul className="space-y-1">
                          {stack.items.map((item) => (
                            <li key={item} className="text-xs text-fui-gray-400">{item}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </Box>
              </div>

              {/* Cost metrics */}
              <Box>
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Infrastructure Cost</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
                  {[
                    { v: '~$0', l: 'Monthly Cost', sub: 'Free tier + credits' },
                    { v: '0\u2192N', l: 'Auto-Scaling', sub: 'Scale-to-zero on idle' },
                    { v: '<200ms', l: 'Warm Latency', sub: 'P95 API response' },
                    { v: 'us-central1', l: 'Region', sub: 'Iowa, USA' },
                  ].map((s, i) => (
                    <motion.div key={s.l}
                      initial={{ opacity: 0, scale: 0.9 }} whileInView={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.1 }} viewport={{ once: true }}
                      className="border border-fui-gray-500/20 bg-black/40 p-4"
                    >
                      <div className="text-2xl text-green-400 font-light">{s.v}</div>
                      <div className="text-[10px] text-fui-gray-400 uppercase tracking-widest mt-1">{s.l}</div>
                      <div className="text-[8px] text-fui-gray-500 mt-0.5">{s.sub}</div>
                    </motion.div>
                  ))}
                </div>
              </Box>
            </motion.div>
          )}

        </AnimatePresence>
      </div>

      {/* ── Footer CTA ── */}
      <section className="py-12 border-t border-fui-gray-500/20">
        <div className="max-w-4xl mx-auto px-4 md:px-6">
          <Box className="text-center">
            <h2 className="text-xl md:text-2xl font-light mb-4">Ready to explore?</h2>
            <p className="text-sm text-fui-gray-400 mb-8">
              Launch the dashboard to analyze drug interactions with our multi-model AI system
            </p>
            <button
              onClick={() => navigate('/dashboard')}
              className="px-6 md:px-8 py-3 md:py-4 border border-cyan-500 text-cyan-400 text-sm uppercase tracking-widest hover:bg-cyan-500/10 transition-all"
              style={{ boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)' }}
            >
              Launch Dashboard
            </button>
          </Box>
        </div>
      </section>
    </div>
  );
}
