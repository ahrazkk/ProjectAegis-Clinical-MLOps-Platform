import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
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
  ExternalLink,
  BookOpen,
  Layers,
  Cpu,
  Globe,
  RefreshCw,
  Search,
  FileText,
  MessageSquare,
  Sparkles,
  Network,
  CircleDot
} from 'lucide-react';

// Container wrapper for all content sections - better readability
const ContentContainer = ({ children, className = "" }) => (
  <div className={`bg-black/80 backdrop-blur-sm border border-fui-gray-500/30 p-6 md:p-8 ${className}`}>
    {children}
  </div>
);

// COMPREHENSIVE SYSTEM FLOW DIAGRAM
const SystemFlowDiagram = () => {
  const flowSteps = [
    { id: 'input', label: 'User Query', sublabel: '"Aspirin + Warfarin"', icon: Search, color: 'cyan', description: 'User submits drug pair for analysis' },
    { id: 'neo4j-lookup', label: 'Neo4j Lookup', sublabel: 'Knowledge Graph', icon: Database, color: 'green', description: 'Check for existing verified interactions' },
    { id: 'context-fetch', label: 'Context Retrieval', sublabel: 'DDI Corpus + PubMed', icon: FileText, color: 'purple', description: 'Fetch relevant clinical literature' },
    { id: 'pubmedbert', label: 'PubMedBERT', sublabel: 'Relation Extraction', icon: Brain, color: 'cyan', description: 'Transformer predicts interaction probability' },
    { id: 'risk-scoring', label: 'Risk Scoring', sublabel: 'Multi-factor Analysis', icon: Activity, color: 'green', description: 'Combine model output with KG evidence' },
    { id: 'response', label: 'Response', sublabel: 'Risk + Mechanism', icon: MessageSquare, color: 'cyan', description: 'Return risk level and recommendations' }
  ];

  const futureSteps = [
    { id: 'rag-pipeline', label: 'RAG Pipeline', sublabel: 'Coming Soon', icon: Sparkles, description: 'Real-time PubMed retrieval' },
    { id: 'neo4j-update', label: 'Neo4j Update', sublabel: 'Auto-enrichment', icon: RefreshCw, description: 'Discovered interactions update graph' }
  ];

  return (
    <ContentContainer>
      <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// System Flow</div>
      
      {/* Current Pipeline */}
      <div className="mb-8">
        <h3 className="text-sm text-fui-gray-100 uppercase tracking-widest mb-6">Current Pipeline</h3>
        
        {/* Desktop Flow */}
        <div className="hidden lg:block">
          <div className="flex items-start justify-between gap-2">
            {flowSteps.map((step, i) => (
              <React.Fragment key={step.id}>
                <div className="flex-1 min-w-0">
                  <div className={`border p-4 text-center h-full ${
                    step.color === 'cyan' ? 'border-cyan-500/50 bg-cyan-500/5' :
                    step.color === 'green' ? 'border-green-500/50 bg-green-500/5' :
                    'border-purple-500/50 bg-purple-500/5'
                  }`} style={{ boxShadow: `0 0 15px rgba(${step.color === 'cyan' ? '0,212,255' : step.color === 'green' ? '0,255,136' : '168,85,247'},0.1)` }}>
                    <step.icon className={`w-6 h-6 mx-auto mb-2 ${
                      step.color === 'cyan' ? 'text-cyan-400' : step.color === 'green' ? 'text-green-400' : 'text-purple-400'
                    }`} />
                    <div className="text-[11px] text-fui-gray-100 uppercase tracking-widest mb-1">{step.label}</div>
                    <div className="text-[9px] text-fui-gray-500">{step.sublabel}</div>
                  </div>
                  <div className="mt-2 text-[9px] text-fui-gray-400 text-center px-1">{step.description}</div>
                </div>
                {i < flowSteps.length - 1 && <div className="flex-shrink-0 pt-8"><ArrowRight className="w-4 h-4 text-fui-gray-500" /></div>}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Mobile Flow */}
        <div className="lg:hidden space-y-3">
          {flowSteps.map((step, i) => (
            <React.Fragment key={step.id}>
              <div className={`border p-4 flex items-center gap-4 ${
                step.color === 'cyan' ? 'border-cyan-500/50 bg-cyan-500/5' :
                step.color === 'green' ? 'border-green-500/50 bg-green-500/5' :
                'border-purple-500/50 bg-purple-500/5'
              }`}>
                <step.icon className={`w-6 h-6 flex-shrink-0 ${
                  step.color === 'cyan' ? 'text-cyan-400' : step.color === 'green' ? 'text-green-400' : 'text-purple-400'
                }`} />
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] text-fui-gray-100 uppercase tracking-widest">{step.label}</div>
                  <div className="text-[9px] text-fui-gray-500">{step.sublabel}</div>
                  <div className="text-[9px] text-fui-gray-400 mt-1">{step.description}</div>
                </div>
              </div>
              {i < flowSteps.length - 1 && <div className="flex justify-center"><ArrowDown className="w-4 h-4 text-fui-gray-500" /></div>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Future Enhancement - RAG Loop */}
      <div className="mt-8 pt-8 border-t border-fui-gray-500/20">
        <h3 className="text-sm text-purple-400 uppercase tracking-widest mb-6 flex items-center gap-2">
          <Sparkles className="w-4 h-4" />
          Future Enhancement: RAG Pipeline
        </h3>
        
        <div className="flex flex-col md:flex-row items-center justify-center gap-4">
          {futureSteps.map((step, i) => (
            <React.Fragment key={step.id}>
              <div className="border border-dashed border-purple-500/30 p-4 text-center opacity-70 w-full md:w-48">
                <step.icon className="w-6 h-6 mx-auto mb-2 text-purple-400" />
                <div className="text-[11px] text-fui-gray-100 uppercase tracking-widest mb-1">{step.label}</div>
                <div className="text-[9px] text-purple-400">{step.sublabel}</div>
                <div className="text-[9px] text-fui-gray-500 mt-2">{step.description}</div>
              </div>
              {i < futureSteps.length - 1 && <ArrowRight className="w-4 h-4 text-purple-500/50 hidden md:block" />}
            </React.Fragment>
          ))}
          <ArrowRight className="w-4 h-4 text-purple-500/50 hidden md:block" />
          <div className="border border-dashed border-green-500/30 p-4 text-center opacity-70 w-full md:w-48">
            <Database className="w-6 h-6 mx-auto mb-2 text-green-400" />
            <div className="text-[11px] text-fui-gray-100 uppercase tracking-widest mb-1">Neo4j Graph</div>
            <div className="text-[9px] text-green-400">Updated</div>
            <div className="text-[9px] text-fui-gray-500 mt-2">Knowledge continuously grows</div>
          </div>
        </div>
        
        <div className="mt-4 text-center">
          <p className="text-[10px] text-fui-gray-500">→ Future RAG system will fetch latest PubMed research and automatically update Neo4j with new discoveries</p>
        </div>
      </div>
    </ContentContainer>
  );
};

// Architecture Diagram Component
const ArchitectureDiagram = () => (
  <ContentContainer className="relative overflow-hidden">
    {/* Blueprint grid */}
    <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ 
      backgroundImage: 'linear-gradient(rgba(102,102,102,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(102,102,102,0.3) 1px, transparent 1px)', 
      backgroundSize: '30px 30px' 
    }}></div>

    <div className="relative z-10">
      <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// System Architecture</div>
      
      {/* Three-tier architecture */}
      <div className="space-y-6">
        {/* Frontend Layer */}
        <div className="border border-cyan-500/30 p-4 relative bg-cyan-500/5">
          <div className="absolute -top-3 left-4 px-2 bg-black text-[10px] text-cyan-400 uppercase tracking-widest">
            Presentation Layer
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-2">
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <Globe className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">React + Vite</div>
            </div>
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <Layers className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">Three.js 3D</div>
            </div>
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <Activity className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">Framer Motion</div>
            </div>
          </div>
        </div>

        {/* Arrow */}
        <div className="flex justify-center">
          <div className="w-px h-6 bg-fui-gray-500/50"></div>
        </div>

        {/* Backend Layer */}
        <div className="border border-purple-500/30 p-4 relative bg-purple-500/5">
          <div className="absolute -top-3 left-4 px-2 bg-black text-[10px] text-purple-400 uppercase tracking-widest">
            Application Layer
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-2">
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <Server className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">Django REST</div>
            </div>
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <Brain className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">PubMedBERT</div>
            </div>
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <BookOpen className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">RAG Pipeline</div>
            </div>
          </div>
        </div>

        {/* Arrow */}
        <div className="flex justify-center">
          <div className="w-px h-6 bg-fui-gray-500/50"></div>
        </div>

        {/* Data Layer */}
        <div className="border border-green-500/30 p-4 relative bg-green-500/5">
          <div className="absolute -top-3 left-4 px-2 bg-black text-[10px] text-green-400 uppercase tracking-widest">
            Data Layer
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-2">
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <Database className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">Neo4j Aura</div>
            </div>
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <Database className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">SQLite</div>
            </div>
            <div className="border border-fui-gray-500/30 p-3 text-center bg-black/50">
              <Cloud className="w-5 h-5 text-fui-gray-400 mx-auto mb-2" />
              <div className="text-[10px] text-fui-gray-400">PubChem API</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </ContentContainer>
);

// Model Pipeline Diagram
const ModelPipelineDiagram = () => (
  <ContentContainer>
    <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// DDI Prediction Pipeline</div>
    
    {/* Desktop Flow */}
    <div className="hidden md:flex items-center justify-between gap-2 flex-wrap">
      {/* Step 1: Input */}
      <div className="flex-1 min-w-[100px] border border-fui-gray-500/30 p-4 text-center bg-black/50">
        <div className="text-[10px] text-cyan-400 uppercase tracking-widest mb-2">Input</div>
        <div className="text-xs text-fui-gray-400">"Drug A + Drug B"</div>
      </div>

      <ArrowRight className="w-4 h-4 text-fui-gray-500 flex-shrink-0" />

      {/* Step 2: Tokenization */}
      <div className="flex-1 min-w-[100px] border border-fui-gray-500/30 p-4 text-center bg-black/50">
        <div className="text-[10px] text-cyan-400 uppercase tracking-widest mb-2">Tokenize</div>
        <div className="text-xs text-fui-gray-400">BioWordPiece</div>
      </div>

      <ArrowRight className="w-4 h-4 text-fui-gray-500 flex-shrink-0" />

      {/* Step 3: Encoder */}
      <div className="flex-1 min-w-[100px] border border-cyan-500/50 p-4 text-center bg-cyan-500/10" style={{ boxShadow: '0 0 15px rgba(0,255,255,0.1)' }}>
        <div className="text-[10px] text-cyan-400 uppercase tracking-widest mb-2">Encoder</div>
        <div className="text-xs text-fui-gray-100">PubMedBERT</div>
      </div>

      <ArrowRight className="w-4 h-4 text-fui-gray-500 flex-shrink-0" />

      {/* Step 4: Classifier */}
      <div className="flex-1 min-w-[100px] border border-fui-gray-500/30 p-4 text-center bg-black/50">
        <div className="text-[10px] text-cyan-400 uppercase tracking-widest mb-2">Classify</div>
        <div className="text-xs text-fui-gray-400">Relation Head</div>
      </div>

      <ArrowRight className="w-4 h-4 text-fui-gray-500 flex-shrink-0" />

      {/* Step 5: Output */}
      <div className="flex-1 min-w-[100px] border border-green-500/50 p-4 text-center bg-green-500/10" style={{ boxShadow: '0 0 15px rgba(0,255,136,0.1)' }}>
        <div className="text-[10px] text-green-400 uppercase tracking-widest mb-2">Output</div>
        <div className="text-xs text-fui-gray-100">Risk Score</div>
      </div>
    </div>

    {/* Mobile Flow */}
    <div className="md:hidden space-y-3">
      {[
        { label: 'Input', sublabel: '"Drug A + Drug B"', highlight: false },
        { label: 'Tokenize', sublabel: 'BioWordPiece', highlight: false },
        { label: 'Encoder', sublabel: 'PubMedBERT', highlight: true, color: 'cyan' },
        { label: 'Classify', sublabel: 'Relation Head', highlight: false },
        { label: 'Output', sublabel: 'Risk Score', highlight: true, color: 'green' },
      ].map((step, i, arr) => (
        <React.Fragment key={step.label}>
          <div className={`border p-4 text-center ${
            step.highlight 
              ? step.color === 'cyan' ? 'border-cyan-500/50 bg-cyan-500/10' : 'border-green-500/50 bg-green-500/10'
              : 'border-fui-gray-500/30 bg-black/50'
          }`}>
            <div className={`text-[10px] uppercase tracking-widest mb-2 ${
              step.color === 'cyan' ? 'text-cyan-400' : step.color === 'green' ? 'text-green-400' : 'text-cyan-400'
            }`}>{step.label}</div>
            <div className={`text-xs ${step.highlight ? 'text-fui-gray-100' : 'text-fui-gray-400'}`}>{step.sublabel}</div>
          </div>
          {i < arr.length - 1 && <div className="flex justify-center"><ArrowDown className="w-4 h-4 text-fui-gray-500" /></div>}
        </React.Fragment>
      ))}
    </div>
  </ContentContainer>
);

// Performance Metrics Chart (Bar visualization)
const MetricsChart = () => {
  const metrics = [
    { label: 'AUC', value: 92.7, color: '#00d4ff' },
    { label: 'Precision', value: 89.2, color: '#00ff88' },
    { label: 'Recall', value: 87.5, color: '#a855f7' },
    { label: 'F1 Score', value: 88.3, color: '#00d4ff' },
  ];

  return (
    <ContentContainer>
      <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Model Performance</div>
      <div className="space-y-4">
        {metrics.map((metric, i) => (
          <motion.div 
            key={metric.label}
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1 }}
            viewport={{ once: true }}
            className="space-y-2"
          >
            <div className="flex justify-between text-xs">
              <span className="text-fui-gray-400 uppercase tracking-widest">{metric.label}</span>
              <span className="text-fui-gray-100">{metric.value}%</span>
            </div>
            <div className="h-3 bg-fui-gray-500/20 overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                whileInView={{ width: `${metric.value}%` }}
                transition={{ duration: 1, delay: i * 0.1 }}
                viewport={{ once: true }}
                className="h-full"
                style={{ backgroundColor: metric.color }}
              />
            </div>
          </motion.div>
        ))}
      </div>
    </ContentContainer>
  );
};

// Deployment Infrastructure
const DeploymentDiagram = () => (
  <ContentContainer>
    <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Cloud Deployment (GCP)</div>
    
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Cloud Run Services */}
      <div className="border border-cyan-500/30 p-4 bg-cyan-500/5">
        <div className="flex items-center gap-2 mb-4">
          <Cloud className="w-5 h-5 text-cyan-400" />
          <span className="text-xs text-fui-gray-100 uppercase tracking-widest">Cloud Run</span>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs text-fui-gray-400">
            <CheckCircle className="w-3 h-3 text-green-400" />
            <span>aegis-frontend</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-fui-gray-400">
            <CheckCircle className="w-3 h-3 text-green-400" />
            <span>aegis-backend</span>
          </div>
        </div>
      </div>

      {/* Container Registry */}
      <div className="border border-fui-gray-500/30 p-4 bg-black/50">
        <div className="flex items-center gap-2 mb-4">
          <Cpu className="w-5 h-5 text-fui-gray-400" />
          <span className="text-xs text-fui-gray-100 uppercase tracking-widest">Containers</span>
        </div>
        <div className="space-y-2 text-xs text-fui-gray-400">
          <div>• nginx + React build</div>
          <div>• Python 3.11 + PyTorch</div>
          <div>• Gunicorn WSGI</div>
        </div>
      </div>

      {/* External Services */}
      <div className="border border-fui-gray-500/30 p-4 bg-black/50">
        <div className="flex items-center gap-2 mb-4">
          <Database className="w-5 h-5 text-fui-gray-400" />
          <span className="text-xs text-fui-gray-100 uppercase tracking-widest">External</span>
        </div>
        <div className="space-y-2 text-xs text-fui-gray-400">
          <div>• Neo4j Aura (Graph DB)</div>
          <div>• PubChem API</div>
          <div>• RxNorm API</div>
        </div>
      </div>
    </div>
  </ContentContainer>
);

export default function ResearchPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'flow', label: 'System Flow' },
    { id: 'model', label: 'Model' },
    { id: 'data', label: 'Data Pipeline' },
    { id: 'deployment', label: 'Deployment' },
  ];

  return (
    <div className="min-h-screen bg-black text-fui-gray-100 font-mono relative">
      {/* Background pattern */}
      <div className="fixed inset-0 opacity-5 pointer-events-none" style={{ 
        backgroundImage: 'linear-gradient(rgba(102,102,102,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(102,102,102,0.3) 1px, transparent 1px)', 
        backgroundSize: '50px 50px' 
      }}></div>

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/95 backdrop-blur-sm border-b border-fui-gray-500/30">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex items-center justify-between">
          <button 
            onClick={() => navigate('/')}
            className="flex items-center gap-2 md:gap-3 text-fui-gray-400 hover:text-fui-gray-100 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-xs uppercase tracking-widest hidden sm:inline">Back to Home</span>
          </button>
          
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 border border-fui-gray-500 flex items-center justify-center">
              <GitBranch className="w-4 h-4 text-fui-gray-400" />
            </div>
            <span className="text-sm tracking-widest uppercase">
              Project<span className="text-cyan-400">Aegis</span>
            </span>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-24 md:pt-32 pb-12 md:pb-16 border-b border-fui-gray-500/20 relative">
        <div className="max-w-7xl mx-auto px-4 md:px-6">
          <ContentContainer>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <span className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-4 block">// Research & Methodology</span>
              <h1 className="text-2xl md:text-4xl lg:text-5xl font-light mb-6 tracking-wide">
                Technical Documentation
              </h1>
              <p className="text-sm text-fui-gray-400 max-w-2xl leading-relaxed">
                Comprehensive overview of Project Aegis architecture, model design, 
                data pipeline, and cloud deployment infrastructure.
              </p>
            </motion.div>
          </ContentContainer>
        </div>
      </section>

      {/* Tab Navigation */}
      <section className="border-b border-fui-gray-500/20 sticky top-[73px] bg-black/95 backdrop-blur-sm z-40">
        <div className="max-w-7xl mx-auto px-4 md:px-6">
          <div className="flex gap-0 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 md:px-6 py-4 text-[10px] md:text-xs uppercase tracking-widest border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id 
                    ? 'border-cyan-400 text-cyan-400' 
                    : 'border-transparent text-fui-gray-500 hover:text-fui-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Content Sections */}
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-12 md:py-16 relative z-10">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-12 md:space-y-16"
          >
            {/* Problem Statement */}
            <div className="grid lg:grid-cols-2 gap-8 md:gap-12">
              <ContentContainer>
                <h2 className="text-xl font-light mb-6 tracking-wide">Problem Statement</h2>
                <p className="text-sm text-fui-gray-400 leading-relaxed mb-4">
                  Adverse drug events (ADEs) from drug-drug interactions represent one of the most 
                  significant challenges in clinical pharmacology. In the United States alone:
                </p>
                <ul className="space-y-3 text-sm text-fui-gray-400">
                  <li className="flex items-start gap-3">
                    <span className="text-cyan-400">→</span>
                    <span><strong className="text-fui-gray-100">195,000+</strong> hospitalizations annually attributed to DDIs</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-cyan-400">→</span>
                    <span><strong className="text-fui-gray-100">$3.5B</strong> in annual healthcare costs</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-cyan-400">→</span>
                    <span><strong className="text-fui-gray-100">4-5%</strong> of hospital admissions involve preventable ADEs</span>
                  </li>
                </ul>
              </ContentContainer>
              <ContentContainer>
                <h2 className="text-xl font-light mb-6 tracking-wide">Our Solution</h2>
                <p className="text-sm text-fui-gray-400 leading-relaxed mb-4">
                  Project Aegis provides real-time DDI prediction using state-of-the-art 
                  natural language processing and knowledge graph technologies:
                </p>
                <ul className="space-y-3 text-sm text-fui-gray-400">
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>Fine-tuned PubMedBERT for biomedical context understanding</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>Neo4j knowledge graph with 2,000+ drugs and verified interactions</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>RAG pipeline with PubMed literature citations</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>Sub-200ms inference for clinical decision support</span>
                  </li>
                </ul>
              </ContentContainer>
            </div>

            {/* System Architecture */}
            <div>
              <h2 className="text-xl font-light mb-6 tracking-wide px-2">System Architecture</h2>
              <ArchitectureDiagram />
            </div>

            {/* Performance Metrics */}
            <div className="grid lg:grid-cols-2 gap-8 md:gap-12">
              <MetricsChart />
              <ContentContainer>
                <div className="text-[10px] text-cyan-400 uppercase tracking-[0.3em] mb-6">// Key Statistics</div>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { value: '2,080', label: 'Drugs' },
                    { value: '1,651', label: 'Interactions' },
                    { value: '60%+', label: 'Classified' },
                    { value: '<200ms', label: 'Latency' },
                  ].map((stat, i) => (
                    <motion.div
                      key={stat.label}
                      initial={{ opacity: 0, scale: 0.9 }}
                      whileInView={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.1 }}
                      viewport={{ once: true }}
                      className="border border-fui-gray-500/30 p-4 text-center bg-black/50"
                    >
                      <div className="text-2xl text-cyan-400">{stat.value}</div>
                      <div className="text-[10px] text-fui-gray-500 uppercase tracking-widest mt-1">{stat.label}</div>
                    </motion.div>
                  ))}
                </div>
              </ContentContainer>
            </div>
          </motion.div>
        )}

        {/* System Flow Tab - NEW */}
        {activeTab === 'flow' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-12 md:space-y-16"
          >
            <div>
              <h2 className="text-xl font-light mb-6 tracking-wide px-2">How Project Aegis Works</h2>
              <p className="text-sm text-fui-gray-400 mb-8 px-2 max-w-3xl">
                From user query to risk assessment - follow the complete data flow through our system.
                The pipeline combines knowledge graph lookups with deep learning inference for accurate predictions.
              </p>
              <SystemFlowDiagram />
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
              <ContentContainer>
                <h3 className="text-lg font-light mb-4 tracking-wide">Current Capabilities</h3>
                <ul className="space-y-3 text-sm text-fui-gray-400">
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>Neo4j graph lookup for verified interactions</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>DDI Corpus context retrieval for model input</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>PubMedBERT inference with 92.7% AUC</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>Risk scoring with mechanism explanations</span>
                  </li>
                </ul>
              </ContentContainer>

              <ContentContainer>
                <h3 className="text-lg font-light mb-4 tracking-wide flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-purple-400" />
                  Future Roadmap
                </h3>
                <ul className="space-y-3 text-sm text-fui-gray-400">
                  <li className="flex items-start gap-3">
                    <CircleDot className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                    <span>Real-time PubMed RAG for latest research</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CircleDot className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                    <span>Automatic Neo4j enrichment from discoveries</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CircleDot className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                    <span>Graph Neural Networks for molecular analysis</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <CircleDot className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                    <span>Continuous learning from FDA FAERS data</span>
                  </li>
                </ul>
              </ContentContainer>
            </div>
          </motion.div>
        )}

        {/* Model Tab */}
        {activeTab === 'model' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-12 md:space-y-16"
          >
            <div>
              <h2 className="text-xl font-light mb-6 tracking-wide px-2">DDI Prediction Pipeline</h2>
              <ModelPipelineDiagram />
            </div>

            <div className="grid lg:grid-cols-2 gap-8 md:gap-12">
              <ContentContainer>
                <h2 className="text-xl font-light mb-6 tracking-wide">Model Components</h2>
                <div className="space-y-4">
                  {[
                    { 
                      title: 'PubMedBERT Encoder',
                      desc: 'Pre-trained on 3.1B words from PubMed abstracts, fine-tuned on DDI Corpus 2013 for biomedical entity recognition and relation extraction.',
                      tech: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
                    },
                    { 
                      title: 'Relation Classification Head',
                      desc: 'Two-layer MLP with dropout, takes concatenated drug embeddings from BERT encoder. Outputs binary interaction probability.',
                      tech: 'Input: 1536d → Hidden: 768d → Output: 1d (sigmoid)'
                    },
                    { 
                      title: 'Auxiliary NER Head',
                      desc: 'Token-level classification for drug entity recognition. Provides multi-task learning signal for improved generalization.',
                      tech: 'Classes: O, B-DRUG, I-DRUG'
                    },
                  ].map((item, i) => (
                    <motion.div
                      key={item.title}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                      viewport={{ once: true }}
                      className="border border-fui-gray-500/30 bg-black/30 p-4"
                    >
                      <h3 className="text-sm text-fui-gray-100 mb-2">{item.title}</h3>
                      <p className="text-xs text-fui-gray-400 mb-3">{item.desc}</p>
                      <code className="text-[10px] text-fui-accent-cyan bg-fui-gray-500/10 px-2 py-1">{item.tech}</code>
                    </motion.div>
                  ))}
                </div>
              </ContentContainer>

              <ContentContainer>
                <h2 className="text-xl font-light mb-6 tracking-wide">Training Details</h2>
                <div className="space-y-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-fui-gray-400">Dataset</span>
                    <span className="text-fui-gray-100">DDI Corpus 2013</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-fui-gray-400">Training Pairs</span>
                    <span className="text-fui-gray-100">27,792</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-fui-gray-400">Epochs</span>
                    <span className="text-fui-gray-100">50</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-fui-gray-400">Optimizer</span>
                    <span className="text-fui-gray-100">AdamW (lr=2e-5)</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-fui-gray-400">Best AUC</span>
                    <span className="text-fui-accent-green">92.7%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-fui-gray-400">Framework</span>
                    <span className="text-fui-gray-100">PyTorch 2.0 + Transformers</span>
                  </div>
                </div>

                <h2 className="text-xl font-light mb-6 mt-8 tracking-wide">Future Roadmap</h2>
                <div className="border border-fui-accent-purple/30 bg-purple-500/5 p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="w-4 h-4 text-fui-accent-purple" />
                    <span className="text-sm text-fui-gray-100">Graph Neural Networks</span>
                  </div>
                  <p className="text-xs text-fui-gray-400">
                    Planned integration of Message Passing Neural Networks (MPNN) for 
                    molecular structure analysis, enabling prediction from SMILES representations.
                  </p>
                </div>
              </ContentContainer>
            </div>
          </motion.div>
        )}

        {/* Data Pipeline Tab */}
        {activeTab === 'data' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-12 md:space-y-16"
          >
            <div className="grid lg:grid-cols-2 gap-8 md:gap-12">
              <ContentContainer>
                <h2 className="text-xl font-light mb-6 tracking-wide">Data Sources</h2>
                <div className="space-y-4">
                  {[
                    { 
                      name: 'DDI Corpus 2013',
                      desc: 'Annotated corpus of drug-drug interactions from biomedical literature',
                      stats: '1,651 verified interactions'
                    },
                    { 
                      name: 'PubChem API',
                      desc: 'Chemical structure data (SMILES) for molecular representation',
                      stats: '35%+ SMILES coverage'
                    },
                    { 
                      name: 'RxNorm API',
                      desc: 'Standardized drug names and therapeutic classifications',
                      stats: '60%+ classified drugs'
                    },
                  ].map((source, i) => (
                    <motion.div
                      key={source.name}
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.1 }}
                      viewport={{ once: true }}
                      className="border border-fui-gray-500/30 bg-black/30 p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm text-fui-gray-100">{source.name}</h3>
                        <span className="text-[10px] text-fui-accent-cyan">{source.stats}</span>
                      </div>
                      <p className="text-xs text-fui-gray-400">{source.desc}</p>
                    </motion.div>
                  ))}
                </div>
              </ContentContainer>

              <ContentContainer>
                <h2 className="text-xl font-light mb-6 tracking-wide">Knowledge Graph Schema</h2>
                <div className="font-mono text-xs">
                  <div className="text-fui-gray-500 mb-4">// Neo4j Cypher Schema</div>
                  <div className="space-y-2 text-fui-gray-400">
                    <div><span className="text-fui-accent-purple">(:Drug)</span> {'{'}</div>
                    <div className="pl-4">name: <span className="text-fui-accent-green">String</span>,</div>
                    <div className="pl-4">smiles: <span className="text-fui-accent-green">String?</span>,</div>
                    <div className="pl-4">therapeutic_class: <span className="text-fui-accent-green">String?</span>,</div>
                    <div className="pl-4">source: <span className="text-fui-accent-green">String</span></div>
                    <div>{'}'}</div>
                    <div className="mt-4"><span className="text-fui-accent-purple">[:INTERACTS_WITH]</span> {'{'}</div>
                    <div className="pl-4">severity: <span className="text-fui-accent-green">String</span>,</div>
                    <div className="pl-4">type: <span className="text-fui-accent-green">String</span>,</div>
                    <div className="pl-4">description: <span className="text-fui-accent-green">String?</span></div>
                    <div>{'}'}</div>
                  </div>
                </div>
              </ContentContainer>
            </div>

            <ContentContainer>
              <h2 className="text-xl font-light mb-6 tracking-wide">Data Enrichment Pipeline</h2>
              <div className="flex flex-wrap items-center gap-4 justify-between">
                {[
                  { step: '1', label: 'Import DDI Corpus', status: 'complete' },
                  { step: '2', label: 'Fetch SMILES', status: 'complete' },
                  { step: '3', label: 'Classify Drugs', status: 'complete' },
                  { step: '4', label: 'Build Graph', status: 'complete' },
                ].map((item, i) => (
                  <React.Fragment key={item.step}>
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 border ${item.status === 'complete' ? 'border-fui-accent-green text-fui-accent-green' : 'border-fui-gray-500 text-fui-gray-400'} flex items-center justify-center text-xs`}>
                        {item.status === 'complete' ? <CheckCircle className="w-4 h-4" /> : item.step}
                      </div>
                      <span className="text-xs text-fui-gray-400">{item.label}</span>
                    </div>
                    {i < 3 && <ArrowRight className="w-4 h-4 text-fui-gray-500 hidden md:block" />}
                  </React.Fragment>
                ))}
              </div>
            </ContentContainer>
          </motion.div>
        )}

        {/* Deployment Tab */}
        {activeTab === 'deployment' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-12 md:space-y-16"
          >
            <div>
              <h2 className="text-xl font-light mb-6 tracking-wide px-2">Cloud Infrastructure</h2>
              <DeploymentDiagram />
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
              <ContentContainer>
                <h2 className="text-xl font-light mb-6 tracking-wide">Service Endpoints</h2>
                <div className="space-y-4">
                  <div className="border border-fui-gray-500/30 bg-black/30 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-fui-gray-100">Frontend</span>
                      <span className="text-[10px] px-2 py-1 border border-fui-accent-green/50 text-fui-accent-green">LIVE</span>
                    </div>
                    <code className="text-xs text-fui-accent-cyan break-all">
                      aegis-frontend-*.us-central1.run.app
                    </code>
                  </div>
                  <div className="border border-fui-gray-500/30 bg-black/30 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-fui-gray-100">Backend API</span>
                      <span className="text-[10px] px-2 py-1 border border-fui-accent-green/50 text-fui-accent-green">LIVE</span>
                    </div>
                    <code className="text-xs text-fui-accent-cyan break-all">
                      aegis-backend-*.us-central1.run.app/api/v1
                    </code>
                  </div>
                  <div className="border border-fui-gray-500/30 bg-black/30 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-fui-gray-100">Neo4j Aura</span>
                      <span className="text-[10px] px-2 py-1 border border-fui-accent-green/50 text-fui-accent-green">CONNECTED</span>
                    </div>
                    <code className="text-xs text-fui-accent-cyan break-all">
                      neo4j+s://*.databases.neo4j.io
                    </code>
                  </div>
                </div>
              </ContentContainer>

              <ContentContainer>
                <h2 className="text-xl font-light mb-6 tracking-wide">Technology Stack</h2>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { category: 'Frontend', items: ['React 18', 'Vite', 'TailwindCSS', 'Three.js'] },
                    { category: 'Backend', items: ['Django 4.2', 'PyTorch 2.0', 'Transformers', 'Gunicorn'] },
                    { category: 'Database', items: ['Neo4j Aura', 'SQLite', 'Redis (planned)'] },
                    { category: 'DevOps', items: ['Docker', 'Cloud Run', 'Cloud Build', 'GitHub'] },
                  ].map((stack) => (
                    <div key={stack.category} className="border border-fui-gray-500/30 bg-black/30 p-4">
                      <h3 className="text-xs text-fui-accent-cyan uppercase tracking-widest mb-3">{stack.category}</h3>
                      <ul className="space-y-1">
                        {stack.items.map((item) => (
                          <li key={item} className="text-xs text-fui-gray-400">{item}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </ContentContainer>
            </div>
          </motion.div>
        )}
      </div>

      {/* Footer CTA */}
      <section className="py-12 md:py-16 border-t border-fui-gray-500/20">
        <div className="max-w-4xl mx-auto px-4 md:px-6">
          <ContentContainer className="text-center">
            <h2 className="text-xl md:text-2xl font-light mb-4">Ready to explore?</h2>
            <p className="text-sm text-fui-gray-400 mb-8">
              Launch the dashboard to start analyzing drug interactions
            </p>
            <button 
              onClick={() => navigate('/dashboard')}
              className="px-6 md:px-8 py-3 md:py-4 border border-fui-accent-cyan text-fui-accent-cyan text-sm uppercase tracking-widest hover:bg-fui-accent-cyan/10 transition-all"
              style={{ boxShadow: '0 0 20px rgba(0, 255, 255, 0.2)' }}
            >
              Launch Dashboard
            </button>
          </ContentContainer>
        </div>
      </section>
    </div>
  );
}
