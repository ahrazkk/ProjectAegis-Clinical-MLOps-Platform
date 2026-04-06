import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ShieldAlert, Home, ArrowLeft, Terminal, Activity, Search } from 'lucide-react';

export default function NotFoundPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [glitch, setGlitch] = useState(false);
  const [scanLine, setScanLine] = useState(0);
  const [typed, setTyped] = useState('');

  const errorMsg = `ERROR 404: Route "${location.pathname}" not found in Aegis network.`;

  // Typewriter effect
  useEffect(() => {
    let i = 0;
    const timer = setInterval(() => {
      if (i <= errorMsg.length) {
        setTyped(errorMsg.slice(0, i));
        i++;
      } else {
        clearInterval(timer);
      }
    }, 30);
    return () => clearInterval(timer);
  }, []);

  // Glitch effect
  useEffect(() => {
    const timer = setInterval(() => {
      setGlitch(true);
      setTimeout(() => setGlitch(false), 150);
    }, 3000);
    return () => clearInterval(timer);
  }, []);

  // Scan line
  useEffect(() => {
    const timer = setInterval(() => {
      setScanLine(prev => (prev + 1) % 100);
    }, 50);
    return () => clearInterval(timer);
  }, []);

  const navLinks = [
    { path: '/', label: 'Landing Page', icon: Home },
    { path: '/dashboard', label: 'Dashboard', icon: Activity },
    { path: '/research', label: 'Research', icon: Search },
  ];

  return (
    <div className="min-h-screen bg-black text-white font-mono relative overflow-hidden flex items-center justify-center">
      {/* Background grid */}
      <div className="fixed inset-0 opacity-[0.03] pointer-events-none" style={{
        backgroundImage: 'linear-gradient(rgba(239,68,68,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(239,68,68,0.4) 1px, transparent 1px)',
        backgroundSize: '60px 60px'
      }} />

      {/* Scan line */}
      <div
        className="fixed left-0 right-0 h-[2px] bg-red-500/10 pointer-events-none z-50"
        style={{ top: `${scanLine}%`, transition: 'top 50ms linear' }}
      />

      {/* Vignette */}
      <div className="fixed inset-0 pointer-events-none" style={{
        background: 'radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.8) 100%)'
      }} />

      <div className="relative z-10 max-w-2xl w-full mx-4">
        {/* Main error card */}
        <div className="border border-red-500/30 bg-black/80 backdrop-blur-sm relative">
          {/* Corner accents */}
          <div className="absolute -top-px -left-px w-4 h-4 border-t-2 border-l-2 border-red-500/70" />
          <div className="absolute -top-px -right-px w-4 h-4 border-t-2 border-r-2 border-red-500/70" />
          <div className="absolute -bottom-px -left-px w-4 h-4 border-b-2 border-l-2 border-red-500/70" />
          <div className="absolute -bottom-px -right-px w-4 h-4 border-b-2 border-r-2 border-red-500/70" />

          {/* Header bar */}
          <div className="flex items-center gap-3 px-6 py-3 border-b border-red-500/20 bg-red-500/5">
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
              <div className="w-2.5 h-2.5 rounded-full bg-red-500/40" />
              <div className="w-2.5 h-2.5 rounded-full bg-red-500/20" />
            </div>
            <span className="text-[10px] text-red-400/60 uppercase tracking-[0.3em]">Aegis Security Protocol</span>
          </div>

          {/* Content */}
          <div className="p-8 md:p-12">
            {/* Shield icon */}
            <div className="flex justify-center mb-8">
              <div className={`relative ${glitch ? 'translate-x-[2px]' : ''} transition-transform`}>
                <ShieldAlert className="w-20 h-20 text-red-500/80" strokeWidth={1} />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-red-400 mt-1">404</span>
                </div>
              </div>
            </div>

            {/* Error code */}
            <div className="text-center mb-8">
              <h1 className={`text-4xl md:text-5xl font-light tracking-tight mb-3 ${glitch ? 'text-red-400 skew-x-1' : 'text-white'} transition-all`}>
                ROUTE NOT FOUND
              </h1>
              <div className="w-24 h-px bg-gradient-to-r from-transparent via-red-500/50 to-transparent mx-auto mb-4" />
              <p className="text-sm text-red-400/60 uppercase tracking-[0.2em]">
                Aegis Network Boundary Violation
              </p>
            </div>

            {/* Terminal output */}
            <div className="bg-black/60 border border-red-500/10 rounded-lg p-4 mb-8 font-mono">
              <div className="flex items-center gap-2 mb-2 text-[10px] text-red-500/40 uppercase tracking-widest">
                <Terminal className="w-3 h-3" />
                System Log
              </div>
              <div className="text-xs text-red-400/70 leading-relaxed">
                <span className="text-red-500/40">$</span> {typed}
                <span className="animate-pulse text-red-400">_</span>
              </div>
              <div className="text-xs text-white/20 mt-2">
                <span className="text-red-500/40">$</span> Redirecting to secure routes...
              </div>
            </div>

            {/* Navigation links */}
            <div className="space-y-2 mb-8">
              <p className="text-[10px] text-white/30 uppercase tracking-[0.2em] mb-3">Available Routes</p>
              {navLinks.map(link => (
                <button
                  key={link.path}
                  onClick={() => navigate(link.path)}
                  className="w-full flex items-center gap-3 px-4 py-3 border border-white/5 hover:border-cyan-500/30 bg-white/[0.02] hover:bg-cyan-500/5 transition-all group text-left"
                >
                  <link.icon className="w-4 h-4 text-white/20 group-hover:text-cyan-400 transition-colors" />
                  <span className="text-sm text-white/40 group-hover:text-cyan-300 transition-colors">{link.label}</span>
                  <span className="text-[10px] text-white/10 group-hover:text-cyan-500/40 ml-auto font-mono transition-colors">{link.path}</span>
                </button>
              ))}
            </div>

            {/* Action buttons */}
            <div className="flex gap-3">
              <button
                onClick={() => navigate(-1)}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 border border-white/10 hover:border-white/30 text-white/40 hover:text-white/80 text-xs uppercase tracking-widest transition-all"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Go Back
              </button>
              <button
                onClick={() => navigate('/dashboard')}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 border border-cyan-500/30 hover:border-cyan-400/60 bg-cyan-500/5 hover:bg-cyan-500/10 text-cyan-400 text-xs uppercase tracking-widest transition-all"
              >
                <Activity className="w-3.5 h-3.5" />
                Dashboard
              </button>
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-2 border-t border-red-500/10 flex items-center justify-between">
            <span className="text-[9px] text-white/10 uppercase tracking-widest">Project Aegis v2.0</span>
            <span className="text-[9px] text-white/10 uppercase tracking-widest">Security Layer Active</span>
          </div>
        </div>
      </div>
    </div>
  );
}
