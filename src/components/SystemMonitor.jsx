import React, { useEffect, useRef } from 'react';
import { useSystemLogs } from '../hooks/useSystemLogs';
import { Terminal, X, Trash2, Activity, Server, Database, Cpu, Workflow, Zap, HardDrive, Brain, FlaskConical } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function SystemMonitor() {
    const { logs, isOpen, toggleMonitor, clearLogs } = useSystemLogs();
    const bottomRef = useRef(null);

    // Auto-scroll to bottom on new log
    useEffect(() => {
        if (isOpen && bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs, isOpen]);

    return (
        <>
            {/* Toggle Button (Floating) */}
            <motion.button
                onClick={toggleMonitor}
                className="fixed bottom-6 right-6 z-50 p-3 bg-black/90 border border-cyan-500/30 rounded-full text-cyan-400 shadow-[0_0_30px_rgba(34,211,238,0.2)] hover:border-cyan-400/50 hover:scale-110 transition-all backdrop-blur-xl"
                whileHover={{ rotate: 15 }}
            >
                <Terminal className="w-6 h-6" />
            </motion.button>

            {/* Monitor Panel */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ x: '100%', opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: '100%', opacity: 0 }}
                        transition={{ type: 'spring', damping: 20, stiffness: 100 }}
                        className="fixed top-0 right-0 h-screen w-[420px] bg-black/95 backdrop-blur-2xl border-l border-white/5 z-40 shadow-2xl flex flex-col font-mono text-xs"
                        style={{
                            backgroundImage: `
                                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)
                            `,
                            backgroundSize: '20px 20px'
                        }}
                    >
                        {/* Header */}
                        <div className="p-4 border-b border-white/5 bg-black/80 flex items-center justify-between">
                            <div className="flex items-center gap-2 text-cyan-400">
                                <Activity className="w-4 h-4 animate-pulse" />
                                <span className="font-bold tracking-widest uppercase">Pipeline Monitor</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={clearLogs}
                                    className="p-1.5 text-zinc-600 hover:text-red-400 transition-colors"
                                    title="Clear Logs"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                                <button
                                    onClick={toggleMonitor}
                                    className="p-1.5 text-zinc-600 hover:text-white transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        {/* Status Grid - Full Pipeline */}
                        <div className="p-3 border-b border-white/5 bg-black/40">
                            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2 font-bold">Infrastructure</div>
                            <div className="grid grid-cols-4 gap-2 mb-3">
                                <StatusBadge icon={Server} label="API" status="online" />
                                <StatusBadge icon={Database} label="NEO4J" status="online" />
                                <StatusBadge icon={Zap} label="REDIS" status="online" />
                                <StatusBadge icon={HardDrive} label="DRUG DB" status="online" />
                            </div>
                            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2 font-bold">AI Models</div>
                            <div className="grid grid-cols-3 gap-2">
                                <StatusBadge icon={Brain} label="PubMedBERT" status="idle" />
                                <StatusBadge icon={Workflow} label="GNN" status="standby" />
                                <StatusBadge icon={FlaskConical} label="RDKit" status="ready" />
                            </div>
                        </div>

                        {/* Log Feed */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-1 font-mono custom-scrollbar">
                            {logs.length === 0 && (
                                <div className="text-center text-zinc-700 mt-20 opacity-50">
                                    // AWAITING SIGNALS...
                                </div>
                            )}
                            {logs.map((log) => (
                                <div key={log.id} className="group flex items-start gap-3 hover:bg-white/[0.02] p-1.5 rounded transition-colors border-l-2 border-transparent hover:border-cyan-500/30">
                                    <span className="text-zinc-600 shrink-0 select-none text-[10px]">
                                        {log.timestamp}
                                    </span>
                                    <div className="flex-1 break-words">
                                        <span className={`font-bold mr-2 ${getSourceColor(log.source)}`}>
                                            {log.source}::
                                        </span>
                                        <span className={`${getTypeColor(log.type)}`}>
                                            {log.message}
                                        </span>
                                    </div>
                                </div>
                            ))}
                            <div ref={bottomRef} />
                        </div>

                        {/* Footer */}
                        <div className="p-3 border-t border-white/5 bg-black/80">
                            <div className="flex items-center gap-2 text-cyan-600/80">
                                <span className="animate-pulse">▶</span>
                                <span className="opacity-50">aegis@core:~$ _</span>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}

function StatusBadge({ icon: Icon, label, status }) {
    const getStatusStyle = (s) => {
        switch (s) {
            case 'online': return 'text-emerald-400 bg-emerald-500/5 border-emerald-500/20';
            case 'offline': return 'text-red-400 bg-red-500/5 border-red-500/20';
            case 'idle': return 'text-cyan-400 bg-cyan-500/5 border-cyan-500/20';
            case 'standby': return 'text-amber-400 bg-amber-500/5 border-amber-500/20';
            case 'ready': return 'text-violet-400 bg-violet-500/5 border-violet-500/20';
            default: return 'text-zinc-400 bg-zinc-500/5 border-zinc-500/20';
        }
    };

    const getDotColor = (s) => {
        switch (s) {
            case 'online': return 'bg-emerald-400';
            case 'offline': return 'bg-red-400';
            case 'idle': return 'bg-cyan-400';
            case 'standby': return 'bg-amber-400';
            case 'ready': return 'bg-violet-400';
            default: return 'bg-zinc-400';
        }
    };

    return (
        <div className={`flex flex-col items-center justify-center p-2 rounded border ${getStatusStyle(status)}`}>
            <Icon className="w-3.5 h-3.5 mb-1 opacity-80" />
            <span className="text-[9px] font-bold uppercase tracking-wide">{label}</span>
            <div className="flex items-center gap-1 mt-1">
                <div className={`w-1 h-1 rounded-full ${getDotColor(status)} ${status === 'online' ? 'animate-pulse' : ''}`} />
                <span className="text-[8px] uppercase opacity-60">{status}</span>
            </div>
        </div>
    );
}

function getSourceColor(source) {
    switch (source) {
        case 'API': return 'text-blue-400';
        case 'NEO4J': return 'text-orange-400';
        case 'REDIS': return 'text-red-400';
        case 'AI': return 'text-violet-400';
        case 'SYSTEM': return 'text-cyan-400';
        case 'DATABASE': return 'text-amber-400';
        default: return 'text-zinc-400';
    }
}

function getTypeColor(type) {
    switch (type) {
        case 'error': return 'text-red-400';
        case 'success': return 'text-emerald-300';
        case 'warning': return 'text-yellow-400';
        default: return 'text-zinc-300';
    }
}

