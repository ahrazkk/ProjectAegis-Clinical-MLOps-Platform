import React, { useEffect, useRef } from 'react';
import { useSystemLogs } from '../hooks/useSystemLogs';
import { Terminal, X, Trash2, Activity, Server, Database, Cpu } from 'lucide-react';
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
                className="fixed bottom-6 right-6 z-50 p-3 bg-black/80 border border-emerald-500/50 rounded-full text-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:bg-emerald-900/20 hover:scale-110 transition-all"
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
                        className="fixed top-0 right-0 h-screen w-96 bg-[#030305]/95 backdrop-blur-xl border-l border-emerald-500/30 z-40 shadow-2xl flex flex-col font-mono text-xs"
                    >
                        {/* Header */}
                        <div className="p-4 border-b border-emerald-500/20 bg-black/50 flex items-center justify-between">
                            <div className="flex items-center gap-2 text-emerald-400">
                                <Activity className="w-4 h-4 animate-pulse" />
                                <span className="font-bold tracking-widest uppercase">System Monitor</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={clearLogs}
                                    className="p-1.5 text-slate-500 hover:text-red-400 transition-colors"
                                    title="Clear Logs"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                                <button
                                    onClick={toggleMonitor}
                                    className="p-1.5 text-slate-500 hover:text-white transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        {/* Status Grid */}
                        <div className="grid grid-cols-3 gap-2 p-2 border-b border-emerald-500/20 bg-emerald-900/5">
                            <StatusBadge icon={Server} label="API" status="online" />
                            <StatusBadge icon={Database} label="NEO4J" status="offline" />
                            <StatusBadge icon={Cpu} label="AI" status="idle" />
                        </div>

                        {/* Log Feed */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-2 font-mono custom-scrollbar">
                            {logs.length === 0 && (
                                <div className="text-center text-slate-600 mt-20 opacity-50">
                  // WAITING FOR SIGNALS...
                                </div>
                            )}
                            {logs.map((log) => (
                                <div key={log.id} className="group flex items-start gap-3 hover:bg-white/5 p-1 rounded transition-colors">
                                    <span className="text-slate-500 shrink-0 select-none">
                                        [{log.timestamp}]
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

                        {/* Footer Input */}
                        <div className="p-3 border-t border-emerald-500/20 bg-black/80">
                            <div className="flex items-center gap-2 text-emerald-600">
                                <span className="animate-pulse">▶</span>
                                <span className="opacity-50">root@aegis-core:~$ _</span>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}

function StatusBadge({ icon: Icon, label, status }) {
    const getStatusColor = (s) => {
        switch (s) {
            case 'online': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
            case 'offline': return 'text-red-400 bg-red-500/10 border-red-500/20';
            case 'idle': return 'text-blue-400 bg-blue-500/10 border-blue-500/20';
            default: return 'text-slate-400 bg-slate-500/10 border-slate-500/20';
        }
    };

    return (
        <div className={`flex flex-col items-center justify-center p-2 rounded border ${getStatusColor(status)}`}>
            <Icon className="w-4 h-4 mb-1" />
            <span className="text-[10px] font-bold">{label}</span>
            <div className="flex items-center gap-1 mt-1">
                <div className={`w-1.5 h-1.5 rounded-full ${status === 'online' ? 'bg-emerald-400 animate-pulse' : status === 'offline' ? 'bg-red-400' : 'bg-blue-400'}`} />
                <span className="text-[9px] uppercase opacity-80">{status}</span>
            </div>
        </div>
    );
}

function getSourceColor(source) {
    switch (source) {
        case 'API': return 'text-blue-400';
        case 'NEO4J': return 'text-orange-400';
        case 'AI': return 'text-purple-400';
        case 'SYSTEM': return 'text-emerald-400';
        default: return 'text-slate-400';
    }
}

function getTypeColor(type) {
    switch (type) {
        case 'error': return 'text-red-400';
        case 'success': return 'text-emerald-300';
        case 'warning': return 'text-yellow-400';
        default: return 'text-slate-300';
    }
}
