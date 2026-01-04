import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Search,
    Activity,
    AlertCircle,
    Settings,
    Zap,
    Shield,
    Share,
    Plus,
    Trash2,
    Hexagon,
    Cpu,
    ArrowRight,
    Layers,
    Maximize2,
    MoreHorizontal,
    Atom,
    MousePointer2,
    GitGraph,
    MessageSquare,
    Send,
    Sparkles,
    Database,
    Network,
    BookOpen,
    FlaskConical,
    LayoutDashboard,
    Users,
    FileText,
    Bell,
    X,
    Heart,
    User,
    Loader2
} from 'lucide-react';
import { predictDDI, analyzePolypharmacy, sendChatMessage, searchDrugs, checkHealth } from './services/api';

// --- API Configuration ---
const USE_REAL_API = true; // Set to false to use mock data

// --- Mock Data (Fallback) ---
const DRUG_DATABASE = [
    { id: 1, name: 'Warfarin', category: 'Anticoagulant', risk: 'High', mw: '308.33', logp: '2.7', smiles: 'CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O' },
    { id: 2, name: 'Aspirin', category: 'NSAID', risk: 'Low', mw: '180.16', logp: '1.2', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
    { id: 3, name: 'Lisinopril', category: 'ACE Inhibitor', risk: 'Moderate', mw: '405.49', logp: '-1.2', smiles: 'NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O' },
    { id: 4, name: 'Simvastatin', category: 'Statin', risk: 'Moderate', mw: '418.57', logp: '4.7', smiles: 'CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12' },
    { id: 5, name: 'Metformin', category: 'Antidiabetic', risk: 'Low', mw: '129.16', logp: '-1.4', smiles: 'CN(C)C(=N)NC(=N)N' },
    { id: 6, name: 'Ibuprofen', category: 'NSAID', risk: 'Low', mw: '206.28', logp: '3.5', smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O' },
    { id: 7, name: 'Omeprazole', category: 'PPI', risk: 'Low', mw: '345.42', logp: '2.2', smiles: 'COC1=CC2=NC(CS(=O)C3=NC4=C(N3)C=CC=C4C)=NC2=CC1OC' },
    { id: 8, name: 'Atorvastatin', category: 'Statin', risk: 'Moderate', mw: '558.64', logp: '4.1', smiles: 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)c(-c2ccccc2)n1CC[C@H](O)C[C@H](O)CC(=O)O' },
];

// Fallback mock for when API is not available
const MOCK_INTERACTIONS_DB = {
    'Warfarin-Aspirin': {
        severity: 'major',
        risk_score: 0.92,
        risk_level: 'critical',
        confidence: 0.984,
        mechanism_hypothesis: 'Pharmacodynamic Synergism - Concurrent use significantly increases bleeding risk due to additive anticoagulant effects and platelet inhibition.',
        affected_systems: [
            { system: 'blood', severity: 0.9, symptoms: ['Increased bleeding risk', 'Prolonged INR'] },
            { system: 'gi', severity: 0.6, symptoms: ['GI bleeding', 'Ulceration'] }
        ],
        citations: ['N Engl J Med 2024; 389:123-135', 'Clin Pharmacol Ther. 2023; 114:88-92']
    }
};

// --- Body Map Component ---
const BodyMap = ({ affectedSystems = {} }) => {
    const getOrganColor = (organ) => {
        const severity = affectedSystems[organ] || 0;
        if (severity > 0.7) return 'fill-red-500 animate-pulse';
        if (severity > 0.4) return 'fill-orange-500';
        if (severity > 0) return 'fill-yellow-500';
        return 'fill-slate-700';
    };

    return (
        <div className="relative w-full h-full flex items-center justify-center">
            <svg viewBox="0 0 200 400" className="w-32 h-64">
                {/* Head */}
                <circle cx="100" cy="35" r="25" className="fill-slate-700 stroke-slate-600" strokeWidth="1" />
                
                {/* Body */}
                <ellipse cx="100" cy="130" rx="45" ry="60" className="fill-slate-700 stroke-slate-600" strokeWidth="1" />
                
                {/* Heart */}
                <circle cx="90" cy="110" r="12" className={`${getOrganColor('heart')} stroke-slate-600 transition-colors`} strokeWidth="1" />
                
                {/* Lungs */}
                <ellipse cx="70" cy="120" rx="15" ry="25" className={`${getOrganColor('lungs')} stroke-slate-600 transition-colors`} strokeWidth="1" />
                <ellipse cx="130" cy="120" rx="15" ry="25" className={`${getOrganColor('lungs')} stroke-slate-600 transition-colors`} strokeWidth="1" />
                
                {/* Liver */}
                <ellipse cx="120" cy="150" rx="20" ry="15" className={`${getOrganColor('liver')} stroke-slate-600 transition-colors`} strokeWidth="1" />
                
                {/* Kidney */}
                <ellipse cx="80" cy="165" rx="10" ry="15" className={`${getOrganColor('kidney')} stroke-slate-600 transition-colors`} strokeWidth="1" />
                <ellipse cx="120" cy="165" rx="10" ry="15" className={`${getOrganColor('kidney')} stroke-slate-600 transition-colors`} strokeWidth="1" />
                
                {/* GI */}
                <ellipse cx="100" cy="185" rx="25" ry="15" className={`${getOrganColor('gi')} stroke-slate-600 transition-colors`} strokeWidth="1" />
                
                {/* Brain indicator */}
                <circle cx="100" cy="30" r="10" className={`${getOrganColor('brain')} stroke-slate-600 transition-colors`} strokeWidth="1" />
                
                {/* Legs */}
                <rect x="80" y="200" width="15" height="80" rx="5" className="fill-slate-700 stroke-slate-600" strokeWidth="1" />
                <rect x="105" y="200" width="15" height="80" rx="5" className="fill-slate-700 stroke-slate-600" strokeWidth="1" />
                
                {/* Arms */}
                <rect x="40" y="100" width="12" height="60" rx="5" className="fill-slate-700 stroke-slate-600" strokeWidth="1" transform="rotate(-15 40 100)" />
                <rect x="148" y="100" width="12" height="60" rx="5" className="fill-slate-700 stroke-slate-600" strokeWidth="1" transform="rotate(15 160 100)" />
                
                {/* Blood indicator (overlay) */}
                {affectedSystems['blood'] > 0 && (
                    <circle cx="100" cy="130" r="50" className="fill-red-500/20 animate-pulse" />
                )}
            </svg>
            
            {/* Legend */}
            <div className="absolute bottom-0 left-0 right-0 flex justify-center gap-4 text-[10px]">
                {Object.entries(affectedSystems).filter(([_, v]) => v > 0).map(([organ, severity]) => (
                    <div key={organ} className="flex items-center gap-1">
                        <div className={`w-2 h-2 rounded-full ${severity > 0.7 ? 'bg-red-500' : severity > 0.4 ? 'bg-orange-500' : 'bg-yellow-500'}`} />
                        <span className="text-slate-400 capitalize">{organ}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// --- Components ---

const GlassCard = ({ children, className = "", noPadding = false }) => (
    <div className={`bg-[#131B2C]/80 backdrop-blur-xl border border-white/10 shadow-2xl rounded-[32px] ${noPadding ? '' : 'p-6'} ${className} transition-all duration-300 hover:shadow-blue-900/20 hover:border-blue-500/30`}>
        {children}
    </div>
);

const Badge = ({ children, type = "neutral" }) => {
    const styles = {
        neutral: "bg-slate-800 text-slate-400 border-slate-700",
        critical: "bg-red-500/10 text-red-400 border-red-500/20 shadow-[0_0_10px_rgba(239,68,68,0.2)]",
        blue: "bg-blue-500/10 text-blue-400 border-blue-500/20 shadow-[0_0_10px_rgba(59,130,246,0.2)]",
        success: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    };
    return (
        <span className={`px-2.5 py-1 rounded-full text-[10px] uppercase font-bold tracking-wider border ${styles[type]}`}>
            {children}
        </span>
    );
};

// --- Stabilized Molecular Viewer ---
const MolecularViewer = ({ active, riskLevel }) => {
    const canvasRef = useRef(null);
    const rotation = useRef({ x: 0.2, y: 0.4 });
    const isDragging = useRef(false);
    const lastMouse = useRef({ x: 0, y: 0 });

    // STATIC GENERATION: Only run once to ensure stable structure
    const moleculeData = useRef(null);

    if (!moleculeData.current) {
        const atoms = [];
        const bonds = [];

        // Create Molecule A (Warfarin-ish structure)
        // Double Ring System
        const cx1 = -2.5, cy1 = 0, cz1 = 0;
        for (let i = 0; i < 6; i++) {
            const ang = (i / 6) * Math.PI * 2;
            atoms.push({ x: cx1 + Math.cos(ang), y: cy1 + Math.sin(ang), z: cz1, type: 'C' });
            bonds.push([i, (i + 1) % 6]); // Ring bonds
        }
        // Side group
        atoms.push({ x: cx1 + 2, y: cy1 + 0.5, z: cz1 + 0.5, type: 'O' });
        bonds.push([0, 6]);

        // Create Molecule B (Aspirin-ish structure)
        const cx2 = 2.5, cy2 = 0.5, cz2 = 1;
        const offset = 7; // Index offset
        for (let i = 0; i < 6; i++) {
            const ang = (i / 6) * Math.PI * 2;
            atoms.push({ x: cx2 + Math.cos(ang), y: cy2 + Math.sin(ang), z: cz2, type: 'C' });
            bonds.push([offset + i, offset + ((i + 1) % 6)]);
        }
        // Side group
        atoms.push({ x: cx2 - 1.5, y: cy2 - 1, z: cz2 - 0.5, type: 'O' });
        bonds.push([offset + 3, offset + 6]);

        moleculeData.current = { atoms, bonds };
    }

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let animationId;

        const resize = () => {
            const parent = canvas.parentElement;
            canvas.width = parent.offsetWidth * window.devicePixelRatio;
            canvas.height = parent.offsetHeight * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };
        resize();
        window.addEventListener('resize', resize);

        const render = () => {
            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;
            ctx.clearRect(0, 0, w, h);

            const cx = w / 2;
            const cy = h / 2;
            const scale = Math.min(w, h) * 0.15; // Zoomed out slightly

            if (!isDragging.current) rotation.current.y += 0.002;

            const cosX = Math.cos(rotation.current.x);
            const sinX = Math.sin(rotation.current.x);
            const cosY = Math.cos(rotation.current.y);
            const sinY = Math.sin(rotation.current.y);

            // Project Atoms
            const projected = moleculeData.current.atoms.map(a => {
                // Rotate Y
                let x = a.x * cosY - a.z * sinY;
                let z = a.x * sinY + a.z * cosY;
                // Rotate X
                let y = a.y * cosX - z * sinX;
                z = a.y * sinX + z * cosX;

                // Interaction Drift (Subtle movement towards each other if risky)
                if (riskLevel === 'Critical') {
                    if (a.x < 0) x += 0.2; // Move left right
                    if (a.x > 0) x -= 0.2; // Move right left
                }

                return {
                    ...a, x, y, z,
                    px: x * scale + cx,
                    py: y * scale + cy,
                    alpha: Math.max(0.1, (z + 5) / 10) // Z-depth opacity
                };
            });

            // Draw Bonds (Behind atoms)
            projected.sort((a, b) => a.z - b.z);

            // Re-project without sort for bond lookup
            const rawProjected = moleculeData.current.atoms.map(a => {
                let x = a.x * cosY - a.z * sinY;
                let z = a.x * sinY + a.z * cosY;
                let y = a.y * cosX - z * sinX;
                z = a.y * sinX + z * cosX;
                if (riskLevel === 'Critical') {
                    if (a.x < 0) x += 0.2;
                    if (a.x > 0) x -= 0.2;
                }
                return { px: x * scale + cx, py: y * scale + cy, alpha: (z + 5) / 10 };
            });

            ctx.lineCap = 'round';
            moleculeData.current.bonds.forEach(([i, j]) => {
                const p1 = rawProjected[i];
                const p2 = rawProjected[j];
                ctx.beginPath();
                ctx.moveTo(p1.px, p1.py);
                ctx.lineTo(p2.px, p2.py);
                ctx.lineWidth = 3;
                ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)'; // Stable bond color
                ctx.stroke();
            });

            // Draw Interaction Zone
            if (riskLevel === 'Critical') {
                const centerIdx1 = 6; // O atom of mol 1
                const centerIdx2 = 13; // O atom of mol 2 (offset 7 + 6)
                const p1 = rawProjected[centerIdx1];
                const p2 = rawProjected[centerIdx2];

                if (p1 && p2) {
                    const mx = (p1.px + p2.px) / 2;
                    const my = (p1.py + p2.py) / 2;

                    const grad = ctx.createRadialGradient(mx, my, 0, mx, my, 80);
                    grad.addColorStop(0, 'rgba(239, 68, 68, 0.2)');
                    grad.addColorStop(1, 'rgba(239, 68, 68, 0)');
                    ctx.fillStyle = grad;
                    ctx.beginPath();
                    ctx.arc(mx, my, 80, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            // Draw Atoms (Sorted)
            projected.forEach(p => {
                ctx.beginPath();
                const r = (p.type === 'O' ? 8 : 6) * p.alpha * 1.5;
                ctx.arc(p.px, p.py, r, 0, Math.PI * 2);

                if (p.type === 'O') ctx.fillStyle = `rgba(239, 68, 68, ${p.alpha})`;
                else ctx.fillStyle = `rgba(59, 130, 246, ${p.alpha})`; // Blue carbons

                ctx.shadowColor = p.type === 'O' ? 'rgba(239, 68, 68, 0.8)' : 'rgba(59, 130, 246, 0.8)';
                ctx.shadowBlur = 15;
                ctx.fill();
                ctx.shadowBlur = 0;

                // Specular highlight
                ctx.beginPath();
                ctx.arc(p.px - r / 3, p.py - r / 3, r / 3, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255,255,255, ${p.alpha * 0.8})`;
                ctx.fill();
            });

            animationId = requestAnimationFrame(render);
        };

        render();

        // Interaction Handlers
        const handleDown = (e) => { isDragging.current = true; lastMouse.current = { x: e.clientX, y: e.clientY }; };
        const handleMove = (e) => {
            if (!isDragging.current) return;
            rotation.current.y += (e.clientX - lastMouse.current.x) * 0.005;
            rotation.current.x += (e.clientY - lastMouse.current.y) * 0.005;
            lastMouse.current = { x: e.clientX, y: e.clientY };
        };
        const handleUp = () => isDragging.current = false;

        canvas.addEventListener('mousedown', handleDown);
        window.addEventListener('mousemove', handleMove);
        window.addEventListener('mouseup', handleUp);

        return () => {
            window.removeEventListener('resize', resize);
            window.removeEventListener('mousemove', handleMove);
            window.removeEventListener('mouseup', handleUp);
            cancelAnimationFrame(animationId);
        };
    }, [riskLevel]);

    return (
        <div className="relative w-full h-full cursor-grab active:cursor-grabbing group">
            <canvas ref={canvasRef} className="w-full h-full" />
            <div className="absolute top-4 right-4 flex gap-2">
                <button className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors shadow-sm text-slate-300">
                    <Maximize2 className="w-4 h-4" />
                </button>
            </div>
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-2 bg-slate-900/50 backdrop-blur-md rounded-full border border-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                <MousePointer2 className="w-3 h-3 text-slate-400" />
                <span className="text-[10px] font-semibold text-slate-400">Interactive Model</span>
            </div>
        </div>
    );
};

export default function Dashboard() {
    const [selectedDrugs, setSelectedDrugs] = useState([DRUG_DATABASE[0], DRUG_DATABASE[1]]);
    const [searchQuery, setSearchQuery] = useState('');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [polypharmacyResult, setPolypharmacyResult] = useState(null);
    const [activeTab, setActiveTab] = useState('structure'); // structure, graph, properties, bodymap
    const [sidebarItem, setSidebarItem] = useState('dashboard');
    const [messages, setMessages] = useState([]);
    const [prompt, setPrompt] = useState('');
    const [apiStatus, setApiStatus] = useState('checking'); // checking, online, offline
    const [chatSessionId, setChatSessionId] = useState(null);
    const [error, setError] = useState(null);

    // Check API status on mount
    useEffect(() => {
        const checkApiStatus = async () => {
            if (!USE_REAL_API) {
                setApiStatus('offline');
                return;
            }
            try {
                await checkHealth();
                setApiStatus('online');
            } catch (err) {
                console.warn('API not available, using mock data:', err);
                setApiStatus('offline');
            }
        };
        checkApiStatus();
    }, []);

    // Search Filter
    const filtered = DRUG_DATABASE.filter(d =>
        d.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !selectedDrugs.some(s => s.id === d.id)
    );

    const addDrug = (drug) => {
        setSelectedDrugs([...selectedDrugs, drug]);
        setSearchQuery('');
        setResult(null);
        setPolypharmacyResult(null);
        setError(null);
    };

    const removeDrug = (id) => {
        setSelectedDrugs(selectedDrugs.filter(d => d.id !== id));
        setResult(null);
        setPolypharmacyResult(null);
        setError(null);
    };

    // Run DDI analysis using real API or mock data
    const runModel = async () => {
        setIsAnalyzing(true);
        setError(null);
        
        try {
            if (apiStatus === 'online' && USE_REAL_API) {
                // Use real API
                if (selectedDrugs.length === 2) {
                    // Two-drug prediction
                    const response = await predictDDI(
                        { name: selectedDrugs[0].name, smiles: selectedDrugs[0].smiles },
                        { name: selectedDrugs[1].name, smiles: selectedDrugs[1].smiles }
                    );
                    setResult(response);
                    setPolypharmacyResult(null);
                } else {
                    // Polypharmacy (N-way) prediction
                    const drugs = selectedDrugs.map(d => ({ name: d.name, smiles: d.smiles }));
                    const response = await analyzePolypharmacy(drugs);
                    setPolypharmacyResult(response);
                    // Also set a summary result
                    if (response.interactions.length > 0) {
                        const topInteraction = response.interactions.sort((a, b) => b.risk_score - a.risk_score)[0];
                        setResult({
                            drug_a: topInteraction.source,
                            drug_b: topInteraction.target,
                            risk_score: response.max_risk_score,
                            risk_level: response.overall_risk_level,
                            severity: topInteraction.severity,
                            confidence: 0.85,
                            mechanism_hypothesis: `${response.total_interactions} interactions detected. ${response.hub_drug} is the hub drug with ${response.hub_interaction_count} interactions.`,
                            affected_systems: Object.entries(response.body_map || {}).map(([system, severity]) => ({
                                system,
                                severity,
                                symptoms: []
                            }))
                        });
                    } else {
                        setResult({
                            severity: 'no_interaction',
                            risk_score: 0,
                            risk_level: 'low',
                            mechanism_hypothesis: 'No significant interactions detected between these drugs.'
                        });
                    }
                }
            } else {
                // Use mock data fallback
                await new Promise(resolve => setTimeout(resolve, 1500));
                const drugNames = selectedDrugs.map(d => d.name).sort();
                const key = drugNames.join('-');
                if (MOCK_INTERACTIONS_DB[key]) {
                    setResult(MOCK_INTERACTIONS_DB[key]);
                } else {
                    // Check for Warfarin-Aspirin in any order
                    const hasWarfarin = selectedDrugs.some(d => d.name === 'Warfarin');
                    const hasAspirin = selectedDrugs.some(d => d.name === 'Aspirin');
                    if (hasWarfarin && hasAspirin) {
                        setResult(MOCK_INTERACTIONS_DB['Warfarin-Aspirin']);
                    } else {
                        setResult({
                            severity: 'no_interaction',
                            risk_score: 0.1,
                            risk_level: 'low',
                            confidence: 0.75,
                            mechanism_hypothesis: 'No significant interaction expected between these drugs.',
                            affected_systems: []
                        });
                    }
                }
            }
        } catch (err) {
            console.error('Prediction error:', err);
            setError('Failed to analyze drug interactions. Please try again.');
            // Fallback to mock
            setResult(MOCK_INTERACTIONS_DB['Warfarin-Aspirin']);
        } finally {
            setIsAnalyzing(false);
        }
    };

    // Send chat message using real API or mock
    const sendPrompt = async (e) => {
        e.preventDefault();
        if (!prompt.trim()) return;
        
        const userMessage = prompt;
        setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
        setPrompt('');
        
        try {
            if (apiStatus === 'online' && USE_REAL_API) {
                // Use real API
                const contextDrugs = selectedDrugs.map(d => d.name);
                const response = await sendChatMessage(userMessage, contextDrugs, chatSessionId);
                setChatSessionId(response.session_id);
                setMessages(prev => [...prev, {
                    role: 'ai',
                    text: response.response,
                    sources: response.sources
                }]);
            } else {
                // Mock response
                await new Promise(resolve => setTimeout(resolve, 1000));
                setMessages(prev => [...prev, {
                    role: 'ai',
                    text: "The interaction is mediated by the displacement of Warfarin from albumin binding sites by Aspirin, combined with pharmacodynamic antiplatelet effects. This can lead to significantly elevated INR values and increased bleeding risk."
                }]);
            }
        } catch (err) {
            console.error('Chat error:', err);
            setMessages(prev => [...prev, {
                role: 'ai',
                text: "I apologize, but I'm currently unable to process your request. Please try again later."
            }]);
        }
    };

    // Helper to get body map data from result
    const getBodyMapData = () => {
        if (!result || !result.affected_systems) return {};
        const bodyMap = {};
        result.affected_systems.forEach(sys => {
            bodyMap[sys.system] = sys.severity || 0.5;
        });
        return bodyMap;
    };

    return (
        <div className="flex h-screen bg-[#0B0F19] font-sans text-slate-200 selection:bg-blue-500/30 overflow-hidden">

            {/* --- Sidebar Navigation --- */}
            <aside className="w-20 lg:w-64 bg-[#0B0F19]/50 backdrop-blur-xl border-r border-white/5 flex flex-col justify-between py-6 z-20">
                <div>
                    <div className="px-6 mb-10 flex items-center gap-3">
                        <div className="w-10 h-10 bg-gradient-to-tr from-blue-600 to-indigo-500 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                            <Hexagon className="w-6 h-6 text-white" />
                        </div>
                        <span className="text-xl font-bold tracking-tight hidden lg:block text-white">Molecule<span className="text-blue-500">AI</span></span>
                    </div>

                    <nav className="px-3 space-y-1">
                        {[
                            { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
                            { id: 'patients', icon: Users, label: 'Patient Records' },
                            { id: 'models', icon: Database, label: 'Model Registry' },
                            { id: 'research', icon: BookOpen, label: 'Research' },
                        ].map(item => (
                            <button
                                key={item.id}
                                onClick={() => setSidebarItem(item.id)}
                                className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium transition-all ${sidebarItem === item.id
                                    ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20 shadow-[0_0_15px_rgba(59,130,246,0.1)]'
                                    : 'text-slate-500 hover:bg-white/5 hover:text-slate-300'
                                    }`}
                            >
                                <item.icon className="w-5 h-5" />
                                <span className="hidden lg:block">{item.label}</span>
                            </button>
                        ))}
                    </nav>
                </div>

                <div className="px-6">
                    <div className="p-4 bg-gradient-to-br from-slate-900 to-black border border-white/10 rounded-2xl hidden lg:block relative overflow-hidden group cursor-pointer">
                        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-indigo-600/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                        <div className="relative z-10">
                            <div className="flex items-center gap-2 text-white mb-1">
                                <Sparkles className="w-4 h-4 text-yellow-400" />
                                <span className="text-xs font-bold uppercase tracking-wider">Pro Plan</span>
                            </div>
                            <p className="text-[10px] text-slate-500 group-hover:text-slate-300">Access full GNN Training pipeline.</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* --- Main Content --- */}
            <main className="flex-1 flex flex-col min-w-0 relative">
                {/* Top Header */}
                <header className="h-20 px-8 flex items-center justify-between z-10 border-b border-white/5">
                    <div>
                        <h1 className="text-2xl font-bold text-white">Drug Interaction Analysis</h1>
                        <p className="text-xs text-slate-500 font-medium">Session ID: #8821-X9 • Project Aegis v1.0</p>
                    </div>
                    <div className="flex items-center gap-4">
                        {/* API Status Indicator */}
                        <div className={`h-10 px-4 bg-[#131B2C] rounded-full border border-white/10 flex items-center gap-2 shadow-sm text-sm font-medium ${
                            apiStatus === 'online' ? 'text-emerald-400' : apiStatus === 'checking' ? 'text-yellow-400' : 'text-orange-400'
                        }`}>
                            <span className={`w-2 h-2 rounded-full ${
                                apiStatus === 'online' ? 'bg-emerald-500 animate-pulse shadow-[0_0_10px_rgba(16,185,129,0.5)]' :
                                apiStatus === 'checking' ? 'bg-yellow-500 animate-pulse' : 'bg-orange-500'
                            }`} />
                            {apiStatus === 'online' ? 'API Connected' : apiStatus === 'checking' ? 'Connecting...' : 'Using Mock Data'}
                        </div>
                        <button className="w-10 h-10 bg-[#131B2C] rounded-full border border-white/10 flex items-center justify-center text-slate-400 hover:text-blue-400 shadow-sm transition-colors">
                            <Bell className="w-5 h-5" />
                        </button>
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 border-2 border-[#0B0F19] shadow-md" />
                    </div>
                </header>

                <div className="flex-1 p-8 grid grid-cols-12 gap-8 overflow-y-auto pb-10 custom-scrollbar">

                    {/* --- Left Column: Input & Controls --- */}
                    <div className="col-span-12 lg:col-span-3 flex flex-col gap-6">
                        <GlassCard className="flex flex-col gap-5 h-full">
                            <div className="flex justify-between items-center">
                                <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Regimen Builder</h2>
                                <Badge type="blue">{selectedDrugs.length} Active</Badge>
                            </div>

                            <div className="relative group z-30">
                                <Search className="absolute left-3 top-3 w-4 h-4 text-slate-500 group-focus-within:text-blue-500 transition-colors" />
                                <input
                                    className="w-full bg-[#0B0F19]/50 border border-white/10 rounded-xl py-2.5 pl-10 pr-3 text-sm text-white focus:bg-[#0B0F19] focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/50 outline-none transition-all placeholder:text-slate-600"
                                    placeholder="Search molecule database..."
                                    value={searchQuery}
                                    onChange={e => setSearchQuery(e.target.value)}
                                />
                                {searchQuery && (
                                    <div className="absolute top-full left-0 w-full mt-2 bg-[#1A2333] backdrop-blur-xl rounded-xl shadow-2xl border border-white/10 overflow-hidden animate-in fade-in slide-in-from-top-2">
                                        {filtered.map(d => (
                                            <button key={d.id} onClick={() => addDrug(d)} className="w-full text-left px-4 py-3 hover:bg-blue-500/10 text-sm font-medium text-slate-300 hover:text-white flex justify-between items-center border-b border-white/5 last:border-0 transition-colors">
                                                {d.name} <Plus className="w-3 h-3 text-blue-500" />
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <div className="flex-1 space-y-3 overflow-y-auto pr-2 custom-scrollbar">
                                {selectedDrugs.map(d => (
                                    <div key={d.id} className="flex items-center justify-between p-3 bg-white/5 border border-white/10 rounded-2xl shadow-sm group hover:bg-white/10 transition-all duration-300">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-xl bg-blue-500/10 text-blue-400 flex items-center justify-center text-[10px] font-bold shadow-inner border border-blue-500/20">
                                                {d.name.substring(0, 2).toUpperCase()}
                                            </div>
                                            <div>
                                                <div className="text-sm font-bold text-slate-200">{d.name}</div>
                                                <div className="text-[10px] font-medium text-slate-500">{d.category}</div>
                                            </div>
                                        </div>
                                        <button onClick={() => removeDrug(d.id)} className="w-8 h-8 flex items-center justify-center rounded-full text-slate-500 hover:bg-red-500/10 hover:text-red-400 transition-colors">
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                ))}
                            </div>

                            <button
                                onClick={runModel}
                                disabled={isAnalyzing || selectedDrugs.length < 2}
                                className={`w-full py-4 rounded-2xl font-bold text-sm shadow-lg flex items-center justify-center gap-2 transition-all duration-300 ${isAnalyzing ? 'bg-slate-800 text-slate-500' : 'bg-blue-600 text-white hover:bg-blue-500 hover:shadow-blue-500/25 active:scale-[0.98]'}`}
                            >
                                {isAnalyzing ? <><div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" /> Analyzing...</> : <><Zap className="w-4 h-4 fill-white" /> Run Analysis</>}
                            </button>
                        </GlassCard>
                    </div>

                    {/* --- Center Column: Visualization & Graph --- */}
                    <div className="col-span-12 lg:col-span-6 flex flex-col gap-6 h-full">
                        <GlassCard className="flex-1 flex flex-col relative overflow-hidden p-0" noPadding={true}>
                            {/* Tab Switcher */}
                            <div className="absolute top-6 left-6 z-10 flex gap-2 flex-wrap">
                                {[
                                    { id: 'structure', label: '3D Structure', icon: Hexagon },
                                    { id: 'graph', label: 'Knowledge Graph', icon: Network },
                                    { id: 'bodymap', label: 'Body Map', icon: User },
                                    { id: 'properties', label: 'Properties', icon: FlaskConical },
                                ].map(tab => (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`px-4 py-2 rounded-full text-xs font-bold backdrop-blur-md border transition-all flex items-center gap-2 ${activeTab === tab.id
                                            ? 'bg-white/10 text-white border-white/20 shadow-lg'
                                            : 'bg-black/20 text-slate-500 border-transparent hover:bg-white/5 hover:text-slate-300'
                                            }`}
                                    >
                                        <tab.icon className="w-3 h-3" /> {tab.label}
                                    </button>
                                ))}
                            </div>

                            <div className="flex-1 bg-gradient-to-b from-[#0B0F19] to-[#131B2C] relative">
                                {activeTab === 'structure' && (
                                    <MolecularViewer active={true} riskLevel={result?.severity || 'None'} />
                                )}

                                {activeTab === 'bodymap' && (
                                    <div className="w-full h-full pt-20 pb-10">
                                        <div className="text-center mb-4">
                                            <h3 className="text-sm font-bold text-slate-300">Affected Organ Systems</h3>
                                            <p className="text-[10px] text-slate-500">Highlighted areas indicate potential interaction effects</p>
                                        </div>
                                        <BodyMap affectedSystems={getBodyMapData()} />
                                    </div>
                                )}

                                {activeTab === 'graph' && (
                                    <div className="w-full h-full flex items-center justify-center relative">
                                        {/* Polypharmacy Network Graph */}
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <div className="relative w-96 h-96">
                                                {/* Drug Nodes - Dynamic based on selected drugs */}
                                                {selectedDrugs.map((drug, idx) => {
                                                    const angle = (idx / selectedDrugs.length) * Math.PI * 2 - Math.PI / 2;
                                                    const radius = 100;
                                                    const x = 50 + (radius * Math.cos(angle) / 2);
                                                    const y = 50 + (radius * Math.sin(angle) / 2);
                                                    return (
                                                        <div 
                                                            key={drug.id}
                                                            className="absolute flex flex-col items-center gap-1 animate-float-slow"
                                                            style={{ 
                                                                left: `${x}%`, 
                                                                top: `${y}%`, 
                                                                transform: 'translate(-50%, -50%)',
                                                                animationDelay: `${idx * 0.5}s` 
                                                            }}
                                                        >
                                                            <div className={`w-14 h-14 bg-[#1A2333] rounded-2xl shadow-xl flex items-center justify-center border z-10 font-bold text-sm ${
                                                                polypharmacyResult?.hub_drug === drug.name 
                                                                    ? 'border-red-500/50 text-red-400' 
                                                                    : 'border-blue-500/30 text-blue-400'
                                                            }`}>
                                                                {drug.name.substring(0, 3).toUpperCase()}
                                                            </div>
                                                            <span className="text-[10px] text-slate-500">{drug.name}</span>
                                                        </div>
                                                    );
                                                })}

                                                {/* Target Nodes */}
                                                <div className="absolute top-10 left-1/2 -translate-x-1/2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-xs font-mono text-emerald-400 shadow-sm">
                                                    Target: VKORC1
                                                </div>
                                                <div className="absolute bottom-10 left-1/2 -translate-x-1/2 px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-lg text-xs font-mono text-purple-400 shadow-sm">
                                                    Enzyme: CYP2C9
                                                </div>

                                                {/* SVG Connections */}
                                                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                                                    <path d="M120 192 L 190 60" stroke="#334155" strokeWidth="2" strokeDasharray="4 4" />
                                                    <path d="M260 192 L 190 60" stroke="#334155" strokeWidth="2" strokeDasharray="4 4" />
                                                    <path d="M120 192 L 190 320" stroke="#334155" strokeWidth="2" strokeDasharray="4 4" />
                                                    <path d="M260 192 L 190 320" stroke="#334155" strokeWidth="2" strokeDasharray="4 4" />
                                                    {/* Direct Interaction */}
                                                    <line x1="130" y1="192" x2="250" y2="192" stroke="#ef4444" strokeWidth="3" opacity="0.2" />
                                                </svg>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {activeTab === 'properties' && (
                                    <div className="p-10 pt-24 grid grid-cols-2 gap-4">
                                        {selectedDrugs.map(d => (
                                            <div key={d.id} className="p-4 bg-white/5 rounded-2xl border border-white/10 shadow-sm">
                                                <h3 className="font-bold text-slate-200 mb-4 flex items-center gap-2">
                                                    <div className="w-2 h-2 bg-blue-500 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.8)]" /> {d.name}
                                                </h3>
                                                <div className="space-y-3">
                                                    <div className="flex justify-between text-sm border-b border-white/5 pb-2">
                                                        <span className="text-slate-500">Mol. Weight</span>
                                                        <span className="font-mono text-slate-300">{d.mw} g/mol</span>
                                                    </div>
                                                    <div className="flex justify-between text-sm border-b border-white/5 pb-2">
                                                        <span className="text-slate-500">LogP</span>
                                                        <span className="font-mono text-slate-300">{d.logp}</span>
                                                    </div>
                                                    <div className="flex justify-between text-sm">
                                                        <span className="text-slate-500">H-Bond Donors</span>
                                                        <span className="font-mono text-slate-300">{Math.floor(Math.random() * 5)}</span>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </GlassCard>
                    </div>

                    {/* --- Right Column: Intelligence & Report --- */}
                    <div className="col-span-12 lg:col-span-3 flex flex-col gap-6">
                        <GlassCard className="flex-1 flex flex-col relative overflow-hidden">
                            <div className="flex justify-between items-center mb-6">
                                <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                                    <Sparkles className="w-3 h-3 text-blue-500" /> AI Synthesis
                                </h2>
                                <div className="flex gap-2">
                                    <Share className="w-4 h-4 text-slate-500 hover:text-blue-400 cursor-pointer" />
                                    <Maximize2 className="w-4 h-4 text-slate-500 hover:text-blue-400 cursor-pointer" />
                                </div>
                            </div>

                            {/* Error Display */}
                            {error && (
                                <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-xs text-red-400">
                                    {error}
                                </div>
                            )}

                            {result && result.severity !== 'no_interaction' ? (
                                <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-6">
                                    {/* Risk Alert Card */}
                                    <div className={`p-5 rounded-2xl shadow-sm relative overflow-hidden ${
                                        result.risk_level === 'critical' || result.severity === 'major'
                                            ? 'bg-red-500/10 border border-red-500/20' 
                                            : result.risk_level === 'high' || result.severity === 'moderate'
                                            ? 'bg-orange-500/10 border border-orange-500/20'
                                            : 'bg-yellow-500/10 border border-yellow-500/20'
                                    }`}>
                                        <div className={`absolute top-0 right-0 w-16 h-16 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2 ${
                                            result.risk_level === 'critical' ? 'bg-red-500/20' : 'bg-orange-500/20'
                                        }`} />
                                        <div className="flex gap-3 relative z-10">
                                            <div className={`p-2 rounded-lg shadow-sm h-fit ${
                                                result.risk_level === 'critical' ? 'bg-red-500/20' : 'bg-orange-500/20'
                                            }`}>
                                                <AlertCircle className={`w-5 h-5 ${
                                                    result.risk_level === 'critical' ? 'text-red-400' : 'text-orange-400'
                                                }`} />
                                            </div>
                                            <div>
                                                <div className={`font-bold leading-tight ${
                                                    result.risk_level === 'critical' ? 'text-red-100' : 'text-orange-100'
                                                }`}>
                                                    {result.risk_level === 'critical' ? 'Critical' : result.severity === 'major' ? 'Major' : 'Moderate'} Interaction
                                                </div>
                                                <div className={`text-xs mt-1 opacity-80 font-medium ${
                                                    result.risk_level === 'critical' ? 'text-red-300' : 'text-orange-300'
                                                }`}>
                                                    {result.drug_a || selectedDrugs[0]?.name} + {result.drug_b || selectedDrugs[1]?.name}
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Risk Score Display */}
                                    <div>
                                        <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2">Risk Score</div>
                                        <div className="flex items-end gap-3">
                                            <span className="text-4xl font-light text-white tracking-tighter">
                                                {((result.risk_score || result.confidence || 0.5) * 100).toFixed(1)}
                                                <span className="text-lg text-slate-500 ml-1">%</span>
                                            </span>
                                            <div className="flex-1 h-2 bg-slate-800 rounded-full mb-3 overflow-hidden">
                                                <div 
                                                    className={`h-full shadow-[0_0_10px_rgba(59,130,246,0.5)] ${
                                                        result.risk_level === 'critical' ? 'bg-gradient-to-r from-red-500 to-red-600' :
                                                        result.risk_level === 'high' ? 'bg-gradient-to-r from-orange-500 to-red-500' :
                                                        'bg-gradient-to-r from-blue-500 to-indigo-600'
                                                    }`}
                                                    style={{ width: `${(result.risk_score || result.confidence || 0.5) * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-3">
                                        {/* Mechanism */}
                                        <div className="p-4 bg-white/5 border border-white/10 rounded-2xl shadow-sm">
                                            <div className="text-[10px] font-bold text-blue-400 uppercase mb-2 flex items-center gap-2">
                                                <Activity className="w-3 h-3" /> Mechanism Hypothesis
                                            </div>
                                            <p className="text-sm text-slate-300 leading-relaxed font-medium">
                                                {result.mechanism_hypothesis || result.mechanism || 'Analyzing interaction mechanism...'}
                                            </p>
                                        </div>

                                        {/* Affected Systems */}
                                        {result.affected_systems && result.affected_systems.length > 0 && (
                                            <div className="p-4 bg-white/5 border border-white/10 rounded-2xl shadow-sm">
                                                <div className="text-[10px] font-bold text-red-400 uppercase mb-2 flex items-center gap-2">
                                                    <Heart className="w-3 h-3" /> Affected Systems
                                                </div>
                                                <div className="flex flex-wrap gap-2">
                                                    {result.affected_systems.map((sys, i) => (
                                                        <span key={i} className="px-2 py-1 bg-red-500/10 border border-red-500/20 rounded-lg text-xs text-red-400 capitalize">
                                                            {sys.system || sys}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* Citations */}
                                        {result.citations && (
                                            <div className="p-4 bg-white/5 border border-white/10 rounded-2xl shadow-sm">
                                                <div className="text-[10px] font-bold text-slate-500 uppercase mb-2 flex items-center gap-2">
                                                    <FileText className="w-3 h-3" /> Key Citations
                                                </div>
                                                <ul className="space-y-2">
                                                    {result.citations.map((c, i) => (
                                                        <li key={i} className="text-[11px] text-blue-400 hover:text-blue-300 hover:underline cursor-pointer flex items-start gap-2">
                                                            <span className="mt-1 w-1 h-1 rounded-full bg-blue-500" /> {c}
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ) : result && result.severity === 'no_interaction' ? (
                                <div className="flex-1 flex flex-col items-center justify-center">
                                    <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl text-center">
                                        <Shield className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                                        <p className="text-sm font-medium text-emerald-300">No Significant Interaction</p>
                                        <p className="text-xs text-slate-500 mt-1">{result.mechanism_hypothesis}</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex-1 flex flex-col items-center justify-center opacity-40">
                                    <Database className="w-12 h-12 text-slate-600 mb-4" />
                                    <p className="text-sm text-slate-500 text-center">Select drugs and run analysis</p>
                                </div>
                            )}

                            <div className="mt-6 pt-4 border-t border-white/5">
                                <div className="h-40 bg-[#0B0F19]/50 rounded-2xl border border-white/5 p-4 mb-3 overflow-y-auto text-sm space-y-3 custom-scrollbar">
                                    {messages.length === 0 && (
                                        <div className="flex flex-col items-center justify-center h-full text-center">
                                            <MessageSquare className="w-6 h-6 text-slate-700 mb-2" />
                                            <p className="text-xs text-slate-600">Ask Dr. AI about this interaction...</p>
                                        </div>
                                    )}
                                    {messages.map((m, i) => (
                                        <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                            <div className={`max-w-[85%] p-3 rounded-2xl text-xs leading-relaxed ${m.role === 'user' ? 'bg-blue-600 text-white rounded-br-none shadow-md' : 'bg-white/10 border border-white/5 rounded-bl-none text-slate-300 shadow-sm'}`}>
                                                {m.text}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                <form onSubmit={sendPrompt} className="relative group">
                                    <input
                                        className="w-full bg-[#0B0F19] border border-white/10 rounded-xl py-3 pl-4 pr-12 text-xs text-white focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/50 outline-none transition-all placeholder:text-slate-600 shadow-sm"
                                        placeholder="Type your clinical query..."
                                        value={prompt}
                                        onChange={e => setPrompt(e.target.value)}
                                    />
                                    <button type="submit" className="absolute right-2 top-2 p-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors shadow-md group-focus-within:scale-105">
                                        <ArrowRight className="w-3 h-3" />
                                    </button>
                                </form>
                            </div>
                        </GlassCard>
                    </div>
                </div>

            </main>
        </div>
    );
}
