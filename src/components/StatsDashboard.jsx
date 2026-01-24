import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Database,
  Activity,
  Pill,
  AlertTriangle,
  TrendingUp,
  Zap,
  BarChart3,
  PieChart,
  Atom,
  Layers,
  RefreshCw,
  ChevronRight,
  Sparkles,
  Target,
  Beaker,
  CheckCircle,
  AlertCircle,
  ArrowUp,
  ArrowDown,
  Shield,
  Clock,
  FileText
} from 'lucide-react';
import { getDatabaseStats } from '../services/api';

// Animated counter hook with smooth easing
function useAnimatedCounter(end, duration = 2000, startOnMount = true) {
  const [count, setCount] = useState(0);
  const rafRef = useRef(null);
  const startTimeRef = useRef(null);

  useEffect(() => {
    if (!startOnMount || end === 0 || typeof end !== 'number') {
      setCount(end);
      return;
    }

    const animate = (timestamp) => {
      if (!startTimeRef.current) startTimeRef.current = timestamp;
      const progress = Math.min((timestamp - startTimeRef.current) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      setCount(Math.floor(eased * end));

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        setCount(end);
      }
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      startTimeRef.current = null;
    };
  }, [end, duration, startOnMount]);

  return count;
}

// Animated Metric Card - FUI Style
function MetricCard({ icon: Icon, label, value, subtitle, trend, color = 'cyan', delay = 0 }) {
  const animatedValue = useAnimatedCounter(typeof value === 'number' ? value : 0);
  const displayValue = typeof value === 'number' ? animatedValue.toLocaleString() : value;

  const colorSchemes = {
    cyan: {
      border: 'border-fui-accent-cyan/30',
      icon: 'text-fui-accent-cyan',
      accent: 'bg-fui-accent-cyan',
    },
    emerald: {
      border: 'border-fui-accent-green/30',
      icon: 'text-fui-accent-green',
      accent: 'bg-fui-accent-green',
    },
    purple: {
      border: 'border-fui-accent-purple/30',
      icon: 'text-fui-accent-purple',
      accent: 'bg-fui-accent-purple',
    },
    amber: {
      border: 'border-fui-accent-orange/30',
      icon: 'text-fui-accent-orange',
      accent: 'bg-fui-accent-orange',
    },
    red: {
      border: 'border-fui-accent-red/30',
      icon: 'text-fui-accent-red',
      accent: 'bg-fui-accent-red',
    },
  };
  const scheme = colorSchemes[color];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className={`relative overflow-hidden bg-theme-panel border ${scheme.border} p-4`}
    >
      {/* Accent line */}
      <div className={`absolute top-0 left-0 right-0 h-[2px] ${scheme.accent} opacity-60`} />

      <div className="relative z-10">
        <div className="flex items-start justify-between">
          <div className={`p-2 border border-fui-gray-500/30 ${scheme.icon}`}>
            <Icon size={18} strokeWidth={1.5} />
          </div>
          {trend !== undefined && (
            <div className={`flex items-center gap-1 text-[10px] ${trend >= 0 ? 'text-fui-accent-green' : 'text-fui-accent-red'}`}>
              {trend >= 0 ? <ArrowUp size={10} /> : <ArrowDown size={10} />}
              {Math.abs(trend)}%
            </div>
          )}
        </div>

        <div className="mt-3">
          <p className="text-[10px] uppercase tracking-widest text-fui-gray-500">{label}</p>
          <p className="text-2xl font-bold text-theme-primary mt-1 font-mono">{displayValue}</p>
          {subtitle && (
            <p className="text-[10px] text-fui-gray-600 mt-1">{subtitle}</p>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// Donut Chart - FUI Style
function DonutChart({ data, size = 160 }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0) || 1;
  const strokeWidth = 20;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;

  const colors = {
    severe: { stroke: '#EF4444' },
    moderate: { stroke: '#F59E0B' },
    minor: { stroke: '#10B981' }
  };

  let cumulativeOffset = 0;
  const segments = Object.entries(data)
    .filter(([_, value]) => value > 0)
    .map(([key, value]) => {
      const percentage = value / total;
      const dashLength = percentage * circumference;
      const offset = cumulativeOffset;
      cumulativeOffset += dashLength;
      return { key, value, percentage, dashLength, offset, ...colors[key] };
    });

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={strokeWidth}
        />
        {/* Data segments */}
        {segments.map((seg, i) => (
          <motion.circle
            key={seg.key}
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={seg.stroke}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            initial={{ strokeDasharray: `0 ${circumference}` }}
            animate={{
              strokeDasharray: `${seg.dashLength} ${circumference - seg.dashLength}`,
              strokeDashoffset: -seg.offset
            }}
            transition={{ duration: 1, delay: 0.2 + i * 0.1 }}
          />
        ))}
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="text-3xl font-bold text-white font-mono"
        >
          {total.toLocaleString()}
        </motion.p>
        <p className="text-[10px] text-fui-gray-500 uppercase tracking-widest mt-1">Total</p>
      </div>
    </div>
  );
}

// Circular Progress Ring - FUI Style
function CircularProgress({ value, max, label, color = '#22D3EE', size = 80 }) {
  const percentage = max > 0 ? (value / max) * 100 : 0;
  const strokeWidth = 6;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.05)"
            strokeWidth={strokeWidth}
          />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1, delay: 0.3 }}
            style={{ strokeDasharray: circumference }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold text-white font-mono">{percentage.toFixed(0)}%</span>
        </div>
      </div>
      <p className="text-[10px] text-fui-gray-400 mt-2 text-center uppercase tracking-widest">{label}</p>
    </div>
  );
}

// Horizontal Bar Chart - FUI Style
function HorizontalBar({ name, value, maxValue, rank, delay = 0, color = 'cyan' }) {
  const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0;

  const colorSchemes = {
    cyan: 'bg-fui-accent-cyan',
    emerald: 'bg-fui-accent-green',
    purple: 'bg-fui-accent-purple',
    amber: 'bg-fui-accent-orange',
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.3 }}
      className="group"
    >
      <div className="flex items-center gap-2 mb-1">
        {rank && (
          <span className={`w-5 h-5 flex items-center justify-center text-[10px] font-bold shrink-0
            ${rank === 1 ? 'bg-fui-accent-orange text-black' :
              rank === 2 ? 'bg-fui-gray-400 text-black' :
                rank === 3 ? 'bg-fui-accent-orange/50 text-white' :
                  'bg-fui-gray-700 text-fui-gray-400'}`}>
            {rank}
          </span>
        )}
        <span className="text-xs text-fui-gray-400 group-hover:text-white transition-colors truncate flex-1">
          {name}
        </span>
        <span className="text-xs font-mono text-fui-accent-cyan tabular-nums shrink-0">{value}</span>
      </div>
      <div className="h-1 bg-fui-gray-700 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.6, delay: delay + 0.1 }}
          className={`h-full ${colorSchemes[color]}`}
        />
      </div>
    </motion.div>
  );
}

// Panel Component - FUI Style
function GlassPanel({ children, className = '', delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className={`bg-theme-panel border border-fui-gray-500/30 overflow-hidden ${className}`}
    >
      {children}
    </motion.div>
  );
}

// Panel Header - FUI Style
function PanelHeader({ icon: Icon, title, iconColor = 'text-fui-accent-cyan' }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3 border-b border-fui-gray-500/30">
      <div className={`p-1.5 border border-fui-gray-500/30 ${iconColor}`}>
        <Icon size={14} strokeWidth={1.5} />
      </div>
      <h3 className="text-[10px] font-normal text-fui-gray-400 uppercase tracking-widest">// {title}</h3>
    </div>
  );
}

// Severity Legend Item - FUI Style
function SeverityItem({ color, label, value, total }) {
  const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
  const colorMap = {
    'bg-red-500': 'bg-fui-accent-red',
    'bg-amber-500': 'bg-fui-accent-orange',
    'bg-emerald-500': 'bg-fui-accent-green'
  };
  return (
    <div className="text-center p-3 border border-fui-gray-500/20 hover:border-fui-gray-500/40 transition-colors">
      <div className={`w-3 h-3 ${colorMap[color] || color} mx-auto mb-2`} />
      <p className="text-[10px] text-fui-gray-500 uppercase tracking-widest">{label}</p>
      <p className="text-lg font-bold text-theme-primary mt-1 font-mono">{value.toLocaleString()}</p>
      <p className="text-[10px] text-fui-gray-600">{percentage}%</p>
    </div>
  );
}

// Main Stats Dashboard Component
export default function StatsDashboard({ compact = false, onExpand }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const data = await getDatabaseStats();
      setStats(data);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Loading State
  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-cyan-500/20 rounded-full" />
            <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-t-cyan-500 rounded-full animate-spin" />
          </div>
          <p className="text-gray-400 text-sm">Loading analytics...</p>
        </div>
      </div>
    );
  }

  // Error State
  if (error && !stats) {
    return (
      <div className="p-6 rounded-2xl bg-red-500/10 border border-red-500/30">
        <div className="flex items-center gap-3">
          <AlertCircle className="text-red-400 shrink-0" size={20} />
          <div>
            <p className="text-red-400 font-medium">Failed to load statistics</p>
            <p className="text-red-400/70 text-sm mt-1">{error}</p>
          </div>
        </div>
        <button onClick={fetchStats} className="mt-4 px-4 py-2 text-sm bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg transition-colors">
          Try again
        </button>
      </div>
    );
  }

  if (!stats) return null;

  const total = (stats.severity_distribution?.severe || 0) + (stats.severity_distribution?.moderate || 0) + (stats.severity_distribution?.minor || 0);

  // Compact Mode for Header - FUI Style matching "Connected" status box
  if (compact) {
    const coverage = stats.total_drugs ? ((stats.drugs_with_smiles / stats.total_drugs) * 100).toFixed(0) : 0;
    return (
      <div className="flex items-center gap-2">
        {/* Drugs Count */}
        <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-normal uppercase tracking-widest border border-theme text-theme-secondary">
          <Pill size={12} />
          <span className="text-theme-muted">Drugs</span>
          <span className="font-bold text-theme-primary">{stats.total_drugs?.toLocaleString()}</span>
        </div>

        {/* Interactions Count */}
        <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-normal uppercase tracking-widest border border-theme text-theme-secondary">
          <Activity size={12} />
          <span className="text-theme-muted">DDIs</span>
          <span className="font-bold text-theme-primary">{stats.total_interactions?.toLocaleString()}</span>
        </div>

        {/* Coverage */}
        <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-normal uppercase tracking-widest border border-theme text-theme-secondary">
          <Atom size={12} />
          <span className="text-theme-muted">Coverage</span>
          <span className="font-bold text-theme-primary">{coverage}%</span>
        </div>

        {onExpand && (
          <button
            onClick={onExpand}
            className="flex items-center gap-1 px-3 py-1.5 text-[10px] font-normal uppercase tracking-widest border border-fui-gray-500/30 text-fui-gray-400 hover:border-fui-accent-cyan/50 hover:text-fui-accent-cyan transition-colors"
          >
            Stats
            <ChevronRight size={12} />
          </button>
        )}
      </div>
    );
  }

  // Full Dashboard View
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="p-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 border border-fui-accent-cyan/30">
            <BarChart3 className="text-fui-accent-cyan" size={20} strokeWidth={1.5} />
          </div>
          <div>
            <h1 className="text-sm font-normal text-theme-primary uppercase tracking-widest">Database Analytics</h1>
            <p className="text-[10px] text-fui-gray-500 uppercase tracking-widest">Real-time insights</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-[10px] text-fui-gray-500 uppercase tracking-widest">
            <Clock size={10} />
            {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Never'}
          </div>
          <button
            onClick={fetchStats}
            disabled={loading}
            className="p-2 border border-fui-gray-500/30 hover:border-fui-accent-cyan/50 transition-colors disabled:opacity-50 group"
          >
            <RefreshCw size={14} className={`text-fui-gray-500 group-hover:text-fui-accent-cyan ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Main Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard icon={Pill} label="Total Drugs" value={stats.total_drugs} subtitle="In knowledge graph" color="cyan" delay={0} />
        <MetricCard icon={Activity} label="Interactions" value={stats.total_interactions} subtitle="Known drug pairs" color="emerald" delay={0.1} />
        <MetricCard icon={Atom} label="With Structures" value={stats.drugs_with_smiles} subtitle={`${((stats.drugs_with_smiles / stats.total_drugs) * 100).toFixed(1)}% coverage`} color="purple" delay={0.2} />
        <MetricCard icon={Zap} label="Today's Predictions" value={stats.recent_predictions || 0} subtitle="Last 24 hours" color="amber" delay={0.3} />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 mt-5">
        {/* Severity Distribution */}
        <GlassPanel delay={0.4}>
          <PanelHeader icon={PieChart} title="Severity Distribution" iconColor="text-fui-accent-red" />
          <div className="p-4">
            <div className="flex justify-center mb-4">
              <DonutChart data={stats.severity_distribution} />
            </div>
            <div className="grid grid-cols-3 gap-2">
              <SeverityItem color="bg-red-500" label="Severe" value={stats.severity_distribution?.severe || 0} total={total} />
              <SeverityItem color="bg-amber-500" label="Moderate" value={stats.severity_distribution?.moderate || 0} total={total} />
              <SeverityItem color="bg-emerald-500" label="Minor" value={stats.severity_distribution?.minor || 0} total={total} />
            </div>
          </div>
        </GlassPanel>

        {/* Data Coverage */}
        <GlassPanel delay={0.5}>
          <PanelHeader icon={Layers} title="Data Coverage" iconColor="text-fui-accent-purple" />
          <div className="p-4">
            <div className="flex justify-around items-center py-2">
              <CircularProgress
                value={stats.drugs_with_smiles}
                max={stats.total_drugs}
                label="SMILES"
                color="#A855F7"
              />
              <CircularProgress
                value={stats.drugs_with_descriptions}
                max={stats.total_drugs}
                label="Docs"
                color="#10B981"
              />
              <CircularProgress
                value={stats.drugs_with_classes}
                max={stats.total_drugs}
                label="Class"
                color="#F59E0B"
              />
            </div>
            <div className="mt-3 pt-3 border-t border-fui-gray-500/20">
              <div className="grid grid-cols-3 gap-3 text-center">
                <div>
                  <p className="text-sm font-bold text-fui-accent-purple font-mono">{stats.drugs_with_smiles}</p>
                  <p className="text-[10px] text-fui-gray-600 uppercase tracking-widest">Molecular</p>
                </div>
                <div>
                  <p className="text-sm font-bold text-fui-accent-green font-mono">{stats.drugs_with_descriptions}</p>
                  <p className="text-[10px] text-fui-gray-600 uppercase tracking-widest">Documented</p>
                </div>
                <div>
                  <p className="text-sm font-bold text-fui-accent-orange font-mono">{stats.drugs_with_classes}</p>
                  <p className="text-[10px] text-fui-gray-600 uppercase tracking-widest">Classified</p>
                </div>
              </div>
            </div>
          </div>
        </GlassPanel>

        {/* Therapeutic Classes */}
        <GlassPanel delay={0.6}>
          <PanelHeader icon={Beaker} title="Therapeutic Classes" iconColor="text-fui-accent-green" />
          <div className="p-4 space-y-3">
            {(stats.top_therapeutic_classes || []).slice(0, 6).map((cls, i) => (
              <HorizontalBar
                key={cls.name}
                name={cls.name}
                value={cls.count}
                maxValue={Math.max(...(stats.top_therapeutic_classes || []).map(c => c.count))}
                delay={0.6 + i * 0.05}
                color="emerald"
              />
            ))}
          </div>
        </GlassPanel>
      </div>

      {/* Top Interacting Drugs */}
      <GlassPanel delay={0.7} className="mt-3">
        <PanelHeader icon={TrendingUp} title="Most Interacting Drugs" iconColor="text-fui-accent-cyan" />
        <div className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
            {(stats.top_interacting_drugs || []).slice(0, 10).map((drug, i) => (
              <HorizontalBar
                key={drug.name}
                name={drug.name}
                value={drug.interactions}
                maxValue={Math.max(...(stats.top_interacting_drugs || []).map(d => d.interactions))}
                rank={i + 1}
                delay={0.7 + i * 0.02}
                color="cyan"
              />
            ))}
          </div>
        </div>
      </GlassPanel>

      {/* Data Sources Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="flex items-center justify-between pt-3 mt-3 border-t border-fui-gray-500/20"
      >
        <div className="flex items-center gap-2">
          <Database size={12} className="text-fui-gray-600" />
          <span className="text-[10px] text-fui-gray-600 uppercase tracking-widest">Sources:</span>
          <div className="flex flex-wrap gap-1">
            {(stats.database_sources || ['DDI Corpus 2013', 'DrugBank', 'SIDER']).map(source => (
              <span
                key={source}
                className="px-2 py-0.5 bg-black/30 text-[10px] text-fui-gray-500 border border-fui-gray-500/20"
              >
                {source}
              </span>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-fui-gray-600 uppercase tracking-widest">
          <div className="w-1.5 h-1.5 bg-fui-accent-green animate-pulse" />
          Live
        </div>
      </motion.div>
    </motion.div>
  );
}
