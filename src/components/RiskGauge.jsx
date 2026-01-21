import React from 'react';
import { motion } from 'framer-motion';

export default function RiskGauge({ score, riskLevel }) {
    // Score is 0-1
    const percentage = Math.round(score * 100);

    // Color logic based on risk level
    const getColors = (level) => {
        switch (level) {
            case 'critical': return { main: '#EF4444', gradient: ['#EF4444', '#B91C1C'], bg: 'rgba(239, 68, 68, 0.1)' };
            case 'high': return { main: '#F97316', gradient: ['#F97316', '#C2410C'], bg: 'rgba(249, 115, 22, 0.1)' };
            case 'medium': return { main: '#EAB308', gradient: ['#EAB308', '#A16207'], bg: 'rgba(234, 179, 8, 0.1)' };
            default: return { main: '#22C55E', gradient: ['#22C55E', '#15803D'], bg: 'rgba(34, 197, 94, 0.1)' };
        }
    };

    const colors = getColors(riskLevel);
    const radius = 80;
    const strokeWidth = 12;
    const circumference = 2 * Math.PI * radius;
    // Arc length (semi-circle + a bit more? let's do 240 degrees)
    // 240 degrees = 2/3 circle. 
    // SVG dasharray: fill, gap.

    // Let's do a standard circular gauge cut at bottom
    // Start angle 135, End angle 405 (270 degrees total)

    return (
        <div className="relative w-full flex flex-col items-center justify-center p-6 bg-white/5 rounded-2xl border border-white/5 backdrop-blur-sm">
            <div className="relative w-48 h-48">
                {/* SVG Container */}
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
                    {/* Background Track */}
                    <circle
                        cx="100"
                        cy="100"
                        r={radius}
                        fill="none"
                        stroke="#1e293b"
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                    />

                    {/* Animated Progress Path */}
                    <motion.circle
                        cx="100"
                        cy="100"
                        r={radius}
                        fill="none"
                        stroke={`url(#gradient-${riskLevel})`}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        initial={{ strokeDashoffset: circumference }}
                        animate={{ strokeDashoffset: circumference - (percentage / 100) * circumference }}
                        transition={{ duration: 1.5, ease: "easeOut" }}
                    />

                    {/* Gradients */}
                    <defs>
                        <linearGradient id={`gradient-${riskLevel}`} x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor={colors.gradient[1]} />
                            <stop offset="100%" stopColor={colors.gradient[0]} />
                        </linearGradient>
                        <filter id={`glow-${riskLevel}`}>
                            <feGaussianBlur stdDeviation="3.5" result="coloredBlur" />
                            <feMerge>
                                <feMergeNode in="coloredBlur" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                    </defs>
                </svg>

                {/* Glow Overlay Circle */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
                        <motion.circle
                            cx="100"
                            cy="100"
                            r={radius}
                            fill="none"
                            stroke={colors.main}
                            strokeWidth={strokeWidth}
                            strokeLinecap="round"
                            strokeDasharray={circumference}
                            initial={{ strokeDashoffset: circumference, opacity: 0 }}
                            animate={{
                                strokeDashoffset: circumference - (percentage / 100) * circumference,
                                opacity: [0.3, 0.6, 0.3]
                            }}
                            transition={{ duration: 2, ease: "easeInOut", repeat: Infinity }}
                            filter={`blur(8px)`}
                        />
                    </svg>
                </div>

                {/* Center Text */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.5 }}
                        className="flex flex-col items-center"
                    >
                        <span className="text-4xl font-bold text-white tracking-tight">
                            {percentage}%
                        </span>
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full mt-2 capitalize bg-white/5 border border-white/10 ${riskLevel === 'critical' ? 'text-red-400' :
                                riskLevel === 'high' ? 'text-orange-400' :
                                    riskLevel === 'medium' ? 'text-yellow-400' : 'text-emerald-400'
                            }`}>
                            {riskLevel} Risk
                        </span>
                    </motion.div>
                </div>
            </div>

            {/* Decorative Elements */}
            <div className="w-full mt-2 flex justify-between text-[10px] text-slate-500 font-medium px-6">
                <span>0%</span>
                <span>100%</span>
            </div>
        </div>
    );
}
