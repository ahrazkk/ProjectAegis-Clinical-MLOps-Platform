import React from 'react';
import { motion } from 'framer-motion';

export default function RiskGauge({ score, riskLevel }) {
    // Score is 0-1
    const percentage = Math.round(score * 100);

    // FUI Blueprint color logic based on risk level
    const getColors = (level) => {
        switch (level) {
            case 'critical': return { main: '#FF0044', gradient: ['#FF0044', '#CC0033'], glow: 'rgba(255, 0, 68, 0.3)' };
            case 'high': return { main: '#FF4444', gradient: ['#FF4444', '#CC3333'], glow: 'rgba(255, 68, 68, 0.3)' };
            case 'medium': return { main: '#FFAA00', gradient: ['#FFAA00', '#CC8800'], glow: 'rgba(255, 170, 0, 0.3)' };
            default: return { main: '#00FF88', gradient: ['#00FF88', '#00CC66'], glow: 'rgba(0, 255, 136, 0.3)' };
        }
    };

    const colors = getColors(riskLevel);
    const radius = 80;
    const strokeWidth = 2;
    const circumference = 2 * Math.PI * radius;

    return (
        <div className="relative w-full flex flex-col items-center justify-center p-6 border border-theme bg-theme-panel relative">
            {/* Corner markers */}
            <div className="absolute -top-px -left-px w-3 h-3 border-t border-l border-fui-gray-500"></div>
            <div className="absolute -top-px -right-px w-3 h-3 border-t border-r border-fui-gray-500"></div>
            <div className="absolute -bottom-px -left-px w-3 h-3 border-b border-l border-fui-gray-500"></div>
            <div className="absolute -bottom-px -right-px w-3 h-3 border-b border-r border-fui-gray-500"></div>

            <div className="relative w-48 h-48">
                {/* SVG Container */}
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
                    {/* Background Track - Blueprint grid style */}
                    <circle
                        cx="100"
                        cy="100"
                        r={radius}
                        fill="none"
                        stroke="rgba(102, 102, 102, 0.3)"
                        strokeWidth={strokeWidth}
                        strokeDasharray="4 4"
                    />

                    {/* Grid circles for FUI effect */}
                    <circle cx="100" cy="100" r={radius * 0.6} fill="none" stroke="rgba(102, 102, 102, 0.15)" strokeWidth="1" />
                    <circle cx="100" cy="100" r={radius * 0.3} fill="none" stroke="rgba(102, 102, 102, 0.15)" strokeWidth="1" />

                    {/* Animated Progress Path */}
                    <motion.circle
                        cx="100"
                        cy="100"
                        r={radius}
                        fill="none"
                        stroke={colors.main}
                        strokeWidth={strokeWidth + 1}
                        strokeLinecap="square"
                        strokeDasharray={circumference}
                        initial={{ strokeDashoffset: circumference }}
                        animate={{ strokeDashoffset: circumference - (percentage / 100) * circumference }}
                        transition={{ duration: 1.5, ease: "easeOut" }}
                        style={{ filter: `drop-shadow(0 0 6px ${colors.glow})` }}
                    />

                    {/* Tick marks */}
                    {[0, 25, 50, 75, 100].map((tick, i) => {
                        const angle = (tick / 100) * 360 - 90;
                        const rad = angle * Math.PI / 180;
                        const x1 = 100 + (radius - 10) * Math.cos(rad);
                        const y1 = 100 + (radius - 10) * Math.sin(rad);
                        const x2 = 100 + (radius + 5) * Math.cos(rad);
                        const y2 = 100 + (radius + 5) * Math.sin(rad);
                        return (
                            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(102, 102, 102, 0.5)" strokeWidth="1" />
                        );
                    })}
                </svg>

                {/* Center Text */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.5 }}
                        className="flex flex-col items-center"
                    >
                        <span
                            className="text-3xl font-light tracking-wider"
                            style={{ color: colors.main, textShadow: `0 0 20px ${colors.glow}` }}
                        >
                            {percentage}%
                        </span>
                        <span
                            className={`text-[10px] font-normal px-3 py-1 mt-2 uppercase tracking-widest border`}
                            style={{ color: colors.main, borderColor: `${colors.main}50` }}
                        >
                            {riskLevel} Risk
                        </span>
                    </motion.div>
                </div>
            </div>

            {/* Decorative Elements */}
            <div className="w-full mt-4 flex justify-between text-[10px] text-fui-gray-500 font-normal px-6 uppercase tracking-widest">
                <span>0%</span>
                <span>100%</span>
            </div>
        </div>
    );
}
