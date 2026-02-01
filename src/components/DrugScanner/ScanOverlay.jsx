/**
 * ScanOverlay Component
 * Visual overlay for the camera view showing scan regions
 */

import React from 'react';
import { motion } from 'framer-motion';

export function ScanOverlay({ mode, isScanning }) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Corner brackets */}
      <div className="absolute inset-8 md:inset-16">
        {/* Top Left */}
        <div className="absolute top-0 left-0 w-16 h-16">
          <div className={`absolute top-0 left-0 w-full h-1 ${isScanning ? 'bg-cyan-400' : 'bg-white/50'} transition-colors`} />
          <div className={`absolute top-0 left-0 w-1 h-full ${isScanning ? 'bg-cyan-400' : 'bg-white/50'} transition-colors`} />
        </div>
        
        {/* Top Right */}
        <div className="absolute top-0 right-0 w-16 h-16">
          <div className={`absolute top-0 right-0 w-full h-1 ${isScanning ? 'bg-cyan-400' : 'bg-white/50'} transition-colors`} />
          <div className={`absolute top-0 right-0 w-1 h-full ${isScanning ? 'bg-cyan-400' : 'bg-white/50'} transition-colors`} />
        </div>
        
        {/* Bottom Left */}
        <div className="absolute bottom-0 left-0 w-16 h-16">
          <div className={`absolute bottom-0 left-0 w-full h-1 ${isScanning ? 'bg-cyan-400' : 'bg-white/50'} transition-colors`} />
          <div className={`absolute bottom-0 left-0 w-1 h-full ${isScanning ? 'bg-cyan-400' : 'bg-white/50'} transition-colors`} />
        </div>
        
        {/* Bottom Right */}
        <div className="absolute bottom-0 right-0 w-16 h-16">
          <div className={`absolute bottom-0 right-0 w-full h-1 ${isScanning ? 'bg-cyan-400' : 'bg-white/50'} transition-colors`} />
          <div className={`absolute bottom-0 right-0 w-1 h-full ${isScanning ? 'bg-cyan-400' : 'bg-white/50'} transition-colors`} />
        </div>
      </div>

      {/* Scanning line animation for barcode mode */}
      {mode === 'barcode' && (
        <motion.div
          className="absolute left-8 right-8 md:left-16 md:right-16 h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
          animate={{
            top: ['30%', '70%', '30%']
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
      )}

      {/* Center crosshair for pill mode */}
      {mode === 'pill' && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative w-48 h-48">
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-dashed border-cyan-400/50"
              animate={{ rotate: 360 }}
              transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
            />
            <div className="absolute inset-4 rounded-full border border-cyan-400/30" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-4 h-0.5 bg-cyan-400/50" />
              <div className="absolute w-0.5 h-4 bg-cyan-400/50" />
            </div>
          </div>
        </div>
      )}

      {/* Text region highlight for OCR mode */}
      {mode === 'ocr' && (
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            className="w-80 h-24 border-2 border-dashed border-cyan-400/50 rounded-lg"
            animate={{
              borderColor: ['rgba(0,212,255,0.3)', 'rgba(0,212,255,0.7)', 'rgba(0,212,255,0.3)']
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
          />
        </div>
      )}

      {/* Mode indicator */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2">
        <div className="px-4 py-2 rounded-full bg-black/50 backdrop-blur-sm border border-white/20">
          <p className="text-white text-sm font-medium">
            {mode === 'auto' && '🔄 Auto Detect'}
            {mode === 'barcode' && '📊 Barcode Mode'}
            {mode === 'ocr' && '📝 Label Mode'}
            {mode === 'pill' && '💊 Pill Mode'}
          </p>
        </div>
      </div>
    </div>
  );
}

export default ScanOverlay;
