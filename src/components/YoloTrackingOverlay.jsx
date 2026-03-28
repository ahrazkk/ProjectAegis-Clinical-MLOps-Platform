import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import * as THREE from 'three';

// ─────────────────────────────────────────────────────────────────────────────
// YOLO v8 CNN-style tracking overlay — REAL-TIME 3D → 2D projection
// Enhanced: vivid dither zones, connecting edges with data flow, corner brackets
// ─────────────────────────────────────────────────────────────────────────────

// Corner bracket SVG
function CornerBrackets({ color, size = 14, strokeWidth = 1.5 }) {
  return (
    <>
      <svg className="absolute -top-px -left-px" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path d={`M0,${size} L0,0 L${size},0`} fill="none" stroke={color} strokeWidth={strokeWidth} />
      </svg>
      <svg className="absolute -top-px -right-px" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path d={`M0,0 L${size},0 L${size},${size}`} fill="none" stroke={color} strokeWidth={strokeWidth} />
      </svg>
      <svg className="absolute -bottom-px -left-px" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path d={`M0,0 L0,${size} L${size},${size}`} fill="none" stroke={color} strokeWidth={strokeWidth} />
      </svg>
      <svg className="absolute -bottom-px -right-px" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path d={`M${size},0 L${size},${size} L0,${size}`} fill="none" stroke={color} strokeWidth={strokeWidth} />
      </svg>
    </>
  );
}

// Telemetry HUD
function TelemetryHUD() {
  const [fps, setFps] = useState(59.97);
  const [latency, setLatency] = useState(12.4);
  const [frameCount, setFrameCount] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setFps(58 + Math.random() * 3.5);
      setLatency(10 + Math.random() * 8);
      setFrameCount(p => p + 1);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 2, duration: 0.8 }}
      className="absolute top-3 right-3 sm:top-4 sm:right-4 font-mono text-[8px] sm:text-[9px] space-y-0.5 pointer-events-none select-none"
      style={{ color: 'rgba(139,92,246,0.45)' }}
    >
      <div>FPS: {fps.toFixed(2)}</div>
      <div>INF: {latency.toFixed(1)}ms</div>
      <div>FRAME: {String(frameCount).padStart(6, '0')}</div>
      <div style={{ color: 'rgba(236,72,153,0.35)' }}>MODEL: DDI-GNN-v3</div>
      <div style={{ color: 'rgba(139,92,246,0.3)' }}>CONF_THRESH: 0.45</div>
    </motion.div>
  );
}

// Status bar
function StatusBar() {
  const [objects, setObjects] = useState(3);
  useEffect(() => {
    const interval = setInterval(() => setObjects(2 + Math.floor(Math.random() * 3)), 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 2.5, duration: 0.6 }}
      className="absolute bottom-3 left-3 sm:bottom-4 sm:left-4 font-mono text-[8px] sm:text-[9px] pointer-events-none select-none"
    >
      <div className="flex items-center gap-3" style={{ color: 'rgba(139,92,246,0.35)' }}>
        <span className="flex items-center gap-1">
          <span className="w-1 h-1 rounded-full bg-purple-400 yolo-crosshair-pulse" />
          TRACKING
        </span>
        <span>{objects} OBJECTS</span>
        <span>NMS: 0.50</span>
        <span style={{ color: 'rgba(236,72,153,0.3)' }}>YOLOv8-DDI</span>
      </div>
    </motion.div>
  );
}

// Scan line
function ScanLine({ color = '#8B5CF6', speed = 5 }) {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div
        className="absolute left-0 w-full yolo-scan-line"
        style={{
          height: '1px',
          background: `linear-gradient(90deg, transparent, ${color}66, ${color}aa, ${color}66, transparent)`,
          boxShadow: `0 0 15px ${color}33`,
          animationDuration: `${speed}s`,
        }}
      />
    </div>
  );
}

// Particle motes (canvas)
function ParticleMotes({ count = 40 }) {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animFrameRef = useRef();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const resize = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resize();
    window.addEventListener('resize', resize);

    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    particlesRef.current = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25 - 0.1,
      size: Math.random() * 1.2 + 0.3,
      opacity: Math.random() * 0.3 + 0.05,
      hue: Math.random() > 0.5 ? 270 : 330,
      life: Math.random(),
      lifeSpeed: 0.002 + Math.random() * 0.003,
    }));

    const animate = () => {
      const cw = canvas.offsetWidth;
      const ch = canvas.offsetHeight;
      ctx.clearRect(0, 0, cw, ch);
      particlesRef.current.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        p.life += p.lifeSpeed;
        if (p.x < 0) p.x = cw;
        if (p.x > cw) p.x = 0;
        if (p.y < 0) p.y = ch;
        if (p.y > ch) p.y = 0;
        const alpha = p.opacity * (0.5 + 0.5 * Math.sin(p.life * Math.PI * 2));
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 80%, 65%, ${alpha})`;
        ctx.fill();
      });
      animFrameRef.current = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [count]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ opacity: 0.5 }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D → 2D Projection utility
// ─────────────────────────────────────────────────────────────────────────────
function project3Dto2D(worldPos, camera, width, height) {
  if (!camera || !worldPos) return null;
  const vec = new THREE.Vector3(worldPos[0], worldPos[1], worldPos[2]);
  vec.project(camera);
  if (vec.z > 1) return null;
  return {
    x: (vec.x * 0.5 + 0.5) * width,
    y: (-vec.y * 0.5 + 0.5) * height,
  };
}

// Detection definitions
const detectionDefs = [
  { id: 'drugA', label: 'mol_warfarin', baseConf: 0.97, color: '#A78BFA', boxSize: 180, showCrosshair: true },
  { id: 'drugB', label: 'mol_aspirin', baseConf: 0.94, color: '#EC4899', boxSize: 160, showCrosshair: true },
  { id: 'center', label: 'interaction_zone', baseConf: 0.86, color: '#818CF8', boxSize: 220, showCrosshair: false },
];

// Connection pair definitions for edge drawing
const connectionPairs = [
  { from: 'drugA', to: 'drugB', gradId: 0 },
  { from: 'drugA', to: 'center', gradId: 1 },
  { from: 'drugB', to: 'center', gradId: 2 },
];

// ─────────────────────────────────────────────────────────────────────────────
// CONNECTION LINES + DATA FLOW CANVAS — draws animated edges between detections
// ─────────────────────────────────────────────────────────────────────────────
function ConnectionCanvas({ screenPositions }) {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;

    // Initialize data flow particles
    particlesRef.current = connectionPairs.flatMap((pair, pairIdx) =>
      Array.from({ length: 5 }, (_, i) => ({
        pairIdx,
        progress: i / 5,
        speed: 0.003 + Math.random() * 0.004,
        size: 1.5 + Math.random() * 1.5,
        opacity: 0.3 + Math.random() * 0.4,
      }))
    );

    const colors = ['#A78BFA', '#EC4899', '#818CF8'];

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.offsetWidth * dpr;
      canvas.height = canvas.offsetHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener('resize', resize);

    const update = () => {
      const positions = screenPositions.current;
      if (!positions) { raf = requestAnimationFrame(update); return; }
      timeRef.current += 0.016;

      const cw = canvas.offsetWidth;
      const ch = canvas.offsetHeight;
      ctx.clearRect(0, 0, cw, ch);

      // Draw connection edges
      connectionPairs.forEach((pair, idx) => {
        const a = positions[pair.from];
        const b = positions[pair.to];
        if (!a || !b) return;

        const dist = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
        const maxDist = 600;
        const strength = Math.max(0, 1 - dist / maxDist);
        if (strength <= 0) return;

        // Main line — gradient with glow
        const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
        const c1 = colors[idx % colors.length];
        const c2 = colors[(idx + 1) % colors.length];
        grad.addColorStop(0, c1);
        grad.addColorStop(1, c2);

        // Glow layer
        ctx.save();
        ctx.globalAlpha = strength * 0.15;
        ctx.strokeStyle = grad;
        ctx.lineWidth = 8;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();

        // Primary line — dashed
        ctx.save();
        ctx.globalAlpha = strength * 0.5;
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([10, 6]);
        ctx.lineDashOffset = -timeRef.current * 40;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();

        // Secondary thinner solid line
        ctx.save();
        ctx.globalAlpha = strength * 0.2;
        ctx.strokeStyle = grad;
        ctx.lineWidth = 0.5;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();

        // Distance label at midpoint
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2;
        ctx.save();
        ctx.globalAlpha = strength * 0.35;
        ctx.fillStyle = '#A78BFA';
        ctx.font = '7px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${dist.toFixed(0)}px`, mx, my - 6);
        ctx.restore();
      });

      // Draw data flow particles along edges
      particlesRef.current.forEach(p => {
        const pair = connectionPairs[p.pairIdx];
        const a = positions[pair.from];
        const b = positions[pair.to];
        if (!a || !b) return;

        const dist = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
        const strength = Math.max(0, 1 - dist / 600);
        if (strength <= 0) return;

        p.progress += p.speed;
        if (p.progress > 1) p.progress -= 1;

        const px = a.x + (b.x - a.x) * p.progress;
        const py = a.y + (b.y - a.y) * p.progress;

        // Particle glow
        ctx.save();
        ctx.globalAlpha = p.opacity * strength;
        const color = colors[p.pairIdx % colors.length];

        // Outer glow
        const glow = ctx.createRadialGradient(px, py, 0, px, py, p.size * 4);
        glow.addColorStop(0, color + '66');
        glow.addColorStop(1, color + '00');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(px, py, p.size * 4, 0, Math.PI * 2);
        ctx.fill();

        // Core
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, p.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      // Node glow halos at each detection point
      Object.entries(positions).forEach(([id, pos]) => {
        if (!pos) return;
        const det = detectionDefs.find(d => d.id === id);
        if (!det) return;

        const pulse = 0.5 + 0.5 * Math.sin(timeRef.current * 2 + detectionDefs.indexOf(det));
        ctx.save();
        ctx.globalAlpha = 0.12 + pulse * 0.08;
        const halo = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 30);
        halo.addColorStop(0, det.color + '88');
        halo.addColorStop(1, det.color + '00');
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 30, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      raf = requestAnimationFrame(update);
    };

    raf = requestAnimationFrame(update);
    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(raf);
    };
  }, [screenPositions]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 1 }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN OVERLAY
// ─────────────────────────────────────────────────────────────────────────────
export default function YoloTrackingOverlay({ moleculePositionsRef, cameraRef, canvasContainerRef }) {
  const boxRefs = useRef({});
  const labelRefs = useRef({});
  const confidenceRefs = useRef({});
  const ditherRefs = useRef({});
  const screenPositionsRef = useRef({});
  const [isActive, setIsActive] = useState(false);

  // Stagger activation
  useEffect(() => {
    const timer = setTimeout(() => setIsActive(true), 1200);
    return () => clearTimeout(timer);
  }, []);

  // 60fps tracking loop — direct DOM manipulation, no React state
  useEffect(() => {
    if (!isActive) return;
    let raf;
    let confFlicker = {};

    const update = () => {
      if (!cameraRef?.current || !moleculePositionsRef?.current || !canvasContainerRef?.current) {
        raf = requestAnimationFrame(update);
        return;
      }

      const container = canvasContainerRef.current;
      const width = container.offsetWidth;
      const height = container.offsetHeight;
      const camera = cameraRef.current;
      const positions = moleculePositionsRef.current;

      detectionDefs.forEach(det => {
        const worldPos = positions[det.id];
        const screen = project3Dto2D(worldPos, camera, width, height);
        const boxEl = boxRefs.current[det.id];
        const confEl = confidenceRefs.current[det.id];
        const ditherEl = ditherRefs.current[det.id];

        if (!screen || !boxEl) return;

        const halfSize = det.boxSize / 2;
        // Subtle jitter for realism
        const jitterX = (Math.random() - 0.5) * 1.5;
        const jitterY = (Math.random() - 0.5) * 1.5;

        const tx = screen.x - halfSize + jitterX;
        const ty = screen.y - halfSize + jitterY;

        boxEl.style.transform = `translate(${tx}px, ${ty}px)`;
        boxEl.style.width = `${det.boxSize}px`;
        boxEl.style.height = `${det.boxSize}px`;

        // Store screen position for connection lines
        screenPositionsRef.current[det.id] = { x: screen.x, y: screen.y };

        // Dither zone — expanded area around box with stronger effect
        if (ditherEl) {
          const expand = 45;
          ditherEl.style.transform = `translate(${tx - expand}px, ${ty - expand}px)`;
          ditherEl.style.width = `${det.boxSize + expand * 2}px`;
          ditherEl.style.height = `${det.boxSize + expand * 2}px`;
        }

        // Fluctuate confidence
        if (!confFlicker[det.id] || Math.random() < 0.05) {
          confFlicker[det.id] = det.baseConf + (Math.random() - 0.5) * 0.06;
        }
        if (confEl) {
          confEl.textContent = confFlicker[det.id].toFixed(2);
        }
      });

      raf = requestAnimationFrame(update);
    };

    raf = requestAnimationFrame(update);
    return () => cancelAnimationFrame(raf);
  }, [isActive, moleculePositionsRef, cameraRef, canvasContainerRef]);

  return (
    <div className="absolute inset-0 pointer-events-none z-[2] overflow-hidden">
      {/* Dither noise — full overlay subtle */}
      <div className="absolute inset-0 yolo-dither animate-grain" style={{ opacity: 0.025 }} />

      {/* Particles */}
      <ParticleMotes count={35} />

      {/* Scan lines */}
      <ScanLine color="#8B5CF6" speed={5} />
      <ScanLine color="#EC4899" speed={8} />

      {/* Connection edges with data flow particles — canvas-based */}
      {isActive && <ConnectionCanvas screenPositions={screenPositionsRef} />}

      {/* Dither zones — vivid noise around each detection */}
      {isActive && detectionDefs.map(det => (
        <div
          key={`dither-${det.id}`}
          ref={el => { ditherRefs.current[det.id] = el; }}
          className="absolute top-0 left-0 pointer-events-none yolo-dither-zone"
          style={{
            willChange: 'transform',
            opacity: 0.12,
            borderRadius: '4px',
          }}
        />
      ))}

      {/* Real-time tracked bounding boxes */}
      {isActive && detectionDefs.map(det => (
        <div
          key={det.id}
          ref={el => { boxRefs.current[det.id] = el; }}
          className="absolute top-0 left-0 pointer-events-none"
          style={{ willChange: 'transform' }}
        >
          {/* Border + glow */}
          <div
            className="absolute inset-0 yolo-box-pulse"
            style={{
              border: `1px solid ${det.color}88`,
              boxShadow: `0 0 20px ${det.color}22, inset 0 0 20px ${det.color}08`,
            }}
          >
            <CornerBrackets color={det.color} size={12} strokeWidth={1.5} />

            {/* Inner scan lines for depth */}
            <div
              className="absolute inset-0 opacity-[0.05]"
              style={{
                backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 3px, ${det.color} 3px, ${det.color} 4px)`,
              }}
            />
          </div>

          {/* Label tag with improved design */}
          <div
            ref={el => { labelRefs.current[det.id] = el; }}
            className="absolute -top-6 left-0 flex items-center gap-1.5 px-2.5 py-1 font-mono text-[8px] tracking-wider rounded-sm"
            style={{
              background: `linear-gradient(135deg, ${det.color}dd, ${det.color}99)`,
              color: '#fff',
              boxShadow: `0 0 16px ${det.color}44, 0 2px 8px rgba(0,0,0,0.3)`,
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-white/50 animate-pulse" />
            <span className="uppercase font-semibold">{det.label}</span>
            <span ref={el => { confidenceRefs.current[det.id] = el; }} className="opacity-70 ml-1 font-normal">
              {det.baseConf.toFixed(2)}
            </span>
          </div>

          {/* Crosshair */}
          {det.showCrosshair && (
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
              <div
                className="w-2 h-2 rounded-full yolo-crosshair-pulse"
                style={{ backgroundColor: det.color, boxShadow: `0 0 12px ${det.color}` }}
              />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-px" style={{ backgroundColor: `${det.color}44` }} />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-px h-6" style={{ backgroundColor: `${det.color}44` }} />
            </div>
          )}

          {/* Data readout below box */}
          <div
            className="absolute -bottom-4 left-0 font-mono text-[7px] tracking-wider"
            style={{ color: `${det.color}55` }}
          >
            {det.id === 'drugA' && 'MW:308.33 | C₁₉H₁₆O₄'}
            {det.id === 'drugB' && 'MW:180.16 | C₉H₈O₄'}
            {det.id === 'center' && 'ΔG:-8.2kcal/mol'}
          </div>
        </div>
      ))}

      {/* HUD */}
      <TelemetryHUD />
      <StatusBar />

      {/* Vignette */}
      <div
        className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at center, transparent 40%, rgba(5,5,6,0.5) 100%)' }}
      />
    </div>
  );
}
