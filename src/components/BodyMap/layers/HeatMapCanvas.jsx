// HeatMapCanvas.jsx — Canvas 2D blurred radial gradients per affected organ
// Creates a thermal-imaging effect behind the SVG body.

import React, { useRef, useEffect } from 'react';
import { ORGAN_SYSTEMS, getSeverityColor } from '../organRegistry';

// Coordinate mapping: SVG viewBox 400×780 → canvas pixels
function mapToCanvas(svgX, svgY, canvasWidth, canvasHeight) {
  return {
    x: (svgX / 400) * canvasWidth,
    y: (svgY / 780) * canvasHeight,
  };
}

export default function HeatMapCanvas({ organs = {}, visible = true, className = '' }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!visible) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const cw = rect.width;
    const ch = rect.height;

    // Clear
    ctx.clearRect(0, 0, cw, ch);

    // Draw radial gradients for each affected organ
    for (const [key, def] of Object.entries(ORGAN_SYSTEMS)) {
      const organData = organs[key];
      const severity = organData?.severity || 0;
      if (severity <= 0) continue;

      const { x, y } = mapToCanvas(def.center.x, def.center.y, cw, ch);
      const colorInfo = getSeverityColor(severity);

      // Radius scales with severity and organ type
      const baseRadius = def.isOverlay ? cw * 0.25 : cw * 0.12;
      const radius = baseRadius * (0.6 + severity * 0.8);

      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);

      // Parse hex color for rgba
      const hex = colorInfo.fill;
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);

      gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${severity * 0.35})`);
      gradient.addColorStop(0.4, `rgba(${r}, ${g}, ${b}, ${severity * 0.15})`);
      gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, cw, ch);
    }
  }, [organs, visible]);

  if (!visible) return null;

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 w-full h-full pointer-events-none ${className}`}
      style={{ filter: 'blur(20px)', opacity: 0.7 }}
    />
  );
}
