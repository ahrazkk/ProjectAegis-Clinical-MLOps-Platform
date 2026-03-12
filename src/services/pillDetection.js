/**
 * Pill Detection Service for Project Aegis
 * 
 * Multi-stage pill identification pipeline:
 * 1. MobileNet-based pill classifier (TF.js) - when trained model available
 * 2. Computer vision fallback - shape, color, imprint analysis
 * 3. Backend API integration for pill ID lookup
 * 
 * @author Project Aegis
 */

import * as tf from '@tensorflow/tfjs';

// API base URL
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Known pill shapes for classification
const PILL_SHAPES = [
  'round', 'oval', 'capsule', 'oblong', 'rectangle',
  'square', 'diamond', 'triangle', 'pentagon', 'hexagon'
];

// Known pill colors
const PILL_COLORS = {
  white: { r: 240, g: 240, b: 240, threshold: 50 },
  pink: { r: 255, g: 182, b: 193, threshold: 60 },
  red: { r: 200, g: 30, b: 30, threshold: 60 },
  orange: { r: 255, g: 140, b: 0, threshold: 55 },
  yellow: { r: 240, g: 230, b: 50, threshold: 55 },
  green: { r: 60, g: 160, b: 60, threshold: 65 },
  blue: { r: 40, g: 80, b: 200, threshold: 65 },
  purple: { r: 130, g: 50, b: 160, threshold: 60 },
  brown: { r: 139, g: 90, b: 43, threshold: 55 },
  gray: { r: 155, g: 155, b: 155, threshold: 45 },
  tan: { r: 210, g: 180, b: 140, threshold: 50 },
  turquoise: { r: 64, g: 224, b: 208, threshold: 65 },
  maroon: { r: 128, g: 0, b: 0, threshold: 55 },
  black: { r: 30, g: 30, b: 30, threshold: 50 }
};

/**
 * PillDetectionModel - Manages TF.js model loading and inference
 */
class PillDetectionModel {
  constructor() {
    this.model = null;
    this.isLoaded = false;
    this.isLoading = false;
    this.labels = [];
  }

  /**
   * Try to load a trained pill detection model.
   * Falls back to computer vision if no model is available.
   */
  async loadModel() {
    if (this.isLoaded || this.isLoading) return this.isLoaded;
    this.isLoading = true;

    try {
      // Try loading from public/models/pill-classifier/
      const modelUrl = '/models/pill-classifier/model.json';
      this.model = await tf.loadLayersModel(modelUrl);

      // Load labels
      const labelsResponse = await fetch('/models/pill-classifier/labels.json');
      if (labelsResponse.ok) {
        this.labels = await labelsResponse.json();
      }

      this.isLoaded = true;
      console.log('Pill detection model loaded successfully');
    } catch {
      console.info('No trained pill model found, using CV-based detection');
      this.isLoaded = false;
    }

    this.isLoading = false;
    return this.isLoaded;
  }

  /**
   * Run model inference on a preprocessed image tensor.
   * Returns top-k predictions with confidence.
   */
  async predict(imageTensor, topK = 5) {
    if (!this.model) return null;

    const predictions = tf.tidy(() => {
      // Resize to model input size (224x224 for MobileNet-based)
      const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
      const normalized = resized.div(255.0);
      const batched = normalized.expandDims(0);
      return this.model.predict(batched);
    });

    const scores = await predictions.data();
    predictions.dispose();

    // Get top-k results
    const indexed = Array.from(scores).map((score, i) => ({ score, index: i }));
    indexed.sort((a, b) => b.score - a.score);

    return indexed.slice(0, topK).map(({ score, index }) => ({
      label: this.labels[index] || `class_${index}`,
      confidence: score,
      index
    }));
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isLoaded = false;
    }
  }
}

// Singleton instance
export const pillModel = new PillDetectionModel();

// ============================================================
// Computer Vision Pipeline (works without trained model)
// ============================================================

/**
 * Full pill analysis pipeline.
 * Extracts color, shape, and imprint from a pill image.
 */
export async function analyzePill(imageSrc) {
  const imageData = await loadImageToCanvas(imageSrc, 300, 300);
  const { canvas, ctx, width, height } = imageData;

  // Step 1: Segment the pill from background
  const mask = segmentPill(ctx, width, height);

  // Step 2: Extract dominant colors (multi-region sampling)
  const colorResult = analyzeColor(ctx, width, height, mask);

  // Step 3: Detect shape from contour
  const shapeResult = analyzeShape(mask, width, height);

  // Step 4: Extract imprint text via preprocessing + OCR region
  const imprintRegion = extractImprintRegion(ctx, width, height, mask);

  // Step 5: Try TF.js model if available
  let modelPrediction = null;
  if (pillModel.isLoaded) {
    const tensor = tf.browser.fromPixels(canvas);
    modelPrediction = await pillModel.predict(tensor);
    tensor.dispose();
  }

  return {
    color: colorResult.primaryColor,
    colorSecondary: colorResult.secondaryColor,
    colorRGB: colorResult.rgb,
    colorConfidence: colorResult.confidence,
    shape: shapeResult.shape,
    shapeConfidence: shapeResult.confidence,
    aspectRatio: shapeResult.aspectRatio,
    imprint: null, // Will be filled by OCR step in the hook
    imprintRegionDataUrl: imprintRegion,
    modelPrediction,
    pillArea: mask.filter(v => v > 0).length / mask.length,
    features: {
      colorHSL: colorResult.hsl,
      contourPoints: shapeResult.contourSample,
      circularity: shapeResult.circularity,
      corners: shapeResult.corners
    }
  };
}

/**
 * Load an image source into a canvas and return the drawing context.
 */
function loadImageToCanvas(imageSrc, targetW, targetH) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, targetW, targetH);
      resolve({ canvas, ctx, width: targetW, height: targetH });
    };
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = imageSrc;
  });
}

// ============================================================
// Step 1: Pill Segmentation (Otsu + morphological cleanup)
// ============================================================

/**
 * Segment the pill from background using adaptive thresholding.
 * Returns a binary mask (Uint8Array, 0 = bg, 255 = pill).
 */
function segmentPill(ctx, w, h) {
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  const gray = new Uint8Array(w * h);

  // Convert to grayscale
  for (let i = 0; i < gray.length; i++) {
    const idx = i * 4;
    gray[i] = Math.round(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]);
  }

  // Compute Otsu threshold
  const threshold = otsuThreshold(gray);

  // Create binary mask - pills are usually lighter than background
  const mask = new Uint8Array(w * h);
  const histogram = [0, 0]; // [dark, light]
  for (let i = 0; i < gray.length; i++) {
    if (gray[i] > threshold) {
      histogram[1]++;
    } else {
      histogram[0]++;
    }
  }

  // Assume pill is the region with less area (not background)
  const pillIsLight = histogram[1] < histogram[0];
  for (let i = 0; i < gray.length; i++) {
    mask[i] = pillIsLight ? (gray[i] > threshold ? 255 : 0) : (gray[i] <= threshold ? 255 : 0);
  }

  // Morphological opening to remove noise (erode then dilate)
  const opened = morphOpen(mask, w, h, 3);

  // Keep only the largest connected component
  return keepLargestComponent(opened, w, h);
}

/**
 * Otsu's thresholding method for automatic binarization.
 */
function otsuThreshold(gray) {
  const histogram = new Array(256).fill(0);
  for (const val of gray) histogram[val]++;

  const total = gray.length;
  let sumTotal = 0;
  for (let i = 0; i < 256; i++) sumTotal += i * histogram[i];

  let sumBg = 0, weightBg = 0;
  let maxVariance = 0, threshold = 0;

  for (let t = 0; t < 256; t++) {
    weightBg += histogram[t];
    if (weightBg === 0) continue;

    const weightFg = total - weightBg;
    if (weightFg === 0) break;

    sumBg += t * histogram[t];
    const meanBg = sumBg / weightBg;
    const meanFg = (sumTotal - sumBg) / weightFg;

    const variance = weightBg * weightFg * (meanBg - meanFg) ** 2;
    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = t;
    }
  }

  return threshold;
}

/**
 * Morphological opening (erode then dilate).
 */
function morphOpen(mask, w, h, kernelSize) {
  const eroded = morphErode(mask, w, h, kernelSize);
  return morphDilate(eroded, w, h, kernelSize);
}

function morphErode(mask, w, h, k) {
  const result = new Uint8Array(w * h);
  const half = Math.floor(k / 2);
  for (let y = half; y < h - half; y++) {
    for (let x = half; x < w - half; x++) {
      let allSet = true;
      for (let dy = -half; dy <= half && allSet; dy++) {
        for (let dx = -half; dx <= half && allSet; dx++) {
          if (mask[(y + dy) * w + (x + dx)] === 0) allSet = false;
        }
      }
      result[y * w + x] = allSet ? 255 : 0;
    }
  }
  return result;
}

function morphDilate(mask, w, h, k) {
  const result = new Uint8Array(w * h);
  const half = Math.floor(k / 2);
  for (let y = half; y < h - half; y++) {
    for (let x = half; x < w - half; x++) {
      let anySet = false;
      for (let dy = -half; dy <= half && !anySet; dy++) {
        for (let dx = -half; dx <= half && !anySet; dx++) {
          if (mask[(y + dy) * w + (x + dx)] > 0) anySet = true;
        }
      }
      result[y * w + x] = anySet ? 255 : 0;
    }
  }
  return result;
}

/**
 * Keep only the largest connected component (flood fill based).
 */
function keepLargestComponent(mask, w, h) {
  const labels = new Int32Array(w * h);
  let currentLabel = 0;
  const componentSizes = new Map();

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (mask[idx] > 0 && labels[idx] === 0) {
        currentLabel++;
        let size = 0;
        const stack = [[x, y]];

        while (stack.length > 0) {
          const [cx, cy] = stack.pop();
          const cidx = cy * w + cx;
          if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
          if (mask[cidx] === 0 || labels[cidx] !== 0) continue;

          labels[cidx] = currentLabel;
          size++;
          stack.push([cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]);
        }

        componentSizes.set(currentLabel, size);
      }
    }
  }

  // Find largest component
  let largestLabel = 0, largestSize = 0;
  for (const [label, size] of componentSizes) {
    if (size > largestSize) {
      largestSize = size;
      largestLabel = label;
    }
  }

  // Keep only largest
  const result = new Uint8Array(w * h);
  for (let i = 0; i < labels.length; i++) {
    result[i] = labels[i] === largestLabel ? 255 : 0;
  }
  return result;
}

// ============================================================
// Step 2: Color Analysis (multi-region + HSL classification)
// ============================================================

/**
 * Analyze pill color using the segmented mask region.
 * Uses multi-region sampling and K-means-like clustering for multi-color pills.
 */
function analyzeColor(ctx, w, h, mask) {
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;

  // Collect pill pixels
  const pixels = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0) {
      const idx = i * 4;
      pixels.push([data[idx], data[idx + 1], data[idx + 2]]);
    }
  }

  if (pixels.length === 0) {
    return { primaryColor: 'unknown', secondaryColor: null, rgb: { r: 128, g: 128, b: 128 }, hsl: { h: 0, s: 0, l: 50 }, confidence: 0 };
  }

  // Compute mean RGB
  let rSum = 0, gSum = 0, bSum = 0;
  for (const [r, g, b] of pixels) {
    rSum += r;
    gSum += g;
    bSum += b;
  }
  const avgR = Math.round(rSum / pixels.length);
  const avgG = Math.round(gSum / pixels.length);
  const avgB = Math.round(bSum / pixels.length);

  // Simple 2-cluster split for two-tone pills
  let secondaryColor = null;
  if (pixels.length > 100) {
    const midY = Math.floor(h / 2);
    const topPixels = [], bottomPixels = [];
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] > 0) {
        const y = Math.floor(i / w);
        const idx = i * 4;
        if (y < midY) topPixels.push([data[idx], data[idx + 1], data[idx + 2]]);
        else bottomPixels.push([data[idx], data[idx + 1], data[idx + 2]]);
      }
    }

    if (topPixels.length > 20 && bottomPixels.length > 20) {
      const topAvg = averagePixels(topPixels);
      const bottomAvg = averagePixels(bottomPixels);
      const colorDist = Math.sqrt(
        (topAvg[0] - bottomAvg[0]) ** 2 +
        (topAvg[1] - bottomAvg[1]) ** 2 +
        (topAvg[2] - bottomAvg[2]) ** 2
      );

      if (colorDist > 60) {
        secondaryColor = classifyColorRGB(bottomAvg[0], bottomAvg[1], bottomAvg[2]);
      }
    }
  }

  const primaryColor = classifyColorRGB(avgR, avgG, avgB);
  const hsl = rgbToHsl(avgR, avgG, avgB);

  return {
    primaryColor,
    secondaryColor,
    rgb: { r: avgR, g: avgG, b: avgB },
    hsl,
    confidence: pixels.length > 500 ? 0.9 : pixels.length > 100 ? 0.7 : 0.4
  };
}

function averagePixels(pixels) {
  let r = 0, g = 0, b = 0;
  for (const [pr, pg, pb] of pixels) { r += pr; g += pg; b += pb; }
  return [Math.round(r / pixels.length), Math.round(g / pixels.length), Math.round(b / pixels.length)];
}

/**
 * Classify RGB to the closest named pill color using perceptual distance.
 */
function classifyColorRGB(r, g, b) {
  let bestColor = 'unknown';
  let minDist = Infinity;

  for (const [name, ref] of Object.entries(PILL_COLORS)) {
    const dist = Math.sqrt((r - ref.r) ** 2 + (g - ref.g) ** 2 + (b - ref.b) ** 2);
    if (dist < minDist) {
      minDist = dist;
      bestColor = name;
    }
  }

  return bestColor;
}

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, l = (max + min) / 2;

  if (max === min) {
    h = s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      default: h = ((r - g) / d + 4) / 6; break;
    }
  }

  return { h: Math.round(h * 360), s: Math.round(s * 100), l: Math.round(l * 100) };
}

// ============================================================
// Step 3: Shape Detection (contour analysis)
// ============================================================

/**
 * Determine pill shape from the binary mask.
 * Uses contour following + shape descriptors.
 */
function analyzeShape(mask, w, h) {
  // Extract contour points
  const contour = extractContour(mask, w, h);

  if (contour.length < 10) {
    return { shape: 'unknown', confidence: 0, aspectRatio: 1, circularity: 0, corners: 0, contourSample: [] };
  }

  // Compute bounding box
  let minX = w, maxX = 0, minY = h, maxY = 0;
  for (const [x, y] of contour) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  const bboxW = maxX - minX + 1;
  const bboxH = maxY - minY + 1;
  const aspectRatio = Math.max(bboxW, bboxH) / Math.max(Math.min(bboxW, bboxH), 1);

  // Compute area & perimeter
  const area = mask.filter(v => v > 0).length;
  const perimeter = contour.length;

  // Circularity = 4π * area / perimeter²
  const circularity = (4 * Math.PI * area) / (perimeter * perimeter);

  // Detect corners using angle-based method
  const corners = detectCorners(contour, 15);

  // Classify shape based on features
  const shape = classifyShape(aspectRatio, circularity, corners.length);

  // Downsample contour for transport
  const step = Math.max(1, Math.floor(contour.length / 50));
  const contourSample = contour.filter((_, i) => i % step === 0);

  return {
    shape: shape.name,
    confidence: shape.confidence,
    aspectRatio: Math.round(aspectRatio * 100) / 100,
    circularity: Math.round(circularity * 1000) / 1000,
    corners: corners.length,
    contourSample
  };
}

/**
 * Extract boundary contour from binary mask using Moore neighborhood tracing.
 */
function extractContour(mask, w, h) {
  const contour = [];

  // Find starting point (first foreground pixel)
  let startX = -1, startY = -1;
  for (let y = 0; y < h && startX < 0; y++) {
    for (let x = 0; x < w && startX < 0; x++) {
      if (mask[y * w + x] > 0) {
        // Check if it's a boundary pixel (has at least one bg neighbor)
        const isBoundary = (
          x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
          mask[y * w + (x - 1)] === 0 || mask[y * w + (x + 1)] === 0 ||
          mask[(y - 1) * w + x] === 0 || mask[(y + 1) * w + x] === 0
        );
        if (isBoundary) {
          startX = x;
          startY = y;
        }
      }
    }
  }

  if (startX < 0) return contour;

  // Moore neighborhood tracing
  const dx = [1, 1, 0, -1, -1, -1, 0, 1];
  const dy = [0, 1, 1, 1, 0, -1, -1, -1];

  let x = startX, y = startY;
  let dir = 7; // Start direction
  const visited = new Set();
  const maxIter = w * h;

  for (let iter = 0; iter < maxIter; iter++) {
    const key = `${x},${y}`;
    if (contour.length > 2 && x === startX && y === startY) break;
    if (visited.has(key) && contour.length > 10) break;

    contour.push([x, y]);
    visited.add(key);

    // Search counterclockwise starting from (dir + 5) % 8
    let found = false;
    let startDir = (dir + 5) % 8;

    for (let i = 0; i < 8; i++) {
      const d = (startDir + i) % 8;
      const nx = x + dx[d], ny = y + dy[d];

      if (nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny * w + nx] > 0) {
        x = nx;
        y = ny;
        dir = d;
        found = true;
        break;
      }
    }

    if (!found) break;
  }

  return contour;
}

/**
 * Detect corners on a contour using angle changes.
 */
function detectCorners(contour, windowSize) {
  if (contour.length < windowSize * 2) return [];

  const corners = [];
  const n = contour.length;

  for (let i = 0; i < n; i++) {
    const prev = contour[(i - windowSize + n) % n];
    const curr = contour[i];
    const next = contour[(i + windowSize) % n];

    const v1 = [prev[0] - curr[0], prev[1] - curr[1]];
    const v2 = [next[0] - curr[0], next[1] - curr[1]];

    const dot = v1[0] * v2[0] + v1[1] * v2[1];
    const mag1 = Math.sqrt(v1[0] ** 2 + v1[1] ** 2);
    const mag2 = Math.sqrt(v2[0] ** 2 + v2[1] ** 2);

    if (mag1 > 0 && mag2 > 0) {
      const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
      const angle = Math.acos(cosAngle) * (180 / Math.PI);

      // Angle below 120° indicates a corner
      if (angle < 120) {
        // Ensure minimum distance between corners
        const tooClose = corners.some(c => {
          const dist = Math.sqrt((c[0] - curr[0]) ** 2 + (c[1] - curr[1]) ** 2);
          return dist < windowSize * 2;
        });
        if (!tooClose) {
          corners.push(curr);
        }
      }
    }
  }

  return corners;
}

/**
 * Classify shape based on computed features.
 */
function classifyShape(aspectRatio, circularity, cornerCount) {
  // Round: high circularity, aspect ratio near 1
  if (circularity > 0.8 && aspectRatio < 1.2) {
    return { name: 'round', confidence: Math.min(circularity, 0.95) };
  }

  // Oval: moderate circularity, aspect ratio 1.2-2.0
  if (circularity > 0.6 && aspectRatio >= 1.2 && aspectRatio < 2.0) {
    return { name: 'oval', confidence: 0.8 };
  }

  // Capsule: high aspect ratio with moderate circularity (rounded ends)
  if (aspectRatio >= 2.0 && circularity > 0.5) {
    return { name: 'capsule', confidence: 0.85 };
  }

  // Oblong: high aspect ratio, lower circularity
  if (aspectRatio >= 1.8 && circularity <= 0.5) {
    return { name: 'oblong', confidence: 0.7 };
  }

  // Rectangle: 4 corners, low circularity
  if (cornerCount === 4 && circularity < 0.8) {
    if (aspectRatio < 1.3) return { name: 'square', confidence: 0.7 };
    return { name: 'rectangle', confidence: 0.75 };
  }

  // Diamond: 4 corners, rotated
  if (cornerCount === 4 && circularity > 0.5 && circularity < 0.8) {
    return { name: 'diamond', confidence: 0.6 };
  }

  // Triangle: 3 corners
  if (cornerCount === 3) {
    return { name: 'triangle', confidence: 0.65 };
  }

  // Pentagon: 5 corners
  if (cornerCount === 5) {
    return { name: 'pentagon', confidence: 0.6 };
  }

  // Hexagon: 6 corners
  if (cornerCount === 6) {
    return { name: 'hexagon', confidence: 0.6 };
  }

  // Default: classify by aspect ratio
  if (aspectRatio < 1.3) return { name: 'round', confidence: 0.5 };
  if (aspectRatio < 2.0) return { name: 'oval', confidence: 0.5 };
  return { name: 'capsule', confidence: 0.4 };
}

// ============================================================
// Step 4: Imprint Region Extraction
// ============================================================

/**
 * Extract and preprocess the pill surface for imprint/text OCR.
 * Returns a data URL of the preprocessed imprint region.
 */
function extractImprintRegion(ctx, w, h, mask) {
  // Get the bounding box of the pill in the mask
  let minX = w, maxX = 0, minY = h, maxY = 0;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0) {
      const x = i % w;
      const y = Math.floor(i / w);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }

  const pillW = maxX - minX + 1;
  const pillH = maxY - minY + 1;

  if (pillW < 10 || pillH < 10) return null;

  // Create a canvas with just the pill area, enhanced for OCR
  const impCanvas = document.createElement('canvas');
  const padding = 10;
  impCanvas.width = pillW + padding * 2;
  impCanvas.height = pillH + padding * 2;
  const impCtx = impCanvas.getContext('2d');

  // White background
  impCtx.fillStyle = 'white';
  impCtx.fillRect(0, 0, impCanvas.width, impCanvas.height);

  // Draw just the pill region
  const sourceData = ctx.getImageData(minX, minY, pillW, pillH);
  const impData = impCtx.createImageData(pillW, pillH);

  for (let y = 0; y < pillH; y++) {
    for (let x = 0; x < pillW; x++) {
      const srcIdx = (y * pillW + x) * 4;
      const maskIdx = (y + minY) * w + (x + minX);

      if (mask[maskIdx] > 0) {
        // Convert to high contrast grayscale for better OCR
        const gray = Math.round(
          0.299 * sourceData.data[srcIdx] +
          0.587 * sourceData.data[srcIdx + 1] +
          0.114 * sourceData.data[srcIdx + 2]
        );

        // Enhance contrast
        const enhanced = gray < 128 ? Math.max(0, gray - 40) : Math.min(255, gray + 40);

        impData.data[srcIdx] = enhanced;
        impData.data[srcIdx + 1] = enhanced;
        impData.data[srcIdx + 2] = enhanced;
        impData.data[srcIdx + 3] = 255;
      } else {
        impData.data[srcIdx] = 255;
        impData.data[srcIdx + 1] = 255;
        impData.data[srcIdx + 2] = 255;
        impData.data[srcIdx + 3] = 255;
      }
    }
  }

  impCtx.putImageData(impData, padding, padding);

  return impCanvas.toDataURL('image/png');
}

// ============================================================
// Backend Integration
// ============================================================

/**
 * Search for pills in the backend using detected features.
 */
export async function searchPillByFeatures(features) {
  try {
    const params = new URLSearchParams();
    if (features.color) params.append('color', features.color);
    if (features.shape) params.append('shape', features.shape);
    if (features.imprint) params.append('imprint', features.imprint);

    const response = await fetch(`${API_BASE}/drugs/pill-search/?${params}`);
    if (response.ok) {
      const data = await response.json();
      return data.results || data || [];
    }
  } catch (err) {
    console.warn('Pill search API failed:', err);
  }

  return [];
}

/**
 * Upload pill image to backend for server-side analysis.
 */
export async function uploadPillImage(imageDataUrl) {
  try {
    const blob = dataUrlToBlob(imageDataUrl);
    const formData = new FormData();
    formData.append('image', blob, 'pill.png');

    const response = await fetch(`${API_BASE}/scanner/analyze-pill/`, {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      return await response.json();
    }
  } catch (err) {
    console.warn('Pill image upload failed:', err);
  }
  return null;
}

function dataUrlToBlob(dataUrl) {
  const [header, base64] = dataUrl.split(',');
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(base64);
  const array = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    array[i] = binary.charCodeAt(i);
  }
  return new Blob([array], { type: mime });
}

export default {
  pillModel,
  analyzePill,
  searchPillByFeatures,
  uploadPillImage,
  PILL_SHAPES,
  PILL_COLORS
};
