/**
 * useDrugScanner Hook
 * 
 * Core logic for multi-modal drug detection:
 * 1. Barcode/NDC scanning (QuaggaJS)
 * 2. OCR text recognition (Tesseract.js)
 * 3. Visual pill identification (TensorFlow.js)
 * 
 * @author OpenClaw Bot for Project Aegis
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import Quagga from '@ericblade/quagga2';
import Tesseract from 'tesseract.js';
import * as tf from '@tensorflow/tfjs';

// API base URL
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Common drug name patterns for OCR extraction
const DRUG_NAME_PATTERNS = [
  // Generic patterns
  /(?:^|\s)([A-Z][a-z]+(?:in|ol|am|ine|ide|ate|one|ril|tan|pam|lam|zole|pril|sartan|statin|mycin|cillin|cycline)\b)/gi,
  // Specific strength patterns
  /(\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|IU|units?))/gi,
  // NDC pattern
  /NDC[:\s]*(\d{4,5}-\d{3,4}-\d{1,2})/gi,
  // Common drug names (will be expanded)
  /\b(Aspirin|Ibuprofen|Acetaminophen|Tylenol|Advil|Warfarin|Metformin|Lisinopril|Atorvastatin|Omeprazole|Amlodipine|Metoprolol|Losartan|Albuterol|Gabapentin|Hydrochlorothiazide|Sertraline|Pantoprazole|Atenolol|Prednisone)\b/gi
];

export function useDrugScanner() {
  const [detectedDrugs, setDetectedDrugs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  
  // TensorFlow model ref (for pill detection)
  const pillModelRef = useRef(null);
  const tesseractWorkerRef = useRef(null);

  // Initialize Tesseract worker
  useEffect(() => {
    const initTesseract = async () => {
      try {
        tesseractWorkerRef.current = await Tesseract.createWorker('eng');
      } catch (err) {
        console.warn('Tesseract initialization failed:', err);
      }
    };
    initTesseract();

    return () => {
      if (tesseractWorkerRef.current) {
        tesseractWorkerRef.current.terminate();
      }
    };
  }, []);

  // Clear results
  const clearResults = useCallback(() => {
    setDetectedDrugs([]);
    setError(null);
  }, []);

  // Lookup drug by NDC code
  const lookupByNDC = async (ndc) => {
    try {
      const response = await fetch(`${API_BASE}/drugs/ndc/${ndc}/`);
      if (response.ok) {
        return await response.json();
      }
    } catch (err) {
      console.warn('NDC lookup failed:', err);
    }
    return null;
  };

  // Lookup drug by name
  const lookupByName = async (name) => {
    try {
      const response = await fetch(`${API_BASE}/drugs/search/?q=${encodeURIComponent(name)}`);
      if (response.ok) {
        const data = await response.json();
        return data.results || data;
      }
    } catch (err) {
      console.warn('Name lookup failed:', err);
    }
    return [];
  };

  // Barcode scanning with QuaggaJS
  const scanBarcode = useCallback(async (imageSrc) => {
    setIsProcessing(true);
    setError(null);

    return new Promise((resolve) => {
      // Create image element
      const img = new Image();
      img.onload = async () => {
        // Create canvas for Quagga
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        try {
          const result = await new Promise((resolveQuagga, rejectQuagga) => {
            Quagga.decodeSingle({
              src: imageSrc,
              numOfWorkers: 0,
              locate: true,
              decoder: {
                readers: [
                  'upc_reader',
                  'upc_e_reader',
                  'code_128_reader',
                  'code_39_reader',
                  'ean_reader',
                  'ean_8_reader'
                ]
              }
            }, (result) => {
              if (result && result.codeResult) {
                resolveQuagga(result.codeResult);
              } else {
                rejectQuagga(new Error('No barcode detected'));
              }
            });
          });

          const barcode = result.code;
          console.log('Barcode detected:', barcode);

          // Try to lookup as NDC first
          let drug = await lookupByNDC(barcode);
          
          if (!drug) {
            // Try external API for UPC/NDC lookup
            drug = await lookupBarcodeExternal(barcode);
          }

          if (drug) {
            const detectedDrug = {
              ...drug,
              detectionMethod: 'barcode',
              confidence: 1.0,
              ndc: barcode
            };
            setDetectedDrugs([detectedDrug]);
            resolve([detectedDrug]);
          } else {
            setError(`Barcode ${barcode} not found in database`);
            resolve([]);
          }
        } catch (err) {
          console.log('Barcode scan failed:', err.message);
          resolve([]);
        } finally {
          setIsProcessing(false);
        }
      };

      img.onerror = () => {
        setIsProcessing(false);
        setError('Failed to load image');
        resolve([]);
      };

      img.src = imageSrc;
    });
  }, []);

  // External barcode lookup (RxNorm, OpenFDA)
  const lookupBarcodeExternal = async (barcode) => {
    try {
      // Try OpenFDA
      const fdaResponse = await fetch(
        `https://api.fda.gov/drug/ndc.json?search=product_ndc:"${barcode}"&limit=1`
      );
      
      if (fdaResponse.ok) {
        const data = await fdaResponse.json();
        if (data.results && data.results.length > 0) {
          const result = data.results[0];
          return {
            name: result.brand_name || result.generic_name,
            genericName: result.generic_name,
            strength: result.active_ingredients?.[0]?.strength,
            therapeuticClass: result.pharm_class?.[0],
            ndc: barcode
          };
        }
      }
    } catch (err) {
      console.warn('External barcode lookup failed:', err);
    }
    return null;
  };

  // OCR scanning with Tesseract
  const scanOCR = useCallback(async (imageSrc) => {
    setIsProcessing(true);
    setError(null);

    try {
      let worker = tesseractWorkerRef.current;
      
      // Create worker if not available
      if (!worker) {
        worker = await Tesseract.createWorker('eng');
        tesseractWorkerRef.current = worker;
      }

      const { data: { text, confidence } } = await worker.recognize(imageSrc);
      
      console.log('OCR Text:', text);
      console.log('OCR Confidence:', confidence);

      if (!text || text.trim().length < 3) {
        setError('No readable text found');
        setIsProcessing(false);
        return [];
      }

      // Extract potential drug names
      const extractedDrugs = extractDrugNames(text);
      console.log('Extracted drug names:', extractedDrugs);

      if (extractedDrugs.length === 0) {
        setError('No drug names detected in text');
        setIsProcessing(false);
        return [];
      }

      // Lookup each extracted drug
      const results = [];
      for (const drugName of extractedDrugs.slice(0, 5)) { // Limit to 5
        const matches = await lookupByName(drugName.name);
        if (matches && matches.length > 0) {
          results.push({
            ...matches[0],
            detectionMethod: 'ocr',
            confidence: (confidence / 100) * drugName.confidence,
            extractedText: drugName.name
          });
        }
      }

      if (results.length > 0) {
        setDetectedDrugs(results);
      } else {
        setError('Detected text did not match any drugs in database');
      }

      setIsProcessing(false);
      return results;
    } catch (err) {
      console.error('OCR error:', err);
      setError('OCR processing failed');
      setIsProcessing(false);
      return [];
    }
  }, []);

  // Extract drug names from OCR text
  const extractDrugNames = (text) => {
    const results = [];
    const seen = new Set();

    for (const pattern of DRUG_NAME_PATTERNS) {
      let match;
      const regex = new RegExp(pattern.source, pattern.flags);
      
      while ((match = regex.exec(text)) !== null) {
        const drugName = match[1] || match[0];
        const normalized = drugName.trim().toLowerCase();
        
        if (!seen.has(normalized) && normalized.length >= 3) {
          seen.add(normalized);
          results.push({
            name: drugName.trim(),
            confidence: 0.8 // Base confidence for pattern match
          });
        }
      }
    }

    // Also try to find any capitalized words that might be drug names
    const words = text.match(/\b[A-Z][a-z]{3,}\b/g) || [];
    for (const word of words) {
      const normalized = word.toLowerCase();
      if (!seen.has(normalized)) {
        seen.add(normalized);
        results.push({
          name: word,
          confidence: 0.5 // Lower confidence for generic matches
        });
      }
    }

    return results.sort((a, b) => b.confidence - a.confidence);
  };

  // Pill visual identification
  const scanPill = useCallback(async (imageSrc) => {
    setIsProcessing(true);
    setError(null);

    try {
      // For now, we'll use a color/shape analysis approach
      // In production, this would use a trained model on pill images
      
      const pillFeatures = await analyzePillImage(imageSrc);
      console.log('Pill features:', pillFeatures);

      // Search by pill characteristics
      const results = await searchByPillFeatures(pillFeatures);

      if (results.length > 0) {
        const drugsWithMethod = results.map(drug => ({
          ...drug,
          detectionMethod: 'pill',
          confidence: drug.confidence || 0.7
        }));
        setDetectedDrugs(drugsWithMethod);
        return drugsWithMethod;
      } else {
        setError('Could not identify pill. Try barcode or label scan.');
        return [];
      }
    } catch (err) {
      console.error('Pill scan error:', err);
      setError('Pill identification failed');
      return [];
    } finally {
      setIsProcessing(false);
    }
  }, []);

  // Analyze pill image for features (color, shape)
  const analyzePillImage = async (imageSrc) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const size = 100; // Analyze at fixed size
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        
        // Draw and scale image
        ctx.drawImage(img, 0, 0, size, size);
        const imageData = ctx.getImageData(0, 0, size, size);
        const data = imageData.data;

        // Calculate dominant color
        let r = 0, g = 0, b = 0, count = 0;
        
        for (let i = 0; i < data.length; i += 4) {
          // Skip very dark (background) and very light pixels
          const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
          if (brightness > 30 && brightness < 240) {
            r += data[i];
            g += data[i + 1];
            b += data[i + 2];
            count++;
          }
        }

        if (count > 0) {
          r = Math.round(r / count);
          g = Math.round(g / count);
          b = Math.round(b / count);
        }

        // Classify color
        const color = classifyColor(r, g, b);

        // Detect shape (simplified - just aspect ratio for now)
        const shape = 'round'; // Would need edge detection for real shape analysis

        resolve({
          color,
          colorRGB: { r, g, b },
          shape,
          // Would include imprint detection here with real model
          imprint: null
        });
      };

      img.onerror = () => {
        resolve({ color: 'unknown', shape: 'unknown', imprint: null });
      };

      img.src = imageSrc;
    });
  };

  // Classify RGB to color name
  const classifyColor = (r, g, b) => {
    const colors = {
      white: { r: 255, g: 255, b: 255 },
      pink: { r: 255, g: 182, b: 193 },
      red: { r: 255, g: 0, b: 0 },
      orange: { r: 255, g: 165, b: 0 },
      yellow: { r: 255, g: 255, b: 0 },
      green: { r: 0, g: 128, b: 0 },
      blue: { r: 0, g: 0, b: 255 },
      purple: { r: 128, g: 0, b: 128 },
      brown: { r: 139, g: 69, b: 19 },
      gray: { r: 128, g: 128, b: 128 }
    };

    let closestColor = 'unknown';
    let minDistance = Infinity;

    for (const [name, rgb] of Object.entries(colors)) {
      const distance = Math.sqrt(
        Math.pow(r - rgb.r, 2) +
        Math.pow(g - rgb.g, 2) +
        Math.pow(b - rgb.b, 2)
      );
      if (distance < minDistance) {
        minDistance = distance;
        closestColor = name;
      }
    }

    return closestColor;
  };

  // Search by pill features (would call backend API)
  const searchByPillFeatures = async (features) => {
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

    // Fallback: suggest user try other methods
    return [];
  };

  // Auto scan - tries all methods
  const scanImage = useCallback(async (imageSrc) => {
    setIsProcessing(true);
    setError(null);
    clearResults();

    // Try barcode first (fastest and most accurate)
    console.log('Trying barcode detection...');
    let results = await scanBarcode(imageSrc);
    
    if (results.length > 0) {
      setIsProcessing(false);
      return results;
    }

    // Try OCR next
    console.log('Trying OCR detection...');
    results = await scanOCR(imageSrc);
    
    if (results.length > 0) {
      setIsProcessing(false);
      return results;
    }

    // Finally try pill identification
    console.log('Trying pill identification...');
    results = await scanPill(imageSrc);
    
    if (results.length === 0) {
      setError('Could not detect any drugs. Please try again with better lighting or a clearer image.');
    }

    setIsProcessing(false);
    return results;
  }, [scanBarcode, scanOCR, scanPill, clearResults]);

  return {
    detectedDrugs,
    isProcessing,
    error,
    scanImage,
    scanBarcode,
    scanOCR,
    scanPill,
    clearResults
  };
}

export default useDrugScanner;
