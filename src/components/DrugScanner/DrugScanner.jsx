/**
 * DrugScanner Component
 * 
 * Multi-modal drug detection system that can identify drugs via:
 * 1. Barcode/NDC scanning
 * 2. OCR text recognition (reading labels)
 * 3. Visual pill identification (shape, color, imprint)
 * 
 * @author OpenClaw Bot for Project Aegis
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Camera, 
  ScanLine, 
  Pill, 
  Type, 
  Barcode, 
  X, 
  CheckCircle, 
  AlertCircle,
  Loader2,
  RotateCcw,
  Zap,
  Upload
} from 'lucide-react';

import { useDrugScanner } from '../../hooks/useDrugScanner';
import { ScanOverlay } from './ScanOverlay';
import { DetectionResults } from './DetectionResults';

const SCAN_MODES = {
  AUTO: 'auto',
  BARCODE: 'barcode',
  OCR: 'ocr',
  PILL: 'pill'
};

export function DrugScanner({ onDrugDetected, onClose }) {
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  
  const [scanMode, setScanMode] = useState(SCAN_MODES.AUTO);
  const [isScanning, setIsScanning] = useState(false);
  const [facingMode, setFacingMode] = useState('environment'); // back camera
  const [capturedImage, setCapturedImage] = useState(null);
  
  const {
    detectedDrugs,
    isProcessing,
    error,
    scanImage,
    scanBarcode,
    scanOCR,
    scanPill,
    clearResults
  } = useDrugScanner();

  // Video constraints
  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: facingMode,
    aspectRatio: 16/9
  };

  // Capture image from webcam
  const captureImage = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setCapturedImage(imageSrc);
      return imageSrc;
    }
    return null;
  }, []);

  // Handle scan based on mode
  const handleScan = useCallback(async () => {
    setIsScanning(true);
    const imageSrc = captureImage();
    
    if (!imageSrc) {
      setIsScanning(false);
      return;
    }

    try {
      switch (scanMode) {
        case SCAN_MODES.AUTO:
          // Try all methods in order of speed
          await scanImage(imageSrc); // This tries barcode → OCR → pill
          break;
        case SCAN_MODES.BARCODE:
          await scanBarcode(imageSrc);
          break;
        case SCAN_MODES.OCR:
          await scanOCR(imageSrc);
          break;
        case SCAN_MODES.PILL:
          await scanPill(imageSrc);
          break;
      }
    } catch (err) {
      console.error('Scan error:', err);
    } finally {
      setIsScanning(false);
    }
  }, [scanMode, captureImage, scanImage, scanBarcode, scanOCR, scanPill]);

  // Handle file upload
  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageSrc = e.target?.result;
      setCapturedImage(imageSrc);
      setIsScanning(true);
      
      try {
        await scanImage(imageSrc);
      } finally {
        setIsScanning(false);
      }
    };
    reader.readAsDataURL(file);
  }, [scanImage]);

  // Toggle camera
  const toggleCamera = useCallback(() => {
    setFacingMode(prev => prev === 'environment' ? 'user' : 'environment');
  }, []);

  // Reset scanner
  const resetScanner = useCallback(() => {
    setCapturedImage(null);
    clearResults();
  }, [clearResults]);

  // Handle drug selection
  const handleDrugSelect = useCallback((drug) => {
    if (onDrugDetected) {
      onDrugDetected(drug);
    }
  }, [onDrugDetected]);

  // Continuous scanning in auto mode
  useEffect(() => {
    let interval;
    if (scanMode === SCAN_MODES.BARCODE && !capturedImage && !isProcessing) {
      interval = setInterval(() => {
        handleScan();
      }, 500); // Scan every 500ms for barcodes
    }
    return () => clearInterval(interval);
  }, [scanMode, capturedImage, isProcessing, handleScan]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-[#0a0a0f]/95 backdrop-blur-xl"
    >
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#4a4a5a]/30">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 
                          flex items-center justify-center border border-cyan-500/30">
              <Camera className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Drug Scanner</h2>
              <p className="text-xs text-gray-400">Scan pills, labels, or barcodes</p>
            </div>
          </div>
          
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          {/* Camera View */}
          <div className="flex-1 relative bg-black">
            {!capturedImage ? (
              <>
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="w-full h-full object-cover"
                />
                <ScanOverlay mode={scanMode} isScanning={isScanning} />
              </>
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <img 
                  src={capturedImage} 
                  alt="Captured" 
                  className="max-w-full max-h-full object-contain"
                />
              </div>
            )}

            {/* Processing Overlay */}
            <AnimatePresence>
              {isProcessing && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 bg-black/60 flex items-center justify-center"
                >
                  <div className="text-center">
                    <Loader2 className="w-12 h-12 text-cyan-400 animate-spin mx-auto mb-3" />
                    <p className="text-white font-medium">Analyzing...</p>
                    <p className="text-gray-400 text-sm">
                      {scanMode === SCAN_MODES.AUTO && 'Trying all detection methods'}
                      {scanMode === SCAN_MODES.BARCODE && 'Scanning barcode'}
                      {scanMode === SCAN_MODES.OCR && 'Reading text'}
                      {scanMode === SCAN_MODES.PILL && 'Identifying pill'}
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Camera Controls */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-3">
              {capturedImage ? (
                <button
                  onClick={resetScanner}
                  className="p-4 rounded-full bg-white/10 hover:bg-white/20 
                           border border-white/20 transition-all"
                >
                  <RotateCcw className="w-6 h-6 text-white" />
                </button>
              ) : (
                <>
                  <button
                    onClick={toggleCamera}
                    className="p-3 rounded-full bg-white/10 hover:bg-white/20 
                             border border-white/20 transition-all"
                  >
                    <RotateCcw className="w-5 h-5 text-white" />
                  </button>
                  
                  <button
                    onClick={handleScan}
                    disabled={isProcessing}
                    className="p-5 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 
                             hover:from-cyan-400 hover:to-purple-400 
                             disabled:opacity-50 transition-all shadow-lg shadow-cyan-500/25"
                  >
                    <Zap className="w-8 h-8 text-white" />
                  </button>
                  
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="p-3 rounded-full bg-white/10 hover:bg-white/20 
                             border border-white/20 transition-all"
                  >
                    <Upload className="w-5 h-5 text-white" />
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </>
              )}
            </div>
          </div>

          {/* Right Panel - Mode Selection & Results */}
          <div className="w-full lg:w-96 border-t lg:border-t-0 lg:border-l border-[#4a4a5a]/30 
                        flex flex-col bg-[#12121a]">
            {/* Scan Mode Selector */}
            <div className="p-4 border-b border-[#4a4a5a]/30">
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-3">Detection Mode</p>
              <div className="grid grid-cols-4 gap-2">
                {[
                  { mode: SCAN_MODES.AUTO, icon: Zap, label: 'Auto' },
                  { mode: SCAN_MODES.BARCODE, icon: Barcode, label: 'Barcode' },
                  { mode: SCAN_MODES.OCR, icon: Type, label: 'Label' },
                  { mode: SCAN_MODES.PILL, icon: Pill, label: 'Pill' },
                ].map(({ mode, icon: Icon, label }) => (
                  <button
                    key={mode}
                    onClick={() => setScanMode(mode)}
                    className={`p-3 rounded-lg border transition-all flex flex-col items-center gap-1
                      ${scanMode === mode 
                        ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400' 
                        : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                      }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="text-xs">{label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Results */}
            <div className="flex-1 overflow-y-auto p-4">
              <DetectionResults 
                results={detectedDrugs}
                error={error}
                onSelect={handleDrugSelect}
              />
            </div>

            {/* Tips */}
            <div className="p-4 border-t border-[#4a4a5a]/30">
              <p className="text-xs text-gray-400">
                <span className="text-cyan-400 font-medium">Tip:</span>
                {scanMode === SCAN_MODES.AUTO && ' Auto mode tries all detection methods automatically.'}
                {scanMode === SCAN_MODES.BARCODE && ' Hold the barcode steady within the frame.'}
                {scanMode === SCAN_MODES.OCR && ' Point at the drug name on the label.'}
                {scanMode === SCAN_MODES.PILL && ' Place the pill on a plain, contrasting background.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default DrugScanner;
