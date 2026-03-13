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
  const [cameraAvailable, setCameraAvailable] = useState(true);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  
  const {
    detectedDrugs,
    isProcessing,
    error,
    scanImage,
    scanBarcode,
    scanOCR,
    scanPill,
    clearResults,
    pillModelStatus,
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
      if (imageSrc) setCapturedImage(imageSrc);
      return imageSrc;
    }
    return null;
  }, []);

  // Auto-dismiss camera error after 4 seconds
  useEffect(() => {
    if (cameraError) {
      const t = setTimeout(() => setCameraError(null), 4000);
      return () => clearTimeout(t);
    }
  }, [cameraError]);

  // Handle scan based on mode
  const handleScan = useCallback(async () => {
    setCameraError(null);
    setIsScanning(true);
    const imageSrc = captureImage();

    if (!imageSrc) {
      setIsScanning(false);
      setCameraError(
        !cameraReady
          ? 'Camera is still starting up—please wait a moment, then try again.'
          : 'Could not capture image. Try moving to better lighting or tap Upload instead.'
      );
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
    setCameraReady(false); // stream will restart
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
      className="fixed inset-0 z-[80] bg-[#0a0a0f]/95 backdrop-blur-xl"
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
              <p className="text-[10px] mt-1">
                {pillModelStatus?.isLoaded
                  ? <span className="text-green-400">AI model ready</span>
                  : pillModelStatus?.isLoading
                    ? <span className="text-cyan-400">Loading AI model…</span>
                    : <span className="text-amber-400">Using CV fallback mode</span>
                }
              </p>
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
                {cameraAvailable ? (
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    onUserMedia={() => setCameraReady(true)}
                    onUserMediaError={() => { setCameraAvailable(false); setCameraReady(false); }}
                    playsInline
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center px-6 text-center">
                    <div>
                      <AlertCircle className="w-10 h-10 text-amber-400 mx-auto mb-3" />
                      <p className="text-white text-sm mb-2">Camera access unavailable on this device/browser.</p>
                      <p className="text-gray-400 text-xs mb-4">Use Upload to analyze a pill image instead.</p>
                      <label
                        htmlFor="scanner-file-upload"
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 border border-white/20 text-white text-sm cursor-pointer"
                      >
                        <Upload className="w-4 h-4" />
                        Upload Image
                      </label>
                    </div>
                  </div>
                )}
                {/* Camera initialising overlay */}
                {cameraAvailable && !cameraReady && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                    <div className="text-center">
                      <Loader2 className="w-8 h-8 text-cyan-400 animate-spin mx-auto mb-2" />
                      <p className="text-white text-sm">Starting camera…</p>
                    </div>
                  </div>
                )}
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

            {/* Camera error notification */}
            <AnimatePresence>
              {cameraError && (
                <motion.div
                  initial={{ opacity: 0, y: -16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -16 }}
                  className="absolute top-4 left-4 right-4 z-20 bg-amber-500/20 border border-amber-500/40 rounded-xl p-3 text-sm text-amber-300"
                >
                  {cameraError}
                </motion.div>
              )}
            </AnimatePresence>

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
            <div
              className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 z-10"
              style={{ paddingBottom: 'env(safe-area-inset-bottom, 0px)' }}
            >
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
                    disabled={!cameraAvailable}
                    className="p-3 rounded-full bg-white/10 hover:bg-white/20 
                             border border-white/20 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <RotateCcw className="w-5 h-5 text-white" />
                  </button>
                  
                  <button
                    onClick={handleScan}
                    disabled={isProcessing || (cameraAvailable && !cameraReady)}
                    className="p-5 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 
                             hover:from-cyan-400 hover:to-purple-400 
                             disabled:opacity-50 transition-all shadow-lg shadow-cyan-500/25"
                    title={cameraAvailable && !cameraReady ? 'Camera initialising…' : 'Scan'}
                  >
                    {cameraAvailable && !cameraReady
                      ? <Loader2 className="w-8 h-8 text-white animate-spin" />
                      : <Zap className="w-8 h-8 text-white" />
                    }
                  </button>
                  
                  <label
                    htmlFor="scanner-file-upload"
                    className="p-3 rounded-full bg-white/10 hover:bg-white/20 
                             border border-white/20 transition-all cursor-pointer"
                  >
                    <Upload className="w-5 h-5 text-white" />
                  </label>
                  <input
                    id="scanner-file-upload"
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="sr-only"
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
