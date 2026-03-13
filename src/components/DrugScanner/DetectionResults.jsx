/**
 * DetectionResults Component
 * Displays detected drugs from scanning
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Pill, 
  CheckCircle, 
  AlertCircle, 
  Plus,
  Barcode,
  Type,
  Eye,
  ChevronRight
} from 'lucide-react';

const METHOD_ICONS = {
  barcode: Barcode,
  ocr: Type,
  pill: Eye,
  pill_multimodal: Eye,
  pill_model: Eye,
  pill_upload: Eye,
  database: Pill
};

const METHOD_LABELS = {
  barcode: 'Barcode',
  ocr: 'Label OCR',
  pill: 'Visual AI',
  pill_multimodal: 'Multimodal AI',
  pill_model: 'AI Model',
  pill_upload: 'Image Upload',
  database: 'Database Match'
};

export function DetectionResults({ results, error, onSelect }) {
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-4">
        <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center mb-4">
          <AlertCircle className="w-8 h-8 text-red-400" />
        </div>
        <p className="text-white font-medium mb-2">Detection Error</p>
        <p className="text-gray-400 text-sm">{error}</p>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-4">
        <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
          <Pill className="w-8 h-8 text-gray-500" />
        </div>
        <p className="text-white font-medium mb-2">No Drugs Detected</p>
        <p className="text-gray-400 text-sm">
          Point the camera at a pill, medication label, or barcode
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs text-gray-400 uppercase tracking-wider">
          {results.every(r => r.isEstimate) ? 'Visual Analysis' : `Detected Drugs (${results.filter(r => !r.isEstimate).length || results.length})`}
        </p>
        <div className="flex items-center gap-1">
          {results.some(r => !r.isEstimate) ? (
            <><CheckCircle className="w-4 h-4 text-green-400" /><span className="text-xs text-green-400">Found</span></>
          ) : (
            <><Eye className="w-4 h-4 text-amber-400" /><span className="text-xs text-amber-400">Estimate</span></>
          )}
        </div>
      </div>

      <AnimatePresence mode="popLayout">
        {results.map((drug, index) => {
          const MethodIcon = METHOD_ICONS[drug.detectionMethod] || Pill;
          
          return (
            <motion.div
              key={drug.id || drug.name + index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.1 }}
              className="group"
            >
              <button
                onClick={() => onSelect(drug)}
                className={`w-full p-4 rounded-xl bg-gradient-to-r transition-all text-left
                  ${
                    drug.isEstimate
                      ? 'from-amber-500/5 to-transparent border border-amber-500/20 hover:border-amber-500/40'
                      : 'from-white/5 to-transparent border border-white/10 hover:border-cyan-500/50'
                  }`}
              >
                <div className="flex items-start gap-3">
                  {/* Drug Icon/Image */}
                  <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 
                                flex items-center justify-center flex-shrink-0 border border-cyan-500/20">
                    {drug.imageUrl ? (
                      <img 
                        src={drug.imageUrl} 
                        alt={drug.name}
                        className="w-full h-full object-cover rounded-lg"
                      />
                    ) : (
                      <Pill className="w-6 h-6 text-cyan-400" />
                    )}
                  </div>

                  {/* Drug Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="text-white font-semibold truncate">
                        {drug.name}
                      </h3>
                      {drug.confidence && (
                        <span className={`text-xs px-2 py-0.5 rounded-full
                          ${drug.confidence > 0.8 
                            ? 'bg-green-500/20 text-green-400' 
                            : drug.confidence > 0.5 
                              ? 'bg-yellow-500/20 text-yellow-400'
                              : 'bg-orange-500/20 text-orange-400'
                          }`}
                        >
                          {Math.round(drug.confidence * 100)}%
                        </span>
                      )}
                    </div>

                    {drug.genericName && drug.genericName !== drug.name && (
                      <p className="text-gray-400 text-sm truncate">
                        Generic: {drug.genericName}
                      </p>
                    )}

                    {drug.strength && (
                      <p className="text-cyan-400 text-sm">
                        {drug.strength}
                      </p>
                    )}

                    {/* Detection method badge */}
                    <div className="flex items-center gap-2 mt-2">
                      <div className="flex items-center gap-1 text-xs text-gray-500">
                        <MethodIcon className="w-3 h-3" />
                        <span>{METHOD_LABELS[drug.detectionMethod]}</span>
                      </div>
                      
                        {drug.ndc && (
                        <span className="text-xs text-gray-500">
                          NDC: {drug.ndc}
                        </span>
                      )}
                      {drug.isEstimate && (
                        <span className="text-xs text-amber-500/70">AI estimate — verify manually</span>
                      )}
                    </div>

                    {/* Pill visual properties */}
                    {drug.pillFeatures && (drug.pillFeatures.color || drug.pillFeatures.shape) && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {drug.pillFeatures.color && drug.pillFeatures.color !== 'unknown' && (
                          <span className="text-xs px-1.5 py-0.5 rounded bg-white/5 text-gray-400">{drug.pillFeatures.color}</span>
                        )}
                        {drug.pillFeatures.shape && drug.pillFeatures.shape !== 'unknown' && (
                          <span className="text-xs px-1.5 py-0.5 rounded bg-white/5 text-gray-400">{drug.pillFeatures.shape}</span>
                        )}
                        {drug.imprint && (
                          <span className="text-xs px-1.5 py-0.5 rounded bg-white/5 text-gray-400">imprint: {drug.imprint}</span>
                        )}
                      </div>
                    )}

                    {drug.confidence && (
                      <div className="mt-2">
                        <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-cyan-400 via-blue-400 to-emerald-400"
                            style={{ width: `${Math.max(6, Math.min(100, Math.round(drug.confidence * 100)))}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {drug.yoloTopDetection && (
                      <div className="mt-2 text-xs text-cyan-300">
                        <span className="px-1.5 py-0.5 rounded bg-cyan-500/15 border border-cyan-500/30 mr-1">
                          YOLO {Math.round((drug.yoloTopDetection.confidence || 0) * 100)}%
                        </span>
                        <span className="text-gray-400">
                          {drug.yoloTopDetection.class_name || 'pill'}
                        </span>
                        {drug.yoloTopDetection.bbox && (
                          <span className="text-gray-500 ml-2">
                            box {Number(drug.yoloTopDetection.bbox.x_norm || 0).toFixed(2)},
                            {Number(drug.yoloTopDetection.bbox.y_norm || 0).toFixed(2)},
                            {Number(drug.yoloTopDetection.bbox.width_norm || 0).toFixed(2)},
                            {Number(drug.yoloTopDetection.bbox.height_norm || 0).toFixed(2)}
                          </span>
                        )}
                      </div>
                    )}

                    {drug.decisionMessage && (
                      <p className={`mt-2 text-xs ${drug.decisionStatus === 'uncertain' ? 'text-amber-400' : 'text-emerald-400'}`}>
                        {drug.decisionMessage}
                      </p>
                    )}
                  </div>

                  {/* Add button */}
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-xs text-cyan-400">Add</span>
                    <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center">
                      <Plus className="w-4 h-4 text-cyan-400" />
                    </div>
                  </div>
                </div>

                {/* Additional info on hover */}
                {drug.therapeuticClass && (
                  <div className="mt-3 pt-3 border-t border-white/5">
                    <p className="text-xs text-gray-500">
                      <span className="text-gray-400">Class:</span> {drug.therapeuticClass}
                    </p>
                  </div>
                )}

                {drug.overlayDataUrl && (
                  <div className="mt-3 pt-3 border-t border-white/5">
                    <p className="text-xs text-gray-400 mb-2">Detected pill region</p>
                    <img
                      src={drug.overlayDataUrl}
                      alt="Pill detection overlay"
                      className="w-full max-h-28 object-contain rounded-lg border border-cyan-500/20"
                    />
                  </div>
                )}
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>

      {/* Multiple drugs hint */}
      {results.length > 1 && (
        <p className="text-xs text-center text-gray-500 mt-4">
          Tap a drug to add it to your interaction check
        </p>
      )}
    </div>
  );
}

export default DetectionResults;
