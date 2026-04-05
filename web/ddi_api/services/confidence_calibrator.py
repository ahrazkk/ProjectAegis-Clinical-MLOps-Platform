"""
Confidence Calibrator for Project Aegis

Uses approved corrections to dynamically adjust GNN prediction confidence
and severity without full model retraining.

Two adjustment strategies:
1. Exact match: If an approved correction exists for a drug pair, use the corrected severity
2. Severity bias: Track systematic over/under-prediction patterns and apply global bias correction
"""
import logging
import time
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Severity numeric mapping for bias computation
SEVERITY_SCORES = {
    'none': 0, 'minor': 1, 'moderate': 2, 'severe': 3, 'critical': 4,
}
SCORE_TO_SEVERITY = {v: k for k, v in SEVERITY_SCORES.items()}


class ConfidenceCalibrator:
    """Singleton calibrator that adjusts GNN outputs based on approved corrections."""

    _instance = None
    CACHE_TTL = 300  # 5 minutes

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._pair_overrides = {}  # (drug_a, drug_b) -> corrected_severity
        self._severity_bias = {}   # original_severity -> average_shift
        self._correction_count = 0
        self._last_refresh = 0
        self._load_corrections()

    def _load_corrections(self):
        """Load approved corrections and compute bias map."""
        try:
            from .correction_memory import get_correction_memory
            mem = get_correction_memory()
            corrections = mem.export_training_data()

            self._pair_overrides.clear()
            shift_accumulator = defaultdict(list)

            for c in corrections:
                drug_a = c.get('drug_a', '').strip().title()
                drug_b = c.get('drug_b', '').strip().title()
                pair_key = tuple(sorted([drug_a, drug_b]))
                corrected = c.get('corrected_severity', '').lower()
                original = c.get('original_severity', '').lower()

                if corrected and corrected in SEVERITY_SCORES:
                    self._pair_overrides[pair_key] = corrected

                # Compute severity shift
                if original in SEVERITY_SCORES and corrected in SEVERITY_SCORES:
                    shift = SEVERITY_SCORES[corrected] - SEVERITY_SCORES[original]
                    shift_accumulator[original].append(shift)

            # Compute average bias per severity level
            self._severity_bias.clear()
            for sev, shifts in shift_accumulator.items():
                if shifts:
                    avg_shift = sum(shifts) / len(shifts)
                    self._severity_bias[sev] = avg_shift

            self._correction_count = len(corrections)
            self._last_refresh = time.time()
            logger.info("Calibrator loaded %d corrections, %d pair overrides, bias map: %s",
                        len(corrections), len(self._pair_overrides), self._severity_bias)
        except Exception as e:
            logger.warning("Calibrator load failed: %s", e)

    def refresh(self):
        """Force reload corrections data."""
        self._load_corrections()

    def _ensure_fresh(self):
        """Auto-refresh if cache has expired."""
        if time.time() - self._last_refresh > self.CACHE_TTL:
            self._load_corrections()

    def adjust(self, drug_a: str, drug_b: str, raw_severity: str,
               raw_confidence: float) -> Tuple[str, float, Dict[str, Any]]:
        """
        Adjust severity and confidence based on corrections.

        Returns: (adjusted_severity, adjusted_confidence, calibration_info)
        """
        self._ensure_fresh()

        drug_a_n = drug_a.strip().title()
        drug_b_n = drug_b.strip().title()
        pair_key = tuple(sorted([drug_a_n, drug_b_n]))
        info = {
            'correction_adjusted': False,
            'adjustment_type': 'none',
            'correction_count': self._correction_count,
        }

        # Strategy 1: Exact pair override
        if pair_key in self._pair_overrides:
            corrected_sev = self._pair_overrides[pair_key]
            info['correction_adjusted'] = True
            info['adjustment_type'] = 'exact_pair_override'
            info['original_severity'] = raw_severity
            info['corrected_severity'] = corrected_sev
            # Boost confidence since we have human-verified data
            adjusted_conf = min(0.95, raw_confidence + 0.15)
            return corrected_sev, adjusted_conf, info

        # Strategy 2: Severity bias correction (only if enough data)
        raw_sev_lower = raw_severity.lower()
        if raw_sev_lower in self._severity_bias and self._correction_count >= 3:
            bias = self._severity_bias[raw_sev_lower]
            if abs(bias) >= 0.3:  # Only apply if bias is meaningful
                current_score = SEVERITY_SCORES.get(raw_sev_lower, 2)
                adjusted_score = max(0, min(4, round(current_score + bias)))
                adjusted_sev = SCORE_TO_SEVERITY.get(adjusted_score, raw_severity)
                if adjusted_sev != raw_sev_lower:
                    info['correction_adjusted'] = True
                    info['adjustment_type'] = 'severity_bias'
                    info['original_severity'] = raw_severity
                    info['corrected_severity'] = adjusted_sev
                    info['bias_applied'] = bias
                    return adjusted_sev, raw_confidence, info

        return raw_severity, raw_confidence, info


def get_confidence_calibrator() -> ConfidenceCalibrator:
    return ConfidenceCalibrator()
