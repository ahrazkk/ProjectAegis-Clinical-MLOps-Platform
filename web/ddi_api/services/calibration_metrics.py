"""Utilities for calibration quality assessment on binary prediction scores."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


def _validate_binary_inputs(labels: Iterable[int], scores: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(list(labels), dtype=float)
    y_score = np.asarray(list(scores), dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError("labels and scores must have identical shapes")
    if y_true.size == 0:
        raise ValueError("labels and scores cannot be empty")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("labels must be binary values 0 or 1")
    if np.any((y_score < 0) | (y_score > 1)):
        raise ValueError("scores must be in [0, 1]")

    return y_true.astype(np.int8), y_score.astype(np.float64)


def brier_score(labels: Iterable[int], scores: Iterable[float]) -> float:
    y_true, y_score = _validate_binary_inputs(labels, scores)
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(
    labels: Iterable[int],
    scores: Iterable[float],
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, float]]]:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    y_true, y_score = _validate_binary_inputs(labels, scores)
    n = y_true.shape[0]

    bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1)
    # Put score==1.0 into the final bucket.
    bin_ids = np.minimum((y_score * n_bins).astype(int), n_bins - 1)

    ece = 0.0
    mce = 0.0
    bins: List[Dict[str, float]] = []

    for idx in range(n_bins):
        mask = bin_ids == idx
        count = int(np.sum(mask))

        if count == 0:
            bins.append(
                {
                    "bin_index": idx,
                    "start": float(bin_edges[idx]),
                    "end": float(bin_edges[idx + 1]),
                    "count": 0,
                    "avg_confidence": 0.0,
                    "empirical_accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        avg_conf = float(np.mean(y_score[mask]))
        acc = float(np.mean(y_true[mask]))
        gap = abs(acc - avg_conf)

        ece += (count / n) * gap
        mce = max(mce, gap)

        bins.append(
            {
                "bin_index": idx,
                "start": float(bin_edges[idx]),
                "end": float(bin_edges[idx + 1]),
                "count": count,
                "avg_confidence": avg_conf,
                "empirical_accuracy": acc,
                "gap": float(gap),
            }
        )

    return float(ece), float(mce), bins


def bootstrap_confidence_interval(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    labels: Iterable[int],
    scores: Iterable[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    if n_bootstrap < 200:
        raise ValueError("n_bootstrap must be >= 200 for stable confidence intervals")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    y_true, y_score = _validate_binary_inputs(labels, scores)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[sample_idx], y_score[sample_idx])

    lower_q = alpha / 2
    upper_q = 1 - (alpha / 2)

    return {
        "lower": float(np.quantile(samples, lower_q)),
        "upper": float(np.quantile(samples, upper_q)),
        "alpha": float(alpha),
        "n_bootstrap": int(n_bootstrap),
    }


def _score_summary(
    labels: Iterable[int],
    scores: Iterable[float],
    n_bins: int,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    y_true, y_score = _validate_binary_inputs(labels, scores)
    ece, mce, bins = expected_calibration_error(y_true, y_score, n_bins=n_bins)
    brier = brier_score(y_true, y_score)

    return {
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "brier_confidence_interval": bootstrap_confidence_interval(
            lambda yt, ys: brier_score(yt, ys),
            y_true,
            y_score,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "ece_confidence_interval": bootstrap_confidence_interval(
            lambda yt, ys: expected_calibration_error(yt, ys, n_bins=n_bins)[0],
            y_true,
            y_score,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "bins": bins,
    }


def generate_calibration_report(
    labels: Iterable[int],
    raw_scores: Iterable[float],
    calibrated_scores: Iterable[float],
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    y_true, raw = _validate_binary_inputs(labels, raw_scores)
    _, calibrated = _validate_binary_inputs(labels, calibrated_scores)

    if raw.shape != calibrated.shape:
        raise ValueError("raw_scores and calibrated_scores must have identical shapes")

    raw_summary = _score_summary(y_true, raw, n_bins=n_bins, n_bootstrap=n_bootstrap, seed=seed)
    calibrated_summary = _score_summary(y_true, calibrated, n_bins=n_bins, n_bootstrap=n_bootstrap, seed=seed)

    return {
        "meta": {
            "n_samples": int(y_true.shape[0]),
            "positive_rate": float(np.mean(y_true)),
            "n_bins": int(n_bins),
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
        },
        "raw": raw_summary,
        "calibrated": calibrated_summary,
        "delta": {
            "brier_improvement": float(raw_summary["brier_score"] - calibrated_summary["brier_score"]),
            "ece_improvement": float(raw_summary["ece"] - calibrated_summary["ece"]),
            "mce_improvement": float(raw_summary["mce"] - calibrated_summary["mce"]),
        },
    }
