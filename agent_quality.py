#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-level image profiling and confidence scoring helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def build_image_profile(gray: np.ndarray, local_roi_mode: bool = False) -> Dict:
    """Summarize image dynamic range and likely level-adjustment artifacts."""
    if gray.ndim == 3:
        gray = np.mean(gray, axis=2)
    img = gray.astype(np.uint8, copy=False)
    flat = img.reshape(-1)
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    occupied = hist > 0

    p1, p50, p99 = np.percentile(flat, [1, 50, 99])
    dynamic_range = float(p99 - p1)
    saturation_fraction = float(np.mean((flat <= 1) | (flat >= 254)))
    occupied_fraction = float(np.mean(occupied))
    zero_bins = int(np.count_nonzero(~occupied))
    dominant_bin_fraction = float(np.max(hist) / max(flat.size, 1))

    likely_clipped = saturation_fraction > 0.20 or (saturation_fraction > 0.04 and dynamic_range > 120)
    likely_sparse_hist = occupied_fraction < 0.35 and dominant_bin_fraction > 0.08
    if local_roi_mode and likely_sparse_hist and not likely_clipped:
        likely_sparse_hist = False
    possibly_adjusted = bool(likely_clipped or likely_sparse_hist)

    noise = float(np.std(flat[flat <= p50])) if np.any(flat <= p50) else float(np.std(flat))
    noise_score = float(np.clip(1.0 - saturation_fraction * 4.0 - max(0.0, noise - 18.0) / 40.0, 0.0, 1.0))

    reasons: List[str] = []
    if likely_clipped:
        reasons.append("clipped_extremes")
    if likely_sparse_hist:
        reasons.append("sparse_histogram")

    return {
        "image_mean": float(np.mean(flat)),
        "image_median": float(p50),
        "image_std": float(np.std(flat)),
        "image_p1": float(p1),
        "image_p99": float(p99),
        "dynamic_range": dynamic_range,
        "saturation_fraction": saturation_fraction,
        "histogram_occupied_fraction": occupied_fraction,
        "histogram_zero_bins": zero_bins,
        "dominant_bin_fraction": dominant_bin_fraction,
        "possibly_adjusted": possibly_adjusted,
        "adjustment_reason": ",".join(reasons),
        "background_noise_score": noise_score,
    }


def _metric01(metrics: Dict, key: str, default: float) -> float:
    value = metrics.get(key, default)
    if value is None:
        value = default
    return float(np.clip(float(value), 0.0, 1.0))


def calculate_composite_confidence(metrics: Dict) -> Dict:
    """Combine CV, grid, PSF, image profile, and batch evidence into one score."""
    legacy_score = float(metrics.get("score", 70.0) or 0.0)
    grid_quality = _metric01(metrics, "grid_signal_quality", 0.65)
    grid_confidence = _metric01(metrics, "grid_auto_confidence", 0.65)
    grid_inlier_fraction = _metric01(metrics, "grid_auto_inlier_fraction", 0.65)
    weak_fraction = _metric01(metrics, "grid_weak_positive_fraction", 0.0)
    psf_pass_rate = _metric01(metrics, "psf_pass_rate", 0.85)
    noise_score = _metric01(metrics, "background_noise_score", 0.75)
    batch_score = _metric01(metrics, "batch_consistency_score", 0.85)

    score = (
        5.0
        +
        0.20 * legacy_score
        + 25.0 * grid_quality
        + 8.0 * grid_confidence
        + 10.0 * grid_inlier_fraction
        + 20.0 * psf_pass_rate
        + 10.0 * noise_score
        + 10.0 * batch_score
    )
    score -= 20.0 * weak_fraction

    flags: List[str] = []
    if metrics.get("grid_unreliable"):
        flags.append("grid_unreliable")
        score -= 12.0
    if grid_inlier_fraction < 0.15:
        flags.append("grid_low_inlier_fraction")
        score -= 10.0
    if psf_pass_rate < 0.60:
        flags.append("psf_low_pass_rate")
        score -= 8.0
    if psf_pass_rate < 0.25:
        flags.append("psf_formal_qc_failed")
        score -= 12.0
    if metrics.get("possibly_adjusted"):
        flags.append("possibly_adjusted_image")
        score -= 8.0
    if weak_fraction > 0.50:
        flags.append("many_weak_grid_positives")
    if batch_score < 0.75:
        flags.append("batch_inconsistency")
    if noise_score < 0.50:
        flags.append("noisy_or_clipped_background")
    if metrics.get("spatial_qc_low_quality") is True:
        flags.append("spatial_qc_low_quality")
        if metrics.get("spatial_qc_status") == "fail":
            score -= 5.0
        elif metrics.get("spatial_qc_status") == "warning":
            score -= 2.0

    return {
        "composite_confidence": float(np.clip(score, 0.0, 100.0)),
        "confidence_flags": flags,
    }
