#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial goodness-of-fit quality control for DeepLISA detections.

This module audits the spatial distribution of accepted positive chambers.
It does not change the detected count.  The goal is to flag images that are
unsuitable for manuscript-level spatial randomness claims.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import chi2 as chi2_distribution

from grid_detection import generate_grid_centers


def contour_centers(contours: Iterable) -> np.ndarray:
    """Return contour centers as an ``N x 2`` float array."""
    centers: List[Tuple[float, float]] = []
    for contour in contours or []:
        ((x, y), _) = cv2.minEnclosingCircle(contour)
        centers.append((float(x), float(y)))
    if not centers:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


def measure_candidate_signal(gray: np.ndarray, contour, radius: int) -> Dict:
    """Measure one candidate using a small local patch.

    The previous exporter allocated two full-image masks for every candidate.
    This local implementation preserves the same inner-disk / outer-annulus
    measurements while keeping work proportional to the chamber size.
    """
    ((x, y), detected_radius) = cv2.minEnclosingCircle(contour)
    ix, iy = int(round(x)), int(round(y))
    radius = max(2, int(radius))
    inner_r = max(2, int(round(radius * 0.7)))
    outer_r = max(inner_r + 3, int(round(radius * 2.0)))
    height, width = gray.shape[:2]

    x1 = max(0, ix - outer_r)
    x2 = min(width, ix + outer_r + 1)
    y1 = max(0, iy - outer_r)
    y2 = min(height, iy + outer_r + 1)
    patch = gray[y1:y2, x1:x2]

    local_x = ix - x1
    local_y = iy - y1
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    dist2 = (xx - local_x) ** 2 + (yy - local_y) ** 2
    inner_pixels = patch[dist2 <= inner_r * inner_r]
    ring_pixels = patch[(dist2 > inner_r * inner_r) & (dist2 <= outer_r * outer_r)]

    inner_mean = float(np.mean(inner_pixels)) if inner_pixels.size else 0.0
    ring_mean = float(np.mean(ring_pixels)) if ring_pixels.size else 0.0
    ring_std = max(1.0, float(np.std(ring_pixels))) if ring_pixels.size else 1.0
    local_delta = inner_mean - ring_mean
    local_z = local_delta / ring_std
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 0.0 if perimeter <= 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
    return {
        "x": float(x),
        "y": float(y),
        "detected_radius": float(detected_radius),
        "area": area,
        "circularity": circularity,
        "inner_mean": inner_mean,
        "ring_mean": ring_mean,
        "ring_std": ring_std,
        "local_delta": local_delta,
        "local_z": local_z,
        "measurement_patch_width": int(patch.shape[1]),
        "measurement_patch_height": int(patch.shape[0]),
    }


def _clamp_roi(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        width: int,
        height: int,
) -> Tuple[float, float, float, float]:
    x_min = float(np.clip(x_min, 0, max(width - 1, 0)))
    x_max = float(np.clip(x_max, 0, max(width - 1, 0)))
    y_min = float(np.clip(y_min, 0, max(height - 1, 0)))
    y_max = float(np.clip(y_max, 0, max(height - 1, 0)))
    if x_max <= x_min:
        x_min, x_max = 0.0, float(max(width - 1, 1))
    if y_max <= y_min:
        y_min, y_max = 0.0, float(max(height - 1, 1))
    return x_min, x_max, y_min, y_max


def estimate_effective_array_roi_from_image(image: np.ndarray) -> Dict:
    """Estimate the visible chamber-array body while excluding borders/artifacts.

    This image-only ROI is intentionally conservative.  It is used as a fallback
    when the fitted grid is not reliable enough to define manuscript-level
    spatial QC bounds.
    """
    if image is None:
        return {
            "qc_roi_source": "full_image",
            "qc_roi_confidence": 0.0,
            "qc_roi_x_min": 0.0,
            "qc_roi_x_max": 1.0,
            "qc_roi_y_min": 0.0,
            "qc_roi_y_max": 1.0,
            "qc_roi_area_fraction": 1.0,
        }

    gray = image
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    height, width = gray.shape[:2]
    if height < 20 or width < 20:
        return {
            "qc_roi_source": "full_image",
            "qc_roi_confidence": 0.15,
            "qc_roi_x_min": 0.0,
            "qc_roi_x_max": float(max(width - 1, 1)),
            "qc_roi_y_min": 0.0,
            "qc_roi_y_max": float(max(height - 1, 1)),
            "qc_roi_area_fraction": 1.0,
        }

    max_dim = 900.0
    scale = min(1.0, max_dim / max(height, width))
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray.copy()

    low = float(np.percentile(small, 2.0))
    high = float(np.percentile(small, 99.2))
    saturated_cutoff = min(250.0, max(high, low + 8.0))
    body_floor = max(5.0, low + 3.0, float(np.percentile(small, 8.0)) + 1.0)

    # Bright saturated blobs and microscope annotations are excluded before
    # morphology.  The chamber array is usually a broad, dim textured region,
    # whereas borders are near-black and bubbles are near-saturated.
    valid = ((small.astype(np.float32) >= body_floor)
             & (small.astype(np.float32) <= saturated_cutoff)).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "qc_roi_source": "full_image",
            "qc_roi_confidence": 0.20,
            "qc_roi_x_min": 0.0,
            "qc_roi_x_max": float(width - 1),
            "qc_roi_y_min": 0.0,
            "qc_roi_y_max": float(height - 1),
            "qc_roi_area_fraction": 1.0,
        }

    image_area_small = float(small.shape[0] * small.shape[1])
    large_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) >= max(25.0, image_area_small * 0.015)
    ]
    if not large_contours:
        large_contours = [max(contours, key=cv2.contourArea)]

    points = np.vstack(large_contours).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(points)
    area_fraction_small = float((w * h) / max(image_area_small, 1.0))
    if area_fraction_small < 0.04:
        confidence = 0.25
    else:
        confidence = float(np.clip(0.45 + min(area_fraction_small, 0.65) * 0.45, 0.0, 0.85))

    margin_x = max(2.0, small.shape[1] * 0.01)
    margin_y = max(2.0, small.shape[0] * 0.01)
    x_min = (x - margin_x) / scale
    x_max = (x + w + margin_x) / scale
    y_min = (y - margin_y) / scale
    y_max = (y + h + margin_y) / scale
    x_min, x_max, y_min, y_max = _clamp_roi(x_min, x_max, y_min, y_max, width, height)
    area = max((x_max - x_min) * (y_max - y_min), 1.0)

    return {
        "qc_roi_source": "effective_array_image",
        "qc_roi_confidence": confidence,
        "qc_roi_x_min": x_min,
        "qc_roi_x_max": x_max,
        "qc_roi_y_min": y_min,
        "qc_roi_y_max": y_max,
        "qc_roi_area_fraction": float(area / max(width * height, 1)),
    }


def estimate_qc_roi(
        image_shape: Tuple[int, int],
        centers: np.ndarray,
        metrics: Optional[Dict] = None,
        image: Optional[np.ndarray] = None,
) -> Dict:
    """Estimate the ROI used only for spatial QC.

    Grid-derived ROI is preferred because it is based on the chamber model
    rather than on the positive distribution itself.
    """
    metrics = metrics or {}
    height, width = int(image_shape[0]), int(image_shape[1])
    radius = max(2.0, float(metrics.get("analysis_radius_px", 0) or 7.0))
    margin = max(8.0, radius * 2.0)
    image_roi = None
    if image is not None:
        image_roi = estimate_effective_array_roi_from_image(image)

    grid_points = np.empty((0, 2), dtype=np.float32)
    pitch = float(metrics.get("grid_pitch_px", 0) or 0)
    target_n = int(
        metrics.get("grid_positions", 0)
        or metrics.get("effective_total_chambers", 0)
        or 0
    )
    grid_unreliable = bool(metrics.get("grid_unreliable", False))
    grid_inlier = float(metrics.get("grid_auto_inlier_fraction", 0) or 0)
    grid_reliable = not grid_unreliable and (grid_inlier == 0 or grid_inlier >= 0.15)
    if metrics.get("grid_mode") and grid_reliable and pitch > 0 and target_n > 0:
        layout = str(metrics.get("grid_layout", "rect") or "rect").lower()
        origin_x = float(metrics.get("grid_origin_x_px", 0) or 0)
        origin_y = float(metrics.get("grid_origin_y_px", 0) or 0)
        angle = float(metrics.get("grid_angle_deg", 0) or 0)
        generated_centers, _ = generate_grid_centers(
            (height, width),
            pitch,
            origin_x,
            origin_y,
            target_n,
            angle_deg=angle,
            layout=layout,
        )
        grid_points = np.asarray(generated_centers, dtype=np.float32)

    if image_roi and image_roi["qc_roi_confidence"] >= 0.50 and not grid_points.size:
        return image_roi
    if grid_points.size:
        source = "grid_model"
        roi_points = grid_points
        confidence = float(np.clip(metrics.get("grid_auto_confidence", 0.75) or 0.75, 0.0, 1.0))
    elif centers.size:
        source = "positive_centers"
        roi_points = centers
        confidence = 0.45
        margin = max(margin, 24.0)
    else:
        source = "full_image"
        roi_points = np.asarray([[0, 0], [width - 1, height - 1]], dtype=np.float32)
        confidence = 0.25

    x_min = float(np.min(roi_points[:, 0])) - margin
    x_max = float(np.max(roi_points[:, 0])) + margin
    y_min = float(np.min(roi_points[:, 1])) - margin
    y_max = float(np.max(roi_points[:, 1])) + margin
    x_min, x_max, y_min, y_max = _clamp_roi(x_min, x_max, y_min, y_max, width, height)
    area = max((x_max - x_min) * (y_max - y_min), 1.0)
    area_fraction = float(area / max(width * height, 1))

    return {
        "qc_roi_source": source,
        "qc_roi_confidence": confidence,
        "qc_roi_x_min": x_min,
        "qc_roi_x_max": x_max,
        "qc_roi_y_min": y_min,
        "qc_roi_y_max": y_max,
        "qc_roi_area_fraction": area_fraction,
    }


def _quadrat_indices(points: np.ndarray, roi: Dict, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = float(roi["qc_roi_x_min"])
    x_max = float(roi["qc_roi_x_max"])
    y_min = float(roi["qc_roi_y_min"])
    y_max = float(roi["qc_roi_y_max"])
    inside = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    pts = points[inside]
    if pts.size == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int), inside
    x_norm = (pts[:, 0] - x_min) / max(x_max - x_min, 1e-6)
    y_norm = (pts[:, 1] - y_min) / max(y_max - y_min, 1e-6)
    x_bin = np.clip(np.floor(x_norm * bins).astype(int), 0, bins - 1)
    y_bin = np.clip(np.floor(y_norm * bins).astype(int), 0, bins - 1)
    return x_bin, y_bin, inside


def _grid_opportunity_counts(
        image_shape: Tuple[int, int],
        roi: Dict,
        metrics: Optional[Dict],
        bins: int,
) -> Tuple[np.ndarray, Dict]:
    """Count theoretical chamber opportunities per quadrat from the grid model."""
    metrics = metrics or {}
    counts = np.zeros((bins, bins), dtype=np.int32)
    info = {
        "spatial_expected_mode": "equal_area",
        "spatial_chamber_opportunity_count": 0,
        "spatial_opportunity_min": "",
        "spatial_opportunity_max": "",
        "spatial_opportunity_cv": "",
        "spatial_opportunity_grid_layout": "",
    }

    if not metrics.get("grid_mode"):
        return counts, info

    pitch = float(metrics.get("grid_pitch_px", 0) or 0)
    target_n = int(
        metrics.get("grid_positions", 0)
        or metrics.get("effective_total_chambers", 0)
        or 0
    )
    if pitch <= 0 or target_n <= 0:
        return counts, info

    height, width = int(image_shape[0]), int(image_shape[1])
    grid_centers, layout = generate_grid_centers(
        (height, width),
        pitch,
        float(metrics.get("grid_origin_x_px", 0) or 0),
        float(metrics.get("grid_origin_y_px", 0) or 0),
        target_n,
        angle_deg=float(metrics.get("grid_angle_deg", 0) or 0),
        layout=str(metrics.get("grid_layout", "hex") or "hex"),
    )
    points = np.asarray(grid_centers, dtype=np.float32)
    if points.size == 0:
        return counts, info

    x_bin, y_bin, inside = _quadrat_indices(points, roi, bins)
    for xb, yb in zip(x_bin, y_bin):
        counts[yb, xb] += 1

    total = int(np.count_nonzero(inside))
    if total <= 0:
        return counts, info

    nonzero = counts[counts > 0]
    info.update({
        "spatial_expected_mode": "grid_opportunity_weighted",
        "spatial_chamber_opportunity_count": total,
        "spatial_opportunity_min": int(np.min(nonzero)) if nonzero.size else 0,
        "spatial_opportunity_max": int(np.max(nonzero)) if nonzero.size else 0,
        "spatial_opportunity_cv": (
            float(np.std(nonzero) / max(float(np.mean(nonzero)), 1e-9))
            if nonzero.size else ""
        ),
        "spatial_opportunity_grid_layout": layout,
    })
    return counts, info


def compute_spatial_qc(
        contours: Iterable,
        image_shape: Tuple[int, int],
        metrics: Optional[Dict] = None,
        bins: int = 10,
        min_count: int = 30,
        image: Optional[np.ndarray] = None,
) -> Dict:
    """Compute 10 x 10 chi-square spatial QC for accepted positives."""
    metrics = metrics or {}
    centers = contour_centers(contours)
    roi = estimate_qc_roi(image_shape, centers, metrics, image=image)
    result = dict(roi)
    result.update({
        "spatial_qc_bins": int(bins),
        "spatial_qc_candidate_count": int(len(centers)),
        "spatial_qc_inside_roi_count": 0,
        "spatial_chi2": "",
        "spatial_chi2_df": "",
        "spatial_df": "",
        "spatial_p_value": "",
        "spatial_pass_p_ge_0_05": "",
        "spatial_qc_status": "not_applicable",
        "spatial_qc_low_quality": True,
        "spatial_qc_flags": [],
        "spatial_max_abs_std_residual": "",
        "spatial_worst_quadrat": "",
        "spatial_edge_center_ratio": "",
        "spatial_edge_density": "",
        "spatial_center_density": "",
        "spatial_expected_per_bin": "",
        "spatial_min_expected_per_bin": "",
        "spatial_expected_mode": "equal_area",
        "spatial_chamber_opportunity_count": 0,
        "spatial_opportunity_min": "",
        "spatial_opportunity_max": "",
        "spatial_opportunity_cv": "",
        "spatial_opportunity_grid_layout": "",
        "spatial_quadrat_counts": [],
        "spatial_quadrat_residuals": [],
    })

    if centers.size == 0:
        result["spatial_qc_flags"] = ["no_positive_centers"]
        return result

    x_bin, y_bin, inside = _quadrat_indices(centers, roi, bins)
    inside_count = int(np.count_nonzero(inside))
    result["spatial_qc_inside_roi_count"] = inside_count
    outside_count = int(len(centers) - inside_count)

    flags: List[str] = []
    if outside_count:
        flags.append("positive_centers_outside_qc_roi")
    if roi["qc_roi_confidence"] < 0.50:
        flags.append("low_confidence_qc_roi")
    if inside_count < min_count:
        flags.append("insufficient_positive_count_for_spatial_qc")

    observed = np.zeros((bins, bins), dtype=np.int32)
    for xb, yb in zip(x_bin, y_bin):
        observed[yb, xb] += 1

    grid_unreliable = bool(metrics.get("grid_unreliable", False))
    grid_inlier = float(metrics.get("grid_auto_inlier_fraction", 0) or 0)
    grid_not_formal = bool(
        metrics.get("grid_mode")
        and (
            grid_unreliable
            or (grid_inlier > 0 and grid_inlier < 0.15)
            or roi["qc_roi_confidence"] < 0.50
        )
    )
    if grid_not_formal:
        if grid_unreliable or (grid_inlier > 0 and grid_inlier < 0.15):
            flags.append("grid_unreliable_for_spatial_qc")
        if roi["qc_roi_confidence"] < 0.50:
            flags.append("roi_unreliable_for_spatial_qc")
        counts_records = []
        residual_records = []
        for y in range(bins):
            for x in range(bins):
                counts_records.append({
                    "x_bin": int(x + 1),
                    "y_bin": int(y + 1),
                    "count": int(observed[y, x]),
                    "expected": "",
                    "chamber_opportunity": "",
                    "opportunity_fraction": "",
                })
                residual_records.append({
                    "x_bin": int(x + 1),
                    "y_bin": int(y + 1),
                    "std_residual": "",
                    "observed": int(observed[y, x]),
                    "expected": "",
                    "chamber_opportunity": "",
                })
        result.update({
            "spatial_qc_inside_roi_count": inside_count,
            "spatial_expected_mode": "not_recommended_grid_unreliable",
            "spatial_qc_status": "not_recommended",
            "spatial_qc_low_quality": True,
            "spatial_qc_flags": sorted(set(flags)),
            "spatial_quadrat_counts": counts_records,
            "spatial_quadrat_residuals": residual_records,
        })
        return result

    opportunity, opportunity_info = _grid_opportunity_counts(image_shape, roi, metrics, bins)
    result.update(opportunity_info)
    if opportunity_info["spatial_expected_mode"] == "grid_opportunity_weighted":
        opportunity_total = max(int(np.sum(opportunity)), 1)
        expected_matrix = inside_count * opportunity.astype(np.float64) / float(opportunity_total)
    else:
        expected_matrix = np.full(
            (bins, bins),
            inside_count / float(bins * bins),
            dtype=np.float64,
        )

    valid = expected_matrix > 0
    if np.any((observed > 0) & ~valid):
        flags.append("positive_in_zero_opportunity_quadrat")
    if not np.any(valid):
        flags.append("no_valid_spatial_expected_bins")
        result["spatial_qc_flags"] = flags
        return result

    residuals = np.zeros((bins, bins), dtype=np.float64)
    residuals[valid] = (
        observed.astype(np.float64)[valid] - expected_matrix[valid]
    ) / np.sqrt(expected_matrix[valid])
    chi2 = float(np.sum(
        (observed.astype(np.float64)[valid] - expected_matrix[valid]) ** 2
        / expected_matrix[valid]
    ))
    df = int(np.count_nonzero(valid) - 1)
    p_value = float(chi2_distribution.sf(chi2, df))
    chi2_df = float(chi2 / max(df, 1))
    expected = inside_count / float(bins * bins)
    min_expected = float(np.min(expected_matrix[valid]))
    statistically_applicable = inside_count >= min_count
    if min_expected < 5.0:
        flags.append("low_expected_count_per_quadrat")

    abs_resid = np.abs(residuals)
    worst_y, worst_x = np.unravel_index(int(np.argmax(abs_resid)), abs_resid.shape)
    max_abs_resid = float(abs_resid[worst_y, worst_x])

    edge_mask = np.zeros((bins, bins), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    center_mask = np.zeros((bins, bins), dtype=bool)
    center_start = max(1, bins // 2 - 2)
    center_end = min(bins, bins // 2 + 2)
    center_mask[center_start:center_end, center_start:center_end] = True
    edge_density = float(np.mean(observed[edge_mask]))
    center_density = float(np.mean(observed[center_mask]))
    edge_center_ratio = float(edge_density / max(center_density, 1e-6))

    if statistically_applicable:
        if p_value < 0.05:
            flags.append("spatial_chi_square_failed")
        if chi2_df > 1.50:
            flags.append("spatial_overdispersed")
        if chi2_df < 0.50:
            flags.append("spatial_underdispersed")
        if max_abs_resid > 4.0:
            flags.append("localized_residual_hotspot")
        if edge_center_ratio > 2.25 or edge_center_ratio < 0.35:
            flags.append("edge_center_imbalance")

    low_quality = bool(flags)
    if inside_count < min_count:
        status = "not_applicable"
    else:
        status = "pass" if not low_quality else ("warning" if p_value >= 0.05 else "fail")
    counts_records = []
    residual_records = []
    for y in range(bins):
        for x in range(bins):
            counts_records.append({
                "x_bin": int(x + 1),
                "y_bin": int(y + 1),
                "count": int(observed[y, x]),
                "expected": float(expected_matrix[y, x]),
                "chamber_opportunity": int(opportunity[y, x]) if opportunity.size else "",
                "opportunity_fraction": (
                    float(opportunity[y, x] / max(np.sum(opportunity), 1))
                    if opportunity.size and np.sum(opportunity) > 0 else ""
                ),
            })
            residual_records.append({
                "x_bin": int(x + 1),
                "y_bin": int(y + 1),
                "std_residual": float(residuals[y, x]),
                "observed": int(observed[y, x]),
                "expected": float(expected_matrix[y, x]),
                "chamber_opportunity": int(opportunity[y, x]) if opportunity.size else "",
            })

    result.update({
        "spatial_chi2": chi2,
        "spatial_chi2_df": chi2_df,
        "spatial_df": df,
        "spatial_p_value": p_value,
        "spatial_pass_p_ge_0_05": bool(p_value >= 0.05),
        "spatial_qc_status": status,
        "spatial_qc_low_quality": low_quality,
        "spatial_qc_flags": flags,
        "spatial_max_abs_std_residual": max_abs_resid,
        "spatial_worst_quadrat": f"{worst_x + 1},{worst_y + 1}",
        "spatial_edge_center_ratio": edge_center_ratio,
        "spatial_edge_density": edge_density,
        "spatial_center_density": center_density,
        "spatial_expected_per_bin": expected,
        "spatial_min_expected_per_bin": min_expected,
        "spatial_quadrat_counts": counts_records,
        "spatial_quadrat_residuals": residual_records,
    })
    return result


def spatial_bin_for_point(x: float, y: float, spatial_qc: Dict) -> Dict:
    """Return quadrat metadata for one point using a computed spatial QC ROI."""
    bins = int(spatial_qc.get("spatial_qc_bins", 10) or 10)
    x_min = float(spatial_qc.get("qc_roi_x_min", 0) or 0)
    x_max = float(spatial_qc.get("qc_roi_x_max", 1) or 1)
    y_min = float(spatial_qc.get("qc_roi_y_min", 0) or 0)
    y_max = float(spatial_qc.get("qc_roi_y_max", 1) or 1)
    inside = x_min <= x <= x_max and y_min <= y <= y_max
    if not inside:
        return {
            "spatial_x_bin": "",
            "spatial_y_bin": "",
            "spatial_bin_expected": spatial_qc.get("spatial_expected_per_bin", ""),
            "spatial_bin_std_residual": "",
            "spatial_inside_qc_roi": False,
        }

    x_bin = int(np.clip(np.floor((x - x_min) / max(x_max - x_min, 1e-6) * bins), 0, bins - 1))
    y_bin = int(np.clip(np.floor((y - y_min) / max(y_max - y_min, 1e-6) * bins), 0, bins - 1))
    residual_lookup = {
        (int(row["x_bin"]), int(row["y_bin"])): row
        for row in spatial_qc.get("spatial_quadrat_residuals", [])
    }
    row = residual_lookup.get((x_bin + 1, y_bin + 1), {})
    return {
        "spatial_x_bin": x_bin + 1,
        "spatial_y_bin": y_bin + 1,
        "spatial_bin_expected": row.get("expected", spatial_qc.get("spatial_expected_per_bin", "")),
        "spatial_bin_std_residual": row.get("std_residual", ""),
        "spatial_inside_qc_roi": True,
    }


def flatten_spatial_qc_for_report(spatial_qc: Dict) -> Dict:
    """Return scalar fields suitable for CSV reports."""
    scalar = dict(spatial_qc or {})
    scalar.pop("spatial_quadrat_counts", None)
    scalar.pop("spatial_quadrat_residuals", None)
    flags = scalar.get("spatial_qc_flags", [])
    if isinstance(flags, list):
        scalar["spatial_qc_flags"] = ";".join(flags)
    scalar["qc_roi_bounds"] = (
        f"{scalar.get('qc_roi_x_min', ''):.1f},"
        f"{scalar.get('qc_roi_y_min', ''):.1f},"
        f"{scalar.get('qc_roi_x_max', ''):.1f},"
        f"{scalar.get('qc_roi_y_max', ''):.1f}"
        if scalar.get("qc_roi_x_min", "") != ""
        else ""
    )
    return scalar
