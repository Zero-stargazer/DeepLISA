#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLISA 工具箱底层 (v14.0 - 梯度敏感版)
修复说明：
1. [评分系统] 修正了对高密度信号的误杀。重点奖励“半径一致性”，不再因为“点多”而盲目扣分。
2. [急救策略] 废弃“休克疗法”，采用“梯度下降”策略。防止从 99999 直接杀到 7。
3. [物理约束] 针对阵列芯片，强化了网格特征的识别权重。
"""

import csv
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import KDTree
from typing import Dict, List, Tuple

from config import settings
from image_utils import load_image
from advanced_filtering import (apply_advanced_preprocessing, binarize_adaptive,
                                advanced_spot_filtering, enhance_image_quality)
from grid_detection import (
    auto_calibrate_grid,
    detect_positive_chambers_on_grid,
    generate_grid_centers,
)
from quality_control import QualityController, QualityMetrics
from spatial_qc import estimate_effective_array_roi_from_image


def get_adaptive_block_size(avg_radius: float) -> int:
    """根据平均腔室半径返回自适应 block_size (v15.0)
    block_size ≈ avg_radius × 2.5，保证奇数"""
    bs = max(3, int(avg_radius * 2.5))
    return bs | 1


def get_chamber_tier_params(chamber_type: str = "medium"):
    """根据腔室类型返回 (min_radius, max_radius, block_size)"""
    tier = settings.CHAMBER_PARAMS.get(f"{chamber_type}_chambers",
                                        settings.CHAMBER_PARAMS["medium_chambers"])
    min_r, max_r = tier["min_radius"], tier["max_radius"]
    avg_r = (min_r + max_r) / 2
    bs = get_adaptive_block_size(avg_r)
    return min_r, max_r, bs


def get_runtime_radius_bounds() -> Tuple[int, int]:
    """Use the GUI/user radius as the primary scale instead of a fixed tier."""
    r = max(2, int(settings.RADIUS))
    return max(1, int(round(r * 0.45))), max(r + 2, int(round(r * 1.8)))


def get_tophat_kernel_size() -> int:
    """Return a kernel wider than the expected fluorescent chamber signal."""
    tier_kernel = int(settings.KERNEL_MAP.get(
        settings.CHAMBER_TYPE,
        settings.TOPHAT_DEFAULT["kernel_size"],
    ))
    radius_kernel = (max(2, int(settings.RADIUS)) * 4 + 1) | 1
    return max(int(settings.TOPHAT_DEFAULT["kernel_size"]), tier_kernel, radius_kernel)


def suppress_close_contours(
        contours: List,
        img: np.ndarray,
        min_distance: float = None,
        return_audit: bool = False,
):
    """Enforce the physical chamber-spacing prior.

    Adjacent chambers on the array cannot overlap or touch. When multiple
    candidates are closer than the expected chamber pitch, keep the strongest
    local candidate and suppress the rest.
    """
    if not contours:
        result = ([], {"suppressed_close": 0, "min_center_distance": min_distance or 0})
        return (*result, []) if return_audit else result

    radius = max(2, int(settings.RADIUS))
    if min_distance is None:
        min_distance = max(radius * 2.2, settings.CHAMBER_PARAMS["medium_chambers"].get("min_distance", 8))

    candidates = []
    h, w = img.shape
    for idx, c in enumerate(contours):
        ((x, y), r) = cv2.minEnclosingCircle(c)
        ix, iy = int(x), int(y)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (ix, iy), max(2, int(round(radius * 0.7))), 255, -1)
        intensity = cv2.mean(img, mask=mask)[0]
        candidates.append((float(intensity), idx, float(x), float(y), c))

    candidates.sort(reverse=True, key=lambda item: item[0])
    kept = []
    kept_xy = []
    audit_records = [] if return_audit else None
    min_dist2 = float(min_distance) ** 2

    for intensity, _, x, y, c in candidates:
        too_close = False
        for kx, ky in kept_xy:
            if (x - kx) ** 2 + (y - ky) ** 2 < min_dist2:
                too_close = True
                break
        if not too_close:
            kept.append(c)
            kept_xy.append((x, y))
        if audit_records is not None:
            audit_records.append({
                "stage": "spacing",
                "x": float(x),
                "y": float(y),
                "intensity": float(intensity),
                "reason": "suppressed_close" if too_close else "accepted",
            })

    stats = {"suppressed_close": len(candidates) - len(kept), "min_center_distance": float(min_distance)}
    return (kept, stats, audit_records) if return_audit else (kept, stats)


def filter_by_local_snr_adaptive(
        img: np.ndarray,
        contours: List,
        return_audit: bool = False,
):
    """Local-background gate for bright-background chips.

    Each candidate is judged against its surrounding annulus rather than a
    whole-image threshold. This prevents high probe/background images from
    erasing weak-but-real chamber positives.
    """
    if not contours:
        result = ([], {"global_threshold": 0.0, "high_background": False})
        return (*result, []) if return_audit else result

    h, w = img.shape
    scale = 0.25
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    bg_base = np.percentile(small, 40)
    bg_median = np.percentile(small, 50)
    bg_p90 = np.percentile(small, 90)
    noise_pixels = small[small < bg_base]
    noise_sigma = np.std(noise_pixels) if noise_pixels.size else np.std(small)
    if noise_sigma < 1: noise_sigma = 1.0
    high_background = bool(bg_median > 35 and (bg_p90 - bg_median) > 10)

    # Global fallback threshold. The main decision uses local annulus contrast.
    min_contrast_threshold = bg_base + max(1.2 * noise_sigma, 2.0)

    valid_contours = []
    local_z_values = []
    local_delta_values = []
    rejected_tiny_support = 0
    audit_records = [] if return_audit else None

    for c in contours:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        ix, iy = int(x), int(y)

        if not (0 <= ix < w and 0 <= iy < h):
            if audit_records is not None:
                audit_records.append({
                    "stage": "local_snr",
                    "x": float(x),
                    "y": float(y),
                    "radius": float(r),
                    "reason": "local_center_outside_image",
                })
            continue

        inner_r = max(2, int(round(max(r * 0.75, settings.RADIUS * 0.55))))
        outer_r = max(inner_r + 3, int(round(max(r * 2.5, settings.RADIUS * 2.0))))

        mask_inner = np.zeros_like(img, dtype=np.uint8)
        mask_outer = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask_inner, (ix, iy), inner_r, 255, -1)
        cv2.circle(mask_outer, (ix, iy), outer_r, 255, -1)
        ring = cv2.subtract(mask_outer, mask_inner)

        inner_mean = cv2.mean(img, mask=mask_inner)[0]
        ring_pixels = img[ring > 0]
        if ring_pixels.size < 8:
            ring_mean = bg_base
            ring_std = noise_sigma
        else:
            ring_mean = float(np.mean(ring_pixels))
            ring_std = float(np.std(ring_pixels))
            if ring_std < 1:
                ring_std = 1.0

        local_delta = inner_mean - ring_mean
        local_z = local_delta / (ring_std + 1e-6)
        local_delta_values.append(local_delta)
        local_z_values.append(local_z)

        # Reject adsorption/hot-pixel artifacts. A true chamber signal should
        # occupy a small disk, not one or two isolated pixels in a channel.
        inner_pixels = img[mask_inner > 0]
        support_cutoff = ring_mean + max(1.0, 0.35 * ring_std)
        support_pixels = int(np.count_nonzero(inner_pixels > support_cutoff))
        min_support_pixels = max(4, int(np.pi * (max(1, inner_r) ** 2) * 0.25))
        if support_pixels < min_support_pixels:
            rejected_tiny_support += 1
            if audit_records is not None:
                audit_records.append({
                    "stage": "local_snr",
                    "x": float(x),
                    "y": float(y),
                    "radius": float(r),
                    "inner_mean": float(inner_mean),
                    "ring_mean": float(ring_mean),
                    "ring_std": float(ring_std),
                    "local_delta": float(local_delta),
                    "local_z": float(local_z),
                    "support_pixels": support_pixels,
                    "min_support_pixels": min_support_pixels,
                    "reason": "local_tiny_support",
                })
            continue

        z_cutoff = 0.35 if high_background else 0.7
        delta_cutoff = max(0.8, 0.20 * ring_std) if high_background else max(1.5, 0.40 * ring_std)
        global_fallback = img[iy, ix] > min_contrast_threshold and local_delta > 0

        accepted = (local_delta >= delta_cutoff and local_z >= z_cutoff) or global_fallback
        if audit_records is not None:
            audit_records.append({
                "stage": "local_snr",
                "x": float(x),
                "y": float(y),
                "radius": float(r),
                "inner_mean": float(inner_mean),
                "ring_mean": float(ring_mean),
                "ring_std": float(ring_std),
                "local_delta": float(local_delta),
                "local_z": float(local_z),
                "support_pixels": support_pixels,
                "min_support_pixels": min_support_pixels,
                "reason": "accepted" if accepted else "local_signal_below_cutoff",
            })
        if accepted:
            valid_contours.append(c)

    stats = {
        "global_threshold": float(min_contrast_threshold),
        "high_background": high_background,
        "local_delta_median": float(np.median(local_delta_values)) if local_delta_values else 0.0,
        "local_z_median": float(np.median(local_z_values)) if local_z_values else 0.0,
        "rejected_tiny_support": rejected_tiny_support,
    }
    return (valid_contours, stats, audit_records) if return_audit else (valid_contours, stats)


def _circular_contours_from_points(points: np.ndarray, radius: int) -> List[np.ndarray]:
    contours = []
    radius = max(2, int(radius))
    for x, y in points:
        poly = cv2.ellipse2Poly((int(x), int(y)), (radius, radius), 0, 0, 360, 30)
        contours.append(poly.reshape((-1, 1, 2)).astype(np.int32))
    return contours


def _local_peak_contours(flat: np.ndarray, max_points: int) -> List[np.ndarray]:
    """Find local intensity peaks without requiring closed binary regions."""
    smooth = cv2.GaussianBlur(flat, (3, 3), 0.6)
    radius = max(2, int(settings.RADIUS))
    ksize = max(3, radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(smooth, kernel)

    # Use a permissive floor; the local-annulus gate downstream remains the
    # main false-positive control.
    threshold = max(float(np.percentile(smooth, 60)), float(np.median(smooth) + 0.05 * np.std(smooth)))
    peak_mask = (smooth == dilated) & (smooth > threshold)
    ys, xs = np.where(peak_mask)
    if len(xs) == 0:
        return []

    values = smooth[ys, xs]
    order = np.argsort(values)[::-1]
    if len(order) > max_points:
        order = order[:max_points]
    points = np.column_stack([xs[order], ys[order]])
    return _circular_contours_from_points(points, radius)


def _contour_center(contour) -> Tuple[float, float]:
    ((x, y), _) = cv2.minEnclosingCircle(contour)
    return float(x), float(y)


def filter_contours_to_roi(contours: List, roi: Dict, min_confidence: float = 0.50) -> Tuple[List, Dict]:
    """Keep contours whose centers lie inside a high-confidence effective ROI."""
    if not roi or float(roi.get("qc_roi_confidence", 0.0) or 0.0) < min_confidence:
        return list(contours or []), {
            "roi_filter_applied": False,
            "roi_filtered_outside_count": 0,
        }

    x_min = float(roi.get("qc_roi_x_min", 0.0) or 0.0)
    x_max = float(roi.get("qc_roi_x_max", 0.0) or 0.0)
    y_min = float(roi.get("qc_roi_y_min", 0.0) or 0.0)
    y_max = float(roi.get("qc_roi_y_max", 0.0) or 0.0)
    kept = []
    rejected = 0
    for contour in contours or []:
        x, y = _contour_center(contour)
        if x_min <= x <= x_max and y_min <= y <= y_max:
            kept.append(contour)
        else:
            rejected += 1
    return kept, {
        "roi_filter_applied": True,
        "roi_filtered_outside_count": int(rejected),
        "roi_filter_source": roi.get("qc_roi_source", ""),
        "roi_filter_confidence": roi.get("qc_roi_confidence", ""),
    }


def _local_spot_evidence(img: np.ndarray, contour, radius: int):
    x, y = _contour_center(contour)
    ix, iy = int(round(x)), int(round(y))
    radius = max(2, int(radius))
    inner_r = max(2, int(round(radius * 0.65)))
    outer_r = max(inner_r + 3, int(round(radius * 1.8)))
    h, w = img.shape
    x1, x2 = ix - outer_r, ix + outer_r + 1
    y1, y2 = iy - outer_r, iy + outer_r + 1
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None

    patch = img[y1:y2, x1:x2]
    yy, xx = np.ogrid[-outer_r:outer_r + 1, -outer_r:outer_r + 1]
    dist2 = xx * xx + yy * yy
    inner_pixels = patch[dist2 <= inner_r * inner_r]
    ring_pixels = patch[(dist2 > inner_r * inner_r) & (dist2 <= outer_r * outer_r)]
    ring_mean = float(np.mean(ring_pixels))
    ring_std = max(1.0, float(np.std(ring_pixels)))
    inner_mean = float(np.mean(inner_pixels))
    local_delta = inner_mean - ring_mean
    local_z = local_delta / ring_std
    support_cutoff = ring_mean + max(1.0, 0.35 * ring_std)
    support_fraction = float(np.count_nonzero(inner_pixels > support_cutoff) / max(inner_pixels.size, 1))
    return {
        "x": x,
        "y": y,
        "local_delta": local_delta,
        "local_z": local_z,
        "support_fraction": support_fraction,
        "ring_std": ring_std,
    }


def detect_spot_first_candidates(
        enhanced: np.ndarray,
        radius: int,
        target_n: int,
) -> Tuple[List[np.ndarray], Dict]:
    """Detect bright circular spots before grid assignment.

    This branch intentionally uses a high-confidence tail threshold.  It is a
    recall path for real bright chambers that do not land exactly on an
    auto-fitted grid center, not a replacement for local background filtering.
    """
    bg = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=max(10.0, float(radius) * 2.5))
    flat = cv2.subtract(enhanced, bg)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    peak_contours = _local_peak_contours(flat, max_points=min(max(int(target_n * 1.2), 5000), 60000))
    component_threshold = float(np.percentile(flat, 98.0))
    component_mask = (flat >= component_threshold).astype("uint8") * 255
    min_r, max_r = get_runtime_radius_bounds()
    component_filtered, _ = advanced_spot_filtering(
        component_mask,
        min_radius=min_r,
        max_radius=max_r,
        min_circularity=0.25,
    )
    component_contours, _ = cv2.findContours(
        component_filtered,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    raw_contours = list(peak_contours) + list(component_contours)
    evidence_rows = []
    for contour in raw_contours:
        evidence = _local_spot_evidence(enhanced, contour, radius)
        if evidence is not None:
            evidence_rows.append((contour, evidence))

    if not evidence_rows:
        return [], {
            "spot_first_peak_candidates": 0,
            "spot_first_component_candidates": 0,
            "spot_first_count": 0,
            "spot_first_delta_threshold": 0.0,
            "spot_first_z_threshold": 0.0,
        }

    deltas = np.asarray([row["local_delta"] for _, row in evidence_rows], dtype=np.float32)
    delta_median = float(np.median(deltas))
    delta_mad = float(np.median(np.abs(deltas - delta_median)) * 1.4826)
    delta_p98 = float(np.percentile(deltas, 98))
    delta_p99 = float(np.percentile(deltas, 99))
    tail_threshold = delta_p98
    robust_threshold = delta_median + 5.0 * max(delta_mad, 0.05)
    delta_threshold = max(5.0, min(tail_threshold, robust_threshold) * 0.98)
    if delta_p99 > delta_threshold * 1.8:
        delta_threshold = max(delta_threshold, delta_p98 + 0.20 * (delta_p99 - delta_p98))
    z_threshold = 1.05
    min_support_fraction = 0.70

    accepted = []
    for contour, evidence in evidence_rows:
        if (
                evidence["local_delta"] >= delta_threshold
                and evidence["local_z"] >= z_threshold
                and evidence["support_fraction"] >= min_support_fraction
        ):
            accepted.append(contour)

    accepted, spacing_stats = suppress_close_contours(
        accepted,
        enhanced,
        min_distance=max(6.0, float(radius) * 1.15),
    )
    return accepted, {
        "spot_first_peak_candidates": len(peak_contours),
        "spot_first_component_candidates": len(component_contours),
        "spot_first_component_threshold": float(component_threshold),
        "spot_first_evaluated_candidates": len(evidence_rows),
        "spot_first_count": len(accepted),
        "spot_first_delta_threshold": float(delta_threshold),
        "spot_first_z_threshold": float(z_threshold),
        "spot_first_min_support_fraction": float(min_support_fraction),
        "spot_first_suppressed_close": spacing_stats.get("suppressed_close", 0),
    }


def merge_spot_and_grid_contours(
        spot_contours: List,
        grid_contours: List,
        radius: int,
) -> Tuple[List, Dict]:
    """Prefer spot centers, then add grid-only positives that are not duplicates."""
    merged = list(spot_contours)
    merge_distance = max(6.0, float(radius) * 1.20)
    merge_distance2 = merge_distance * merge_distance
    spot_centers = [_contour_center(c) for c in merged]
    grid_only = 0

    for contour in grid_contours:
        gx, gy = _contour_center(contour)
        duplicate = any((gx - sx) ** 2 + (gy - sy) ** 2 <= merge_distance2 for sx, sy in spot_centers)
        if not duplicate:
            merged.append(contour)
            spot_centers.append((gx, gy))
            grid_only += 1

    return merged, {
        "hybrid_detection": True,
        "hybrid_merged_count": len(merged),
        "hybrid_grid_only_count": grid_only,
        "hybrid_duplicate_grid_count": max(0, len(grid_contours) - grid_only),
        "hybrid_merge_distance_px": float(merge_distance),
    }


def calculate_spot_grid_association(
        spot_contours: List,
        image_shape: Tuple[int, int],
        grid_stats: Dict,
        radius: int,
) -> Dict:
    pitch = float(grid_stats.get("grid_pitch_px", 0) or 0)
    if not spot_contours or pitch <= 0:
        return {
            "spot_grid_associated_count": 0,
            "spot_off_grid_count": len(spot_contours),
            "spot_grid_association_radius_px": 0.0,
        }

    centers, _ = generate_grid_centers(
        image_shape,
        pitch,
        float(grid_stats.get("grid_origin_x_px", 0) or 0),
        float(grid_stats.get("grid_origin_y_px", 0) or 0),
        int(grid_stats.get("grid_positions", 0) or 0),
        angle_deg=float(grid_stats.get("grid_angle_deg", 0) or 0),
        layout=str(grid_stats.get("grid_layout", "hex")),
    )
    if not centers:
        return {
            "spot_grid_associated_count": 0,
            "spot_off_grid_count": len(spot_contours),
            "spot_grid_association_radius_px": 0.0,
        }

    tree = KDTree(np.asarray(centers, dtype=np.float32))
    spot_centers = np.asarray([_contour_center(c) for c in spot_contours], dtype=np.float32)
    distances, _ = tree.query(spot_centers)
    association_radius = max(float(radius) * 1.25, min(pitch * 0.55, pitch - float(radius) * 0.20))
    associated = int(np.count_nonzero(distances <= association_radius))
    return {
        "spot_grid_associated_count": associated,
        "spot_off_grid_count": int(len(spot_contours) - associated),
        "spot_grid_association_radius_px": float(association_radius),
        "spot_grid_distance_median_px": float(np.median(distances)) if distances.size else 0.0,
    }


def detect_high_background_rescue(raw_gray: np.ndarray, enhanced: np.ndarray, base_params: Dict,
                                  target_N: int = 21600) -> Tuple[List, Dict]:
    """Second-pass candidate recovery for high-background/high-concentration chips.

    The normal TopHat path can remove weak positives when the whole image is
    bright. This rescue pass uses a gentler background subtraction and a more
    permissive adaptive threshold, then still applies shape and local-annulus
    gates to control false positives.
    """
    bg = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=18, sigmaY=18)
    flat = cv2.subtract(enhanced, bg)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    min_r, max_r = get_runtime_radius_bounds()

    blk = max(9, int(base_params.get("block_size", 15)) | 1)
    relaxed_offset = min(int(base_params.get("c_offset", -5)) + 7, 4)
    binary = binarize_adaptive(flat, blk, relaxed_offset)

    filtered, _ = advanced_spot_filtering(
        binary,
        min_radius=min_r,
        max_radius=max_r,
        min_circularity=max(0.25, float(base_params.get("min_circularity", 0.5)) - 0.2)
    )
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    peak_contours = _local_peak_contours(flat, max_points=min(int(target_N * 1.2), 30000))
    combined = contours + peak_contours
    final_contours, stats = filter_by_local_snr_adaptive(enhanced, combined)
    final_contours, spacing_stats = suppress_close_contours(final_contours, enhanced)
    stats.update(spacing_stats)
    stats["rescue_candidates"] = len(combined)
    stats["rescue_threshold_candidates"] = len(contours)
    stats["rescue_peak_candidates"] = len(peak_contours)
    stats["rescue_offset"] = relaxed_offset
    stats["rescue_block_size"] = blk
    return final_contours, stats


DIST_CV_MONTE_CARLO_REFERENCE = (
    (0.002, 0.5570, 0.4094, 0.7267),
    (0.005, 0.5446, 0.4506, 0.6443),
    (0.010, 0.5344, 0.4668, 0.6047),
    (0.020, 0.5158, 0.4728, 0.5607),
    (0.050, 0.4795, 0.4541, 0.5055),
    (0.100, 0.4268, 0.4131, 0.4400),
    (0.200, 0.3288, 0.3211, 0.3366),
)


def get_expected_dist_cv(fill_ratio: float) -> Tuple[float, float, float]:
    """Interpolate the full-contour Monte Carlo reference curve."""
    ratios = np.array([item[0] for item in DIST_CV_MONTE_CARLO_REFERENCE])
    means = np.array([item[1] for item in DIST_CV_MONTE_CARLO_REFERENCE])
    lows = np.array([item[2] for item in DIST_CV_MONTE_CARLO_REFERENCE])
    highs = np.array([item[3] for item in DIST_CV_MONTE_CARLO_REFERENCE])
    return (
        float(np.interp(fill_ratio, ratios, means)),
        float(np.interp(fill_ratio, ratios, lows)),
        float(np.interp(fill_ratio, ratios, highs)),
    )


def _write_png(path: Path, img: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Unable to encode diagnostic image: {path}")
    encoded.tofile(str(path))


def export_detection_audit(
        audit_dir: str,
        img_path: str,
        stage_images: Dict[str, np.ndarray],
        metrics: Dict,
        records: List[Dict],
) -> str:
    """Write optional per-stage evidence without changing default outputs."""
    safe_name = re.sub(r'[<>:"/\\|?*]+', "_", Path(img_path).stem)
    output_dir = Path(audit_dir) / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage_name, stage_img in stage_images.items():
        _write_png(output_dir / f"{stage_name}.png", stage_img)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    if records:
        fieldnames = sorted({key for record in records for key in record.keys()})
        with (output_dir / "candidate_decisions.csv").open(
                "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    return str(output_dir)


def calculate_distribution_metrics(contours, target_N: int = 21600) -> Dict:
    """Calculate deterministic geometry metrics from every accepted contour."""
    metrics = {"count": 0, "radius_mean": 0, "radius_std": 0, "radius_cv": 0.0,
               "dist_cv": 0.0, "area_ratio": None, "high_density_mode": False,
               "fill_ratio": 0.0, "dist_cv_applicable": False,
               "dist_cv_expected": None, "dist_cv_ci_low": None,
               "dist_cv_ci_high": None}
    count = len(contours)
    metrics["count"] = count

    if count == 0: return metrics

    # 1. 半径一致性。使用全部轮廓，保证同一张图重复运行得到相同结果。
    radii = [cv2.minEnclosingCircle(c)[1] for c in contours]

    q1, q3 = np.percentile(radii, [25, 75])
    iqr = q3 - q1
    clean_radii = [r for r in radii if (q1 - 1.5 * iqr) <= r <= (q3 + 1.5 * iqr)]

    if clean_radii and len(clean_radii) > 5:
        metrics["radius_mean"] = np.mean(clean_radii)
        metrics["radius_std"] = np.std(clean_radii)
        metrics["radius_cv"] = metrics["radius_std"] / (metrics["radius_mean"] + 1e-6)

    # 2. 高密度切换检测
    fill_ratio = count / max(target_N, 1)
    metrics["fill_ratio"] = fill_ratio
    metrics["dist_cv_applicable"] = 0.01 <= fill_ratio <= 0.2
    if metrics["dist_cv_applicable"]:
        expected, ci_low, ci_high = get_expected_dist_cv(fill_ratio)
        metrics["dist_cv_expected"] = expected
        metrics["dist_cv_ci_low"] = ci_low
        metrics["dist_cv_ci_high"] = ci_high
    if fill_ratio > 0.5:
        metrics["high_density_mode"] = True
        # 面积法替代最近邻CV
        total_contour_area = sum(cv2.contourArea(c) for c in contours)
        expected_area = count * np.pi * (settings.RADIUS ** 2)
        metrics["area_ratio"] = total_contour_area / (expected_area + 1e-6)
        metrics["dist_cv"] = None  # 标记不可用
        return metrics

    # 3. 距离 CV (低中密度模式)
    if count < 10:
        metrics["dist_cv"] = 0.0
        return metrics

    calc_centers = [cv2.minEnclosingCircle(c)[0] for c in contours]
    tree = KDTree(calc_centers)
    dists, _ = tree.query(calc_centers, k=2)
    neighbor_dists = dists[:, 1]
    valid_dists = neighbor_dists[neighbor_dists > 1.0]

    if len(valid_dists) > 5:
        mean_dist = np.mean(valid_dists)
        std_dist = np.std(valid_dists)
        metrics["dist_cv"] = std_dist / (mean_dist + 1e-6)
    else:
        metrics["dist_cv"] = 1.0

    return metrics


def calculate_grid_signal_quality(grid_stats: Dict) -> float:
    """Score how far accepted grid positives sit above adaptive cutoffs."""
    if not grid_stats.get("grid_mode") or grid_stats.get("grid_accepted", 0) <= 0:
        return 0.0

    def unit_margin(value: float, threshold: float) -> float:
        threshold = max(float(threshold), 1e-6)
        ratio = max(0.0, float(value) / threshold)
        return float(ratio / (ratio + 1.0))

    delta_q = unit_margin(
        grid_stats.get("grid_accepted_delta_median", 0.0),
        grid_stats.get("grid_delta_threshold", 1.0),
    )
    z_q = unit_margin(
        grid_stats.get("grid_accepted_z_median", 0.0),
        grid_stats.get("grid_z_threshold", 1.0),
    )
    support_q = unit_margin(
        grid_stats.get("grid_accepted_support_median", 0.0),
        grid_stats.get("grid_min_support_fraction", 0.32),
    )
    weak_fraction = float(grid_stats.get("grid_weak_positive_fraction", 0.0))
    return float(np.clip(
        0.35 * delta_q
        + 0.35 * z_q
        + 0.20 * support_q
        + 0.10 * (1.0 - weak_fraction),
        0.0,
        1.0,
    ))


def calculate_score(count, d_met, i_met, target_N, img_std) -> float:
    """
    [重构] 评分函数 v15.0
    逻辑：radius_cv(物理一致性) > 数量合理性 > dist_cv(泊松区间验证)
    Monte Carlo correction: dist_cv is fill-ratio dependent. Compare against
    the simulated curve only for calibrated low-to-moderate occupancy.
    """
    if count == 0: return 10.0
    if count > (target_N * 1.5): return 10.0

    radius_cv = d_met.get('radius_cv', 1.0)
    r_std = d_met.get('radius_std', 9.9)
    dist_cv = d_met.get('dist_cv', None)
    area_ratio = d_met.get('area_ratio', None)
    high_density = d_met.get('high_density_mode', False)
    dist_cv_applicable = d_met.get('dist_cv_applicable', False)
    dist_cv_expected = d_met.get('dist_cv_expected')
    dist_cv_ci_low = d_met.get('dist_cv_ci_low')
    dist_cv_ci_high = d_met.get('dist_cv_ci_high')
    grid_mode = d_met.get('grid_mode', False)

    score = 50.0

    # 1. 半径一致性奖励 (核心物理指标 — radius_cv)
    if grid_mode:
        # Grid contours are fixed-radius by construction. Use signal margins
        # and auto-grid confidence instead of rewarding radius_cv=0.
        grid_signal_quality = float(d_met.get("grid_signal_quality", 0.0))
        grid_auto_confidence = d_met.get("grid_auto_confidence")
        grid_inlier_fraction = float(d_met.get("grid_auto_inlier_fraction", 0.0) or 0.0)
        weak_fraction = float(d_met.get("grid_weak_positive_fraction", 0.0))
        grid_unreliable = bool(d_met.get("grid_unreliable", False))
        if grid_signal_quality >= 0.65:
            score += 15
        elif grid_signal_quality >= 0.50:
            score += 10
        elif grid_signal_quality >= 0.35:
            score += 5
        else:
            score -= 10
        if weak_fraction > 0.60:
            score -= 18
        elif weak_fraction > 0.35:
            score -= 8
        if grid_inlier_fraction and grid_inlier_fraction < 0.15:
            score -= 18
        elif grid_inlier_fraction and grid_inlier_fraction < 0.25:
            score -= 8
        if grid_unreliable:
            score -= 15
        if grid_auto_confidence is not None:
            if grid_auto_confidence >= 0.65:
                score += 5
            elif grid_auto_confidence < 0.45:
                score -= 10
    else:
        if count > 50:
            if radius_cv < 0.15:
                score += 25  # 半径高度一致，腔室尺寸均匀
            elif radius_cv < 0.25:
                score += 15
            elif radius_cv > 0.5:
                score -= 20  # 半径离散度过大，可能是噪声
        else:
            # 低计数时用 radius_std 作为备选
            if r_std < 0.5: score += 25
            elif r_std < 0.8: score += 15
            elif r_std > 2.0: score -= 20

    # 2. 数量合理性奖励
    ratio = count / target_N
    if 0.0001 < ratio < 0.8:
        score += 10

    # 3. 分布一致性验证
    if high_density and area_ratio is not None:
        # 高密度模式: 用面积比
        if 0.7 < area_ratio < 1.3:
            score += 10
        elif area_ratio < 0.5 or area_ratio > 2.0:
            score -= 10
    elif (
            not high_density
            and dist_cv is not None
            and count > 100
            and dist_cv_applicable
            and dist_cv_expected is not None
    ):
        # 低中密度模式: 与相同填充率的 Monte Carlo 参考曲线比较。
        # 使用置信区间并保留少量图像定位误差余量。
        tolerance = max(0.04, 0.15 * dist_cv_expected)
        lower = float(dist_cv_ci_low) - tolerance
        upper = float(dist_cv_ci_high) + tolerance
        if lower <= dist_cv <= upper:
            score += 5
        else:
            score -= 5

    # 4. 惩罚极低数量的伪高分
    if count < 20 and score > 80:
        score -= 15

    return max(0.0, min(100.0, float(score)))


class DeepLisaTools:
    @staticmethod
    def inspect_image(img_path: str) -> Dict:
        try:
            img = load_image(img_path)
            mean_bright = np.mean(img)
            return {
                "status": "success",
                "meta": {
                    "brightness": round(float(mean_bright), 2),
                    "description": "Normal",
                    "height": img.shape[0], "width": img.shape[1]
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def execute_detection(img_path: str, params: Dict, target_N: int = 21600) -> Dict:
        try:
            raw_gray = load_image(img_path)
            img_std = np.std(raw_gray)
            audit_dir = params.get("audit_dir")
            if not audit_dir and settings.EXPORT_DETECTION_AUDIT:
                audit_dir = settings.DETECTION_AUDIT_DIR
            collect_audit = bool(audit_dir)
            collect_grid_records = bool(
                collect_audit
                or params.get("export_grid_candidates")
                or settings.EXPORT_GRID_CANDIDATES
            )

            # 1. 增强
            enhanced = enhance_image_quality(
                raw_gray,
                gamma=params.get('gamma', 1.0),
                contrast=1.1,
                sharpen_amount=0.5
            )

            # 2. 预处理。TopHat 尺度必须与腔室类型一致。
            kernel_size = get_tophat_kernel_size()
            pre = apply_advanced_preprocessing(enhanced, tophat_params={"kernel_size": kernel_size})

            # 3. 二值化 (block_size 自适应)
            blk = int(params.get('block_size', 25)) | 1
            off = int(params.get('c_offset', -2))
            binary = binarize_adaptive(pre, blk, off)
            binary_contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            binary_component_count = len(binary_contours)
            detection_mode = str(params.get("detection_mode", settings.DETECTION_MODE)).strip().lower()

            # [一级熔断] 纯白检测
            white_pixel_ratio = cv2.countNonZero(binary) / binary.size
            grid_modes = {"grid_calibrated", "grid_auto"}
            is_grid_mode = detection_mode in grid_modes
            if not is_grid_mode and white_pixel_ratio > 0.15:  # 放宽一点到 15%
                return {
                    "status": "success",
                    "metrics": {"count": 99999, "ratio": 99999 / target_N, "score": 10.0,
                                "dist_cv": 1.0, "radius_cv": 0, "radius_std": 0,
                                "high_density": True, "fill_ratio": 99999 / target_N,
                                "dist_cv_applicable": False, "msg": "Noise Explosion",
                                "detection_mode": params.get("detection_mode", settings.DETECTION_MODE),
                                "tophat_kernel_size": kernel_size,
                                "white_pixel_ratio": round(float(white_pixel_ratio), 6),
                                "binary_components": binary_component_count,
                                "shape_survivors": 0, "local_snr_survivors": 0,
                                "spacing_survivors": 0},
                    "internal_data": {"contours": [], "qc_metrics": QualityMetrics(confidence_score=10)}
                }

            audit_records = []
            local_stats = {}
            grid_stats = {"grid_mode": False}
            filtered = np.zeros_like(binary)
            shape_survivors = 0
            local_snr_survivors = 0

            if is_grid_mode:
                auto_stats = {}
                if detection_mode == "grid_auto":
                    auto_stats = auto_calibrate_grid(
                        enhanced,
                        radius=settings.RADIUS,
                        target_n=target_N,
                        pitch_hint_px=float(params.get("grid_pitch_px", 0) or 0),
                        angle_hint_deg=float(params.get("grid_angle_deg", 0) or 0),
                        layout_hint=str(params.get("grid_layout", settings.GRID_LAYOUT)),
                    )
                    grid_pitch_px = auto_stats["grid_pitch_px"]
                    grid_origin_x_px = auto_stats["grid_origin_x_px"]
                    grid_origin_y_px = auto_stats["grid_origin_y_px"]
                    grid_angle_deg = auto_stats["grid_angle_deg"]
                    grid_layout = auto_stats.get("grid_layout", "hex")
                else:
                    grid_pitch_px = float(params.get("grid_pitch_px", settings.GRID_PITCH_PX))
                    grid_origin_x_px = float(params.get("grid_origin_x_px", settings.GRID_ORIGIN_X_PX))
                    grid_origin_y_px = float(params.get("grid_origin_y_px", settings.GRID_ORIGIN_Y_PX))
                    grid_angle_deg = float(params.get("grid_angle_deg", settings.GRID_ANGLE_DEG))
                    grid_layout = str(params.get("grid_layout", settings.GRID_LAYOUT))

                grid_contours, detected_grid_stats, grid_audit = detect_positive_chambers_on_grid(
                    enhanced,
                    pitch_px=grid_pitch_px,
                    origin_x_px=grid_origin_x_px,
                    origin_y_px=grid_origin_y_px,
                    radius=settings.RADIUS,
                    target_n=target_N,
                    angle_deg=grid_angle_deg,
                    layout=grid_layout,
                    c_offset=off,
                    min_circularity=params.get('min_circularity', 0.5),
                    return_audit=collect_grid_records,
                )
                grid_stats.update(detected_grid_stats)
                grid_stats.update(auto_stats)
                grid_stats["grid_signal_quality"] = calculate_grid_signal_quality(grid_stats)
                if collect_audit:
                    audit_records.extend(grid_audit)

                spot_contours, spot_stats = detect_spot_first_candidates(
                    enhanced,
                    radius=settings.RADIUS,
                    target_n=target_N,
                )
                association_stats = calculate_spot_grid_association(
                    spot_contours,
                    enhanced.shape,
                    grid_stats,
                    settings.RADIUS,
                )
                grid_inlier = float(grid_stats.get("grid_auto_inlier_fraction", 0.0) or 0.0)
                grid_unreliable = bool(grid_stats.get("grid_unreliable", False))
                use_spot_fallback = bool(
                    detection_mode == "grid_auto"
                    and (
                        grid_unreliable
                        or (grid_inlier > 0.0 and grid_inlier < 0.15)
                    )
                )
                if use_spot_fallback:
                    effective_roi = estimate_effective_array_roi_from_image(enhanced)
                    final_contours, roi_filter_stats = filter_contours_to_roi(
                        spot_contours,
                        effective_roi,
                    )
                    hybrid_stats = {
                        "hybrid_detection": False,
                        "hybrid_merged_count": len(final_contours),
                        "hybrid_grid_only_count": 0,
                        "hybrid_duplicate_grid_count": 0,
                        "hybrid_merge_distance_px": 0.0,
                        "formal_detection_source": "spot_first_grid_unreliable_fallback",
                        "grid_auxiliary_only": True,
                        "grid_formal_count_suppressed": len(grid_contours),
                        **roi_filter_stats,
                    }
                else:
                    final_contours, hybrid_stats = merge_spot_and_grid_contours(
                        spot_contours,
                        grid_contours,
                        settings.RADIUS,
                    )
                    hybrid_stats.update({
                        "formal_detection_source": "hybrid_spot_grid",
                        "grid_auxiliary_only": False,
                        "grid_formal_count_suppressed": 0,
                    })
                grid_stats.update(spot_stats)
                grid_stats.update(hybrid_stats)
                grid_stats.update(association_stats)
                local_stats.update(grid_stats)
                local_stats.update({
                    "local_delta_median": grid_stats.get("grid_local_delta_median", 0),
                    "local_z_median": grid_stats.get("grid_local_z_median", 0),
                })
                local_stats.update({"suppressed_close": 0, "min_center_distance": 0})
                contours = final_contours
                shape_survivors = len(spot_contours)
                local_snr_survivors = len(spot_contours)
            else:
                # 4. 形态学筛选。每个连通区域独立计算距离变换阈值，
                # 避免强亮腔室抬高全图阈值后误杀弱阳性。
                min_r, max_r = get_runtime_radius_bounds()
                if collect_audit:
                    filtered, _, shape_audit = advanced_spot_filtering(
                        binary,
                        min_radius=min_r,
                        max_radius=max_r,
                        min_circularity=max(0.3, params.get('min_circularity', 0.5) - 0.1),
                        return_audit=True,
                    )
                    audit_records.extend(shape_audit)
                else:
                    filtered, _ = advanced_spot_filtering(
                        binary,
                        min_radius=min_r,
                        max_radius=max_r,
                        min_circularity=max(0.3, params.get('min_circularity', 0.5) - 0.1),
                    )

                # 5. 轮廓提取
                contours, _ = cv2.findContours(
                    filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                shape_survivors = len(contours)

            if len(contours) > 25000:
                return {
                    "status": "success",
                    "metrics": {"count": len(contours), "ratio": len(contours) / target_N, "score": 15.0,
                                "dist_cv": 1.0, "radius_cv": 0, "radius_std": 0,
                                "high_density": True, "fill_ratio": len(contours) / target_N,
                                "dist_cv_applicable": False, "msg": "Too Many Contours",
                                "detection_mode": detection_mode,
                                "tophat_kernel_size": kernel_size,
                                "white_pixel_ratio": round(float(white_pixel_ratio), 6),
                                "binary_components": binary_component_count,
                                "shape_survivors": len(contours),
                                "local_snr_survivors": 0, "spacing_survivors": 0},
                    "internal_data": {"contours": [], "qc_metrics": QualityMetrics(confidence_score=15)}
                }

            # 6. 信噪比过滤
            if not is_grid_mode:
                if collect_audit:
                    final_contours, local_stats, local_audit = filter_by_local_snr_adaptive(
                        enhanced, contours, return_audit=True
                    )
                    audit_records.extend(local_audit)
                else:
                    final_contours, local_stats = filter_by_local_snr_adaptive(
                        enhanced, contours
                    )
                local_snr_survivors = len(final_contours)
                if collect_audit:
                    final_contours, spacing_stats, spacing_audit = suppress_close_contours(
                        final_contours, enhanced, return_audit=True
                    )
                    audit_records.extend(spacing_audit)
                else:
                    final_contours, spacing_stats = suppress_close_contours(
                        final_contours, enhanced
                    )
                local_stats.update(spacing_stats)
            final_count = len(final_contours)
            rescue_used = False

            img_p50 = float(np.percentile(raw_gray, 50))
            img_p90 = float(np.percentile(raw_gray, 90))
            likely_high_background = bool(img_p50 > 25 and (img_p90 - img_p50) > 8)
            if not is_grid_mode and likely_high_background and final_count < 500:
                rescue_contours, rescue_stats = detect_high_background_rescue(raw_gray, enhanced, params, target_N)
                if len(rescue_contours) > final_count:
                    final_contours = rescue_contours
                    final_count = len(final_contours)
                    local_stats.update(rescue_stats)
                    local_stats["high_background"] = True
                    rescue_used = True

            # 7. 计算指标
            dist_metrics = calculate_distribution_metrics(final_contours, target_N)
            if is_grid_mode:
                dist_metrics.update({
                    "grid_mode": True,
                    "grid_signal_quality": local_stats.get("grid_signal_quality", 0.0),
                    "grid_auto_confidence": local_stats.get("grid_auto_confidence"),
                    "grid_auto_inlier_fraction": local_stats.get("grid_auto_inlier_fraction", 0.0),
                    "grid_unreliable": local_stats.get("grid_unreliable", False),
                    "grid_weak_positive_fraction": local_stats.get("grid_weak_positive_fraction", 0.0),
                })
            int_metrics = {"contrast_ratio": 1.0}  # 简化，不再作为核心权重
            score = calculate_score(final_count, dist_metrics, int_metrics, target_N, img_std)

            qc = QualityController().evaluate(enhanced, final_contours, final_count)
            qc.confidence_score = score
            result_metrics = {
                    "count": final_count,
                    "ratio": round(final_count / target_N, 4),
                    "score": round(score, 1),
                    "dist_cv": float(round(dist_metrics.get('dist_cv', 0) or 0, 2)),
                    "radius_cv": float(round(dist_metrics.get('radius_cv', 0), 3)),
                    "radius_std": float(round(dist_metrics.get('radius_std', 0), 2)),
                    "fill_ratio": float(round(dist_metrics.get('fill_ratio', final_count / target_N), 6)),
                    "dist_cv_applicable": dist_metrics.get('dist_cv_applicable', False),
                    "dist_cv_expected": float(round(dist_metrics.get('dist_cv_expected'), 3)) if dist_metrics.get('dist_cv_expected') is not None else None,
                    "dist_cv_ci_low": float(round(dist_metrics.get('dist_cv_ci_low'), 3)) if dist_metrics.get('dist_cv_ci_low') is not None else None,
                    "dist_cv_ci_high": float(round(dist_metrics.get('dist_cv_ci_high'), 3)) if dist_metrics.get('dist_cv_ci_high') is not None else None,
                    "area_ratio": float(round(dist_metrics.get('area_ratio'), 3)) if dist_metrics.get('area_ratio') is not None else None,
                    "high_density": dist_metrics.get('high_density_mode', False),
                    "high_background": local_stats.get("high_background", False),
                    "rescue_used": rescue_used,
                    "rescue_candidates": local_stats.get("rescue_candidates", 0),
                    "rescue_peak_candidates": local_stats.get("rescue_peak_candidates", 0),
                    "suppressed_close": local_stats.get("suppressed_close", 0),
                    "min_center_distance": round(local_stats.get("min_center_distance", 0), 3),
                    "rejected_tiny_support": local_stats.get("rejected_tiny_support", 0),
                    "local_delta_median": round(local_stats.get("local_delta_median", 0), 3),
                    "local_z_median": round(local_stats.get("local_z_median", 0), 3),
                    "grid_signal_quality": round(local_stats.get("grid_signal_quality", 0), 3),
                    "detection_mode": detection_mode,
                    "tophat_kernel_size": kernel_size,
                    "white_pixel_ratio": round(float(white_pixel_ratio), 6),
                    "binary_components": binary_component_count,
                    "shape_survivors": shape_survivors,
                    "local_snr_survivors": local_snr_survivors,
                    "spacing_survivors": final_count,
                    "shape_retention_ratio": round(shape_survivors / max(binary_component_count, 1), 6),
                    **grid_stats,
                }

            if audit_dir:
                overlay = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2BGR)
                for contour in final_contours:
                    ((x, y), _) = cv2.minEnclosingCircle(contour)
                    cv2.circle(
                        overlay,
                        (int(round(x)), int(round(y))),
                        max(2, int(settings.RADIUS)),
                        (0, 255, 0),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                result_metrics["audit_dir"] = export_detection_audit(
                    str(audit_dir),
                    img_path,
                    {
                        "01_raw": raw_gray,
                        "02_enhanced": enhanced,
                        "03_preprocessed": pre,
                        "04_binary": binary,
                        "05_shape_filtered": filtered,
                        "06_final_overlay": overlay,
                    },
                    result_metrics,
                    audit_records,
                )

            return {
                "status": "success",
                "metrics": result_metrics,
                "internal_data": {
                    "contours": final_contours,
                    "qc_metrics": qc,
                    "candidate_records": grid_audit if is_grid_mode and collect_grid_records else [],
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def suggest_heuristic_mutation(params: Dict, metrics: Dict, target_N: int) -> Tuple[Dict, str]:
        """[工具 3] 启发式突变 (温和版, v15.0 — 使用统一边界)"""
        new_p = params.copy()
        count = metrics.get('count', 0)
        current_off = new_p.get('c_offset', -2)
        bounds = settings.AI_TUNING_BOUNDS

        if metrics.get("grid_mode"):
            if count < 10:
                strategy_name = "🔦 网格低计数放宽 (Grid Boost)"
                new_p['gamma'] = min(bounds.get('gamma_max', 2.0), 1.2)
                new_p['c_offset'] = min(current_off + 2, bounds.get('c_offset_max', 0))
                new_p['min_circularity'] = max(0.35, float(new_p.get('min_circularity', 0.5)) - 0.05)
            else:
                grid_positions = max(metrics.get("grid_positions", target_N), 1)
                acceptance_ratio = count / grid_positions
                weak_fraction = float(metrics.get("grid_weak_positive_fraction", 0.0))
                signal_quality = float(metrics.get("grid_signal_quality", 0.0))
                grid_inlier = float(metrics.get("grid_auto_inlier_fraction", 0.0) or 0.0)
                grid_unreliable = bool(metrics.get("grid_unreliable", False))
                likely_under_recall = (
                    acceptance_ratio < 0.08
                    and weak_fraction <= 0.35
                    and signal_quality >= 0.45
                    and not grid_unreliable
                    and (grid_inlier == 0.0 or grid_inlier >= 0.20)
                )
                needs_confirmation = (
                    grid_unreliable
                    or (grid_inlier > 0.0 and grid_inlier < 0.20)
                    or weak_fraction > 0.55
                    or acceptance_ratio > 0.18
                    or (acceptance_ratio > 0.08 and weak_fraction > 0.55)
                    or (acceptance_ratio > 0.05 and signal_quality < 0.45)
                )
                if likely_under_recall:
                    strategy_name = "🔎 网格召回恢复 (Grid Recall Probe)"
                    new_p['gamma'] = min(bounds.get('gamma_max', 2.0), max(1.15, float(new_p.get('gamma', 1.0))))
                    new_p['c_offset'] = min(current_off + 3, bounds.get('c_offset_max', 6))
                    new_p['min_circularity'] = max(0.35, float(new_p.get('min_circularity', 0.5)) - 0.05)
                elif needs_confirmation:
                    strategy_name = "📉 网格保守确认 (Grid Confirm)"
                    new_p['c_offset'] = max(current_off - 2, bounds.get('c_offset_min', -18))
                    new_p['min_circularity'] = min(0.75, float(new_p.get('min_circularity', 0.5)) + 0.05)
                    if current_off < -8:
                        new_p['gamma'] = max(bounds.get('gamma_min', 0.5), 0.9)
                else:
                    strategy_name = "🛑 网格信号稳定 (Grid Stable)"
            return new_p, strategy_name

        # 策略 A: 噪声爆炸 -> 逐步收紧 (不要直接跳崖!)
        if count > 20000:
            strategy_name = "📉 梯度收紧 (Step Down)"
            # 如果当前是 -3，下次试 -6，而不是 -15
            # 如果已经是 -6，试 -9
            step_down = 3
            target_off = current_off - step_down

            new_p['c_offset'] = max(target_off, bounds.get('c_offset_min', -18))

            # 如果还没有压住，稍微动一下 block
            if current_off < -8:
                new_p['block_size'] = 35
                new_p['gamma'] = max(bounds.get('gamma_min', 0.5), 0.8)
            else:
                new_p['gamma'] = 1.0  # 保持原样

        # 策略 B: 假阴性 -> 提亮
        elif count < 10:
            strategy_name = "🔦 深度激活 (Boost)"
            new_p['gamma'] = min(bounds.get('gamma_max', 2.0), 1.2)
            new_p['c_offset'] = min(current_off + 2, bounds.get('c_offset_max', 0))
            new_p['min_circularity'] = 0.4  # 形状放宽

        # 策略 C: 正常波动。优先轻微放宽以恢复弱阳性，不再无条件越跑越严。
        else:
            strategy_name = "🔎 召回恢复 (Recall Probe)"
            new_p['c_offset'] = min(current_off + 1, -2)
            if metrics.get("shape_retention_ratio", 1.0) < 0.2:
                new_p['min_circularity'] = max(
                    0.35,
                    float(new_p.get('min_circularity', 0.5)) - 0.05,
                )

        return new_p, strategy_name
