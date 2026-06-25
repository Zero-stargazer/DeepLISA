#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrated hex-grid chamber classification for DeepLISA."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

CAD_STAGGERED_X_GAP_RATIOS = (0.205 / 0.140, 0.145 / 0.140)
CAD_STAGGERED_Y_PHASE_RATIO = 0.030 / 0.140


def _circle_contour(x: float, y: float, radius: int) -> np.ndarray:
    poly = cv2.ellipse2Poly(
        (int(round(x)), int(round(y))),
        (radius, radius),
        0,
        0,
        360,
        30,
    )
    return poly.reshape((-1, 1, 2)).astype(np.int32)


def _median_mad(values: np.ndarray) -> Tuple[float, float]:
    """Return robust center and scaled MAD for noisy grid statistics."""
    if values.size == 0:
        return 0.0, 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) * 1.4826)
    return median, max(mad, 0.05)


def _weighted_phase(values: np.ndarray, period: float, weights: np.ndarray = None) -> Tuple[float, float]:
    """Return the dominant modulo phase and a 0..1 concentration score."""
    if values.size == 0 or period <= 0:
        return 0.0, 0.0

    phases = np.mod(values, period)
    if weights is None:
        weights = np.ones_like(phases, dtype=np.float64)
    else:
        weights = weights.astype(np.float64)
    weights = np.maximum(weights, 1e-6)

    hist_bins = 96
    hist, edges = np.histogram(phases, bins=hist_bins, range=(0.0, period), weights=weights)
    peak = int(np.argmax(hist))
    bin_width = period / hist_bins
    peak_center = (edges[peak] + edges[peak + 1]) * 0.5
    circular_delta = np.abs(((phases - peak_center + period * 0.5) % period) - period * 0.5)
    local = circular_delta <= max(bin_width * 3.0, period / 24.0)
    if np.count_nonzero(local) >= 3:
        phases = phases[local]
        weights = weights[local]

    angles = phases / period * 2.0 * np.pi
    vector = np.sum(weights * np.exp(1j * angles)) / (np.sum(weights) + 1e-9)
    phase = (np.angle(vector) % (2.0 * np.pi)) / (2.0 * np.pi) * period
    confidence = float(np.clip(np.abs(vector), 0.0, 1.0))
    return float(phase), confidence


def _dominant_distance(distances: np.ndarray) -> Tuple[float, float]:
    """Find the main nearest-neighbor distance cluster."""
    if distances.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(distances, [5, 95])
    clipped = distances[(distances >= lo) & (distances <= hi)]
    if clipped.size < 8:
        clipped = distances
    bins = max(20, min(120, int(np.sqrt(clipped.size) * 2)))
    hist, edges = np.histogram(clipped, bins=bins)
    peak = int(np.argmax(hist))
    center = (edges[peak] + edges[peak + 1]) * 0.5
    window = max((edges[1] - edges[0]) * 2.0, center * 0.12)
    local = distances[np.abs(distances - center) <= window]
    if local.size < 8:
        local = clipped
    confidence = float(np.clip(local.size / max(distances.size, 1), 0.0, 1.0))
    return float(np.median(local)), confidence


def apply_grid_reliability_gate(stats: Dict) -> Dict:
    """Penalize grid confidence when geometric inlier support is weak."""
    stats = dict(stats or {})
    raw_confidence = float(stats.get("grid_auto_confidence", 0.0) or 0.0)
    inlier_fraction = float(stats.get("grid_auto_inlier_fraction", 0.0) or 0.0)
    flags = list(stats.get("grid_reliability_flags", []) or [])

    gated_confidence = raw_confidence
    if inlier_fraction < 0.15:
        gated_confidence = min(gated_confidence, 0.35)
        flags.append("low_grid_inlier_fraction")
    elif inlier_fraction < 0.25:
        gated_confidence = min(gated_confidence, 0.55)
        flags.append("borderline_grid_inlier_fraction")

    if raw_confidence < 0.45:
        flags.append("low_grid_raw_confidence")

    stats["grid_auto_confidence_raw"] = raw_confidence
    stats["grid_auto_confidence"] = float(np.clip(gated_confidence, 0.0, 1.0))
    stats["grid_unreliable"] = bool(
        inlier_fraction < 0.15
        or stats["grid_auto_confidence"] < 0.45
    )
    stats["grid_reliability_flags"] = sorted(set(flags))
    return stats


def _extract_calibration_points(
        img: np.ndarray,
        radius: int,
        max_dim: int = 1200,
        max_points: int = 60000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract bright periodic chamber-like points for grid calibration."""
    gray = img
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray.copy()

    if small.dtype != np.uint8:
        small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    low = float(np.percentile(small, 1.0))
    high = float(np.percentile(small, 99.5))
    if high - low >= 3.0:
        small = np.clip((small.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
    sigma = max(5.0, float(radius) * scale * 3.0)
    background = cv2.GaussianBlur(clahe, (0, 0), sigmaX=sigma, sigmaY=sigma)
    flat = cv2.subtract(clahe, background)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    smooth = cv2.GaussianBlur(flat, (3, 3), 0.6)

    nms_radius = max(1, int(round(radius * scale * 0.55)))
    ksize = max(3, nms_radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    peaks = smooth == cv2.dilate(smooth, kernel)

    floor = max(float(np.percentile(smooth, 65)), float(np.mean(smooth) + 0.10 * np.std(smooth)))
    peak_mask = peaks & (smooth >= floor)
    ys, xs = np.where(peak_mask)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), scale

    values = smooth[ys, xs].astype(np.float32)
    if xs.size > max_points:
        tile_size = max(24, int(round(radius * scale * 10.0)))
        tile_x = xs // tile_size
        tile_y = ys // tile_size
        tile_ids = tile_y * (small.shape[1] // tile_size + 2) + tile_x
        unique_tiles = np.unique(tile_ids)
        quota = max(3, int(np.ceil(max_points / max(len(unique_tiles), 1))))
        keep_chunks = []
        for tile_id in unique_tiles:
            idx = np.where(tile_ids == tile_id)[0]
            if idx.size > quota:
                idx = idx[np.argsort(values[idx])[-quota:]]
            keep_chunks.append(idx)
        keep = np.concatenate(keep_chunks) if keep_chunks else np.arange(xs.size)
        if keep.size > max_points:
            keep = keep[np.argsort(values[keep])[-max_points:]]
        xs, ys, values = xs[keep], ys[keep], values[keep]

    points = np.column_stack([xs.astype(np.float32) / scale, ys.astype(np.float32) / scale])
    return points, values, scale


def auto_calibrate_hex_grid(
        img: np.ndarray,
        radius: int,
        target_n: int,
        pitch_hint_px: float = 0.0,
        angle_hint_deg: float = 0.0,
) -> Dict:
    """Estimate pitch, origin, and rotation for a hex chamber array."""
    points, weights, _ = _extract_calibration_points(img, radius)
    if len(points) < 40:
        raise ValueError(f"Grid auto calibration needs >=40 chamber-like points, got {len(points)}")

    tree = cKDTree(points)
    k = min(8, len(points))
    dists, indices = tree.query(points, k=k)
    neighbor_dists = dists[:, 1:].reshape(-1)
    min_dist = max(6.0, float(radius) * 1.45)
    max_dist = max(min_dist + 4.0, float(radius) * 6.5)
    valid_distances = neighbor_dists[
        (neighbor_dists >= min_dist)
        & (neighbor_dists <= max_dist)
    ]

    if pitch_hint_px > 0:
        pitch = float(pitch_hint_px)
        pitch_conf = 1.0
    else:
        pitch, pitch_conf = _dominant_distance(valid_distances)
    if pitch <= 0:
        raise ValueError("Grid auto calibration could not estimate chamber pitch")

    vectors = []
    for i in range(len(points)):
        for j_pos in range(1, k):
            j = indices[i, j_pos]
            dist = dists[i, j_pos]
            if abs(dist - pitch) <= max(2.0, pitch * 0.25):
                vectors.append(points[j] - points[i])
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        angle_deg = float(angle_hint_deg)
        angle_conf = 0.0
    else:
        angles = (np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) + 180.0) % 180.0
        folded = np.mod(angles, 60.0)
        angle_phase, angle_conf = _weighted_phase(folded, 60.0)
        angle_deg = angle_phase if angle_phase <= 30.0 else angle_phase - 60.0
        if abs(float(angle_hint_deg)) > 1e-6 and angle_conf < 0.25:
            angle_deg = float(angle_hint_deg)

    theta = np.deg2rad(angle_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    u = points[:, 0] * cos_t + points[:, 1] * sin_t
    v = -points[:, 0] * sin_t + points[:, 1] * cos_t

    row_step = pitch * np.sqrt(3.0) / 2.0
    origin_v, row_phase_conf = _weighted_phase(v, row_step, weights)
    row_idx = np.rint((v - origin_v) / row_step).astype(np.int64)
    x_corrected = u - np.where(row_idx % 2 == 0, 0.0, 0.5 * pitch)
    origin_u, col_phase_conf = _weighted_phase(x_corrected, pitch, weights)

    row_residual = np.abs(((v - origin_v + row_step * 0.5) % row_step) - row_step * 0.5)
    col_residual = np.abs(((x_corrected - origin_u + pitch * 0.5) % pitch) - pitch * 0.5)
    residual = np.sqrt(row_residual ** 2 + col_residual ** 2)
    inlier_cutoff = max(2.0, float(radius) * 0.45)
    inlier_fraction = float(np.mean(residual <= inlier_cutoff))

    origin_x = origin_u * cos_t - origin_v * sin_t
    origin_y = origin_u * sin_t + origin_v * cos_t
    confidence = float(np.clip(
        0.25 * pitch_conf
        + 0.25 * angle_conf
        + 0.20 * row_phase_conf
        + 0.20 * col_phase_conf
        + 0.10 * inlier_fraction,
        0.0,
        1.0,
    ))

    return apply_grid_reliability_gate({
        "grid_auto": True,
        "grid_layout": "hex",
        "grid_auto_points": int(len(points)),
        "grid_auto_pitch_confidence": float(pitch_conf),
        "grid_auto_angle_confidence": float(angle_conf),
        "grid_auto_row_phase_confidence": float(row_phase_conf),
        "grid_auto_col_phase_confidence": float(col_phase_conf),
        "grid_auto_inlier_fraction": inlier_fraction,
        "grid_auto_confidence": confidence,
        "grid_pitch_px": float(pitch),
        "grid_origin_x_px": float(origin_x),
        "grid_origin_y_px": float(origin_y),
        "grid_angle_deg": float(angle_deg),
        "grid_target_n": int(target_n),
    })


def auto_calibrate_rect_grid(
        img: np.ndarray,
        radius: int,
        target_n: int,
        pitch_hint_px: float = 0.0,
        angle_hint_deg: float = 0.0,
) -> Dict:
    """Estimate a rectangular chamber array from image points."""
    points, weights, _ = _extract_calibration_points(img, radius)
    if len(points) < 40:
        raise ValueError(f"Grid auto calibration needs >=40 chamber-like points, got {len(points)}")

    tree = cKDTree(points)
    k = min(8, len(points))
    dists, indices = tree.query(points, k=k)
    neighbor_dists = dists[:, 1:].reshape(-1)
    min_dist = max(6.0, float(radius) * 1.45)
    max_dist = max(min_dist + 4.0, float(radius) * 6.5)
    valid_distances = neighbor_dists[
        (neighbor_dists >= min_dist)
        & (neighbor_dists <= max_dist)
    ]

    if pitch_hint_px > 0:
        pitch = float(pitch_hint_px)
        pitch_conf = 1.0
    else:
        pitch, pitch_conf = _dominant_distance(valid_distances)
    if pitch <= 0:
        raise ValueError("Grid auto calibration could not estimate chamber pitch")

    vectors = []
    for i in range(len(points)):
        for j_pos in range(1, k):
            j = indices[i, j_pos]
            dist = dists[i, j_pos]
            if abs(dist - pitch) <= max(2.0, pitch * 0.25):
                vectors.append(points[j] - points[i])
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        angle_deg = float(angle_hint_deg)
        angle_conf = 0.0
    else:
        angles = (np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) + 180.0) % 180.0
        folded = np.mod(angles, 90.0)
        angle_phase, angle_conf = _weighted_phase(folded, 90.0)
        angle_deg = angle_phase if angle_phase <= 45.0 else angle_phase - 90.0
        if abs(float(angle_hint_deg)) > 1e-6 and angle_conf < 0.25:
            angle_deg = float(angle_hint_deg)

    theta = np.deg2rad(angle_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    u = points[:, 0] * cos_t + points[:, 1] * sin_t
    v = -points[:, 0] * sin_t + points[:, 1] * cos_t
    origin_u, col_phase_conf = _weighted_phase(u, pitch, weights)
    origin_v, row_phase_conf = _weighted_phase(v, pitch, weights)

    col_residual = np.abs(((u - origin_u + pitch * 0.5) % pitch) - pitch * 0.5)
    row_residual = np.abs(((v - origin_v + pitch * 0.5) % pitch) - pitch * 0.5)
    residual = np.sqrt(row_residual ** 2 + col_residual ** 2)
    inlier_cutoff = max(2.0, float(radius) * 0.45)
    inlier_fraction = float(np.mean(residual <= inlier_cutoff))

    origin_x = origin_u * cos_t - origin_v * sin_t
    origin_y = origin_u * sin_t + origin_v * cos_t
    confidence = float(np.clip(
        0.25 * pitch_conf
        + 0.25 * angle_conf
        + 0.20 * row_phase_conf
        + 0.20 * col_phase_conf
        + 0.10 * inlier_fraction,
        0.0,
        1.0,
    ))

    return apply_grid_reliability_gate({
        "grid_auto": True,
        "grid_layout": "rect",
        "grid_auto_points": int(len(points)),
        "grid_auto_pitch_confidence": float(pitch_conf),
        "grid_auto_angle_confidence": float(angle_conf),
        "grid_auto_row_phase_confidence": float(row_phase_conf),
        "grid_auto_col_phase_confidence": float(col_phase_conf),
        "grid_auto_inlier_fraction": inlier_fraction,
        "grid_auto_confidence": confidence,
        "grid_pitch_px": float(pitch),
        "grid_origin_x_px": float(origin_x),
        "grid_origin_y_px": float(origin_y),
        "grid_angle_deg": float(angle_deg),
        "grid_target_n": int(target_n),
    })


def _fit_staggered_origin(
        image_shape: Tuple[int, int],
        points: np.ndarray,
        weights: np.ndarray,
        pitch: float,
        angle_deg: float,
        origin_x: float,
        origin_y: float,
        target_n: int,
        radius: int,
) -> Dict:
    """Fit CAD-staggered origin by nearest-neighbor support to extracted points."""
    if points.size == 0 or pitch <= 0:
        return {
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
            "inlier_fraction": 0.0,
            "median_distance": 0.0,
            "fit_points": 0,
        }

    max_points = 4000
    if len(points) > max_points:
        order = np.argsort(weights)[-max_points:] if weights.size == len(points) else np.arange(max_points)
        fit_points = points[order]
    else:
        fit_points = points

    period_x = (CAD_STAGGERED_X_GAP_RATIOS[0] + CAD_STAGGERED_X_GAP_RATIOS[1]) * pitch
    period_y = pitch
    cutoff = max(2.0, float(radius) * 0.45)

    def score_origin(ox: float, oy: float) -> Tuple[float, float]:
        centers = generate_staggered_grid(
            image_shape,
            pitch,
            ox,
            oy,
            target_n,
            angle_deg=angle_deg,
        )
        if not centers:
            return 0.0, float("inf")
        tree = cKDTree(np.asarray(centers, dtype=np.float32))
        distances, _ = tree.query(fit_points, k=1)
        return float(np.mean(distances <= cutoff)), float(np.median(distances))

    best = {
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "inlier_fraction": -1.0,
        "median_distance": float("inf"),
    }
    x_offsets = np.linspace(-0.5 * period_x, 0.5 * period_x, 7)
    y_offsets = np.linspace(-0.5 * period_y, 0.5 * period_y, 7)
    for dx in x_offsets:
        for dy in y_offsets:
            inlier, median_distance = score_origin(origin_x + dx, origin_y + dy)
            if (inlier, -median_distance) > (best["inlier_fraction"], -best["median_distance"]):
                best.update({
                    "origin_x": float(origin_x + dx),
                    "origin_y": float(origin_y + dy),
                    "inlier_fraction": inlier,
                    "median_distance": median_distance,
                })

    refine_x = np.linspace(-period_x / 12.0, period_x / 12.0, 5)
    refine_y = np.linspace(-period_y / 12.0, period_y / 12.0, 5)
    coarse_x, coarse_y = best["origin_x"], best["origin_y"]
    for dx in refine_x:
        for dy in refine_y:
            inlier, median_distance = score_origin(coarse_x + dx, coarse_y + dy)
            if (inlier, -median_distance) > (best["inlier_fraction"], -best["median_distance"]):
                best.update({
                    "origin_x": float(coarse_x + dx),
                    "origin_y": float(coarse_y + dy),
                    "inlier_fraction": inlier,
                    "median_distance": median_distance,
                })

    best["fit_points"] = int(len(fit_points))
    return best


def auto_calibrate_staggered_grid(
        img: np.ndarray,
        radius: int,
        target_n: int,
        pitch_hint_px: float = 0.0,
        angle_hint_deg: float = 0.0,
) -> Dict:
    """Estimate a CAD-derived staggered lattice with CAD geometry phase fitting."""
    points, weights, _ = _extract_calibration_points(img, radius)
    if len(points) < 40:
        raise ValueError(f"Grid auto calibration needs >=40 chamber-like points, got {len(points)}")

    estimate = auto_calibrate_rect_grid(
        img,
        radius,
        target_n,
        pitch_hint_px=pitch_hint_px,
        angle_hint_deg=angle_hint_deg,
    )
    pitch = float(estimate["grid_pitch_px"])
    angle_deg = float(estimate["grid_angle_deg"])
    fit = _fit_staggered_origin(
        img.shape,
        points,
        weights,
        pitch,
        angle_deg,
        float(estimate["grid_origin_x_px"]),
        float(estimate["grid_origin_y_px"]),
        target_n,
        radius,
    )
    estimate.update({
        "grid_layout": "staggered_alternating_lattice",
        "grid_design_prior_layout": "staggered_alternating_lattice",
        "grid_origin_x_px": fit["origin_x"],
        "grid_origin_y_px": fit["origin_y"],
        "grid_auto_inlier_fraction": fit["inlier_fraction"],
        "grid_staggered_fit_median_distance_px": fit["median_distance"],
        "grid_staggered_fit_points": fit["fit_points"],
        "grid_staggered_phase_fit": True,
        "grid_staggered_x_gap_ratio_even": CAD_STAGGERED_X_GAP_RATIOS[0],
        "grid_staggered_x_gap_ratio_odd": CAD_STAGGERED_X_GAP_RATIOS[1],
        "grid_staggered_y_phase_ratio": CAD_STAGGERED_Y_PHASE_RATIO,
    })
    return apply_grid_reliability_gate(estimate)


def auto_calibrate_grid(
        img: np.ndarray,
        radius: int,
        target_n: int,
        pitch_hint_px: float = 0.0,
        angle_hint_deg: float = 0.0,
        layout_hint: str = "auto",
) -> Dict:
    """Estimate the best supported grid model for the image."""
    layout_hint = str(layout_hint or "auto").strip().lower()
    if layout_hint in {"staggered", "staggered_alternating_lattice", "cad", "cad_staggered"}:
        return auto_calibrate_staggered_grid(img, radius, target_n, pitch_hint_px, angle_hint_deg)
    if layout_hint == "hex":
        return auto_calibrate_hex_grid(img, radius, target_n, pitch_hint_px, angle_hint_deg)
    if layout_hint in {"rect", "rectangle", "rectangular"}:
        return auto_calibrate_rect_grid(img, radius, target_n, pitch_hint_px, angle_hint_deg)

    estimates = [
        auto_calibrate_hex_grid(img, radius, target_n, pitch_hint_px, angle_hint_deg),
        auto_calibrate_rect_grid(img, radius, target_n, pitch_hint_px, angle_hint_deg),
    ]
    return max(
        estimates,
        key=lambda item: (
            item.get("grid_auto_inlier_fraction", 0.0),
            item.get("grid_auto_confidence", 0.0),
        ),
    )


def staggered_lattice_chip_coordinates(
        col: int,
        row: int,
        pitch_px: float,
        x_gap_ratios: Tuple[float, float] = CAD_STAGGERED_X_GAP_RATIOS,
        y_phase_ratio: float = CAD_STAGGERED_Y_PHASE_RATIO,
) -> Tuple[float, float]:
    """Return CAD-staggered chip coordinates for a column and row-pair index."""
    if pitch_px <= 0:
        raise ValueError("pitch_px must be > 0")
    col = int(col)
    row = int(row)
    x_gap_ratios = tuple(float(v) for v in x_gap_ratios)
    if len(x_gap_ratios) != 2:
        x_gap_ratios = CAD_STAGGERED_X_GAP_RATIOS

    even_gap = x_gap_ratios[0] * pitch_px
    odd_gap = x_gap_ratios[1] * pitch_px
    period_x = even_gap + odd_gap
    if col >= 0:
        x = (col // 2) * period_x + (even_gap if col % 2 else 0.0)
    else:
        n = -col
        x = -((n // 2) * period_x + (odd_gap if n % 2 else 0.0))

    y = row * pitch_px
    if col % 2:
        y += float(y_phase_ratio) * pitch_px
    return float(x), float(y)


def generate_staggered_grid(
        image_shape: Tuple[int, int],
        pitch_px: float,
        origin_x_px: float,
        origin_y_px: float,
        target_n: int,
        angle_deg: float = 0.0,
) -> List[Tuple[float, float]]:
    """Generate CAD-derived staggered lattice centers inside image bounds."""
    if pitch_px <= 0:
        raise ValueError("GRID_PITCH_PX must be > 0 for staggered grid mode")

    height, width = image_shape
    theta = np.deg2rad(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    def to_image(dx: float, dy: float) -> Tuple[float, float]:
        return (
            origin_x_px + dx * cos_t - dy * sin_t,
            origin_y_px + dx * sin_t + dy * cos_t,
        )

    corners = np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [0.0, float(height - 1)],
            [float(width - 1), float(height - 1)],
        ],
        dtype=np.float64,
    )
    dx = corners[:, 0] - float(origin_x_px)
    dy = corners[:, 1] - float(origin_y_px)
    chip_x = dx * cos_t + dy * sin_t
    chip_y = -dx * sin_t + dy * cos_t
    margin = 4.0 * pitch_px
    min_x_gap = min(CAD_STAGGERED_X_GAP_RATIOS) * pitch_px
    col_min = int(np.floor((float(np.min(chip_x)) - margin) / max(min_x_gap, 1e-6))) - 2
    col_max = int(np.ceil((float(np.max(chip_x)) + margin) / max(min_x_gap, 1e-6))) + 2
    row_min = int(np.floor((float(np.min(chip_y)) - margin) / pitch_px)) - 2
    row_max = int(np.ceil((float(np.max(chip_y)) + margin) / pitch_px)) + 2

    candidates = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            chip_x, chip_y = staggered_lattice_chip_coordinates(col, row, pitch_px)
            x, y = to_image(chip_x, chip_y)
            if 0 <= x < width and 0 <= y < height:
                candidates.append((row, col, float(x), float(y)))

    candidates.sort(key=lambda item: (item[0], item[1]))
    if target_n and target_n > 0:
        candidates = candidates[:target_n]
    return [(x, y) for _, _, x, y in candidates]


def generate_hex_grid(
        image_shape: Tuple[int, int],
        pitch_px: float,
        origin_x_px: float,
        origin_y_px: float,
        target_n: int,
        angle_deg: float = 0.0,
) -> List[Tuple[float, float]]:
    """Generate calibrated chamber centers inside the image bounds.

    The grid is defined in chip coordinates and then rotated into image
    coordinates.  This keeps the chamber model usable when the microscope image
    is not perfectly upright.
    """
    if pitch_px <= 0:
        raise ValueError("GRID_PITCH_PX must be > 0 for grid_calibrated mode")

    height, width = image_shape
    row_step = pitch_px * np.sqrt(3.0) / 2.0
    theta = np.deg2rad(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    def to_image(dx: float, dy: float) -> Tuple[float, float]:
        return (
            origin_x_px + dx * cos_t - dy * sin_t,
            origin_y_px + dx * sin_t + dy * cos_t,
        )

    # For rotated grids, rows/columns that start outside the image can rotate
    # back into view.  Search a padded chip-coordinate extent instead of only
    # walking positive x/y as the old upright-only path did.
    diagonal = float(np.hypot(width, height))
    origin_extent = max(abs(float(origin_x_px)), abs(float(origin_y_px)))
    extent = diagonal + origin_extent + 4.0 * pitch_px
    col_min = int(np.floor(-extent / pitch_px)) - 2
    col_max = int(np.ceil(extent / pitch_px)) + 2
    row_min = int(np.floor(-extent / row_step)) - 2
    row_max = int(np.ceil(extent / row_step)) + 2

    candidates = []
    for row in range(row_min, row_max + 1):
        row_y = row * row_step
        row_x_offset = 0.5 * pitch_px if row % 2 else 0.0
        for col in range(col_min, col_max + 1):
            chip_x = col * pitch_px + row_x_offset
            x, y = to_image(chip_x, row_y)
            if 0 <= x < width and 0 <= y < height:
                candidates.append((row, col, float(x), float(y)))

    candidates.sort(key=lambda item: (item[0], item[1]))
    if target_n and target_n > 0:
        candidates = candidates[:target_n]
    return [(x, y) for _, _, x, y in candidates]


def generate_rect_grid(
        image_shape: Tuple[int, int],
        pitch_px: float,
        origin_x_px: float,
        origin_y_px: float,
        target_n: int,
        angle_deg: float = 0.0,
) -> List[Tuple[float, float]]:
    """Generate calibrated rectangular-grid centers inside the image bounds."""
    if pitch_px <= 0:
        raise ValueError("GRID_PITCH_PX must be > 0 for grid mode")

    height, width = image_shape
    theta = np.deg2rad(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    def to_image(dx: float, dy: float) -> Tuple[float, float]:
        return (
            origin_x_px + dx * cos_t - dy * sin_t,
            origin_y_px + dx * sin_t + dy * cos_t,
        )

    diagonal = float(np.hypot(width, height))
    origin_extent = max(abs(float(origin_x_px)), abs(float(origin_y_px)))
    extent = diagonal + origin_extent + 4.0 * pitch_px
    col_min = int(np.floor(-extent / pitch_px)) - 2
    col_max = int(np.ceil(extent / pitch_px)) + 2
    row_min = int(np.floor(-extent / pitch_px)) - 2
    row_max = int(np.ceil(extent / pitch_px)) + 2

    candidates = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            x, y = to_image(col * pitch_px, row * pitch_px)
            if 0 <= x < width and 0 <= y < height:
                candidates.append((row, col, float(x), float(y)))

    candidates.sort(key=lambda item: (item[0], item[1]))
    if target_n and target_n > 0:
        candidates = candidates[:target_n]
    return [(x, y) for _, _, x, y in candidates]


def generate_grid_centers(
        image_shape: Tuple[int, int],
        pitch_px: float,
        origin_x_px: float,
        origin_y_px: float,
        target_n: int,
        angle_deg: float = 0.0,
        layout: str = "hex",
) -> Tuple[List[Tuple[float, float]], str]:
    """Generate centers for the requested supported grid layout."""
    layout_key = str(layout or "hex").strip().lower()
    if layout_key in {"staggered", "staggered_alternating_lattice", "cad", "cad_staggered"}:
        return (
            generate_staggered_grid(
                image_shape,
                pitch_px,
                origin_x_px,
                origin_y_px,
                target_n,
                angle_deg=angle_deg,
            ),
            "staggered_alternating_lattice",
        )
    if layout_key in {"rect", "rectangle", "rectangular"}:
        return (
            generate_rect_grid(
                image_shape,
                pitch_px,
                origin_x_px,
                origin_y_px,
                target_n,
                angle_deg=angle_deg,
            ),
            "rect",
        )
    return (
        generate_hex_grid(
            image_shape,
            pitch_px,
            origin_x_px,
            origin_y_px,
            target_n,
            angle_deg=angle_deg,
        ),
        "hex",
    )


def detect_positive_chambers_on_grid(
        img: np.ndarray,
        pitch_px: float,
        origin_x_px: float,
        origin_y_px: float,
        radius: int,
        target_n: int,
        angle_deg: float = 0.0,
        layout: str = "hex",
        c_offset: int = -5,
        min_circularity: float = 0.5,
        return_audit: bool = False,
) -> Tuple[List[np.ndarray], Dict, List[Dict]]:
    """Classify every calibrated chamber against robust local-annulus evidence.

    Grid mode creates fixed-radius contours, so shape metrics cannot tell true
    chamber signals from weak background texture.  The decision is therefore a
    two-pass local contrast gate: first measure every grid position, then set a
    robust image-specific threshold from the full grid distribution.
    """
    radius = max(2, int(radius))
    centers, layout = generate_grid_centers(
        img.shape,
        pitch_px,
        origin_x_px,
        origin_y_px,
        target_n,
        angle_deg=angle_deg,
        layout=layout,
    )
    inner_radius = max(2, int(round(radius * 0.65)))
    outer_radius = max(inner_radius + 3, int(round(radius * 1.8)))
    contours = []
    records = [] if return_audit else None
    rejected_boundary = 0
    rejected_signal = 0
    candidates = []

    for x, y in centers:
        ix, iy = int(round(x)), int(round(y))
        x1, x2 = ix - outer_radius, ix + outer_radius + 1
        y1, y2 = iy - outer_radius, iy + outer_radius + 1
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            rejected_boundary += 1
            if records is not None:
                records.append({
                    "stage": "grid",
                    "source": "grid",
                    "x": x,
                    "y": y,
                    "inner_mean": None,
                    "ring_mean": None,
                    "ring_std": None,
                    "local_delta": None,
                    "local_delta_cutoff": None,
                    "local_z": None,
                    "local_z_cutoff": None,
                    "psf_like": False,
                    "psf_center_delta": None,
                    "support_pixels": 0,
                    "support_fraction": 0.0,
                    "min_support_pixels": None,
                    "min_support_fraction": None,
                    "classification": "artifact",
                    "formal_positive": False,
                    "candidate_positive": False,
                    "grid_distance_px": None,
                    "reason": "grid_boundary",
                })
            continue

        patch = img[y1:y2, x1:x2]
        yy, xx = np.ogrid[-outer_radius:outer_radius + 1, -outer_radius:outer_radius + 1]
        dist2 = xx * xx + yy * yy
        inner_pixels = patch[dist2 <= inner_radius * inner_radius]
        ring_pixels = patch[
            (dist2 > inner_radius * inner_radius)
            & (dist2 <= outer_radius * outer_radius)
        ]
        inner_mean = float(np.mean(inner_pixels))
        ring_mean = float(np.mean(ring_pixels))
        ring_std = max(1.0, float(np.std(ring_pixels)))
        local_delta = inner_mean - ring_mean
        local_z = local_delta / ring_std
        center_val = float(patch[outer_radius, outer_radius])
        support_cutoff = ring_mean + max(1.0, 0.35 * ring_std)
        support_pixels = int(np.count_nonzero(inner_pixels > support_cutoff))
        psf_like = bool(
            center_val >= ring_mean + max(4.0, 1.2 * ring_std)
            and local_delta >= max(2.0, 0.45 * ring_std)
            and local_z >= 0.9
            and support_pixels / max(inner_pixels.size, 1) >= 0.20
        )
        candidates.append({
            "x": x,
            "y": y,
            "center_value": center_val,
            "inner_mean": inner_mean,
            "ring_mean": ring_mean,
            "ring_std": ring_std,
            "local_delta": local_delta,
            "local_z": local_z,
            "psf_center_delta": center_val - ring_mean,
            "psf_like": psf_like,
            "support_pixels": support_pixels,
            "support_fraction": float(support_pixels / max(inner_pixels.size, 1)),
            "inner_pixel_count": int(inner_pixels.size),
        })

    deltas = np.asarray([item["local_delta"] for item in candidates], dtype=np.float32)
    z_values = np.asarray([item["local_z"] for item in candidates], dtype=np.float32)
    ring_stds = np.asarray([item["ring_std"] for item in candidates], dtype=np.float32)
    delta_median, delta_mad = _median_mad(deltas)
    z_median, z_mad = _median_mad(z_values)
    ring_std_median = float(np.median(ring_stds)) if ring_stds.size else 1.0

    c_offset = int(c_offset)
    strict_boost = max(0.0, (-5.0 - float(c_offset)) / 3.0)
    recall_relax = max(0.0, (float(c_offset) + 5.0) / 3.0)
    robust_k = 3.2 + 0.55 * strict_boost - 0.35 * recall_relax
    z_k = 2.8 + 0.35 * strict_boost - 0.25 * recall_relax
    delta_threshold = max(
        1.8,
        delta_median + robust_k * delta_mad,
        0.45 * ring_std_median,
    )
    z_threshold = max(0.95, z_median + z_k * z_mad)
    min_circularity = float(min_circularity)
    min_support_fraction = float(np.clip(
        0.32
        + 0.25 * (min_circularity - 0.5)
        + 0.025 * strict_boost
        - 0.020 * recall_relax,
        0.22,
        0.55,
    ))

    accepted_deltas = []
    accepted_z_values = []
    accepted_support = []
    high_confidence_count = 0
    weak_count = 0
    uncertain_count = 0
    artifact_count = 0
    candidate_positive_count = 0
    min_support_pixels_ref = 0
    high_support_fraction_cutoff = float(max(min_support_fraction + 0.25, 0.60))

    for item in candidates:
        min_support = max(4, int(round(item["inner_pixel_count"] * min_support_fraction)))
        min_support_pixels_ref = max(min_support_pixels_ref, min_support)
        local_delta_cutoff = max(delta_threshold, 0.45 * item["ring_std"])
        candidate_positive = (
            item["local_delta"] >= local_delta_cutoff
            and item["local_z"] >= z_threshold
            and item["support_pixels"] >= min_support
        )
        high_confidence = (
            candidate_positive
            and item["psf_like"]
            and item["local_delta"] >= local_delta_cutoff * 1.25
            and item["local_z"] >= z_threshold * 1.20
            and item["support_pixels"] >= max(min_support + 2, int(round(min_support * 1.10)))
            and item["support_fraction"] >= high_support_fraction_cutoff
        )
        weak_positive = bool(candidate_positive and not high_confidence)
        uncertain = bool(
            not candidate_positive
            and item["local_delta"] >= local_delta_cutoff * 0.80
            and item["local_z"] >= z_threshold * 0.75
            and item["support_pixels"] >= max(3, int(round(min_support * 0.65)))
        )
        if high_confidence:
            classification = "high_confidence"
            reason = "accepted"
        elif weak_positive:
            classification = "weak"
            reason = "weak_not_formal_positive"
        elif uncertain:
            classification = "uncertain"
            reason = "uncertain_below_formal_cutoff"
        else:
            classification = "artifact"
            reason = "grid_local_signal_below_cutoff"
        formal_positive = high_confidence

        if records is not None:
            record = {
                "stage": "grid",
                "source": "grid",
                "x": item["x"],
                "y": item["y"],
                "inner_mean": item["inner_mean"],
                "ring_mean": item["ring_mean"],
                "ring_std": item["ring_std"],
                "local_delta": item["local_delta"],
                "local_delta_cutoff": local_delta_cutoff,
                "local_z": item["local_z"],
                "local_z_cutoff": z_threshold,
                "psf_like": bool(item["psf_like"]),
                "psf_center_delta": item["psf_center_delta"],
                "support_pixels": item["support_pixels"],
                "support_fraction": item["support_fraction"],
                "min_support_pixels": min_support,
                "min_support_fraction": min_support_fraction,
                "classification": classification,
                "formal_positive": bool(formal_positive),
                "candidate_positive": bool(candidate_positive),
                "grid_distance_px": 0.0,
                "reason": reason,
            }
            records.append(record)
        if candidate_positive:
            candidate_positive_count += 1
        if formal_positive:
            contours.append(_circle_contour(item["x"], item["y"], radius))
            accepted_deltas.append(item["local_delta"])
            accepted_z_values.append(item["local_z"])
            accepted_support.append(item["support_fraction"])
            high_confidence_count += 1
        elif weak_positive:
            weak_count += 1
            rejected_signal += 1
        elif uncertain:
            uncertain_count += 1
            rejected_signal += 1
        else:
            artifact_count += 1
            rejected_signal += 1

    accepted_deltas_arr = np.asarray(accepted_deltas, dtype=np.float32)
    accepted_z_arr = np.asarray(accepted_z_values, dtype=np.float32)
    accepted_support_arr = np.asarray(accepted_support, dtype=np.float32)
    weak_positive_fraction = float(weak_count / max(candidate_positive_count, 1))

    stats = {
        "grid_mode": True,
        "grid_positions": len(centers),
        "grid_accepted": len(contours),
        "grid_formal_positive_count": len(contours),
        "grid_candidate_positive_count": int(candidate_positive_count),
        "grid_high_confidence_count": int(high_confidence_count),
        "grid_weak_count": int(weak_count),
        "grid_uncertain_count": int(uncertain_count),
        "grid_artifact_count": int(artifact_count),
        "grid_rejected_boundary": rejected_boundary,
        "grid_rejected_signal": rejected_signal,
        "grid_pitch_px": float(pitch_px),
        "grid_origin_x_px": float(origin_x_px),
        "grid_origin_y_px": float(origin_y_px),
        "grid_angle_deg": float(angle_deg),
        "grid_layout": layout,
        "grid_local_delta_median": float(delta_median),
        "grid_local_delta_mad": float(delta_mad),
        "grid_local_z_median": float(z_median),
        "grid_local_z_mad": float(z_mad),
        "grid_ring_std_median": float(ring_std_median),
        "grid_delta_threshold": float(delta_threshold),
        "grid_z_threshold": float(z_threshold),
        "grid_min_support_fraction": float(min_support_fraction),
        "grid_high_support_fraction_cutoff": float(high_support_fraction_cutoff),
        "grid_min_support_pixels": int(min_support_pixels_ref),
        "grid_accepted_delta_median": float(np.median(accepted_deltas_arr)) if accepted_deltas_arr.size else 0.0,
        "grid_accepted_z_median": float(np.median(accepted_z_arr)) if accepted_z_arr.size else 0.0,
        "grid_accepted_support_median": float(np.median(accepted_support_arr)) if accepted_support_arr.size else 0.0,
        "grid_weak_positive_fraction": weak_positive_fraction,
        "grid_acceptance_ratio": float(len(contours) / max(len(centers), 1)),
    }
    return contours, stats, records or []
