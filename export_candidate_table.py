#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export per-candidate chamber data for manuscript figures.

This creates the data layer needed for Fig. 3C chi-square spatial checks and
Fig. 4 method-comparison/quality-control panels.
"""

import argparse
import csv
import json
import os
from pathlib import Path

from sample_analysis import SampleAnalyzer
from image_utils import load_image
from config import settings
from spatial_qc import (
    compute_spatial_qc,
    flatten_spatial_qc_for_report,
    measure_candidate_signal,
    spatial_bin_for_point,
)


def parse_channel(name: str) -> str:
    if "_42" in name or "-42" in name:
        return "Aβ42"
    if "_40" in name or "-40" in name:
        return "Aβ40"
    return ""


def contour_record(gray, contour, radius):
    return measure_candidate_signal(gray, contour, radius)


def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image file or folder")
    parser.add_argument("--out", default="results/candidate_export")
    parser.add_argument("--radius", type=int, default=settings.RADIUS)
    parser.add_argument("--mw", type=float, default=settings.DEFAULT_M)
    parser.add_argument("--volume-pl", type=float, default=settings.DEFAULT_V)
    parser.add_argument("--n-total", type=int, default=settings.DEFAULT_N)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--local-roi", action="store_true", help="Treat input images as local ROI crops; counts only, not formal concentration.")
    args = parser.parse_args()

    settings.RADIUS = int(args.radius)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_dir():
        files = [p for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}]
    else:
        files = [input_path]

    analyzer = SampleAnalyzer()
    image_rows = []
    candidate_rows = []
    all_candidate_rows = []
    quadrat_count_rows = []
    quadrat_residual_rows = []
    old_export_grid_candidates = settings.EXPORT_GRID_CANDIDATES
    settings.EXPORT_GRID_CANDIDATES = True

    try:
        for file_path in files:
            result = analyzer.analyze_single_atomic(
                str(file_path),
                M=args.mw,
                V=args.volume_pl,
                N=args.n_total,
                method="advanced",
                ai=not args.no_ai,
                save_result=True,
                output_dir=str(out),
                local_roi_mode=args.local_roi,
            )
            if "error" in result:
                image_rows.append({
                    "file": file_path.name,
                    "status": "ERROR",
                    "error": result["error"],
                })
                continue

            gray = load_image(str(file_path))
            metrics = result.get("metrics", {}) or {}
            conc = result.get("concentration_info", {}) or {}
            contours = result.get("contours", []) or []
            effective_n = int(result.get("effective_N") or metrics.get("effective_total_chambers") or args.n_total)
            sample_id = file_path.stem
            spatial_qc = result.get("spatial_qc") or compute_spatial_qc(
                contours,
                gray.shape,
                metrics=metrics,
            )
            spatial_report = flatten_spatial_qc_for_report(spatial_qc)

            image_rows.append({
                "file": file_path.name,
                "sample_id": sample_id,
                "channel": parse_channel(file_path.name),
                "status": "OK",
                "K_positive": result.get("actual_count", 0),
                "N_total": effective_n,
                "K_over_N": conc.get("ratio", ""),
                "lambda": conc.get("lambda", ""),
                "calculated_concentration": conc.get("concentration", ""),
                "molar_concentration": conc.get("molar_concentration", ""),
                "score": getattr(result.get("qc_metrics"), "confidence_score", ""),
                "analysis_scope": metrics.get("analysis_scope", ""),
                "detection_region_count": metrics.get("detection_region_count", ""),
                "complete_chip_count": metrics.get("complete_chip_count", ""),
                "chambers_per_region": metrics.get("chambers_per_region", ""),
                "chambers_per_chip": metrics.get("chambers_per_chip", ""),
                "regions_per_chip": metrics.get("regions_per_chip", ""),
                "chips_per_mask": metrics.get("chips_per_mask", ""),
                "design_prior_used": metrics.get("design_prior_used", ""),
                "design_layout_type": metrics.get("design_layout_type", ""),
                "design_effective_x_columns": metrics.get("design_effective_x_columns", ""),
                "design_effective_y_row_pairs": metrics.get("design_effective_y_row_pairs", ""),
                "do_not_interpret_regions_as_independent_chips": metrics.get("do_not_interpret_regions_as_independent_chips", ""),
                "grid_auto_confidence": metrics.get("grid_auto_confidence", ""),
                "grid_auto_confidence_raw": metrics.get("grid_auto_confidence_raw", ""),
                "grid_auto_inlier_fraction": metrics.get("grid_auto_inlier_fraction", ""),
                "grid_unreliable": metrics.get("grid_unreliable", ""),
                "grid_reliability_flags": ";".join(metrics.get("grid_reliability_flags", []) or []) if isinstance(metrics.get("grid_reliability_flags"), list) else metrics.get("grid_reliability_flags", ""),
                "grid_high_confidence_count": metrics.get("grid_high_confidence_count", ""),
                "grid_weak_count": metrics.get("grid_weak_count", ""),
                "grid_uncertain_count": metrics.get("grid_uncertain_count", ""),
                "grid_artifact_count": metrics.get("grid_artifact_count", ""),
                "grid_weak_positive_fraction": metrics.get("grid_weak_positive_fraction", ""),
                "psf_formal_qc_status": metrics.get("psf_formal_qc_status", ""),
                "quantitative_use": metrics.get("quantitative_use", conc.get("quantitative_use", "")),
                "spatial_qc_use": metrics.get("spatial_qc_use", ""),
                "formal_qc_flags": metrics.get("formal_qc_flags", ""),
                "radius_cv": metrics.get("radius_cv", ""),
                "dist_cv": metrics.get("dist_cv", ""),
                "dist_cv_applicable": metrics.get("dist_cv_applicable", ""),
                "high_background": metrics.get("high_background", ""),
                "rescue_used": metrics.get("rescue_used", ""),
                "suppressed_close": metrics.get("suppressed_close", ""),
                "rejected_tiny_support": metrics.get("rejected_tiny_support", ""),
                "psf_pass_rate": result.get("psf_pass_rate", ""),
                "llm_intervened": result.get("llm_intervened", ""),
                "visual_path": result.get("visual_path", ""),
                "spatial_qc_status": spatial_report.get("spatial_qc_status", ""),
                "spatial_chi2": spatial_report.get("spatial_chi2", ""),
                "spatial_df": spatial_report.get("spatial_df", ""),
                "spatial_chi2_df": spatial_report.get("spatial_chi2_df", ""),
                "spatial_p_value": spatial_report.get("spatial_p_value", ""),
                "spatial_pass_p_ge_0_05": spatial_report.get("spatial_pass_p_ge_0_05", ""),
                "spatial_expected_mode": spatial_report.get("spatial_expected_mode", ""),
                "spatial_chamber_opportunity_count": spatial_report.get("spatial_chamber_opportunity_count", ""),
                "spatial_opportunity_cv": spatial_report.get("spatial_opportunity_cv", ""),
                "spatial_qc_low_quality": spatial_report.get("spatial_qc_low_quality", ""),
                "spatial_qc_flags": spatial_report.get("spatial_qc_flags", ""),
                "spatial_max_abs_std_residual": spatial_report.get("spatial_max_abs_std_residual", ""),
                "spatial_worst_quadrat": spatial_report.get("spatial_worst_quadrat", ""),
                "spatial_edge_center_ratio": spatial_report.get("spatial_edge_center_ratio", ""),
                "qc_roi_source": spatial_report.get("qc_roi_source", ""),
                "qc_roi_confidence": spatial_report.get("qc_roi_confidence", ""),
                "qc_roi_bounds": spatial_report.get("qc_roi_bounds", ""),
            })

            for row in spatial_qc.get("spatial_quadrat_counts", []):
                quadrat_count_rows.append({
                    "file": file_path.name,
                    "sample_id": sample_id,
                    "channel": parse_channel(file_path.name),
                    **row,
                })
            for row in spatial_qc.get("spatial_quadrat_residuals", []):
                quadrat_residual_rows.append({
                    "file": file_path.name,
                    "sample_id": sample_id,
                    "channel": parse_channel(file_path.name),
                    **row,
                })

            for idx, record in enumerate(result.get("candidate_records", []) or [], start=1):
                all_candidate_rows.append({
                    "file": file_path.name,
                    "sample_id": sample_id,
                    "channel": parse_channel(file_path.name),
                    "candidate_id": idx,
                    **record,
                })

            for idx, contour in enumerate(contours, start=1):
                rec = contour_record(gray, contour, settings.RADIUS)
                rec.update(spatial_bin_for_point(rec["x"], rec["y"], spatial_qc))
                rec.update({
                    "file": file_path.name,
                    "sample_id": sample_id,
                    "channel": parse_channel(file_path.name),
                    "candidate_id": idx,
                    "positive": 1,
                })
                candidate_rows.append(rec)

            conv = result.get("convergence", [])
            if conv:
                safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)
                with open(out / f"{safe}_convergence.json", "w", encoding="utf-8") as fh:
                    json.dump(conv, fh, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    finally:
        settings.EXPORT_GRID_CANDIDATES = old_export_grid_candidates

    image_fields = [
        "file", "sample_id", "channel", "status", "K_positive", "N_total", "K_over_N",
        "lambda", "calculated_concentration", "molar_concentration", "score",
        "analysis_scope", "detection_region_count", "complete_chip_count",
        "chambers_per_region", "chambers_per_chip", "regions_per_chip",
        "chips_per_mask", "design_prior_used", "design_layout_type",
        "design_effective_x_columns", "design_effective_y_row_pairs",
        "do_not_interpret_regions_as_independent_chips",
        "grid_auto_confidence", "grid_auto_confidence_raw",
        "grid_auto_inlier_fraction", "grid_unreliable", "grid_reliability_flags",
        "grid_high_confidence_count", "grid_weak_count", "grid_uncertain_count",
        "grid_artifact_count", "grid_weak_positive_fraction",
        "psf_formal_qc_status", "quantitative_use", "spatial_qc_use",
        "formal_qc_flags",
        "radius_cv", "dist_cv", "dist_cv_applicable", "high_background", "rescue_used",
        "suppressed_close", "rejected_tiny_support", "psf_pass_rate", "llm_intervened",
        "visual_path",
        "spatial_qc_status", "spatial_chi2", "spatial_df", "spatial_chi2_df",
        "spatial_p_value", "spatial_pass_p_ge_0_05", "spatial_expected_mode",
        "spatial_chamber_opportunity_count", "spatial_opportunity_cv", "spatial_qc_low_quality",
        "spatial_qc_flags", "spatial_max_abs_std_residual", "spatial_worst_quadrat",
        "spatial_edge_center_ratio", "qc_roi_source", "qc_roi_confidence",
        "qc_roi_bounds", "error",
    ]
    candidate_fields = [
        "file", "sample_id", "channel", "candidate_id", "positive", "x", "y",
        "detected_radius", "area", "circularity", "inner_mean", "ring_mean",
        "ring_std", "local_delta", "local_z", "spatial_x_bin", "spatial_y_bin",
        "spatial_bin_expected", "spatial_bin_std_residual", "spatial_inside_qc_roi",
        "measurement_patch_width", "measurement_patch_height",
    ]
    quadrat_count_fields = [
        "file", "sample_id", "channel", "x_bin", "y_bin", "count", "expected",
        "chamber_opportunity", "opportunity_fraction",
    ]
    quadrat_residual_fields = [
        "file", "sample_id", "channel", "x_bin", "y_bin", "std_residual",
        "observed", "expected", "chamber_opportunity",
    ]
    all_candidate_fields = [
        "file", "sample_id", "channel", "candidate_id", "stage", "source",
        "x", "y", "reason", "classification", "formal_positive",
        "candidate_positive", "grid_distance_px", "inner_mean", "ring_mean",
        "ring_std", "local_delta", "local_delta_cutoff", "local_z",
        "local_z_cutoff", "psf_like", "psf_center_delta",
        "support_pixels", "support_fraction",
        "min_support_pixels", "min_support_fraction",
    ]
    write_rows(out / "image_summary.csv", image_rows, image_fields)
    write_rows(out / "candidate_table.csv", candidate_rows, candidate_fields)
    write_rows(out / "all_candidate_table.csv", all_candidate_rows, all_candidate_fields)
    write_rows(out / "spatial_quadrat_counts.csv", quadrat_count_rows, quadrat_count_fields)
    write_rows(out / "spatial_quadrat_residuals.csv", quadrat_residual_rows, quadrat_residual_fields)
    print(f"Saved image summary and candidate table to {out}")


if __name__ == "__main__":
    main()
