#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infer whether an imported image covers one or two chip regions."""

from __future__ import annotations

from typing import Dict, Tuple

from config import settings
from design_prior import load_design_prior, summarize_design_prior
from image_utils import load_image


def infer_region_layout_from_shape(
        image_shape: Tuple[int, int],
        base_region_chambers: int,
        single_region_pixel_area: float = None,
        single_region_aspect: float = None,
) -> Dict:
    """Infer detection-region and chip coverage from image geometry.

    The CAD prior defines two chips per mask, with two detection regions per
    chip.  This function reports detection-region counts explicitly so that four
    regions are never interpreted as four chips.
    """
    design = summarize_design_prior(load_design_prior())
    height, width = int(image_shape[0]), int(image_shape[1])
    pixel_area = max(1, width * height)
    short_side = max(1, min(width, height))
    long_side = max(width, height)
    aspect = long_side / short_side

    base_region_chambers = int(base_region_chambers) if base_region_chambers else design["chambers_per_region"]
    if base_region_chambers <= 0:
        base_region_chambers = design["chambers_per_region"]

    single_area_hint = (
        float(single_region_pixel_area)
        if single_region_pixel_area is not None
        else float(settings.SINGLE_REGION_PIXEL_AREA)
    )
    single_aspect_hint = (
        float(single_region_aspect)
        if single_region_aspect is not None
        else float(design.get("single_region_aspect_height_over_width") or settings.SINGLE_REGION_ASPECT)
    )
    single_aspect_hint = max(1.05, single_aspect_hint)
    area_ratio = pixel_area / single_area_hint if single_area_hint > 0 else 1.0

    double_stacked_aspect = single_aspect_hint * 2.0
    double_side_by_side_aspect = max(2.0 / single_aspect_hint, single_aspect_hint / 2.0)
    single_aspect_error = abs(aspect - single_aspect_hint) / single_aspect_hint
    stacked_aspect_error = abs(aspect - double_stacked_aspect) / double_stacked_aspect
    side_by_side_aspect_error = (
        abs(aspect - double_side_by_side_aspect) / double_side_by_side_aspect
    )

    detection_region_count = 1
    complete_chip_count = 0
    reason = "single_detection_region_default"
    confidence = 0.55

    full_mask_area_ratio_min = float(getattr(settings, "FOUR_REGION_AREA_RATIO_MIN", 3.20))
    if area_ratio >= full_mask_area_ratio_min:
        detection_region_count = min(design["regions_per_mask"], 4)
        complete_chip_count = min(design["chips_per_mask"], 2)
        reason = "pixel_area_near_four_detection_regions_two_chips"
        confidence = min(0.96, 0.72 + min(abs(area_ratio - 4.0), 1.0) * 0.10)
    elif area_ratio >= settings.TWO_REGION_AREA_RATIO_MIN:
        detection_region_count = 2
        complete_chip_count = 1
        reason = "pixel_area_near_two_detection_regions_one_chip"
        confidence = min(0.95, 0.60 + min(abs(area_ratio - 2.0), 1.0) * 0.20)
    elif area_ratio >= settings.TWO_REGION_AREA_RATIO_SOFT_MIN and (
            side_by_side_aspect_error < single_aspect_error * 0.85
            or stacked_aspect_error < single_aspect_error * 0.85
    ):
        detection_region_count = 2
        complete_chip_count = 1
        reason = "aspect_and_area_suggest_two_detection_regions_one_chip"
        confidence = 0.72
    elif aspect >= settings.TWO_REGION_STACKED_ASPECT_MIN and area_ratio >= 1.15:
        detection_region_count = 2
        complete_chip_count = 1
        reason = "long_aspect_suggests_two_detection_regions_one_chip"
        confidence = 0.78
    else:
        if area_ratio <= 1.25:
            confidence = 0.80
        elif single_aspect_error <= 0.20:
            confidence = 0.70

    analysis_scope = {
        1: "single_detection_region",
        2: "single_chip_two_detection_regions",
        4: "full_mask_two_chips_four_detection_regions",
    }.get(detection_region_count, "unknown_region_coverage")

    return {
        "auto_region_inference": True,
        "region_count": int(detection_region_count),
        "detection_region_count": int(detection_region_count),
        "complete_chip_count": int(complete_chip_count),
        "chip_count": int(complete_chip_count),
        "base_region_chambers": int(base_region_chambers),
        "effective_total_chambers": int(base_region_chambers * detection_region_count),
        "chambers_per_region": int(base_region_chambers),
        "chambers_per_chip": int(design["chambers_per_chip"]),
        "regions_per_chip": int(design["regions_per_chip"]),
        "chips_per_mask": int(design["chips_per_mask"]),
        "regions_per_mask": int(design["regions_per_mask"]),
        "image_width": int(width),
        "image_height": int(height),
        "image_pixel_area": int(pixel_area),
        "image_aspect_long_short": float(aspect),
        "single_region_pixel_area_hint": float(single_area_hint),
        "single_region_aspect_hint": float(single_aspect_hint),
        "area_ratio_to_single_region": float(area_ratio),
        "region_inference_reason": reason,
        "region_inference_confidence": float(confidence),
        "analysis_scope_inferred": analysis_scope,
        "design_prior_used": bool(design["design_prior_used"]),
        "design_layout_type": design["layout_type"],
        "design_effective_x_columns": int(design["effective_x_columns"]),
        "design_effective_y_row_pairs": int(design["effective_y_row_pairs"]),
        "design_unique_y_coordinates": int(design["unique_y_coordinates"]),
        "do_not_interpret_regions_as_independent_chips": bool(
            design["do_not_interpret_regions_as_independent_chips"]
        ),
    }


def infer_region_layout_from_image(path: str, base_region_chambers: int) -> Dict:
    gray = load_image(path)
    return infer_region_layout_from_shape(gray.shape, base_region_chambers)
