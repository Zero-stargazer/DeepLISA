#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAD-derived design prior helpers for DeepLISA.

The prior is a soft design model.  It tells the software how the mask is
organized, but it must not be used as absolute microscope-image coordinates.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN_PRIOR_PATH = (
    PROJECT_DIR
    / "cad_prior_extraction_20260608"
    / "design_prior_v8A_provisional.json"
)


FALLBACK_PRIOR: Dict = {
    "status": "fallback_defaults",
    "mask_layout": {
        "chips_per_mask": 2,
        "regions_per_chip": 2,
        "regions_per_mask": 4,
        "chip_region_groups": {
            "upper_chip": {
                "regions": ["chip_upper_left", "chip_upper_right"],
                "regions_per_chip": 2,
                "chambers_per_chip": 43200,
            },
            "lower_chip": {
                "regions": ["chip_lower_left", "chip_lower_right"],
                "regions_per_chip": 2,
                "chambers_per_chip": 43200,
            },
        },
        "do_not_interpret_regions_as_independent_chips": True,
    },
    "single_region": {
        "chambers_per_region": 21600,
        "chamber_diameter_um": 100,
        "center_bbox_aspect_height_over_width": 1.4870701146360816,
        "topology": {
            "layout_type": "staggered_alternating_lattice",
            "not_simple_orthogonal_grid": True,
            "effective_x_columns": 108,
            "effective_y_row_pairs": 200,
            "unique_y_coordinates": 400,
            "y_phases_per_row_pair": 2,
        },
        "pitch_mm": {
            "x_adjacent_column_gaps_alternating": [0.205, 0.145],
            "same_x_column_y_pitch": 0.14,
            "y_phase_offset_within_row_pair": 0.03,
            "row_pair_pitch": 0.14,
        },
    },
}


def _configured_prior_path(path: Optional[str] = None) -> Path:
    if path:
        return Path(path)
    env_path = os.getenv("DESIGN_PRIOR_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_DESIGN_PRIOR_PATH


@lru_cache(maxsize=4)
def _load_design_prior_cached(path_text: str) -> Dict:
    path = Path(path_text)
    if not path.exists():
        prior = deepcopy(FALLBACK_PRIOR)
        prior["design_prior_path"] = str(path)
        prior["design_prior_loaded"] = False
        prior["design_prior_error"] = "prior_file_not_found"
        return prior
    with path.open("r", encoding="utf-8") as fh:
        prior = json.load(fh)
    prior["design_prior_path"] = str(path)
    prior["design_prior_loaded"] = True
    prior["design_prior_error"] = ""
    return prior


def load_design_prior(path: Optional[str] = None, *, force_reload: bool = False) -> Dict:
    """Load the CAD-derived prior, falling back to embedded safe defaults."""
    resolved = str(_configured_prior_path(path))
    if force_reload:
        _load_design_prior_cached.cache_clear()
    try:
        return deepcopy(_load_design_prior_cached(resolved))
    except Exception as exc:
        prior = deepcopy(FALLBACK_PRIOR)
        prior["design_prior_path"] = resolved
        prior["design_prior_loaded"] = False
        prior["design_prior_error"] = str(exc)
        return prior


def summarize_design_prior(prior: Optional[Dict] = None) -> Dict:
    """Return a flat summary used by inference, reporting, and tests."""
    prior = prior or load_design_prior()
    layout = prior.get("mask_layout", {}) or {}
    region = prior.get("single_region", {}) or {}
    topology = region.get("topology", {}) or {}
    pitch = region.get("pitch_mm", {}) or {}

    chambers_per_region = int(region.get("chambers_per_region") or 21600)
    regions_per_chip = int(layout.get("regions_per_chip") or 2)
    chips_per_mask = int(layout.get("chips_per_mask") or 2)
    regions_per_mask = int(layout.get("regions_per_mask") or chips_per_mask * regions_per_chip)
    chambers_per_chip = int(chambers_per_region * regions_per_chip)
    chambers_per_mask = int(chambers_per_region * regions_per_mask)

    return {
        "design_prior_used": bool(prior.get("design_prior_loaded", True)),
        "design_prior_status": prior.get("status", ""),
        "design_prior_path": prior.get("design_prior_path", ""),
        "chips_per_mask": chips_per_mask,
        "regions_per_chip": regions_per_chip,
        "regions_per_mask": regions_per_mask,
        "chambers_per_region": chambers_per_region,
        "chambers_per_chip": chambers_per_chip,
        "chambers_per_mask": chambers_per_mask,
        "chip_region_groups": layout.get("chip_region_groups", {}),
        "do_not_interpret_regions_as_independent_chips": bool(
            layout.get("do_not_interpret_regions_as_independent_chips", True)
        ),
        "layout_type": topology.get("layout_type", "staggered_alternating_lattice"),
        "not_simple_orthogonal_grid": bool(topology.get("not_simple_orthogonal_grid", True)),
        "effective_x_columns": int(topology.get("effective_x_columns") or 108),
        "effective_y_row_pairs": int(topology.get("effective_y_row_pairs") or 200),
        "unique_y_coordinates": int(topology.get("unique_y_coordinates") or 400),
        "y_phases_per_row_pair": int(topology.get("y_phases_per_row_pair") or 2),
        "single_region_aspect_height_over_width": float(
            region.get("center_bbox_aspect_height_over_width") or 1.4870701146360816
        ),
        "chamber_diameter_um": float(region.get("chamber_diameter_um") or 100.0),
        "x_gap_pattern_mm": list(pitch.get("x_adjacent_column_gaps_alternating") or [0.205, 0.145]),
        "same_x_column_y_pitch_mm": float(
            pitch.get("same_x_column_y_pitch") or pitch.get("row_pair_pitch") or 0.14
        ),
        "y_phase_offset_mm": float(pitch.get("y_phase_offset_within_row_pair") or 0.03),
    }
