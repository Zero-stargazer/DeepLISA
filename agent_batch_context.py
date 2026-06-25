#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight batch-level consistency hints for DeepLISA Agent."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class BatchContext:
    """Track stable batch-level grid and region properties."""

    def __init__(self):
        self.records: List[Dict] = []

    def observe(self, metrics: Dict) -> Dict:
        assessment = self.assess(metrics)
        if metrics.get("local_roi_mode"):
            return assessment
        merged = dict(metrics)
        merged.update(assessment)
        self.records.append(merged)
        return assessment

    def assess(self, metrics: Dict) -> Dict:
        warnings: List[str] = []
        if metrics.get("local_roi_mode"):
            return {
                "batch_consistency_score": 1.0,
                "batch_warnings": warnings,
                "batch_reference_count": len(self.records),
            }
        if len(self.records) < 2:
            return {
                "batch_consistency_score": 1.0,
                "batch_warnings": warnings,
                "batch_reference_count": len(self.records),
            }

        score = 1.0
        pitch_values = [float(r["grid_pitch_px"]) for r in self.records if r.get("grid_pitch_px")]
        angle_values = [float(r["grid_angle_deg"]) for r in self.records if r.get("grid_angle_deg") is not None]
        layouts = [r.get("grid_layout") for r in self.records if r.get("grid_layout")]
        region_counts = [r.get("region_count") for r in self.records if r.get("region_count")]
        backgrounds = [bool(r.get("high_background", False)) for r in self.records]

        if metrics.get("grid_mode") and pitch_values and metrics.get("grid_pitch_px"):
            pitch_ref = float(np.median(pitch_values))
            pitch = float(metrics.get("grid_pitch_px"))
            if abs(pitch - pitch_ref) > max(2.0, 0.10 * pitch_ref):
                warnings.append("grid_pitch_outlier")
                score -= 0.25

        if metrics.get("grid_mode") and angle_values and metrics.get("grid_angle_deg") is not None:
            angle_ref = float(np.median(angle_values))
            angle = float(metrics.get("grid_angle_deg"))
            if abs(angle - angle_ref) > 3.0:
                warnings.append("grid_angle_outlier")
                score -= 0.20

        if layouts and metrics.get("grid_layout"):
            layout_ref = max(set(layouts), key=layouts.count)
            if metrics.get("grid_layout") != layout_ref:
                warnings.append("grid_layout_mismatch")
                score -= 0.20

        if region_counts and metrics.get("region_count"):
            region_ref = max(set(region_counts), key=region_counts.count)
            if metrics.get("region_count") != region_ref:
                warnings.append("region_count_mismatch")
                score -= 0.15

        if backgrounds and bool(metrics.get("high_background", False)) != (sum(backgrounds) >= len(backgrounds) / 2):
            warnings.append("background_state_differs")
            score -= 0.10

        return {
            "batch_consistency_score": float(np.clip(score, 0.0, 1.0)),
            "batch_warnings": warnings,
            "batch_reference_count": len(self.records),
        }
