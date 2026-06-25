#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import Dict
from config import settings
from processing_pipeline import DeepLisaTools

logger = logging.getLogger(__name__)


class DeepLisaAgentTools:
    """Agent 工具接口适配器"""

    @staticmethod
    def inspect_image(path: str) -> Dict:
        return DeepLisaTools.inspect_image(path)

    @staticmethod
    def execute_detection(
            path: str,
            gamma: float,
            c_offset: int,
            block_size: int,
            min_circularity: float,
            target_N: int = None,
            audit_dir: str = ""
    ) -> Dict:
        if target_N is None: target_N = settings.DEFAULT_N

        params = {
            "gamma": float(gamma),
            "c_offset": int(c_offset),
            "block_size": int(block_size),
            "min_circularity": float(min_circularity),
            "detection_mode": settings.DETECTION_MODE,
            "grid_pitch_px": settings.GRID_PITCH_PX,
            "grid_origin_x_px": settings.GRID_ORIGIN_X_PX,
            "grid_origin_y_px": settings.GRID_ORIGIN_Y_PX,
            "grid_angle_deg": settings.GRID_ANGLE_DEG,
            "grid_layout": settings.GRID_LAYOUT,
            "export_grid_candidates": settings.EXPORT_GRID_CANDIDATES,
        }
        if audit_dir:
            params["audit_dir"] = audit_dir
        return DeepLisaTools.execute_detection(path, params, target_N)

    @staticmethod
    def suggest_heuristic_mutation(params: Dict, metrics: Dict, target_N: int):
        return DeepLisaTools.suggest_heuristic_mutation(params, metrics, target_N)
