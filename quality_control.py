#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLISA 质量控制模块 (v5.1 - 键值安全版)
功能：评估图像质量，提供置信度扣分依据。
修复：全面使用 .get() 访问配置，防止 KeyError。
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List
from config import settings


@dataclass
class QualityMetrics:
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    confidence_score: float = 100.0  # 初始满分


class QualityController:
    def __init__(self, rules=None):
        self.rules = rules or settings.QUALITY_RULES

    def evaluate(self, gray, contours, count) -> QualityMetrics:
        m = QualityMetrics()
        m.confidence_score = 100.0  # 重置

        # 1. 数量检查
        if count == 0:
            # 即使是 0，也不一定是错的（可能是阴性），所以不给 False
            m.warnings.append("未检测到阳性信号 (可能是阴性)")
            m.confidence_score = 60.0  # 及格分
            return m

        min_c = self.rules.get("min_chamber_count", 10)
        max_c = self.rules.get("max_chamber_count", 25000)

        if count < min_c:
            m.warnings.append("信号数量极少")
            # 不扣分，因为可能是低浓度

        if count > max_c:
            m.passed = False
            m.errors.append("信号数量爆表 (可能为噪点)")
            m.confidence_score -= 40

        # 2. 亮度异常检查
        mean_brightness = float(np.mean(gray))
        min_b = self.rules.get("min_brightness", 10)
        max_b = self.rules.get("max_brightness", 255)

        if mean_brightness < min_b:
            m.warnings.append("图像过暗")
            m.confidence_score -= 10
        if mean_brightness > max_b:
            m.warnings.append("图像过曝")
            m.confidence_score -= 10

        # 3. 面积占比 (防止大光斑)
        img_area = gray.size
        total_area = sum(cv2.contourArea(c) for c in contours)
        area_ratio = total_area / (img_area + 1)

        max_ratio = self.rules.get("max_area_ratio", 0.6)
        if area_ratio > max_ratio:
            m.warnings.append("荧光区域占比过高")
            m.confidence_score -= 20

        # 钳制分数
        m.confidence_score = max(0.0, min(100.0, m.confidence_score))
        return m