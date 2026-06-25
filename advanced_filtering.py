#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLISA 高级滤波与图像增强模块 (v6.0 - 细节保真版)
功能：提供底层的 OpenCV 图像处理算子。
修复：优化 adaptiveThreshold 的参数，减少背景噪点对阈值的干扰。
"""

import cv2
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Dict, Optional, List
from config import settings


def enhance_image_quality(
        gray: np.ndarray,
        gamma: float = 1.0,
        contrast: float = 1.0,
        brightness: int = 0,
        sharpen_amount: float = 0.0
) -> np.ndarray:
    """图像增强管线"""
    # 1. 归一化
    norm_img = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 2. Gamma 校正
    if gamma != 1.0:
        invGamma = 1.0 / (gamma + 1e-6)
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        norm_img = cv2.LUT(norm_img, table)

    # 3. 线性对比度
    if contrast != 1.0 or brightness != 0:
        norm_img = cv2.convertScaleAbs(norm_img, alpha=contrast, beta=brightness)

    return norm_img


def apply_advanced_preprocessing(
        gray: np.ndarray,
        use_tophat: bool = True,
        use_clahe: bool = True,
        clahe_params: Optional[Dict] = None,
        tophat_params: Optional[Dict] = None
) -> np.ndarray:
    """预处理管线"""
    result = gray.copy()

    # 1. TopHat 去除不均匀背景 (核心)
    if use_tophat:
        ksize = tophat_params.get("kernel_size", 25) if tophat_params else 25
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        result = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel)

    # 2. 高斯模糊 (微弱) - 压制单像素热噪点
    # 在二值化之前做一点点模糊，可以极大改善 "Noise Explosion"
    result = cv2.GaussianBlur(result, (3, 3), 0.5)

    if use_clahe:
        clip = clahe_params.get("clip_limit", 2.0) if clahe_params else 2.0
        grid = clahe_params.get("tile_grid_size", (8, 8)) if clahe_params else (8, 8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        result = clahe.apply(result)

    return result


def binarize_adaptive(
        image: np.ndarray,
        block_size: int = 15,
        c_offset: int = -2
) -> np.ndarray:
    """自适应二值化"""
    if block_size % 2 == 0: block_size += 1
    if block_size < 3: block_size = 3

    # 注意：cv2.adaptiveThreshold 中，C 是减去的值。
    # 也就是 Threshold = Mean - C
    # 我们希望 Threshold 更高（更严），所以 C 应该是负数 (Mean - (-5) = Mean + 5)
    # 或者 C 是正数但我们要减去它？
    # OpenCV定义: dst(x,y) = src(x,y) > T(x,y)
    # T(x,y) = mean(x,y) - C
    # 如果我们要更严格 (只留很亮的)，T 应该很高。
    # T = mean - C。如果要 T > mean，则 C 必须是负数。
    # 例如 C = -10 => T = mean + 10。只有比均值亮 10 的才是 255。正确。

    binary = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_offset  # 传入负数表示更严格
    )

    # 形态学操作：去除毛刺
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # 开运算：先腐蚀后膨胀，断开粘连，去掉噪点
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return clean_binary


def advanced_spot_filtering(
        binary_img: np.ndarray,
        min_radius: int = 2,
        max_radius: int = 15,
        min_circularity: float = 0.5,
        dt_ratio: float = 0.2,
        min_distance: int = 5,
        return_audit: bool = False
):
    """Split and filter bright regions without using a whole-image DT maximum.

    A whole-image distance-transform threshold lets the strongest chamber set
    the cutoff for every other candidate. Compute the threshold per connected
    bright region so weak but geometrically valid chambers remain eligible.
    """
    result = np.zeros_like(binary_img)
    count = 0
    audit_records = [] if return_audit else None

    min_area = np.pi * (min_radius ** 2) * 0.5  # 允许稍微小一点点
    max_area = np.pi * (max_radius ** 2) * 1.5

    components, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for component in components:
        x, y, w, h = cv2.boundingRect(component)
        roi = np.zeros((h + 2, w + 2), dtype=np.uint8)
        shifted = component.copy()
        shifted[:, 0, 0] -= x - 1
        shifted[:, 0, 1] -= y - 1
        cv2.drawContours(roi, [shifted], -1, 255, -1)

        dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        local_max = float(dist_transform.max())
        if local_max <= 0:
            continue

        _, dist_norm = cv2.threshold(
            dist_transform, max(0.8, dt_ratio * local_max), 255, 0
        )
        seeds, _ = cv2.findContours(
            np.uint8(dist_norm), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for seed in seeds:
            c = seed.copy()
            c[:, 0, 0] += x - 1
            c[:, 0, 1] += y - 1
            area = cv2.contourArea(c)
            ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)
            perimeter = cv2.arcLength(c, True)
            circularity = 0.0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)
            reason = "accepted"

            if area < min_area:
                reason = "shape_area_too_small"
            elif area > max_area:
                reason = "shape_area_too_large"
            elif radius < (min_radius * 0.8):
                reason = "shape_radius_too_small"
            elif perimeter == 0:
                reason = "shape_zero_perimeter"
            elif circularity < min_circularity:
                reason = "shape_low_circularity"

            if audit_records is not None:
                audit_records.append({
                    "stage": "shape",
                    "x": float(center_x),
                    "y": float(center_y),
                    "radius": float(radius),
                    "area": float(area),
                    "circularity": float(circularity),
                    "reason": reason,
                })
            if reason != "accepted":
                continue

            cv2.drawContours(result, [c], -1, 255, -1)
            count += 1

    if return_audit:
        return result, count, audit_records
    return result, count
