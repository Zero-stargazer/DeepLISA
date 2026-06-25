#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLISA PSF 物理校验模块 (v1.0)
功能：基于点扩散函数 (PSF) 进行高斯拟合，验证信号真伪。
"""

import cv2
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Dict


class PSFVerifier:
    @staticmethod
    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
        x, y = xy
        a = (x - x0) ** 2 / (2 * sigma_x ** 2)
        b = (y - y0) ** 2 / (2 * sigma_y ** 2)
        return amplitude * np.exp(-(a + b)) + offset

    @staticmethod
    def verify_candidates(img_gray: np.ndarray, candidates: List[tuple], radius: int) -> List[int]:
        """
        验证候选点是否符合高斯分布
        返回：保留点的索引列表
        """
        keep_indices = []
        h_img, w_img = img_gray.shape
        pad = int(radius * 1.5)

        for i, (x, y, r) in enumerate(candidates):
            x_int, y_int = int(x), int(y)
            x1 = max(0, x_int - pad)
            y1 = max(0, y_int - pad)
            x2 = min(w_img, x_int + pad + 1)
            y2 = min(h_img, y_int + pad + 1)

            roi = img_gray[y1:y2, x1:x2]

            if roi.shape[0] < 3 or roi.shape[1] < 3: continue

            # 简化的判断：中心亮度必须大于边缘
            center_val = roi[roi.shape[0] // 2, roi.shape[1] // 2]
            mean_val = np.mean(roi)

            # 如果中心比平均值亮显著，且不是过曝点
            if center_val > mean_val * 1.1:
                keep_indices.append(i)

        return keep_indices

    @staticmethod
    def filter_contours_by_psf(img_gray: np.ndarray, contours: List, radius: int):
        """Return contours whose local signal has centered disk-like support."""
        if not contours:
            return [], {
                "psf_input_count": 0,
                "psf_kept_count": 0,
                "psf_rejected_count": 0,
                "psf_pass_rate": None,
            }

        keep = []
        h_img, w_img = img_gray.shape
        radius = max(2, int(radius))
        pad = max(5, int(round(radius * 2.0)))
        inner_r = max(2, int(round(radius * 0.65)))
        outer_r = max(inner_r + 2, int(round(radius * 1.5)))

        yy, xx = np.ogrid[-pad:pad + 1, -pad:pad + 1]
        dist2 = xx * xx + yy * yy
        inner_mask = dist2 <= inner_r * inner_r
        ring_mask = (dist2 > inner_r * inner_r) & (dist2 <= outer_r * outer_r)

        for contour in contours:
            ((x, y), _) = cv2.minEnclosingCircle(contour)
            ix, iy = int(round(x)), int(round(y))
            x1, x2 = ix - pad, ix + pad + 1
            y1, y2 = iy - pad, iy + pad + 1
            if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
                continue

            roi = img_gray[y1:y2, x1:x2]
            inner_pixels = roi[inner_mask]
            ring_pixels = roi[ring_mask]
            if inner_pixels.size < 8 or ring_pixels.size < 8:
                continue

            center_val = float(roi[pad, pad])
            inner_mean = float(np.mean(inner_pixels))
            ring_mean = float(np.mean(ring_pixels))
            ring_std = max(1.0, float(np.std(ring_pixels)))
            local_delta = inner_mean - ring_mean
            local_z = local_delta / ring_std
            support_cutoff = ring_mean + max(2.0, 0.8 * ring_std)
            support_fraction = float(np.count_nonzero(inner_pixels > support_cutoff) / inner_pixels.size)

            if (
                    center_val >= ring_mean + max(4.0, 1.2 * ring_std)
                    and local_delta >= max(2.0, 0.45 * ring_std)
                    and local_z >= 0.9
                    and support_fraction >= 0.20
            ):
                keep.append(contour)

        stats = {
            "psf_input_count": len(contours),
            "psf_kept_count": len(keep),
            "psf_rejected_count": len(contours) - len(keep),
            "psf_pass_rate": len(keep) / len(contours) if contours else None,
        }
        return keep, stats
