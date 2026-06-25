#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像加载、预处理与可视化工具模块 (v4.0 - 半径标准化版)
核心修复：
1. [可视化] 强制使用 settings.RADIUS 进行标注，解决亮点半径大小不一的问题。
2. [UI] 优化底部文字排版，防止遮挡。
"""

import cv2
import numpy as np
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
from config import settings

# 尝试加载字体
try:
    FONT_PATH = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
    if not os.path.exists(FONT_PATH):
        FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    # 稍微调小字体以容纳更多信息
    DEFAULT_FONT = ImageFont.truetype(FONT_PATH, 20)
    SMALL_FONT = ImageFont.truetype(FONT_PATH, 16)
except Exception:
    DEFAULT_FONT = ImageFont.load_default()
    SMALL_FONT = ImageFont.load_default()


def load_image(path: str) -> np.ndarray:
    """加载图像并转换为 8位灰度图"""
    # Use np.fromfile + cv2.imdecode instead of cv2.imread so Windows paths
    # containing Chinese characters remain readable.
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像文件未找到: {path}")

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV 无法解码图像: {path}")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype == np.uint16:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    elif gray.dtype != np.uint8:
        gray = gray.astype('uint8')

    return gray


def evaluate_image_quality(gray_img: np.ndarray) -> dict:
    """简易图像质量评估"""
    mean_brightness = np.mean(gray_img)
    contrast = np.std(gray_img)
    snr_estimate = mean_brightness / (contrast + 1e-6)

    quality_status = "正常"
    warnings = []

    if mean_brightness < 20:
        quality_status = "极暗"
        warnings.append("图像严重曝光不足")
    elif mean_brightness > 220:
        quality_status = "过曝"
        warnings.append("图像整体过亮")

    return {
        "brightness": mean_brightness,
        "contrast": contrast,
        "snr": snr_estimate,
        "status": quality_status,
        "warnings": warnings
    }


def create_visualization_base(
        img_path: str,
        contours: list,
        count: int,
        text_info: str = "",
        save_path: str = ""
) -> str:  # 返回保存路径
    """
    生成可视化结果
    [修复] 强制使用固定半径绘制标记，确保整齐划一。
    """
    try:
        # 1. 准备底图
        img_data = load_image(img_path)
        # 转为彩色以便绘制高对比度标记
        vis_img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)

        # 2. 绘制阳性腔室细轮廓。避免实心圆遮盖原始荧光信号，
        # 便于 Fig. 3A 使用 raw/processed overlay 做论文图。
        if contours and count > 0:
            highlight_color = (0, 255, 0)  # green in BGR

            # [关键修复] 获取配置中的标准半径
            fixed_radius = settings.RADIUS
            if fixed_radius < 1: fixed_radius = 3  # 保底
            for c in contours:
                ((x, y), _) = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                cv2.circle(vis_img, center, int(fixed_radius), highlight_color, -1, lineType=cv2.LINE_AA)

        # 转为 PIL
        pil_content = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))

        # 3. 绘制底部 Footer (解决文字压住图片问题)
        # 计算文字高度
        char_width = 12  # 估算
        max_chars = pil_content.width // char_width
        wrapped_lines = []
        for line in text_info.split('\n'):
            wrapped_lines.extend(textwrap.wrap(line, width=max(30, max_chars)))

        line_height = 28
        padding = 15
        footer_height = len(wrapped_lines) * line_height + (padding * 2)

        # 创建新画布 = 原图 + 底部白边
        total_height = pil_content.height + footer_height
        final_canvas = Image.new("RGB", (pil_content.width, total_height), "#FFFFFF")
        final_canvas.paste(pil_content, (0, 0))

        # 绘制文字
        draw = ImageDraw.Draw(final_canvas)
        draw.line([(0, pil_content.height), (pil_content.width, pil_content.height)], fill="#E2E8F0", width=2)

        y_text = pil_content.height + padding
        for line in wrapped_lines:
            fill_color = "#1E293B"  # 默认深蓝灰
            if "Count" in line:
                fill_color = "#0F172A"  # 强调
            elif "Score" in line:
                fill_color = "#059669"  # 绿色分数

            draw.text((15, y_text), line, font=DEFAULT_FONT, fill=fill_color)
            y_text += line_height

        # 4. 保存
        if not save_path:
            # 默认路径
            filename = f"result_{os.path.basename(img_path)}"
            save_path = os.path.join(settings.RESULT_DIR, filename)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_canvas.save(save_path)
        print(f"   可视化已保存: {save_path}")

        return save_path

    except Exception as e:
        print(f"可视化生成失败: {e}")
        return ""
