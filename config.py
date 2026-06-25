#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLISA Configuration Module (v5.1 - Final Stable)
"""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_app_path(path: str) -> str:
    """Resolve relative app paths from the code directory, not launch cwd."""
    path = (path or "").strip()
    if not path:
        return BASE_DIR
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


class AppSettings:
    def __init__(self):
        # 1. 物理常数
        self.AVOGADRO_NUMBER = 6.02214076e23

        # 2. 路径配置
        self.TEMP_DIR = resolve_app_path(os.getenv("TEMP_DIR", "./tmp"))
        self.RESULT_DIR = resolve_app_path(os.getenv("RESULT_DIR", "./results"))

        # 3. UI 界面设置
        self.RADIUS = int(os.getenv("RADIUS", "7"))

        self.THEME = {
            'bg_window': '#E8EDF2',
            'bg_card': '#2B3743',
            'bg_header': '#172637',
            'bg_panel': '#FFFFFF',
            'bg_panel_alt': '#F7F9FB',
            'bg_input': '#1F2B38',
            'bg_drop': '#243241',
            'bg_log': '#FFFFFF',
            'accent_color': '#365F83',
            'accent_hover': '#2F5578',
            'accent_soft': '#B8CFE4',
            'text_primary': '#EAF0F5',
            'text_secondary': '#B8C6D4',
            'text_dark': '#17212C',
            'text_muted': '#667789',
            'border_color': '#BDC9D5',
            'border_dark': '#1E2A36',
            'success_color': '#23D66D',
            'warning_color': '#B7791F',
            'error_color': '#C2413A',
            'font_family': 'Microsoft YaHei UI',
            'font_mono': 'Consolas'
        }
        self.LANGUAGE_OPTIONS = {"zh": "中文", "en": "English"}
        self.LANGUAGE = os.getenv("LANGUAGE", "zh").strip().lower()
        if self.LANGUAGE not in self.LANGUAGE_OPTIONS:
            self.LANGUAGE = "zh"

        # 4. 腔室识别参数
        self.CHAMBER_PARAMS = {
            "small_chambers": {"min_radius": 2, "max_radius": 5, "min_distance": 6},
            "medium_chambers": {"min_radius": 3, "max_radius": 8, "min_distance": 8},
            "large_chambers": {"min_radius": 6, "max_radius": 15, "min_distance": 12}
        }

        self.DT_RATIO = float(os.getenv("DT_RATIO", "0.4"))

        # 5. AI 配置
        self.GLM_API_KEY = os.getenv("GLM_API_KEY", "").strip()
        self.GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip()
        self.AI_MODEL = os.getenv("AI_MODEL", "glm-4-flash").strip()
        self.AI_TEMPERATURE = 0.8
        self.AI_TIMEOUT = 30
        self.AI_MAX_RETRIES = 2

        # 6. 物理实验参数 (默认值)
        # 腔室体积: r=0.05mm, h=100μm → V=π×0.05²×0.1=0.000785mm³=785pL (2026-05-13 实测修正)
        self.DEFAULT_M = float(os.getenv("DEFAULT_MW_KDA", "4.3"))
        self.DEFAULT_V = float(os.getenv("DEFAULT_VOL_PL", "785"))
        self.DEFAULT_N = int(os.getenv("DEFAULT_TOTAL_CHAMBERS", "21600"))
        self.DESIGN_PRIOR_PATH = os.getenv(
            "DESIGN_PRIOR_PATH",
            os.path.abspath(os.path.join(BASE_DIR, "..", "cad_prior_extraction_20260608", "design_prior_v8A_provisional.json"))
        ).strip()
        self.AUTO_REGION_INFERENCE = os.getenv("AUTO_REGION_INFERENCE", "1").strip() == "1"
        self.SINGLE_REGION_PIXEL_AREA = float(os.getenv("SINGLE_REGION_PIXEL_AREA", "13405647"))
        self.SINGLE_REGION_ASPECT = float(os.getenv("SINGLE_REGION_ASPECT", "1.487"))
        self.TWO_REGION_AREA_RATIO_MIN = float(os.getenv("TWO_REGION_AREA_RATIO_MIN", "1.55"))
        self.TWO_REGION_AREA_RATIO_SOFT_MIN = float(os.getenv("TWO_REGION_AREA_RATIO_SOFT_MIN", "1.25"))
        self.TWO_REGION_STACKED_ASPECT_MIN = float(os.getenv("TWO_REGION_STACKED_ASPECT_MIN", "2.35"))
        self.FOUR_REGION_AREA_RATIO_MIN = float(os.getenv("FOUR_REGION_AREA_RATIO_MIN", "3.20"))

        # 7. 质量控制规则
        self.QUALITY_RULES = {
            "min_chamber_count": 10,
            "max_chamber_count": 25000,
            "min_circularity": 0.4,
            "min_brightness": 5,
            "max_brightness": 255,
            "min_area_ratio": 0.001,
            "max_area_ratio": 0.8,
            "expected_count_min": 0,
            "expected_count_max": 0
        }

        self.TOPHAT_DEFAULT = {"kernel_size": 25}
        self.CLAHE_DEFAULT = {"clip_limit": 2.0, "tile_grid_size": (8, 8)}

        # 8. AI 参数调优边界
        self.AI_TUNING_BOUNDS = {
            "c_offset_min": -18,
            "c_offset_max": 6,
            "gamma_min": 0.5,
            "gamma_max": 2.0
        }

        # 9. 腔室类型 → TopHat kernel 映射
        self.KERNEL_MAP = {"small": 7, "medium": 13, "large": 25}
        self.CHAMBER_TYPE = os.getenv("CHAMBER_TYPE", "medium")

        # 10. 检测与审计模式
        # candidate: 默认兼容路径；grid_calibrated: 使用已标定六方网格逐腔室判定。
        self.DETECTION_MODE = os.getenv("DETECTION_MODE", "grid_auto").strip().lower()
        self.EXPORT_DETECTION_AUDIT = os.getenv("EXPORT_DETECTION_AUDIT", "0").strip() == "1"
        self.DETECTION_AUDIT_DIR = os.getenv("DETECTION_AUDIT_DIR", "./results/audit").strip()
        self.EXPORT_GRID_CANDIDATES = os.getenv("EXPORT_GRID_CANDIDATES", "1").strip() == "1"

        # 网格模式必须使用显式标定参数，避免在未知 ROI 上静默推断错误网格。
        self.GRID_PITCH_PX = float(os.getenv("GRID_PITCH_PX", "0"))
        self.GRID_ORIGIN_X_PX = float(os.getenv("GRID_ORIGIN_X_PX", "0"))
        self.GRID_ORIGIN_Y_PX = float(os.getenv("GRID_ORIGIN_Y_PX", "0"))
        self.GRID_ANGLE_DEG = float(os.getenv("GRID_ANGLE_DEG", "0"))
        self.GRID_LAYOUT = os.getenv("GRID_LAYOUT", "staggered_alternating_lattice").strip().lower()


settings = AppSettings()
