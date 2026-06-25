#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLISA 智能分析系统 (v5.2 - 稳定性修复版)
修复说明：
1. [CRITICAL] 修复 'QualityMetrics' object has no attribute 'get' 崩溃问题。
   增加了 _get_score 辅助函数，兼容对象属性访问和字典键值访问。
2. [Debug] 启动时检查并显示 API Key 状态，帮助排查 AI 无响应问题。
"""

import os
import threading
import queue
import logging
import csv
import json
import sys
import traceback
from datetime import datetime
import tkinter.scrolledtext as st
from tkinter import StringVar, BooleanVar, DoubleVar, IntVar, Frame, Toplevel, Label, Canvas
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# 依赖检查与导入
try:
    from config import settings
    from sample_analysis import SampleAnalyzer
except ImportError as e:
    print(f"[Fatal] 核心模块缺失: {e}")
    sys.exit(1)

# 尝试加载拖拽库
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    BaseTk = TkinterDnD.Tk
    DND_AVAILABLE = True
except ImportError:
    from tkinter import Tk as BaseTk

    DND_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


LANGUAGE_LABEL_TO_KEY = {"中文": "zh", "English": "en"}

UI_TEXT = {
    "zh": {
        "window_title": "DeepLISA 智能分析系统 - 高通量数字 ELISA 分析平台",
        "app_title": "DeepLISA 智能分析系统",
        "app_subtitle": "High-Throughput Digital Analysis Platform",
        "section_params": "📐 物理参数配置 (Parameters)",
        "language_label": "界面语言:",
        "mw_label": "分子量 (kDa):",
        "vol_label": "单腔体积 (pL):",
        "total_n_label": "单区域腔室数:",
        "radius_label": "识别半径 (px):",
        "local_roi_mode": "局部 ROI 模式（只看计数，不用于正式浓度）",
        "detection_mode_label": "检测模式:",
        "section_strategy": "🔍 分析策略 (Strategy)",
        "standard_scan": "快速初筛 (Standard Scan)",
        "advanced_ai": "深度智能分析 (Advanced AI)",
        "ai_enable": "启用智能参数优化 (AI Optimization)",
        "ai_hint": "  *调用历史优化数据辅助计算",
        "drop_ready": "📥 样本载入区域\n(拖拽图像文件至此)",
        "drop_no_dnd": "📥 样本载入区域\n(当前环境仅支持按钮导入)",
        "import_image": "导入单张图像 (Import Image)",
        "batch_process": "批量文件夹处理 (Batch Process)",
        "system_log": "📜 系统运行日志 (System Log)",
        "log_ready": ">>> 系统初始化完成。DeepLISA Ready.",
        "status_ready": "Ready",
        "result_title": "📊 分析报告",
        "visual_missing": "[可视化文件丢失]",
        "load_failed": "加载失败",
        "result_concentration": "浓度",
        "result_count": "计数",
        "result_region": "区域",
        "result_confidence": "置信度",
        "result_params": "参数",
    },
    "en": {
        "window_title": "DeepLISA Intelligent Analysis System - High-Throughput Digital ELISA Platform",
        "app_title": "DeepLISA Intelligent Analysis System",
        "app_subtitle": "High-Throughput Digital Analysis Platform",
        "section_params": "📐 Physical Parameters",
        "language_label": "Language:",
        "mw_label": "MW (kDa):",
        "vol_label": "Volume (pL):",
        "total_n_label": "Chambers:",
        "radius_label": "Radius (px):",
        "local_roi_mode": "Local ROI mode (count only)",
        "detection_mode_label": "Detection mode:",
        "section_strategy": "🔍 Analysis Strategy",
        "standard_scan": "Standard Scan",
        "advanced_ai": "Advanced AI Analysis",
        "ai_enable": "Enable AI Optimization",
        "ai_hint": "  *Uses historical optimization data",
        "drop_ready": "📥 Sample Loading Area\n(Drop image files here)",
        "drop_no_dnd": "📥 Sample Loading Area\n(Use import buttons in this environment)",
        "import_image": "Import Image",
        "batch_process": "Batch Process Folder",
        "system_log": "📜 System Log",
        "log_ready": ">>> System initialized. DeepLISA Ready.",
        "status_ready": "Ready",
        "result_title": "📊 Analysis Report",
        "visual_missing": "[Visualization file missing]",
        "load_failed": "Load failed",
        "result_concentration": "Concentration",
        "result_count": "Count",
        "result_region": "Region",
        "result_confidence": "Confidence",
        "result_params": "Params",
    },
}


class ModernAutoCVApp(BaseTk):
    def __init__(self):
        super().__init__()

        # 启动诊断
        print("-" * 40)
        print(">>> DeepLISA 系统启动诊断")
        if settings.GLM_API_KEY:
            print(f"[OK] API Key 已加载: {settings.GLM_API_KEY[:6]}******")
        else:
            print("[WARN] 未检测到 API Key！AI 功能将无法使用。请检查 .env 文件。")
        print("-" * 40)

        self.current_language = settings.LANGUAGE if settings.LANGUAGE in UI_TEXT else "zh"
        self.var_language = StringVar(value=settings.LANGUAGE_OPTIONS[self.current_language])
        self._apply_window_title()
        self.geometry("1280x850")
        self.configure(bg=settings.THEME['bg_window'])

        self._setup_styles()

        self.analyzer = SampleAnalyzer()
        self.gui_queue = queue.Queue()
        self.is_processing = False

        self.var_radius = DoubleVar(value=settings.RADIUS)
        self.var_mw = DoubleVar(value=settings.DEFAULT_M)
        self.var_vol = DoubleVar(value=settings.DEFAULT_V)
        self.var_total_n = IntVar(value=settings.DEFAULT_N)
        self.var_local_roi_mode = BooleanVar(value=False)
        self.var_ai_enable = BooleanVar(value=True)
        self.var_method = StringVar(value="advanced")
        self.var_detection_mode = StringVar(value=settings.DETECTION_MODE)

        self._create_header()
        self._create_main_content()
        self.after(100, self._process_gui_queue)

    def _t(self, key):
        return UI_TEXT.get(self.current_language, UI_TEXT["zh"]).get(key, UI_TEXT["zh"].get(key, key))

    def _apply_window_title(self):
        self.title(self._t("window_title"))

    def _rebuild_static_ui(self):
        if hasattr(self, "left_canvas"):
            self.left_canvas.unbind_all('<MouseWheel>')
        for attr in ("header_frame", "main_container", "status_bar"):
            widget = getattr(self, attr, None)
            if widget is not None and widget.winfo_exists():
                widget.destroy()
        self._apply_window_title()
        self._create_header()
        self._create_main_content()

    def _on_language_change(self, _event=None):
        selected = self.var_language.get()
        language_key = LANGUAGE_LABEL_TO_KEY.get(selected, "zh")
        if language_key == self.current_language:
            return
        self.current_language = language_key
        settings.LANGUAGE = language_key
        self.var_language.set(settings.LANGUAGE_OPTIONS[language_key])
        self._rebuild_static_ui()

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            pass

        theme = settings.THEME
        style.configure('TFrame', background=theme['bg_window'])
        style.configure('Card.TFrame', background=theme['bg_card'], relief='flat')
        style.configure('Panel.TFrame', background=theme['bg_panel'], relief='flat')
        style.configure('TLabel', background=theme['bg_window'], foreground=theme['text_dark'],
                        font=(theme['font_family'], 10))
        style.configure('Card.TLabel', background=theme['bg_card'], foreground=theme['text_primary'])
        style.configure('Panel.TLabel', background=theme['bg_panel'], foreground=theme['text_dark'])
        style.configure('Title.TLabel', background=theme['bg_header'], foreground='#F6FAFD',
                        font=(theme['font_family'], 18, 'bold'))
        style.configure('SubTitle.TLabel', background=theme['bg_card'], foreground=theme['accent_soft'],
                        font=(theme['font_family'], 11, 'bold'))
        style.configure('PanelSubTitle.TLabel', background=theme['bg_panel'], foreground=theme['accent_color'],
                        font=(theme['font_family'], 11, 'bold'))
        style.configure('TButton', background=theme['accent_color'], foreground='#FFFFFF',
                        font=(theme['font_family'], 10, 'bold'), borderwidth=0, padding=(10, 7),
                        focuscolor=theme['bg_window'])
        style.map('TButton', background=[('active', theme['accent_hover'])], relief=[('pressed', 'sunken')])
        style.configure('TCheckbutton', background=theme['bg_card'], foreground=theme['text_primary'])
        style.map('TCheckbutton', background=[('active', theme['bg_card'])])
        style.configure('TRadiobutton', background=theme['bg_card'], foreground=theme['text_primary'])
        style.map('TRadiobutton', background=[('active', theme['bg_card'])])
        style.configure('TEntry', fieldbackground=theme['bg_input'], background=theme['bg_input'],
                        foreground='white', insertcolor='white', bordercolor=theme['border_dark'],
                        lightcolor=theme['border_dark'], darkcolor=theme['border_dark'])
        style.configure('TCombobox', fieldbackground=theme['bg_input'], background=theme['bg_input'],
                        foreground='white', arrowcolor=theme['text_secondary'],
                        bordercolor=theme['border_dark'], lightcolor=theme['border_dark'],
                        darkcolor=theme['border_dark'])
        style.map('TCombobox', fieldbackground=[('readonly', theme['bg_input'])],
                  foreground=[('readonly', 'white')], selectbackground=[('readonly', theme['bg_input'])],
                  selectforeground=[('readonly', 'white')])
        style.configure('TSeparator', background=theme['border_color'])
        style.configure('Side.Vertical.TScrollbar', background=theme['border_color'],
                        troughcolor=theme['bg_card'], bordercolor=theme['bg_card'],
                        arrowcolor=theme['text_secondary'], relief='flat')
        style.map('Side.Vertical.TScrollbar', background=[('active', theme['accent_color'])])

    def _create_header(self):
        header = Frame(self, bg=settings.THEME['bg_header'], height=64)
        self.header_frame = header
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        ttk.Label(header, text=self._t("app_title"), style='Title.TLabel').pack(side='left', padx=22)
        Label(header, text=f"| {self._t('app_subtitle')}", bg=settings.THEME['bg_header'],
              fg=settings.THEME['text_secondary'], font=('Arial', 10)).pack(side='left', pady=(7, 0))

    def _create_main_content(self):
        main_container = Frame(self, bg=settings.THEME['bg_window'])
        self.main_container = main_container
        main_container.pack(fill='both', expand=True, padx=20, pady=18)

        left_outer = ttk.Frame(main_container, style='Card.TFrame', padding=0)
        left_outer.pack(side='left', fill='y', padx=(0, 20))

        self.left_canvas = Canvas(
            left_outer,
            bg=settings.THEME['bg_card'],
            highlightthickness=0,
            bd=0,
            width=330,
        )
        left_scrollbar = ttk.Scrollbar(
            left_outer,
            orient='vertical',
            command=self.left_canvas.yview,
            style='Side.Vertical.TScrollbar',
        )
        left_panel = ttk.Frame(self.left_canvas, style='Card.TFrame', padding=14)
        self.left_scroll_window = self.left_canvas.create_window((0, 0), window=left_panel, anchor='nw')
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.pack(side='left', fill='both', expand=True)
        left_scrollbar.pack(side='right', fill='y')

        left_panel.bind(
            '<Configure>',
            lambda event: self.left_canvas.configure(scrollregion=self.left_canvas.bbox('all')),
        )
        self.left_canvas.bind(
            '<Configure>',
            lambda event: self.left_canvas.itemconfigure(self.left_scroll_window, width=event.width),
        )
        self.left_canvas.bind('<Enter>', self._bind_left_panel_mousewheel)
        self.left_canvas.bind('<Leave>', self._unbind_left_panel_mousewheel)

        ttk.Label(left_panel, text=self._t("section_params"), style='SubTitle.TLabel').pack(anchor='w',
                                                                                            pady=(0, 8))
        self._add_language_row(left_panel)
        ttk.Separator(left_panel).pack(fill='x', pady=8)
        self._add_input_row(left_panel, self._t("mw_label"), self.var_mw)
        self._add_input_row(left_panel, self._t("vol_label"), self.var_vol)
        self._add_input_row(left_panel, self._t("total_n_label"), self.var_total_n)
        self._add_input_row(left_panel, self._t("radius_label"), self.var_radius)
        ttk.Checkbutton(
            left_panel,
            text=self._t("local_roi_mode"),
            variable=self.var_local_roi_mode,
        ).pack(anchor='w', pady=(6, 2))

        self._add_input_row(left_panel, self._t("detection_mode_label"), self.var_detection_mode)

        ttk.Separator(left_panel).pack(fill='x', pady=12)

        ttk.Label(left_panel, text=self._t("section_strategy"), style='SubTitle.TLabel').pack(anchor='w', pady=(0, 8))
        mode_frame = ttk.Frame(left_panel, style='Card.TFrame')
        mode_frame.pack(fill='x')
        ttk.Radiobutton(mode_frame, text=self._t("standard_scan"), variable=self.var_method, value="original").pack(
            anchor='w')
        ttk.Radiobutton(mode_frame, text=self._t("advanced_ai"), variable=self.var_method, value="advanced").pack(
            anchor='w')

        ai_frame = ttk.Frame(left_panel, style='Card.TFrame')
        ai_frame.pack(fill='x', pady=8)
        chk = ttk.Checkbutton(ai_frame, text=self._t("ai_enable"), variable=self.var_ai_enable)
        chk.pack(anchor='w')
        Label(ai_frame, text=self._t("ai_hint"), bg=settings.THEME['bg_card'],
              fg=settings.THEME['text_secondary'], font=('微软雅黑', 8)).pack(anchor='w')

        ttk.Separator(left_panel).pack(fill='x', pady=12)

        drop_frame = Frame(left_panel, bg=settings.THEME['bg_drop'], bd=1, relief='solid', height=100)
        drop_frame.pack(fill='x', pady=8)
        drop_frame.pack_propagate(False)
        lbl_drop = Label(drop_frame, text=self._t("drop_ready"), bg=settings.THEME['bg_drop'],
                         fg=settings.THEME['text_secondary'], font=('微软雅黑', 11))
        lbl_drop.place(relx=0.5, rely=0.5, anchor='center')

        if DND_AVAILABLE:
            drop_frame.drop_target_register(DND_FILES)
            drop_frame.dnd_bind('<<Drop>>', self._on_drop)
        else:
            lbl_drop.config(text=self._t("drop_no_dnd"))

        btn_frame = ttk.Frame(left_panel, style='Card.TFrame')
        btn_frame.pack(fill='x', pady=12)
        ttk.Button(btn_frame, text=self._t("import_image"), command=self._open_file_dialog).pack(fill='x', pady=4)
        ttk.Button(btn_frame, text=self._t("batch_process"), command=self._open_batch_dialog).pack(fill='x', pady=4)

        self._bind_left_scroll_targets(left_panel)

        right_panel = ttk.Frame(main_container, style='TFrame')
        right_panel.pack(side='right', fill='both', expand=True)

        term_frame = ttk.Frame(right_panel, style='Panel.TFrame', padding=14)
        term_frame.pack(fill='both', expand=True, pady=(0, 10))
        ttk.Label(term_frame, text=self._t("system_log"), style='PanelSubTitle.TLabel').pack(anchor='w')
        self.log_area = st.ScrolledText(
            term_frame,
            bg=settings.THEME['bg_log'],
            fg='#263542',
            insertbackground=settings.THEME['accent_color'],
            font=('Consolas', 10),
            relief='solid',
            bd=1,
        )
        self.log_area.pack(fill='both', expand=True, pady=8)
        self.log_area.insert('end', f"{self._t('log_ready')}\n")

        self.status_var = StringVar(value=self._t("status_ready"))
        self.status_bar = Label(self, textvariable=self.status_var, bg='#D7E0E9',
                                fg=settings.THEME['text_muted'], anchor='e', padx=10)
        self.status_bar.pack(side='bottom', fill='x')

    def _add_language_row(self, parent):
        f = ttk.Frame(parent, style='Card.TFrame')
        f.pack(fill='x', pady=4)
        ttk.Label(f, text=self._t("language_label"), width=15, style='Card.TLabel').pack(side='left')
        selector = ttk.Combobox(
            f,
            textvariable=self.var_language,
            values=tuple(settings.LANGUAGE_OPTIONS.values()),
            state='readonly',
            width=12,
        )
        selector.pack(side='right')
        selector.bind('<<ComboboxSelected>>', self._on_language_change)

    def _add_input_row(self, parent, label_text, variable):
        f = ttk.Frame(parent, style='Card.TFrame')
        f.pack(fill='x', pady=4)
        ttk.Label(f, text=label_text, width=15, style='Card.TLabel').pack(side='left')
        ttk.Entry(f, textvariable=variable, width=12).pack(side='right')

    def _bind_left_scroll_targets(self, widget):
        widget.bind('<MouseWheel>', self._on_left_panel_mousewheel, add='+')
        for child in widget.winfo_children():
            self._bind_left_scroll_targets(child)

    def _bind_left_panel_mousewheel(self, _event):
        self.left_canvas.bind_all('<MouseWheel>', self._on_left_panel_mousewheel)

    def _unbind_left_panel_mousewheel(self, _event):
        self.left_canvas.unbind_all('<MouseWheel>')

    def _on_left_panel_mousewheel(self, event):
        if event.delta:
            self.left_canvas.yview_scroll(int(-event.delta / 120), 'units')
        return 'break'

    def log(self, msg: str):
        self.gui_queue.put(("log", msg))

    def _process_gui_queue(self):
        while not self.gui_queue.empty():
            task = self.gui_queue.get()
            t_type = task[0]
            if t_type == "log":
                self.log_area.insert('end', f"> {task[1]}\n")
                self.log_area.see('end')
            elif t_type == "show_result":
                self._show_result_window(task[1], task[2])
            elif t_type == "error":
                messagebox.showerror("分析错误", task[1])
            elif t_type == "status":
                self.status_var.set(task[1])
        self.after(100, self._process_gui_queue)

    def _on_drop(self, event):
        files = self.tk.splitlist(event.data)
        if files: self._start_analysis(files[0])

    def _open_file_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.tif;*.bmp")])
        if path: self._start_analysis(path)

    def _open_batch_dialog(self):
        if self.is_processing: return
        folder_path = filedialog.askdirectory()
        folder_path = folder_path.replace('\\', '/')
        if not folder_path: return
        self._start_thread(folder_path, is_batch=True)

    def _start_analysis(self, path):
        if self.is_processing: return
        self._start_thread(path, is_batch=False)

    def _start_thread(self, path, is_batch):
        self.is_processing = True
        self.log(f"🔎 载入目标: {path}")
        params = {
            "M": self.var_mw.get(), "V": self.var_vol.get(),
            "N": self.var_total_n.get(), "method": self.var_method.get(),
            "ai": self.var_ai_enable.get(), "radius_set": self.var_radius.get(),
            "local_roi_mode": self.var_local_roi_mode.get(),
            "detection_mode": self.var_detection_mode.get(),
        }
        settings.RADIUS = int(params['radius_set'])
        settings.DETECTION_MODE = str(params["detection_mode"]).strip().lower()
        target = self._batch_analysis_thread if is_batch else self._analysis_thread
        threading.Thread(target=target, args=(path, params), daemon=True).start()

    def _get_score(self, qc_data):
        """[修复] 安全获取置信度分数，兼容 Object 和 Dict"""
        if hasattr(qc_data, 'confidence_score'):
            return qc_data.confidence_score
        elif isinstance(qc_data, dict):
            return qc_data.get('confidence_score', 0)
        return 0

    def _batch_analysis_thread(self, folder_path, params):
        self.gui_queue.put(("status", "Batch Processing..."))
        try:
            valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
            self.analyzer.reset_batch_context()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = os.path.join(settings.RESULT_DIR, f"batch_{timestamp}")
            os.makedirs(batch_dir, exist_ok=True)
            csv_path = os.path.join(batch_dir, "batch_report.csv")
            self._ensure_candidate_export_files(batch_dir)
            self.log(f"📂 输出目录: {batch_dir}")

            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'FileName', 'SampleID', 'NominalConcentration', 'Channel', 'ReplicateID',
                    'Concentration', 'Display', 'MolarConcentration', 'Lambda',
                    'K_positive', 'N_total', 'RegionCount', 'DetectionRegionCount',
                    'CompleteChipCount', 'ChambersPerChip', 'DesignLayout',
                    'RegionConfidence', 'RegionReason', 'K_over_N', 'Score', 'RadiusCV', 'DistCV',
                    'DistCVApplicable', 'DistCVExpected', 'DistCVCILow', 'DistCVCIHigh',
                    'HighDensity', 'HighBackground', 'PSFPassRate',
                    'RawCountBeforePSF', 'PSFKeptCount', 'CompositeConfidence',
                    'ConfidenceFlags', 'PossiblyAdjusted', 'AdjustmentReason',
                    'BatchConsistencyScore', 'BatchWarnings',
                    'SpatialQCStatus', 'SpatialChi2', 'SpatialDF', 'SpatialChi2DF',
                    'SpatialPValue', 'SpatialPassPGe005', 'SpatialLowQuality',
                    'SpatialQCFlags', 'SpatialMaxAbsStdResidual',
                    'SpatialWorstQuadrat', 'SpatialEdgeCenterRatio',
                    'QCROISource', 'QCROIConfidence', 'QCROIBounds',
                    'GridSignalQuality', 'GridWeakPositiveFraction',
                    'HybridDetection', 'SpotFirstCount', 'GridAccepted',
                    'SpotGridAssociatedCount', 'SpotOffGridCount',
                    'HybridGridOnlyCount', 'SpotFirstDeltaThreshold',
                    'DetectionMode', 'TopHatKernel', 'WhitePixelRatio',
                    'BinaryComponents', 'ShapeSurvivors', 'LocalSNRSurvivors',
                    'SpacingSurvivors', 'ShapeRetentionRatio', 'AuditDir',
                    'LLMIntervened', 'Mode', 'Params', 'Image', 'ConvergenceJSON', 'Warning'
                    , 'AnalysisScope', 'LocalTotalChambers', 'QuantitativeUse'
                ])

                for idx, filename in enumerate(image_files):
                    filepath = os.path.join(folder_path, filename)
                    self.gui_queue.put(("status", f"Processing {idx + 1}/{len(image_files)}"))
                    self.log(f"[{idx + 1}] 分析: {filename} ...")

                    try:
                        result = self.analyzer.analyze_single_atomic(
                            filepath, M=params['M'], V=params['V'], N=params['N'],
                            method=params['method'], ai=params['ai'], save_result=True,
                            output_dir=batch_dir,
                            local_roi_mode=params.get('local_roi_mode', False),
                        )
                        self._handle_result(result, writer, filename, params, output_dir=batch_dir)
                    except Exception as e:
                        logger.error(f"Error {filename}: {e}")
                        writer.writerow([filename, "", "", "", "", "ERROR", "-", "", "",
                                         0, params.get('N', 0), "", "", "", "", 0, "", "", "", "", "",
                                         "", "", "", "", "", "", "", "", "", "", "", "",
                                         "", "", "", "", "", "", str(e)])
            self.log(f"🎉 批量完成！")
        except Exception as e:
            self.log(f"❌ 批量异常: {e}")
        finally:
            self.is_processing = False
            self.gui_queue.put(("status", "Ready"))

    def _analysis_thread(self, path, params):
        try:
            self.gui_queue.put(("status", "Analyzing..."))
            self.analyzer.reset_batch_context()
            result = self.analyzer.analyze_single_atomic(
                path, M=params['M'], V=params['V'], N=params['N'],
                method=params['method'], ai=params['ai'], save_result=True,
                local_roi_mode=params.get('local_roi_mode', False),
            )

            vis_path = result.get('visual_path')
            if not vis_path:
                vis_path = os.path.join(settings.RESULT_DIR, f"result_{os.path.basename(path)}")
                if os.path.exists(vis_path): result['visual_path'] = vis_path

            os.makedirs(settings.RESULT_DIR, exist_ok=True)
            self._handle_result(
                result,
                None,
                os.path.basename(path),
                params,
                show_window=True,
                output_dir=settings.RESULT_DIR,
            )
        except Exception as e:
            self.log(f"❌ 系统异常: {e}")
            logger.exception("Analysis Error")
        finally:
            self.is_processing = False
            self.gui_queue.put(("status", "Ready"))

    def _candidate_export_paths(self, output_dir):
        return {
            "candidate": os.path.join(output_dir, "candidate_table.csv"),
            "all_candidates": os.path.join(output_dir, "all_candidate_table.csv"),
            "counts": os.path.join(output_dir, "spatial_quadrat_counts.csv"),
            "residuals": os.path.join(output_dir, "spatial_quadrat_residuals.csv"),
        }

    def _candidate_export_fields(self):
        return {
            "candidate": [
                "FileName", "SampleID", "Channel", "CandidateID", "Positive",
                "X", "Y", "DetectedRadius", "Area", "Circularity",
                "InnerMean", "RingMean", "RingStd", "LocalDelta", "LocalZ",
                "SpatialXBin", "SpatialYBin", "SpatialBinExpected",
                "SpatialBinStdResidual", "SpatialInsideQCROI",
            ],
            "counts": [
                "FileName", "SampleID", "Channel", "XBin", "YBin", "Count", "Expected",
                "ChamberOpportunity", "OpportunityFraction",
            ],
            "residuals": [
                "FileName", "SampleID", "Channel", "XBin", "YBin",
                "StdResidual", "Observed", "Expected", "ChamberOpportunity",
            ],
            "all_candidates": [
                "FileName", "SampleID", "Channel", "CandidateID", "Stage", "Source",
                "X", "Y", "Reason", "Classification", "FormalPositive",
                "CandidatePositive", "GridDistancePx", "InnerMean", "RingMean",
                "RingStd", "LocalDelta", "LocalDeltaCutoff", "LocalZ",
                "LocalZCutoff", "PSFLike", "PSFCenterDelta", "SupportPixels",
                "SupportFraction", "MinSupportPixels", "MinSupportFraction",
            ],
        }

    def _ensure_candidate_export_files(self, output_dir):
        paths = self._candidate_export_paths(output_dir)
        fields = self._candidate_export_fields()
        for key, path in paths.items():
            if os.path.exists(path):
                continue
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields[key])
                writer.writeheader()

    def _candidate_measurement_record(self, gray, contour, radius):
        from spatial_qc import measure_candidate_signal

        return measure_candidate_signal(gray, contour, radius)

    def _append_candidate_exports(self, result, filename, sample_id, channel, output_dir):
        if not output_dir:
            return
        contours = result.get("contours") or []
        spatial_qc = result.get("spatial_qc") or {}
        paths = self._candidate_export_paths(output_dir)
        fields = self._candidate_export_fields()
        self._ensure_candidate_export_files(output_dir)

        for key in ("counts", "residuals"):
            rows = spatial_qc.get(
                "spatial_quadrat_counts" if key == "counts" else "spatial_quadrat_residuals",
                [],
            )
            if not rows:
                continue
            with open(paths[key], "a", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields[key])
                for row in rows:
                    if key == "counts":
                        writer.writerow({
                            "FileName": filename, "SampleID": sample_id, "Channel": channel,
                            "XBin": row.get("x_bin", ""), "YBin": row.get("y_bin", ""),
                            "Count": row.get("count", ""), "Expected": row.get("expected", ""),
                            "ChamberOpportunity": row.get("chamber_opportunity", ""),
                            "OpportunityFraction": row.get("opportunity_fraction", ""),
                        })
                    else:
                        writer.writerow({
                            "FileName": filename, "SampleID": sample_id, "Channel": channel,
                            "XBin": row.get("x_bin", ""), "YBin": row.get("y_bin", ""),
                            "StdResidual": row.get("std_residual", ""),
                            "Observed": row.get("observed", ""), "Expected": row.get("expected", ""),
                            "ChamberOpportunity": row.get("chamber_opportunity", ""),
                        })

        candidate_records = result.get("candidate_records") or []
        if candidate_records:
            with open(paths["all_candidates"], "a", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields["all_candidates"])
                for idx, rec in enumerate(candidate_records, start=1):
                    writer.writerow({
                        "FileName": filename,
                        "SampleID": sample_id,
                        "Channel": channel,
                        "CandidateID": idx,
                        "Stage": rec.get("stage", ""),
                        "Source": rec.get("source", ""),
                        "X": rec.get("x", ""),
                        "Y": rec.get("y", ""),
                        "Reason": rec.get("reason", ""),
                        "Classification": rec.get("classification", ""),
                        "FormalPositive": rec.get("formal_positive", ""),
                        "CandidatePositive": rec.get("candidate_positive", ""),
                        "GridDistancePx": rec.get("grid_distance_px", ""),
                        "InnerMean": rec.get("inner_mean", ""),
                        "RingMean": rec.get("ring_mean", ""),
                        "RingStd": rec.get("ring_std", ""),
                        "LocalDelta": rec.get("local_delta", ""),
                        "LocalDeltaCutoff": rec.get("local_delta_cutoff", ""),
                        "LocalZ": rec.get("local_z", ""),
                        "LocalZCutoff": rec.get("local_z_cutoff", ""),
                        "PSFLike": rec.get("psf_like", ""),
                        "PSFCenterDelta": rec.get("psf_center_delta", ""),
                        "SupportPixels": rec.get("support_pixels", ""),
                        "SupportFraction": rec.get("support_fraction", ""),
                        "MinSupportPixels": rec.get("min_support_pixels", ""),
                        "MinSupportFraction": rec.get("min_support_fraction", ""),
                    })

        if not contours:
            return
        from image_utils import load_image
        from spatial_qc import spatial_bin_for_point

        gray = load_image(result.get("file_path", ""))
        with open(paths["candidate"], "a", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields["candidate"])
            for idx, contour in enumerate(contours, start=1):
                rec = self._candidate_measurement_record(gray, contour, settings.RADIUS)
                bin_rec = spatial_bin_for_point(rec["x"], rec["y"], spatial_qc)
                writer.writerow({
                    "FileName": filename,
                    "SampleID": sample_id,
                    "Channel": channel,
                    "CandidateID": idx,
                    "Positive": 1,
                    "X": rec["x"],
                    "Y": rec["y"],
                    "DetectedRadius": rec["detected_radius"],
                    "Area": rec["area"],
                    "Circularity": rec["circularity"],
                    "InnerMean": rec["inner_mean"],
                    "RingMean": rec["ring_mean"],
                    "RingStd": rec["ring_std"],
                    "LocalDelta": rec["local_delta"],
                    "LocalZ": rec["local_z"],
                    "SpatialXBin": bin_rec.get("spatial_x_bin", ""),
                    "SpatialYBin": bin_rec.get("spatial_y_bin", ""),
                    "SpatialBinExpected": bin_rec.get("spatial_bin_expected", ""),
                    "SpatialBinStdResidual": bin_rec.get("spatial_bin_std_residual", ""),
                    "SpatialInsideQCROI": bin_rec.get("spatial_inside_qc_roi", ""),
                })

    def _handle_result(self, result, writer, filename, params, show_window=False, output_dir=None):
        if "error" in result:
            self.log(f"   ⚠️ 失败: {result['error']}")
            if writer:
                writer.writerow([filename, "", "", "", "", "ERROR", "-", "", "",
                                 0, params.get('N', 0), "", "", "", "", 0, "", "", "", "", "",
                                 "", "", "", "", "", "", "", "", "", "", "", "",
                                 "", "", "", "", "", "", result['error']])
            return

        count = result.get('actual_count', 0)
        effective_n = result.get('effective_N', params.get('N', ''))
        region_info = result.get('region_info', {}) or {}
        conc_info = result.get('concentration_info')
        if not conc_info:
            try:
                conc_info = self.analyzer.calculator.calculate_concentration(count, params['M'], params['V'],
                                                                             effective_n or params['N'])
                result['concentration_info'] = conc_info
            except:
                pass

        disp_str = conc_info.get('display_str', 'N/A') if conc_info else 'N/A'
        raw_conc = conc_info.get('concentration', 0) if conc_info else 0
        molar_conc = conc_info.get('molar_concentration', '') if conc_info else ''
        lambda_val = conc_info.get('lambda', '') if conc_info else ''
        ratio_val = conc_info.get('ratio', '') if conc_info else ''
        warning_msg = conc_info.get('warning_msg', '') if conc_info else ''

        # [关键修复] 使用 _get_score 安全获取分数
        qc_data = result.get('qc_metrics', {})
        score = self._get_score(qc_data)

        ai_p = result.get('used_ai_params', {})
        ai_str = f"G={ai_p.get('gamma', 0):.1f}/O={ai_p.get('c_offset', 0)}"
        vis_name = os.path.basename(result.get('visual_path', ''))
        metrics = result.get('metrics', {}) or {}
        region_count = region_info.get('region_count', metrics.get('region_count', ''))
        detection_region_count = region_info.get(
            'detection_region_count',
            metrics.get('detection_region_count', region_count),
        )
        complete_chip_count = region_info.get('complete_chip_count', metrics.get('complete_chip_count', ''))
        chambers_per_chip = region_info.get('chambers_per_chip', metrics.get('chambers_per_chip', ''))
        design_layout = region_info.get('design_layout_type', metrics.get('design_layout_type', ''))
        psf_pass = result.get('psf_pass_rate')
        psf_pass = "" if psf_pass is None else f"{psf_pass:.4f}"
        llm_intervened = bool(result.get('llm_intervened', False))
        mode = result.get('mode', '')
        channel = ""
        if "_42" in filename or "-42" in filename:
            channel = "Aβ42"
        elif "_40" in filename or "-40" in filename:
            channel = "Aβ40"
        sample_id = os.path.splitext(filename)[0]
        convergence_name = ""

        if writer:
            convergence = result.get('convergence', [])
            if convergence and output_dir:
                try:
                    safe_base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)
                    convergence_name = f"{safe_base}_convergence.json"
                    conv_path = os.path.join(output_dir, convergence_name)
                    with open(conv_path, "w", encoding="utf-8") as fh:
                        json.dump(convergence, fh, ensure_ascii=False, indent=2,
                                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
                except Exception as e:
                    warning_msg = f"{warning_msg}; convergence export failed: {e}".strip("; ")

            writer.writerow([
                filename, sample_id, "", channel, "",
                raw_conc, disp_str, molar_conc, lambda_val,
                count, effective_n,
                region_count,
                detection_region_count,
                complete_chip_count,
                chambers_per_chip,
                design_layout,
                region_info.get('region_inference_confidence', metrics.get('region_inference_confidence', '')),
                region_info.get('region_inference_reason', metrics.get('region_inference_reason', '')),
                ratio_val, f"{score:.1f}",
                metrics.get('radius_cv', ''), metrics.get('dist_cv', ''),
                metrics.get('dist_cv_applicable', ''), metrics.get('dist_cv_expected', ''),
                metrics.get('dist_cv_ci_low', ''), metrics.get('dist_cv_ci_high', ''),
                metrics.get('high_density', ''), metrics.get('high_background', ''), psf_pass,
                metrics.get('raw_count_before_psf', ''), metrics.get('psf_kept_count', ''),
                metrics.get('composite_confidence', ''),
                ";".join(metrics.get('confidence_flags', [])) if isinstance(metrics.get('confidence_flags'), list) else metrics.get('confidence_flags', ''),
                metrics.get('possibly_adjusted', ''), metrics.get('adjustment_reason', ''),
                metrics.get('batch_consistency_score', ''),
                ";".join(metrics.get('batch_warnings', [])) if isinstance(metrics.get('batch_warnings'), list) else metrics.get('batch_warnings', ''),
                metrics.get('spatial_qc_status', ''), metrics.get('spatial_chi2', ''),
                metrics.get('spatial_df', ''), metrics.get('spatial_chi2_df', ''),
                metrics.get('spatial_p_value', ''), metrics.get('spatial_pass_p_ge_0_05', ''),
                metrics.get('spatial_qc_low_quality', ''),
                ";".join(metrics.get('spatial_qc_flags', [])) if isinstance(metrics.get('spatial_qc_flags'), list) else metrics.get('spatial_qc_flags', ''),
                metrics.get('spatial_max_abs_std_residual', ''),
                metrics.get('spatial_worst_quadrat', ''),
                metrics.get('spatial_edge_center_ratio', ''),
                metrics.get('qc_roi_source', ''), metrics.get('qc_roi_confidence', ''),
                metrics.get('qc_roi_bounds', ''),
                metrics.get('grid_signal_quality', ''), metrics.get('grid_weak_positive_fraction', ''),
                metrics.get('hybrid_detection', ''), metrics.get('spot_first_count', ''),
                metrics.get('grid_accepted', ''), metrics.get('spot_grid_associated_count', ''),
                metrics.get('spot_off_grid_count', ''), metrics.get('hybrid_grid_only_count', ''),
                metrics.get('spot_first_delta_threshold', ''),
                metrics.get('detection_mode', ''), metrics.get('tophat_kernel_size', ''),
                metrics.get('white_pixel_ratio', ''), metrics.get('binary_components', ''),
                metrics.get('shape_survivors', ''), metrics.get('local_snr_survivors', ''),
                metrics.get('spacing_survivors', ''), metrics.get('shape_retention_ratio', ''),
                metrics.get('audit_dir', ''),
                llm_intervened, mode, ai_str, vis_name, convergence_name, warning_msg
                , metrics.get('analysis_scope', ''), metrics.get('local_total_chambers', ''),
                metrics.get('quantitative_use', '')
            ])
            try:
                self._append_candidate_exports(result, filename, sample_id, channel, output_dir)
            except Exception as e:
                logger.warning(f"Candidate export failed for {filename}: {e}")
        elif output_dir:
            try:
                self._append_candidate_exports(result, filename, sample_id, channel, output_dir)
            except Exception as e:
                logger.warning(f"Candidate export failed for {filename}: {e}")

        scope = metrics.get('analysis_scope', 'full_region')
        if scope == "local_roi":
            region_text = f" | ROI N: {effective_n}"
        else:
            region_text = f" | 区域: {region_count} | N: {effective_n}" if region_count else f" | N: {effective_n}"
        self.log(f"   ✅ 完成 | 浓度: {disp_str} | 计数: {count}{region_text}")

        if show_window and result.get('visual_path'):
            self.gui_queue.put(("show_result", result['visual_path'], result))

    def _show_result_window(self, img_path, data):
        top = Toplevel(self)
        top.title(self._t("result_title"))
        top.configure(bg=settings.THEME['bg_window'])

        img_frame = Frame(top, bg=settings.THEME['bg_panel'])
        img_frame.pack(fill='both', expand=True, padx=10, pady=10)

        try:
            if img_path: img_path = os.path.abspath(img_path)
            if img_path and os.path.exists(img_path):
                pil_img = Image.open(img_path)
                pil_img.thumbnail((900, 700))
                tk_img = ImageTk.PhotoImage(pil_img)
                lbl = Label(img_frame, image=tk_img, bg='#000000')
                lbl.image = tk_img
                lbl.pack(fill='both', expand=True)
            else:
                Label(img_frame, text=self._t("visual_missing"), fg=settings.THEME['warning_color'],
                      bg=settings.THEME['bg_panel']).pack(pady=50)
        except Exception as e:
            Label(img_frame, text=f"{self._t('load_failed')}: {e}", fg=settings.THEME['error_color'],
                  bg=settings.THEME['bg_panel']).pack()

        info_frame = Frame(top, bg=settings.THEME['bg_panel'], pady=15)
        info_frame.pack(fill='x', side='bottom')

        conc = data.get('concentration_info', {})
        region_info = data.get('region_info', {}) or {}
        region_count = region_info.get('region_count', '')
        detection_region_count = region_info.get('detection_region_count', region_count)
        complete_chip_count = region_info.get('complete_chip_count', '')
        effective_n = data.get('effective_N', '')
        # [关键修复] 使用 _get_score 安全获取分数
        qc_data = data.get('qc_metrics', {})
        score = self._get_score(qc_data)

        region_text = f" | {self._t('result_region')}: {region_count} | N: {effective_n}" if region_count else ""
        metrics = data.get('metrics', {}) or {}
        if metrics.get('analysis_scope') == "local_roi":
            region_text = f" | ROI N: {effective_n}"
        elif region_count:
            region_text = (
                f" | detection regions: {detection_region_count} | chips: {complete_chip_count} | N: {effective_n}"
            )
        text = (
            f"🧪 {self._t('result_concentration')}: {conc.get('display_str')} | "
            f"🔢 {self._t('result_count')}: {data.get('actual_count')}{region_text} | "
            f"📈 {self._t('result_confidence')}: {score:.1f}\n"
            f"⚙️ {self._t('result_params')}: {data.get('used_ai_params')}"
        )
        Label(info_frame, text=text, bg=settings.THEME['bg_panel'], fg=settings.THEME['text_dark'],
              font=('微软雅黑', 12), wraplength=1000).pack()


if __name__ == "__main__":
    try:
        app = ModernAutoCVApp()
        app.mainloop()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
