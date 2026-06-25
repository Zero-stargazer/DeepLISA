#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Optional
from agent_batch_context import BatchContext
from agent_quality import calculate_composite_confidence
from config import settings
from concentration_calculator import ConcentrationCalculator
from image_utils import load_image
from region_inference import infer_region_layout_from_image
from spatial_qc import compute_spatial_qc, flatten_spatial_qc_for_report
# [修改] 导入新的 Agent 类
from agent_core import DeepLisaAgent


class SampleAnalyzer:
    def __init__(self, loading_efficiency: float = 0.8):
        self.calculator = ConcentrationCalculator(loading_efficiency)
        # [修改] 初始化 Agent
        self.agent = DeepLisaAgent()
        self.batch_context = BatchContext()

    @staticmethod
    def apply_formal_qc_use_flags(metrics: Dict) -> Dict:
        """Mark outputs that should not be used as formal quantitative/QC evidence."""
        flags = []
        if metrics.get("grid_unreliable"):
            flags.append("grid_unreliable")
        try:
            weak_fraction = float(metrics.get("grid_weak_positive_fraction") or 0.0)
        except (TypeError, ValueError):
            weak_fraction = 0.0
        if weak_fraction > 0.60:
            flags.append("high_weak_positive_fraction")
        if metrics.get("psf_formal_qc_status") == "fail":
            flags.append("psf_formal_qc_failed")

        metrics["spatial_qc_use"] = (
            "not_recommended"
            if bool(metrics.get("spatial_qc_low_quality", False))
            else "standard"
        )
        if flags:
            metrics["quantitative_use"] = "not_recommended"
        else:
            metrics.setdefault("quantitative_use", "standard")

        existing = str(metrics.get("formal_qc_flags", "") or "")
        merged = [part for part in existing.split(";") if part]
        for flag in flags:
            if flag not in merged:
                merged.append(flag)
        metrics["formal_qc_flags"] = ";".join(merged)
        return metrics

    def reset_batch_context(self):
        self.batch_context = BatchContext()

    def apply_batch_context(self, result_data: Dict) -> Dict:
        metrics = result_data.get("metrics", {}) or {}
        batch_assessment = self.batch_context.observe(metrics)
        metrics.update(batch_assessment)
        composite = calculate_composite_confidence(metrics)
        metrics.update(composite)
        qc_metrics = result_data.get("qc_metrics")
        if qc_metrics is not None and hasattr(qc_metrics, "confidence_score"):
            qc_metrics.confidence_score = metrics["composite_confidence"]
        result_data["metrics"] = metrics
        return result_data

    def analyze_single_atomic(
            self,
            path: str,
            M: float,
            V: float,
            N: int,
            method: str = "advanced",
            ai: bool = True,
            save_result: bool = True,
            output_dir: str = "",
            local_roi_mode: bool = False,
    ) -> Dict:

        # 参数校验
        M = float(M) if M and M > 0 else settings.DEFAULT_M
        V = float(V) if V and V > 0 else settings.DEFAULT_V
        base_region_n = int(N) if N and N > 0 else settings.DEFAULT_N
        if local_roi_mode:
            region_info = {
                "auto_region_inference": False,
                "local_roi_mode": True,
                "region_count": 0,
                "detection_region_count": 0,
                "complete_chip_count": 0,
                "chip_count": 0,
                "base_region_chambers": base_region_n,
                "effective_total_chambers": base_region_n,
                "region_inference_reason": "local_roi_pending_grid_positions",
                "region_inference_confidence": 0.0,
                "analysis_scope_inferred": "local_roi",
            }
        elif settings.AUTO_REGION_INFERENCE:
            region_info = infer_region_layout_from_image(path, base_region_n)
        else:
            region_info = {
                "auto_region_inference": False,
                "region_count": 1,
                "base_region_chambers": base_region_n,
                "effective_total_chambers": base_region_n,
                "region_inference_reason": "disabled",
                "region_inference_confidence": 1.0,
            }
        N = int(region_info["effective_total_chambers"])

        # 1. 调用 Agent 进行分析
        if ai:
            # 这里的 run_analysis 必须在 agent_core.py 中存在
            result_data = self.agent.run_analysis(
                path,
                N,
                output_dir=output_dir if save_result else "",
                local_roi_mode=local_roi_mode,
            )
        else:
            # 如果不启用 AI，可以使用 Agent 的工具执行一次默认参数
            default_params = {"gamma": 1.0, "c_offset": -2, "block_size": 15, "min_circularity": 0.5}
            raw_res = self.agent.tools.execute_detection(
                path,
                default_params['gamma'],
                default_params['c_offset'],
                default_params['block_size'],
                default_params['min_circularity'],
                N
            )
            # 手动包装一下格式以匹配
            result_data = {
                "actual_count": raw_res['metrics']['count'],
                "qc_metrics": raw_res['internal_data']['qc_metrics'],
                "used_ai_params": default_params,
                "metrics": raw_res.get("metrics", {}),
                "contours": raw_res['internal_data']['contours'],
                "candidate_records": raw_res.get("internal_data", {}).get("candidate_records", []),
                "psf_pass_rate": None,
                "llm_intervened": False,
                "mode": "heuristic_single_pass",
                "convergence": [],
                "visual_path": ""  # 后续处理
            }
            # 如果需要保存图片
            if save_result:
                from image_utils import create_visualization_base
                vis_save_path = ""
                if output_dir:
                    vis_save_path = os.path.join(output_dir, f"result_{os.path.basename(path)}")
                vis_path = create_visualization_base(
                    path,
                    result_data['contours'],
                    result_data['actual_count'],
                    save_path=vis_save_path,
                )
                result_data['visual_path'] = vis_path

        if "error" in result_data:
            return result_data

        count = result_data.get("actual_count", 0)
        metrics = result_data.get("metrics", {}) or {}
        metrics.update({
            "region_count": region_info.get("region_count", 1),
            "detection_region_count": region_info.get("detection_region_count", region_info.get("region_count", 1)),
            "complete_chip_count": region_info.get("complete_chip_count", 0),
            "chip_count": region_info.get("chip_count", region_info.get("complete_chip_count", 0)),
            "base_region_chambers": region_info.get("base_region_chambers", base_region_n),
            "effective_total_chambers": N,
            "chambers_per_region": region_info.get("chambers_per_region", base_region_n),
            "chambers_per_chip": region_info.get("chambers_per_chip", base_region_n * 2),
            "regions_per_chip": region_info.get("regions_per_chip", 2),
            "chips_per_mask": region_info.get("chips_per_mask", 2),
            "regions_per_mask": region_info.get("regions_per_mask", 4),
            "region_inference_reason": region_info.get("region_inference_reason", ""),
            "region_inference_confidence": region_info.get("region_inference_confidence", 0),
            "analysis_scope_inferred": region_info.get("analysis_scope_inferred", ""),
            "design_prior_used": region_info.get("design_prior_used", False),
            "design_layout_type": region_info.get("design_layout_type", ""),
            "design_effective_x_columns": region_info.get("design_effective_x_columns", ""),
            "design_effective_y_row_pairs": region_info.get("design_effective_y_row_pairs", ""),
            "design_unique_y_coordinates": region_info.get("design_unique_y_coordinates", ""),
            "do_not_interpret_regions_as_independent_chips": region_info.get(
                "do_not_interpret_regions_as_independent_chips",
                True,
            ),
        })

        if local_roi_mode:
            grid_positions = int(metrics.get("grid_positions", 0) or 0)
            boundary_rejected = int(metrics.get("grid_rejected_boundary", 0) or 0)
            local_total = max(1, grid_positions - boundary_rejected) if grid_positions else base_region_n
            N = local_total
            region_info.update({
                "local_roi_mode": True,
                "region_count": 0,
                "effective_total_chambers": local_total,
                "local_total_chambers": local_total,
                "region_inference_reason": (
                    "local_roi_grid_positions"
                    if grid_positions
                    else "local_roi_fallback_to_input_n"
                ),
                "region_inference_confidence": metrics.get("grid_auto_confidence", 0.0),
            })
            metrics.update({
                "analysis_scope": "local_roi",
                "local_roi_mode": True,
                "local_total_chambers": local_total,
                "effective_total_chambers": local_total,
                "region_count": 0,
                "detection_region_count": 0,
                "complete_chip_count": 0,
                "chip_count": 0,
                "region_inference_reason": region_info["region_inference_reason"],
                "region_inference_confidence": region_info["region_inference_confidence"],
                "quantitative_use": "not_recommended",
            })
        else:
            metrics.update({
                "analysis_scope": region_info.get("analysis_scope_inferred", "full_region"),
                "local_roi_mode": False,
                "quantitative_use": "standard",
            })

        metrics["analysis_radius_px"] = settings.RADIUS
        try:
            raw_gray = load_image(path)
            spatial_qc = compute_spatial_qc(
                result_data.get("contours") or [],
                raw_gray.shape,
                metrics=metrics,
                image=raw_gray,
            )
            metrics.update(flatten_spatial_qc_for_report(spatial_qc))
            result_data["spatial_qc"] = spatial_qc
        except Exception as exc:
            metrics.update({
                "spatial_qc_status": "error",
                "spatial_qc_low_quality": True,
                "spatial_qc_flags": f"spatial_qc_error:{exc}",
            })
            result_data["spatial_qc"] = {}

        metrics = self.apply_formal_qc_use_flags(metrics)
        result_data["metrics"] = metrics
        result_data = self.apply_batch_context(result_data)
        metrics = result_data.get("metrics", {}) or {}

        # 2. 计算浓度
        conc_info = self.calculator.calculate_concentration(count, M, V, N)
        if local_roi_mode:
            original_display = conc_info.get("display_str", "")
            roi_warning = (
                "[ROI] 局部截图模式：N 使用截图内可分类腔室数，"
                "浓度仅供参考，不建议用于正式定量。"
            )
            conc_info["roi_reference_display"] = original_display
            conc_info["display_str"] = "ROI mode: count only"
            conc_info["quantitative_use"] = "not_recommended"
            conc_info["warning_msg"] = (
                f"{roi_warning} {conc_info.get('warning_msg', '')}".strip()
            )
        else:
            conc_info["quantitative_use"] = metrics.get("quantitative_use", "standard")
            if conc_info["quantitative_use"] != "standard":
                conc_info["warning_msg"] = (
                    f"Formal QC not recommended: {metrics.get('formal_qc_flags', '')}"
                )

        conc_info.update({
            "calculation_N": int(N),
            "base_region_chambers": int(region_info.get("base_region_chambers", base_region_n) or base_region_n),
            "region_count": int(region_info.get("region_count", 0) or 0),
            "detection_region_count": int(region_info.get("detection_region_count", region_info.get("region_count", 0)) or 0),
            "complete_chip_count": int(region_info.get("complete_chip_count", 0) or 0),
            "chambers_per_chip": int(region_info.get("chambers_per_chip", base_region_n * 2) or base_region_n * 2),
            "region_inference_reason": region_info.get("region_inference_reason", ""),
            "region_inference_confidence": region_info.get("region_inference_confidence", 0),
            "analysis_scope": metrics.get("analysis_scope", "full_region"),
        })

        # 3. 组装结果
        final_result = {
            "file_path": path,
            "visual_path": result_data.get("visual_path"),
            "actual_count": count,
            "effective_N": N,
            "region_info": region_info,
            "concentration_info": conc_info,
            "qc_metrics": result_data.get("qc_metrics"),
            "used_ai_params": result_data.get("used_ai_params"),
            "metrics": metrics,
            "psf_pass_rate": result_data.get("psf_pass_rate"),
            "candidate_records": result_data.get("candidate_records", []),
            "llm_intervened": result_data.get("llm_intervened", False),
            "mode": result_data.get("mode"),
            "convergence": result_data.get("convergence", []),
            "contours": result_data.get("contours"),
            "spatial_qc": result_data.get("spatial_qc", {}),
        }

        return final_result
