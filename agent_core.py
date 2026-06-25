import json
import logging
import os
import sys
from typing import Dict
import cv2
from config import settings
from agent_quality import build_image_profile, calculate_composite_confidence
from ai_enhancement import AIParameterOptimizer
from agent_tools import DeepLisaAgentTools
from image_utils import create_visualization_base
from processing_pipeline import get_chamber_tier_params
from psf_verification import PSFVerifier

logger = logging.getLogger(__name__)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class DeepLisaAgent:
    def __init__(self, mode: str = "llm_enhanced"):
        """
        mode: "heuristic" — 纯规则引擎
              "llm_enhanced" — 规则引擎 + LLM 在 step≥3 且 score<70 时介入
        """
        self.llm = AIParameterOptimizer()
        self.tools = DeepLisaAgentTools()
        self.max_steps = 5
        self.mode = mode

    @staticmethod
    def apply_final_quality_gates(
            result: Dict,
            raw_gray,
            radius: int,
            target_n: int = None,
            psf_policy: str = "filter",
    ) -> Dict:
        """Apply final PSF filtering, image profiling, and composite confidence."""
        metrics = result.get("metrics", {}) or {}
        internal = result.get("internal_data", {}) or {}
        contours = internal.get("contours", []) or []
        psf_policy = str(psf_policy or "filter").strip().lower()
        if psf_policy not in {"filter", "advisory"}:
            psf_policy = "filter"
        metrics["psf_filter_policy"] = psf_policy

        raw_count = len(contours)
        kept_contours, psf_stats = PSFVerifier.filter_contours_by_psf(raw_gray, contours, radius)
        if contours:
            final_contours = contours if psf_policy == "advisory" else kept_contours
            internal["contours"] = final_contours
            metrics["raw_count_before_psf"] = raw_count
            metrics["count"] = len(final_contours)
            denominator = int(target_n or metrics.get("effective_total_chambers") or metrics.get("grid_positions") or 1)
            metrics["ratio"] = round(len(final_contours) / max(denominator, 1), 6)
            metrics["spacing_survivors"] = len(final_contours)
        metrics.update(psf_stats)
        if psf_stats.get("psf_pass_rate") is not None:
            metrics["psf_pass_rate"] = psf_stats["psf_pass_rate"]
            if psf_stats["psf_pass_rate"] < 0.25:
                metrics["psf_formal_qc_status"] = "fail"
                metrics["quantitative_use"] = "not_recommended"
            else:
                metrics["psf_formal_qc_status"] = "pass"

        profile = build_image_profile(raw_gray, local_roi_mode=(psf_policy == "advisory"))
        metrics.update(profile)

        composite = calculate_composite_confidence(metrics)
        metrics.update(composite)
        qc_metrics = internal.get("qc_metrics")
        if qc_metrics is not None and hasattr(qc_metrics, "confidence_score"):
            qc_metrics.confidence_score = metrics["composite_confidence"]

        result["metrics"] = metrics
        result["internal_data"] = internal
        result["psf_pass_rate"] = metrics.get("psf_pass_rate")
        return result

    @staticmethod
    def choose_psf_policy(local_roi_mode: bool, metrics: Dict = None) -> str:
        """Choose whether PSF is a formal filter or advisory audit."""
        metrics = metrics or {}
        if local_roi_mode:
            return "advisory"
        if metrics.get("grid_mode") or metrics.get("hybrid_detection"):
            return "filter"
        return "filter"

    @staticmethod
    def should_prefer_grid_recall_candidate(
            previous_metrics: Dict,
            current_metrics: Dict,
            target_N: int,
            local_roi_mode: bool = False,
    ) -> bool:
        """Prefer a higher-recall grid candidate when quality remains comparable."""
        if local_roi_mode:
            return False
        previous_metrics = previous_metrics or {}
        current_metrics = current_metrics or {}
        if not current_metrics.get("grid_mode"):
            return False

        prev_count = int(previous_metrics.get("count") or 0)
        curr_count = int(current_metrics.get("count") or 0)
        if prev_count <= 0 or curr_count <= max(prev_count + 50, int(prev_count * 1.12)):
            return False
        if curr_count >= 20000:
            return False
        if current_metrics.get("grid_unreliable"):
            return False
        grid_inlier = float(current_metrics.get("grid_auto_inlier_fraction") or 0.0)
        if grid_inlier and grid_inlier < 0.20:
            return False

        grid_positions = int(
            current_metrics.get("grid_positions")
            or previous_metrics.get("grid_positions")
            or target_N
            or 1
        )
        acceptance_ratio = prev_count / max(grid_positions, 1)
        weak_fraction = max(
            float(previous_metrics.get("grid_weak_positive_fraction") or 0.0),
            float(current_metrics.get("grid_weak_positive_fraction") or 0.0),
        )
        if weak_fraction > 0.55:
            return False
        if acceptance_ratio >= 0.12 or weak_fraction <= 0.25:
            return False

        prev_quality = float(previous_metrics.get("grid_signal_quality") or 0.0)
        curr_quality = float(current_metrics.get("grid_signal_quality") or 0.0)
        prev_score = float(previous_metrics.get("score") or 0.0)
        curr_score = float(current_metrics.get("score") or 0.0)
        return curr_quality >= prev_quality - 0.08 and curr_score >= prev_score - 3.0

    def run_analysis(self, img_path: str, target_N: int, output_dir: str = "", local_roi_mode: bool = False) -> Dict:
        print(f"🤖 [Agent] 启动深度研判: {os.path.basename(img_path)} (mode={self.mode})")

        obs = self.tools.inspect_image(img_path)
        if obs.get('status') == 'error':
            return {"error": f"Image load failed: {obs.get('message')}"}

        history = []
        best_result = None
        best_score = -1.0
        llm_intervened = False
        convergence_data = []
        audit_dir = ""
        if output_dir and settings.EXPORT_DETECTION_AUDIT:
            audit_dir = os.path.join(output_dir, "audit")

        # --- 1. 智能参数初始化 (Memory Guided + Chamber Adaptive) ---
        # 根据腔室尺寸计算自适应 block_size
        chamber_type = settings.CHAMBER_TYPE
        min_r, max_r, adaptive_blk = get_chamber_tier_params(chamber_type)
        memory_suggest = self.llm.memory_bank.suggest_initial_guess(
            chamber_type=chamber_type,
            detection_mode=settings.DETECTION_MODE,
        )

        if memory_suggest:
            print(f"   🧠 [Memory] 唤醒历史经验: {memory_suggest}")
            current_params = memory_suggest
            # 用自适应 block_size 覆盖 memory 中的旧值
            current_params['block_size'] = adaptive_blk
        else:
            current_params = {
                "gamma": 1.0, "c_offset": -5, "block_size": adaptive_blk, "min_circularity": 0.5
            }
        print(f"   📐 [Chamber] type={chamber_type}, r={min_r}-{max_r}px → block_size={adaptive_blk}")

        # 安全钳制
        bounds = settings.AI_TUNING_BOUNDS
        if current_params['c_offset'] > -4:
            current_params['c_offset'] = -5

        for step in range(self.max_steps):
            res = self.tools.execute_detection(
                img_path,
                current_params['gamma'],
                current_params['c_offset'],
                current_params['block_size'],
                current_params['min_circularity'],
                target_N,
                audit_dir=audit_dir
            )

            if res.get('status') == 'error': break

            metrics = res['metrics']
            curr_score = metrics['score']
            curr_count = metrics['count']
            radius_cv = metrics.get('radius_cv', 0)
            dist_cv = metrics.get('dist_cv', 0)
            high_density = metrics.get('high_density', False)
            msg = metrics.get('msg', '')

            log_str = f"   🔧 Round {step + 1}: G={current_params['gamma']:.1f}, Off={current_params['c_offset']}, Blk={current_params['block_size']} → Count={curr_count}, Score={curr_score}, R_cv={radius_cv:.3f}"
            if high_density: log_str += " [HD]"
            if msg: log_str += f" [{msg}]"
            print(log_str)

            # 收敛数据记录
            convergence_data.append({
                "step": step + 1,
                "count": curr_count,
                "score": curr_score,
                "radius_cv": radius_cv,
                "dist_cv": dist_cv,
                "high_density": high_density,
                "binary_components": metrics.get("binary_components"),
                "shape_survivors": metrics.get("shape_survivors"),
                "local_snr_survivors": metrics.get("local_snr_survivors"),
                "spacing_survivors": metrics.get("spacing_survivors"),
                "shape_retention_ratio": metrics.get("shape_retention_ratio"),
                "grid_signal_quality": metrics.get("grid_signal_quality"),
                "grid_weak_positive_fraction": metrics.get("grid_weak_positive_fraction"),
                "grid_delta_threshold": metrics.get("grid_delta_threshold"),
                "grid_z_threshold": metrics.get("grid_z_threshold"),
                "grid_auto_confidence": metrics.get("grid_auto_confidence"),
                "grid_auto_inlier_fraction": metrics.get("grid_auto_inlier_fraction"),
                "grid_unreliable": metrics.get("grid_unreliable"),
                "params": current_params.copy()
            })

            # --- 决策逻辑 v2.0 ---
            is_better = False
            if best_result is None:
                if curr_count < 20000:
                    is_better = True
            else:
                prev_score = best_score
                if curr_score > prev_score:
                    prev_count = best_result['metrics']['count']
                    if curr_count < 20 and prev_count > 100 and prev_count < 20000:
                        print("      ⚠️ [警惕] 分数提升但数量骤降，疑似过拟合，保留旧结果。")
                    elif curr_count < prev_count * 0.85 and curr_score <= prev_score + 5:
                        print("      ⚠️ [警惕] 分数小幅变化但召回明显下降，保留旧结果。")
                    else:
                        is_better = True
                elif abs(curr_score - prev_score) < 5:
                    prev_rcv = best_result['metrics'].get('radius_cv', 9.9)
                    prev_count = best_result['metrics']['count']
                    if metrics.get("grid_mode"):
                        prev_quality = best_result['metrics'].get("grid_signal_quality", 0.0)
                        curr_quality = metrics.get("grid_signal_quality", 0.0)
                        roi_count_gain = curr_count > max(prev_count + 2, prev_count * 1.10)
                        if (
                                local_roi_mode
                                and roi_count_gain
                                and curr_count < 20000
                                and curr_quality >= prev_quality - 0.03
                        ):
                            is_better = True
                            print("      ROI mode: keeping higher-count candidate with similar grid quality")
                        elif self.should_prefer_grid_recall_candidate(
                                best_result['metrics'],
                                metrics,
                                target_N,
                                local_roi_mode=local_roi_mode,
                        ):
                            is_better = True
                            print("      Grid recall: keeping higher-count candidate with similar grid quality")
                        elif curr_quality > prev_quality + 0.03:
                            is_better = True
                            print("      ✅ [优化] 网格信号质量改善")
                        elif (
                                not local_roi_mode
                                and not metrics.get("hybrid_detection")
                                and curr_count < prev_count * 0.95
                                and curr_quality >= prev_quality - 0.05
                        ):
                            is_better = True
                            print("      ✅ [优化] 网格模式选择更保守计数")
                    elif curr_count > prev_count * 1.05 and radius_cv <= prev_rcv + 0.05:
                        is_better = True
                        print("      ✅ [优化] 分数稳定且召回改善")
                    elif curr_count >= prev_count * 0.9 and radius_cv < prev_rcv:
                        is_better = True
                        print("      ✅ [优化] radius_cv 改善")

            if is_better:
                best_score = curr_score
                best_result = res
                best_result['used_ai_params'] = current_params.copy()

            history.append({"params": current_params.copy(), "metrics": metrics})

            # 阈值收紧后若分数不升反而明显丢失候选，立即停止并保留上一轮。
            if len(history) >= 2:
                previous_metrics = history[-2]["metrics"]
                previous_params = history[-2]["params"]
                previous_count = previous_metrics.get("count", 0)
                previous_score = previous_metrics.get("score", 0)
                if (
                        20 <= previous_count < 20000
                        and curr_count < previous_count * 0.85
                        and curr_score <= previous_score
                ):
                    print("   🛑 召回下降且评分未改善，停止继续收紧。")
                    break
                if (
                        20 <= previous_count < 20000
                        and current_params.get("c_offset", -2) > previous_params.get("c_offset", -2)
                ):
                    grid_positions = max(metrics.get("grid_positions", target_N), 1)
                    acceptance_ratio = curr_count / grid_positions
                    continue_grid_recall = (
                        metrics.get("grid_mode")
                        and acceptance_ratio < 0.08
                        and metrics.get("grid_weak_positive_fraction", 0.0) <= 0.35
                        and metrics.get("grid_signal_quality", 0.0) >= 0.45
                        and not metrics.get("grid_unreliable", False)
                        and (
                            not metrics.get("grid_auto_inlier_fraction")
                            or metrics.get("grid_auto_inlier_fraction", 0.0) >= 0.20
                        )
                        and current_params.get("c_offset", -2) < bounds.get("c_offset_max", 6)
                    )
                    if continue_grid_recall:
                        print("   🔎 网格召回仍偏低，继续下一档召回探测。")
                    else:
                        print("   🛑 已完成一档召回探测，保留最佳结果并停止扩张。")
                        break

            # --- 停止条件 ---
            if best_score >= 90 and step >= 2:
                if best_result['metrics']['count'] > 50:
                    print("   ✅ 结果优异且稳定，停止优化。")
                    break
                elif best_result['metrics']['count'] <= 50 and step >= 4:
                    print("   ✅ 确认低浓度样本。")
                    break

            # --- 双层决策：LLM 增强 (v15.0 激活) ---
            if self.mode == "llm_enhanced" and step >= 3 and best_score < 70 and self.llm.is_available():
                print("   🤖 [LLM介入] 常规搜索未见效 (step≥3, score<70)，调用GLM-4战略分析...")
                try:
                    llm_suggest = self.llm.suggest_next_params(step + 1, history, current_params, target_N)
                    if llm_suggest:
                        # 保留自适应 block_size
                        llm_suggest['block_size'] = adaptive_blk
                        current_params = llm_suggest
                        llm_intervened = True
                        print(f"   🤖 [GLM-4] 提出新参数: {current_params}")
                        continue
                except Exception as e:
                    print(f"   ⚠️ [LLM] API 调用失败: {e}，fallback 到启发式规则")

            mutated, strategy = self.tools.suggest_heuristic_mutation(current_params, metrics, target_N)
            if mutated == current_params:
                print("   🛑 参数已稳定，停止重复搜索。")
                break
            current_params = mutated
            print(f"      👉 Next: {strategy}")

        if not best_result:
            best_result = res
            best_result['used_ai_params'] = current_params

        # --- PSF / image profile / composite confidence final gates ---
        try:
            raw_gray = __import__('image_utils', fromlist=['load_image']).load_image(img_path)
            best_result.setdefault("metrics", {})["local_roi_mode"] = bool(local_roi_mode)
            psf_policy = self.choose_psf_policy(local_roi_mode, best_result.get("metrics", {}))
            best_result = self.apply_final_quality_gates(
                best_result,
                raw_gray,
                settings.RADIUS,
                target_N,
                psf_policy=psf_policy,
            )
            psf_pass_rate = best_result.get("psf_pass_rate")
            raw_count = best_result.get("metrics", {}).get("raw_count_before_psf", best_result['metrics']['count'])
            final_count = best_result['metrics']['count']
            if psf_pass_rate is not None and psf_policy == "advisory":
                kept_count = best_result.get("metrics", {}).get("psf_kept_count", final_count)
                print(f"   [PSF] ROI advisory: {kept_count}/{raw_count} would pass ({psf_pass_rate:.1%})")
            if psf_pass_rate is not None and psf_policy != "advisory":
                print(f"   🔬 [PSF] 物理过滤: {final_count}/{raw_count} 保留 ({psf_pass_rate:.1%})")
            if best_result.get("metrics", {}).get("possibly_adjusted"):
                print(f"   ⚠️ [Image] 疑似色阶/强增强图: {best_result['metrics'].get('adjustment_reason', '')}")
        except Exception as e:
            print(f"   ⚠️ [Quality] 终局质量门控跳过: {e}")

        # 保存记忆
        best_metrics = best_result.get('metrics', {})
        if best_metrics.get("grid_mode"):
            print("   🧠 [Memory] 网格模式跳过参数记忆：阳性判定由自动网格与局部信号分布驱动。")
        elif (
                best_metrics.get("composite_confidence", best_score) > 80
                and best_result['used_ai_params']['c_offset'] > bounds.get('c_offset_min', -18)
        ):
            self.llm.memory_bank.save_success(
                best_result['used_ai_params'],
                best_metrics.get("composite_confidence", best_score),
                best_metrics
            )

        # 生成可视化
        metric_info = best_result.get('metrics', {})
        grid_info = ""
        if metric_info.get("grid_mode"):
            grid_info = (
                f"\nGrid: {metric_info.get('grid_layout', '')}, "
                f"pitch={metric_info.get('grid_pitch_px', 0):.2f}px, "
                f"angle={metric_info.get('grid_angle_deg', 0):.2f}deg, "
                f"auto_conf={metric_info.get('grid_auto_confidence', 0):.2f}, "
                f"inlier={metric_info.get('grid_auto_inlier_fraction', 0):.2f}, "
                f"signal_q={metric_info.get('grid_signal_quality', 0):.2f}, "
                f"weak_frac={metric_info.get('grid_weak_positive_fraction', 0):.2f}"
            )
        psf_label = "PSF(advisory)" if metric_info.get("psf_filter_policy") == "advisory" else "PSF"
        info = (
            f"Score: {metric_info.get('composite_confidence', best_score):.1f}\n"
            f"Params: {best_result['used_ai_params']}\n"
            f"R_cv: {metric_info.get('radius_cv', 'N/A')}, "
            f"{psf_label}={metric_info.get('psf_kept_count', metric_info.get('count', 0))}/"
            f"{metric_info.get('raw_count_before_psf', metric_info.get('count', 0))}"
            f"{grid_info}"
        )
        vis_save_path = ""
        if output_dir:
            vis_save_path = os.path.join(output_dir, f"result_{os.path.basename(img_path)}")
        vis_path = create_visualization_base(
            img_path,
            best_result['internal_data']['contours'],
            best_result['metrics']['count'],
            info,
            save_path=vis_save_path
        )

        return {
            "visual_path": vis_path,
            "actual_count": best_result['metrics']['count'],
            "metrics": best_result.get('metrics', {}),
            "qc_metrics": best_result['internal_data']['qc_metrics'],
            "used_ai_params": best_result['used_ai_params'],
            "contours": best_result['internal_data']['contours'],
            "candidate_records": best_result.get("internal_data", {}).get("candidate_records", []),
            "psf_pass_rate": best_result.get('psf_pass_rate'),
            "llm_intervened": llm_intervened,
            "mode": self.mode,
            "convergence": convergence_data
        }
