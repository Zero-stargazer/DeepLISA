#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLISA 浓度计算模块 (v5.1 - 饱和度保护版)
功能：基于泊松分布将阳性计数转换为浓度。
修复：增加了 count >= N 时的数学保护，防止 log 负数报错。
"""

import math
from typing import Dict
from config import settings


class ConcentrationCalculator:
    def __init__(self, loading_efficiency: float = 0.8):
        self.loading_efficiency = loading_efficiency
        self.avogadro = settings.AVOGADRO_NUMBER

    def calculate_concentration(
            self,
            on_count: int,
            M: float,
            V: float,
            N: int
    ) -> Dict:
        """
        计算浓度
        :param on_count: 阳性数
        :param M: 分子量 (kDa)
        :param V: 腔室体积 (pL)
        :param N: 总腔室数
        """
        # 1. 基础参数校验
        if N <= 0: N = settings.DEFAULT_N
        if V <= 0: V = settings.DEFAULT_V
        if M <= 0: M = settings.DEFAULT_M

        # 2. 饱和度检查与截断
        # dELISA 原理决定了当阳性率 >= 99% 时，泊松分布精度开始下降
        # >= 99.9% 时数学上截断，99%-99.8% 区间标记为近饱和
        ratio = on_count / N
        is_saturated = False
        is_near_saturated = False
        is_zero_count = (on_count == 0)

        if ratio >= 0.999:
            ratio = 0.999
            is_saturated = True
        elif ratio >= 0.990:
            is_near_saturated = True

        # 3. 泊松反演公式
        # Lambda = -ln(1 - K_on/N)
        # Average enzymes per bead/chamber
        try:
            lambda_val = -math.log(1 - ratio)
        except ValueError:
            lambda_val = 0.0

        # 4. 摩尔浓度 (M = mol/L)
        # Conc = (Lambda / (V * NA)) * (1/Efficiency)
        # V (pL) -> L: * 1e-12
        vol_liter = V * 1e-12
        molar_conc = (lambda_val) / (vol_liter * self.avogadro * self.loading_efficiency)

        # 5. 质量浓度 (g/mL)
        # 1 Da = 1 g/mol
        # M (kDa) -> g/mol: * 1000
        # g/L -> g/mL: / 1000
        # 综合：Molar * M * 1000 / 1000 = Molar * M (g/L) = Molar * M * 1e-3 (g/mL)
        # 让我们一步步来：
        # Mass (g/L) = Molar (mol/L) * M (g/mol)
        # Mass (g/L) = Molar * (M_kDa * 1000)
        # Mass (ng/mL) = Mass (g/L) * 1e9 / 1000 = Mass (g/L) * 1e6
        #              = Molar * M_kDa * 1000 * 1e6 = Molar * M_kDa * 1e9

        # 简单推导：
        # 1 M = 1 mol/L
        # 1 ng/mL = 1e-6 g/L
        # Conc (ng/mL) = Molar * MW(g/mol) * 1e6
        #              = Molar * (M_kDa * 1000) * 1e6
        #              = Molar * M_kDa * 1e9

        mass_conc_ng_ml = molar_conc * M * 1e9

        # 6. 格式化显示
        display_str = self._format_concentration(mass_conc_ng_ml)
        if is_saturated:
            display_str = "> " + display_str + " (Sat)"

        warning_msg = self._get_warning(is_saturated, is_near_saturated, is_zero_count)

        return {
            "concentration": mass_conc_ng_ml,  # 单位 ng/mL
            "molar_concentration": molar_conc,
            "lambda": lambda_val,
            "display_str": display_str,
            "is_saturated": is_saturated,
            "is_near_saturated": is_near_saturated,
            "is_zero_count": is_zero_count,
            "warning_msg": warning_msg,
            "ratio": round(ratio, 6)
        }

    @staticmethod
    def _get_warning(is_sat: bool, is_near: bool, is_zero: bool) -> str:
        if is_sat:
            return "[CRITICAL] 阳性率 >99.9%，泊松分布失效。浓度为下界估计值，建议稀释样本。"
        elif is_near:
            return "[WARNING] 阳性率 99%-99.9%，泊松反演精度下降。建议稀释样本后复测。"
        elif is_zero:
            return "[INFO] 零阳性计数。可能为阴性样本或检测失败，建议复查原始图像。"
        return ""

    def _format_concentration(self, conc_ng_ml: float) -> str:
        """单位自适应格式化"""
        if conc_ng_ml <= 1e-9:
            return "0.000 ng/mL"

        if conc_ng_ml < 1e-3:
            # 转为 fg/mL (1 ng = 1e6 fg)
            val = conc_ng_ml * 1e6
            return f"{val:.3f} fg/mL"
        elif conc_ng_ml < 1:
            # 转为 pg/mL (1 ng = 1000 pg)
            val = conc_ng_ml * 1000
            return f"{val:.3f} pg/mL"
        elif conc_ng_ml < 1000:
            return f"{conc_ng_ml:.3f} ng/mL"
        else:
            # 转为 ug/mL
            val = conc_ng_ml / 1000
            return f"{val:.3f} ug/mL"