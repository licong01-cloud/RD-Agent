"""v25 Two-Stage执行器: 前30分钟 + 后210分钟

v25方案B: 基于Oracle权重的two-stage执行
- Early模型: 预测前30分钟分布（权重约88.79%）
- Late模型: 基于前30统计预测后210分钟（权重约11.21%）
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rl_execution.network.execution_plan_net_v25_early_enhanced import EarlyPlanNetEnhanced
from rl_execution.network.execution_plan_net_v25_late import LatePlanNet


def _get_limit_pct(stock_id: str) -> float:
    code = stock_id.split(".")[0]
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


class V25TwoStageExecutor:
    """v25 Two-Stage执行器"""

    EARLY_LEN = 30
    LATE_LEN = 210
    TOTAL_LEN = 240

    def __init__(self,
                 early_model_path: str | None = None,
                 late_model_path: str | None = None,
                 device: str = "cpu"):
        if not early_model_path:
            raise FileNotFoundError("V25 early_model_path is required")
        if not late_model_path:
            raise FileNotFoundError("V25 late_model_path is required")
        if not Path(early_model_path).exists():
            raise FileNotFoundError(f"V25 early_model_path does not exist: {early_model_path}")
        if not Path(late_model_path).exists():
            raise FileNotFoundError(f"V25 late_model_path does not exist: {late_model_path}")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("V25 requested CUDA device but torch.cuda.is_available() is false")
        self.device = torch.device(device)

        # Load early model.
        self.early_model = EarlyPlanNetEnhanced().to(self.device)
        ckpt = torch.load(early_model_path, map_location=self.device, weights_only=False)
        self.early_model.load_state_dict(ckpt["model"])
        self.early_model.eval()

        # Load late model.
        self.late_model = LatePlanNet().to(self.device)
        ckpt = torch.load(late_model_path, map_location=self.device, weights_only=False)
        self.late_model.load_state_dict(ckpt["model"])
        self.late_model.eval()

        # Runtime state.
        self._current_plan = None
        self._gap_ratio = 0.0
        self._limit_pct = 0.10
        self._stock_id = ""
        self._is_buy = True

    def reset(self, total_amount: float, open_price: float, prev_close: float,
              stock_id: str, is_buy: bool, day_features: Optional[np.ndarray] = None):
        """每个新订单开始时重置"""
        self._current_plan = None
        self._limit_pct = _get_limit_pct(stock_id)
        self._stock_id = stock_id
        self._is_buy = is_buy

        # 计算gap_ratio
        if prev_close > 1e-4 and self._limit_pct > 1e-4:
            gap_pct = (open_price - prev_close) / prev_close
            gap_pct = np.clip(gap_pct, -0.20, 0.20)
            self._gap_ratio = gap_pct / self._limit_pct
        else:
            self._gap_ratio = 0.0

        # Generate the execution plan; failure must stop the run.
        self._generate_plan(day_features)

    def generate_plan(self, full_day_close: np.ndarray, full_day_volume: np.ndarray,
                      full_day_high: np.ndarray, full_day_low: np.ndarray,
                      prev_close: float, stock_id: str, is_buy: bool,
                      day_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate and return the authoritative 240-minute V25 plan."""
        close_arr = np.asarray(full_day_close, dtype=np.float64)
        if close_arr.ndim != 1 or len(close_arr) < self.TOTAL_LEN:
            raise ValueError("V25 generate_plan requires at least 240 close bars")
        open_price = float(close_arr[0])
        if open_price <= 0 or prev_close <= 0:
            raise ValueError("V25 generate_plan requires positive open_price and prev_close")
        self.reset(
            total_amount=1.0,
            open_price=open_price,
            prev_close=float(prev_close),
            stock_id=stock_id,
            is_buy=is_buy,
            day_features=day_features,
        )
        if self._current_plan is None:
            raise RuntimeError("V25 plan generation failed")
        return self._current_plan.copy()

    def _generate_plan(self, day_features: Optional[np.ndarray] = None):
        """生成全天240分钟执行计划"""
        if day_features is None:
            day_features = np.zeros(10, dtype=np.float32)
        else:
            day_features = np.asarray(day_features, dtype=np.float32)
            if day_features.shape != (10,) or np.isnan(day_features).any():
                raise ValueError("V25 day_features must be a 10-element finite array")

        # 准备输入
        gap_bucket = self._gap_ratio_to_bucket(self._gap_ratio)
        gap_ratio_signed = self._gap_ratio

        with torch.no_grad():
            gb = torch.LongTensor([gap_bucket]).to(self.device)
            gr = torch.FloatTensor([abs(self._gap_ratio)]).to(self.device)
            grs = torch.FloatTensor([gap_ratio_signed]).to(self.device)
            lp = torch.FloatTensor([self._limit_pct]).to(self.device)
            ib = torch.FloatTensor([1.0 if self._is_buy else 0.0]).to(self.device)
            df = torch.FloatTensor([day_features]).to(self.device)

            # 预测前30分钟
            pred_early = self.early_model(gb, gr, grs, lp, ib, df).cpu().numpy()[0]

            # 计算前30统计
            early_weight = pred_early.sum()
            early_peak_pos = pred_early.argmax() / (self.EARLY_LEN - 1)
            early_mean = pred_early.mean()
            early_concentration = pred_early.max() / (early_mean + 1e-8)

            # 预测后210分钟
            ew = torch.FloatTensor([early_weight]).to(self.device)
            epp = torch.FloatTensor([early_peak_pos]).to(self.device)
            ec = torch.FloatTensor([early_concentration]).to(self.device)

            pred_late = self.late_model(gb, gr, ib, ew, epp, ec).cpu().numpy()[0]

            # 拼接全天计划
            # Apply the same oracle segment weights used during v25 training.
            pred_early_weighted = pred_early * 0.8879
            pred_late_weighted = pred_late * 0.1121
            self._current_plan = np.concatenate([pred_early_weighted, pred_late_weighted])

            # 归一化
            total = self._current_plan.sum()
            if total <= 1e-8:
                raise RuntimeError("V25 plan sum is zero")
            self._current_plan = self._current_plan / total
            early_sum = float(self._current_plan[:self.EARLY_LEN].sum())
            late_sum = float(self._current_plan[self.EARLY_LEN:].sum())
            if abs(early_sum - 0.8879) > 1e-4 or abs(late_sum - 0.1121) > 1e-4:
                raise RuntimeError(
                    f"V25 plan weight mismatch: early={early_sum:.6f} late={late_sum:.6f}"
                )

    def _gap_ratio_to_bucket(self, gap_ratio: float) -> int:
        edges = [-0.70, -0.50, -0.30, -0.10, 0.10, 0.30, 0.50, 0.70]
        for i, edge in enumerate(edges):
            if gap_ratio < edge:
                return i
        return len(edges)

    def decide(self, cur_step: int, remaining: float, is_buy: bool,
               cur_price: float, prev_close: float, limit_pct: float,
               is_limit_up: bool, is_limit_down: bool,
               close_arr: np.ndarray, vol_arr: np.ndarray,
               high_arr: np.ndarray, low_arr: np.ndarray,
               limit_up_arr: np.ndarray, limit_down_arr: np.ndarray) -> tuple[float, float]:
        """决策当前分钟执行比例

        Returns:
            (frac, urgency_bps): 执行比例和紧急度
        """
        # Layer 0: 硬规则
        if cur_step >= self.TOTAL_LEN:
            return 1.0, 0.0

        if self._current_plan is None:
            raise RuntimeError("V25 plan is not initialized; refusing silent TWAP fallback")

        if is_limit_up and is_buy:
            return 0.0, 0.0
        if is_limit_down and not is_buy:
            return 0.0, 0.0

        # Use the plan as a fraction of the current remaining amount.
        if self._current_plan is not None and cur_step < self.TOTAL_LEN:
            remaining_weight = float(self._current_plan[cur_step:].sum())
            if remaining_weight <= 1e-8:
                raise RuntimeError("V25 remaining plan weight is zero")
            frac = float(self._current_plan[cur_step]) / remaining_weight
            return frac, 0.0

        raise RuntimeError("V25 plan lookup failed; refusing silent TWAP fallback")
