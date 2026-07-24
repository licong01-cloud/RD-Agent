"""
Score-Weighted TopK Strategy with Dynamic n_drop

新增可选策略，不替换 EnhancedTopkDropoutStrategy 默认。
提供 4 种加权方式和受控动态 n_drop，供 A/B 实验探索。

Created: 2026-04-06
"""

import json
import logging
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

logger = logging.getLogger(__name__)

HMM_COEFFICIENT_SCHEMA_V2 = "hmm_sector_coefficients_v2"


def _validate_hmm_coefficient_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("HMM coefficient artifact must be a JSON object")
    daily = payload.get("daily_coefficients")
    if not isinstance(daily, dict) or not daily:
        raise RuntimeError("HMM coefficient artifact has no daily_coefficients")
    date_keys = list(daily)
    if date_keys != sorted(date_keys):
        raise RuntimeError("HMM coefficient dates must be sorted in canonical ISO order")
    for date_key, coefficients in daily.items():
        try:
            date.fromisoformat(date_key)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"HMM coefficient date is not ISO YYYY-MM-DD: {date_key!r}") from exc
        if not isinstance(coefficients, dict) or not coefficients:
            raise RuntimeError(f"HMM coefficient sector map is empty for {date_key}")
        for sector_code, coefficient in coefficients.items():
            if not isinstance(sector_code, str) or not sector_code.strip():
                raise RuntimeError(f"HMM coefficient sector identity is invalid for {date_key}")
            if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
                raise RuntimeError(f"HMM coefficient is not numeric for {date_key}/{sector_code}")
            if not np.isfinite(coefficient) or coefficient <= 0:
                raise RuntimeError(f"HMM coefficient must be finite and positive for {date_key}/{sector_code}")

    schema_version = payload.get("schema_version")
    if schema_version is None:
        mapping = payload.get("stock_sector_map")
        if not isinstance(mapping, dict) or not mapping:
            raise RuntimeError("legacy HMM coefficient artifact has no stock_sector_map")
        payload["_detected_mapping_mode"] = "static_legacy_v1"
    elif schema_version == HMM_COEFFICIENT_SCHEMA_V2:
        if payload.get("mapping_mode") != "pit_by_trade_date_v1":
            raise RuntimeError("HMM coefficient v2 mapping_mode must be pit_by_trade_date_v1")
        mapping_by_date = payload.get("stock_sector_map_by_date")
        daily_states = payload.get("daily_states")
        if not isinstance(mapping_by_date, dict) or set(mapping_by_date) != set(date_keys):
            raise RuntimeError("HMM coefficient v2 PIT mapping dates must exactly match coefficient dates")
        if not isinstance(daily_states, dict) or set(daily_states) != set(date_keys):
            raise RuntimeError("HMM coefficient v2 state dates must exactly match coefficient dates")
        preset_coeffs = payload.get("preset_coeffs")
        if not isinstance(preset_coeffs, dict):
            raise RuntimeError("HMM coefficient v2 requires preset_coeffs")
        for date_key in date_keys:
            mapping = mapping_by_date[date_key]
            states = daily_states[date_key]
            if not isinstance(mapping, dict) or not mapping or not isinstance(states, dict) or not states:
                raise RuntimeError(f"HMM coefficient v2 evidence is empty for {date_key}")
            if set(states) != set(daily[date_key]):
                raise RuntimeError(f"HMM coefficient v2 state/sector sets differ for {date_key}")
            for sector_code, state in states.items():
                if state not in {"trending", "neutral", "fading"}:
                    raise RuntimeError(f"HMM coefficient v2 has unknown state for {date_key}/{sector_code}")
                expected = preset_coeffs.get(state)
                if expected is None or not np.isclose(daily[date_key][sector_code], expected, atol=1e-12):
                    raise RuntimeError(f"HMM coefficient v2 state/coefficient mismatch for {date_key}/{sector_code}")
        payload["_detected_mapping_mode"] = "pit_by_trade_date_v1"
    else:
        raise RuntimeError(f"unsupported HMM coefficient schema_version: {schema_version}")
    return payload


class ScoreWeightedTopkStrategy(TopkDropoutStrategy):
    """
    可调参数的分数加权策略 + 受控动态 n_drop。

    加权参数:
        weight_method: "softmax" | "linear" | "rank" | "equal"
        temperature: Softmax 温度 (仅 softmax 生效)
        score_clip_quantile: 极端值 clip 分位数, 0=不clip
        max_weight: 单票仓位上限
        min_weight: 单票仓位下限
        max_position_ratio: 组合最大仓位比例

    动态 n_drop 参数:
        enable_dynamic_ndrop: 总开关
        max_n_drop: 硬上限 (关键风控, 默认=n_drop=5)
        min_n_drop: 硬下限 (0=允许完全不换仓)
        threshold_method: "adaptive" | "fixed" | "percentile"
        min_improvement: fixed 阈值
        adaptive_multiplier: adaptive 乘数 (threshold = score_std * multiplier)
        threshold_floor: 自适应阈值最小值保护
    """

    def __init__(
        self,
        signal=None,
        topk=50,
        n_drop=5,
        # ── 加权 ──
        weight_method="softmax",
        temperature=1.0,
        score_clip_quantile=0.0,
        max_weight=0.05,
        min_weight=0.005,
        max_position_ratio=0.95,
        # ── 动态 n_drop ──
        enable_dynamic_ndrop=True,
        max_n_drop=5,
        min_n_drop=0,
        threshold_method="adaptive",
        min_improvement=0.01,
        adaptive_multiplier=0.5,
        threshold_floor=0.005,
        # ── HMM 行业热度调整 ──
        enable_sector_hmm=False,
        sector_hmm_model_path=None,
        hmm_signal_preset=None,
        hmm_signal_presets=None,
        # ── 通用 ──
        min_trade_price=0.5,
        max_trade_price=5000.0,
        max_single_order_value=5_000_000.0,
        lot_size=100.0,
        **kwargs,
    ):
        self.hmm_coefficients_file = kwargs.pop("hmm_coefficients_file", None)
        super().__init__(signal=signal, topk=topk, n_drop=n_drop, **kwargs)
        # 加权 — 参数校验
        _VALID_WEIGHT_METHODS = ("softmax", "linear", "rank", "equal")
        if weight_method not in _VALID_WEIGHT_METHODS:
            raise ValueError(
                f"weight_method='{weight_method}' 无效, 必须是 {_VALID_WEIGHT_METHODS}"
            )
        _VALID_THRESHOLD_METHODS = ("adaptive", "fixed", "percentile")
        if threshold_method not in _VALID_THRESHOLD_METHODS:
            raise ValueError(
                f"threshold_method='{threshold_method}' 无效, 必须是 {_VALID_THRESHOLD_METHODS}"
            )
        self.weight_method = weight_method
        self.temperature = max(float(temperature), 0.01)
        self.score_clip_quantile = float(score_clip_quantile)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.max_position_ratio = float(max_position_ratio)
        # 动态 n_drop
        self.enable_dynamic_ndrop = bool(enable_dynamic_ndrop)
        self.max_n_drop = int(max_n_drop)
        self.min_n_drop = int(min_n_drop)
        self.threshold_method = threshold_method
        self.min_improvement = float(min_improvement)
        self.adaptive_multiplier = float(adaptive_multiplier)
        self.threshold_floor = float(threshold_floor)
        # 通用
        self.min_trade_price = float(min_trade_price)
        self.max_trade_price = float(max_trade_price)
        self.max_single_order_value = float(max_single_order_value)
        self.lot_size = float(lot_size)
        # 诊断
        self._last_diag_date = None
        self._diag_stats = {}
        # 备选股列表（供 inner_strategy 的 TAIL_SUBSTITUTE 使用）
        self._backup_candidates = []
        # HMM 行业热度调整
        self.enable_sector_hmm = bool(enable_sector_hmm)
        self.sector_hmm_model_path = sector_hmm_model_path
        self.hmm_signal_preset = hmm_signal_preset
        self.hmm_signal_presets = hmm_signal_presets or {}
        self._hmm_config: Optional[Dict] = None
        self._hmm_config_loaded = False
        self._last_hmm_adjustment_trace: dict | None = None

    # ── 辅助方法（复用 EnhancedTopkDropoutStrategy 模式）──

    def _get_current_price(self, stock_id, trade_step, direction):
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        return self.trade_exchange.get_deal_price(
            stock_id=stock_id, start_time=start_time,
            end_time=end_time, direction=direction,
        )

    def _get_current_factor(self, stock_id, trade_step):
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        return self.trade_exchange.get_factor(
            stock_id=stock_id, start_time=start_time, end_time=end_time,
        )

    def _shares_to_adjusted_amount(self, shares, factor):
        if shares is None:
            return 0.0
        if factor is None or not np.isfinite(factor) or factor <= 0:
            return float(shares)
        if getattr(self.trade_exchange, "trade_w_adj_price", False) or \
           getattr(self.trade_exchange, "trade_unit", None) is None:
            return float(shares)
        return float(shares) / float(factor)

    def _normalize_signal_scores(self, all_pred_scores, end_time):
        """将 signal 输出归一化为单层 pd.Series(stock_id -> score)"""
        if all_pred_scores is None:
            return pd.Series(dtype=float)

        scores_obj = all_pred_scores
        if isinstance(scores_obj, pd.DataFrame):
            if "score" in scores_obj.columns:
                scores_obj = scores_obj["score"]
            else:
                raise ValueError(
                    f"Signal 返回 DataFrame 但缺少 'score' 列, "
                    f"实际列: {list(scores_obj.columns)}"
                )

        if not isinstance(scores_obj, pd.Series):
            raise TypeError(
                f"Signal 返回了不支持的类型 {type(scores_obj).__name__}, "
                f"期望 pd.Series 或含 'score' 列的 pd.DataFrame"
            )

        if isinstance(scores_obj.index, pd.MultiIndex):
            dt = pd.to_datetime(end_time)
            dt_level = "datetime" if "datetime" in scores_obj.index.names else 0
            try:
                scores_obj = scores_obj.xs(dt, level=dt_level)
            except KeyError:
                # 目标日期不在 signal 中 → 使用最近一天（Qlib signal 日期对齐偏移时正常）
                available_dts = scores_obj.index.get_level_values(dt_level).unique()
                if len(available_dts) == 0:
                    raise ValueError(
                        f"Signal MultiIndex 在 level='{dt_level}' 中无任何日期, "
                        f"end_time={end_time}"
                    )
                last_dt = max(available_dts)
                logger.warning(
                    "[ScoreWeighted] Signal 中不存在 dt=%s, 回退到最近日期 %s",
                    dt, last_dt,
                )
                scores_obj = scores_obj.xs(last_dt, level=dt_level)

        scores_obj = scores_obj.dropna()
        scores_obj.index = scores_obj.index.astype(str)
        return scores_obj

    # ── 核心方法 1: 阈值计算 ──

    def _compute_threshold(self, current_holdings_scores):
        """根据 threshold_method 计算换仓显著性阈值"""
        if self.threshold_method == "fixed":
            return self.min_improvement
        elif self.threshold_method == "adaptive":
            if len(current_holdings_scores) > 1:
                score_std = float(np.std(current_holdings_scores))
                return max(score_std * self.adaptive_multiplier, self.threshold_floor)
            return self.threshold_floor
        elif self.threshold_method == "percentile":
            if len(current_holdings_scores) > 1:
                q75, q25 = np.percentile(current_holdings_scores, [75, 25])
                return max((q75 - q25) * 0.5, self.threshold_floor)
            return self.threshold_floor
        raise ValueError(f"Unknown threshold_method: {self.threshold_method}")

    # ── 核心方法 2: 动态 n_drop ──

    def _filter_dynamic_ndrop(self, sell_candidates, buy_candidates, current_scores):
        """
        动态 n_drop: 在 [min_n_drop, max_n_drop] 区间内按显著性筛选。

        Args:
            sell_candidates: [(stock_id, score), ...] 按 score 升序
            buy_candidates:  [(stock_id, score), ...] 按 score 降序

        Returns:
            (final_sell_ids, final_buy_ids)
        """
        if not self.enable_dynamic_ndrop:
            n = min(self.max_n_drop, len(sell_candidates), len(buy_candidates))
            return ([s[0] for s in sell_candidates[:n]],
                    [b[0] for b in buy_candidates[:n]])

        # Step 1: 硬上限截断
        sell_cands = sell_candidates[:self.max_n_drop]
        buy_cands = buy_candidates[:self.max_n_drop]

        # Step 2: 显著性过滤
        threshold = self._compute_threshold(current_scores)
        self._diag_stats["threshold"] = threshold
        actual_sells, actual_buys = [], []
        n = min(len(sell_cands), len(buy_cands))
        for i in range(n):
            sell_sym, sell_score = sell_cands[i]
            buy_sym, buy_score = buy_cands[i]
            if buy_score - sell_score > threshold:
                actual_sells.append(sell_sym)
                actual_buys.append(buy_sym)
            else:
                break

        # Step 3: 若不足 min_n_drop 则强制补足
        while len(actual_sells) < self.min_n_drop:
            idx = len(actual_sells)
            if idx < len(sell_cands) and idx < len(buy_cands):
                actual_sells.append(sell_cands[idx][0])
                actual_buys.append(buy_cands[idx][0])
            else:
                break

        return actual_sells, actual_buys

    # ── 核心方法 3: 权重计算 ──

    # ------------------------------------------------------------------
    # HMM 行业热度调整（与 TopkDropoutWithRiskControlStrategy 一致）
    # ------------------------------------------------------------------

    def _load_hmm_config(self) -> Dict:
        """从预计算 JSON 文件加载 HMM 系数。enable_sector_hmm=True 时必须成功。"""
        if self._hmm_config_loaded:
            return self._hmm_config

        if not self.hmm_coefficients_file:
            raise RuntimeError(
                "enable_sector_hmm=True 但未指定 hmm_coefficients_file，"
                "无法加载 HMM 系数。请检查实验配置。"
            )

        with open(self.hmm_coefficients_file, encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"HMM 配置文件不是有效 UTF-8 JSON: {self.hmm_coefficients_file}") from exc
        self._hmm_config = _validate_hmm_coefficient_payload(payload)

        self._hmm_config_loaded = True
        logger.info(
            "Loaded HMM config: %d sectors, preset=%s",
            self._hmm_config.get("sector_count", 0),
            self._hmm_config.get("preset_key"),
        )
        return self._hmm_config

    def _apply_hmm_adjustment(self, pred_score: pd.Series, trade_date_str: str) -> pd.Series:
        """对 pred_score 乘以行业热度系数。不启用 HMM 时直接返回原值。"""
        if not self.enable_sector_hmm:
            return pred_score
        if pred_score is None or (hasattr(pred_score, "empty") and pred_score.empty):
            raise RuntimeError(f"HMM adjustment input score is empty: date={trade_date_str}")

        hmm_config = self._load_hmm_config()
        daily_coefficients = hmm_config["daily_coefficients"]
        if hmm_config["_detected_mapping_mode"] == "pit_by_trade_date_v1":
            stock_sector_map = hmm_config["stock_sector_map_by_date"].get(trade_date_str)
            daily_states = hmm_config["daily_states"].get(trade_date_str)
        else:
            stock_sector_map = hmm_config["stock_sector_map"]
            daily_states = None

        day_coeffs = daily_coefficients.get(trade_date_str)
        if not day_coeffs:
            raise RuntimeError(
                f"HMM 配置中缺少交易日 {trade_date_str} 的系数。"
                f"预计算覆盖范围: {hmm_config.get('test_start')} ~ {hmm_config.get('backtest_end')}。"
                f"请检查预计算日期范围是否覆盖回测区间。"
            )

        if not stock_sector_map:
            raise RuntimeError(f"HMM 配置中缺少交易日 {trade_date_str} 的 PIT 股票行业映射")
        if not np.isfinite(pred_score.to_numpy(dtype=float)).all():
            raise RuntimeError(f"HMM 调整前评分包含非有限值: date={trade_date_str}")

        missing_mapping = [stock_id for stock_id in pred_score.index if stock_id not in stock_sector_map]
        missing_coefficient = [
            (stock_id, stock_sector_map[stock_id])
            for stock_id in pred_score.index
            if stock_id in stock_sector_map and stock_sector_map[stock_id] not in day_coeffs
        ]
        if missing_mapping:
            raise RuntimeError(
                f"HMM 股票行业映射缺失: date={trade_date_str} count={len(missing_mapping)} "
                f"sample={missing_mapping[:10]}",
            )
        if missing_coefficient:
            raise RuntimeError(
                f"HMM 行业系数缺失: date={trade_date_str} count={len(missing_coefficient)} "
                f"sample={missing_coefficient[:10]}",
            )

        adjusted = pred_score.copy()
        n_adjusted = 0
        trace_rows = []
        for stock_id in adjusted.index:
            sector_code = stock_sector_map[stock_id]
            coeff = float(day_coeffs[sector_code])
            raw_score = float(adjusted[stock_id])
            adjusted_score = raw_score * coeff
            if not np.isfinite(adjusted_score):
                raise RuntimeError(
                    f"HMM 调整后评分非有限: date={trade_date_str} stock={stock_id} sector={sector_code}",
                )
            adjusted[stock_id] = adjusted_score
            if not np.isclose(coeff, 1.0, atol=1e-12):
                n_adjusted += 1
            trace_rows.append(
                {
                    "stock_id": str(stock_id),
                    "sector_code": sector_code,
                    "state": daily_states.get(sector_code) if daily_states is not None else None,
                    "coefficient": coeff,
                    "raw_score": raw_score,
                    "adjusted_score": adjusted_score,
                    "reason": "hmm_sector_coefficient_applied",
                },
            )

        self._last_hmm_adjustment_trace = {
            "trade_date": trade_date_str,
            "mapping_mode": hmm_config["_detected_mapping_mode"],
            "row_count": len(trace_rows),
            "rows": trace_rows,
        }

        if n_adjusted > 0:
            logger.info(
                "HMM adjustment: date=%s, adjusted %d stocks, %d sectors with non-neutral coeff",
                trade_date_str, n_adjusted,
                len([v for v in day_coeffs.values() if abs(v - 1.0) > 1e-6]),
            )

        return adjusted

    # ------------------------------------------------------------------
    # 权重计算
    # ------------------------------------------------------------------

    def _compute_weights(self, scores):
        """
        根据 weight_method 计算持仓权重。

        Returns:
            np.ndarray of weights, sum <= max_position_ratio
        """
        if len(scores) == 0:
            return np.array([])

        s = scores.copy().astype(float)

        # clip 极端值
        if self.score_clip_quantile > 0:
            lo = np.percentile(s, self.score_clip_quantile * 100)
            hi = np.percentile(s, (1 - self.score_clip_quantile) * 100)
            s = np.clip(s, lo, hi)

        if self.weight_method == "equal":
            weights = np.ones(len(s)) / len(s)
        elif self.weight_method == "rank":
            ranks = s.argsort().argsort() + 1.0
            weights = ranks / ranks.sum()
        elif self.weight_method == "linear":
            shifted = s - s.min() + 1e-8
            weights = shifted / shifted.sum()
        elif self.weight_method == "softmax":
            s_norm = s / self.temperature
            s_norm = s_norm - np.max(s_norm)
            exp_s = np.exp(s_norm)
            weights = exp_s / exp_s.sum()
        else:
            raise ValueError(f"Unknown weight_method: {self.weight_method}")

        # 迭代 clip 到 [min_weight, max_weight]
        for _ in range(10):
            over = weights > self.max_weight
            under = (weights < self.min_weight) & (weights > 0)
            if not over.any() and not under.any():
                break
            if over.any():
                excess = (weights[over] - self.max_weight).sum()
                weights[over] = self.max_weight
                free = ~over
                if free.any() and free.sum() > 0:
                    weights[free] += excess * weights[free] / (weights[free].sum() + 1e-12)
            if under.any():
                deficit = (self.min_weight - weights[under]).sum()
                weights[under] = self.min_weight
                free = ~over & ~under
                if free.any() and free.sum() > 0:
                    reduce = deficit * weights[free] / (weights[free].sum() + 1e-12)
                    weights[free] = np.maximum(weights[free] - reduce, self.min_weight)

        # 归一化后乘以 max_position_ratio
        total = weights.sum()
        if total > 0:
            weights = weights / total * self.max_position_ratio

        return weights

    # ── 主方法: generate_trade_decision ──

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        cur_dt = pd.Timestamp(trade_start_time).date() if trade_start_time is not None else None
        if cur_dt != self._last_diag_date:
            self._diag_stats = {"sells": 0, "buys": 0, "ndrop_filtered": 0, "threshold": 0.0}
            self._last_diag_date = cur_dt

        # 1. 获取预测评分
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        all_pred_scores = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)

        scores = self._normalize_signal_scores(all_pred_scores, pred_end_time)

        if scores is None or scores.empty:
            return TradeDecisionWO([], self)

        # 1.5 HMM 行业热度调整（normalize 后 stock_id 是干净的字符串）
        trade_date_str = str(trade_start_time)[:10]
        scores = self._apply_hmm_adjustment(scores, trade_date_str)

        # 2. 当前持仓
        current_holdings = list(self.trade_position.get_stock_list())

        # 3. TopK 候选
        ranked = scores.sort_values(ascending=False)
        topk_stocks = set(ranked.head(self.topk).index.tolist())

        # 4. 确定 sell/buy 候选
        # sell: 持仓中不在 topk 的 (score 升序)
        sell_candidates = []
        for sid in current_holdings:
            if sid not in topk_stocks:
                sc = scores.get(sid, -999.0)
                sell_candidates.append((sid, float(sc)))
        sell_candidates.sort(key=lambda x: x[1])

        # buy: topk 中未持仓的 (score 降序)
        buy_candidates = []
        for sid in topk_stocks:
            if sid not in current_holdings:
                sc = scores.get(sid, 0.0)
                buy_candidates.append((sid, float(sc)))
        buy_candidates.sort(key=lambda x: -x[1])

        # 5. 建仓/补仓 vs 动态 n_drop
        #    空仓或欠仓时直接补足到 topk，不走 sell-buy 配对逻辑
        #    （父类 TopkDropoutStrategy 通过 `topk - len(last)` 实现同等效果）
        if len(current_holdings) < self.topk:
            # 补仓模式：买入 topk 中未持仓的股票，补足空位
            slots = self.topk - len(current_holdings)
            actual_sells = []
            actual_buys = [b[0] for b in buy_candidates[:slots]]
            logger.info(
                "[ScoreWeighted] 补仓模式: holdings=%d < topk=%d, 买入 %d 只. date=%s",
                len(current_holdings), self.topk, len(actual_buys), cur_dt,
            )
        else:
            # 正常换仓模式：sell-buy 配对 + 动态 n_drop
            current_scores_arr = np.array(
                [float(scores.get(s, 0.0)) for s in current_holdings if s in scores.index]
            )
            actual_sells, actual_buys = self._filter_dynamic_ndrop(
                sell_candidates, buy_candidates, current_scores_arr,
            )
        self._diag_stats["ndrop_filtered"] = len(sell_candidates) - len(actual_sells)

        # 6. 计算最终持仓列表和权重
        final_holdings = [s for s in current_holdings if s not in actual_sells] + actual_buys
        final_holdings = [s for s in final_holdings if s in scores.index]
        final_scores = np.array([float(scores[s]) for s in final_holdings])
        weights = self._compute_weights(final_scores)

        if len(weights) == 0:
            return TradeDecisionWO([], self)

        # 构建 symbol -> weight 映射
        weight_map = {}
        for i, sid in enumerate(final_holdings):
            if i < len(weights):
                weight_map[sid] = float(weights[i])

        # 7. 生成卖出订单 (全部卖出)
        sell_orders = []
        for sid in actual_sells:
            amount = self.trade_position.get_stock_amount(sid)
            if amount is not None and float(amount) > 0:
                sell_orders.append(
                    Order(sid, amount, OrderDir.SELL, trade_start_time, trade_end_time)
                )
            else:
                logger.warning(
                    "[ScoreWeighted] 卖出目标 %s 的持仓量为 %s, 跳过. date=%s",
                    sid, amount, cur_dt,
                )
        self._diag_stats["sells"] = len(sell_orders)

        # 8. 生成买入订单 (按权重分配资金给新增持仓)
        #    基于总账户价值 (持仓市值 + 现金) 计算目标仓位，而非仅用现金
        total_cash = self.trade_position.get_cash()
        if total_cash is None:
            raise RuntimeError(
                f"trade_position.get_cash() 返回 None, 持仓状态异常. date={cur_dt}"
            )
        # 计算总账户价值 = 现金 + 所有持仓市值
        total_account_value = float(total_cash)
        for sid in current_holdings:
            stock_price = self._get_current_price(sid, trade_step, OrderDir.SELL)
            if stock_price is not None and not (isinstance(stock_price, float) and np.isnan(stock_price)) and stock_price > 0:
                amt = self.trade_position.get_stock_amount(sid)
                if amt is not None and float(amt) > 0:
                    total_account_value += float(amt) * float(stock_price)

        buy_orders = []
        for sid in actual_buys:
            w = weight_map.get(sid, 0.0)
            if w <= 0:
                continue

            target_value = total_account_value * w
            target_value = min(target_value, self.max_single_order_value)

            buy_price = self._get_current_price(sid, trade_step, OrderDir.BUY)
            if buy_price is None or (isinstance(buy_price, float) and np.isnan(buy_price)) or buy_price <= 0:
                logger.warning("[ScoreWeighted] %s 买入价格无效: %s, 跳过. date=%s", sid, buy_price, cur_dt)
                continue
            if buy_price < self.min_trade_price or buy_price > self.max_trade_price:
                logger.warning(
                    "[ScoreWeighted] %s 价格 %.2f 超出范围 [%.2f, %.2f], 跳过. date=%s",
                    sid, buy_price, self.min_trade_price, self.max_trade_price, cur_dt,
                )
                continue

            factor = self._get_current_factor(sid, trade_step)
            target_shares = target_value / float(buy_price)
            buy_amount = self._shares_to_adjusted_amount(target_shares, factor)

            # 资金检查（不兜底：get_cash 失败必须暴露）
            avail_cash = float(self.trade_position.get_cash())
            affordable_shares = avail_cash / float(buy_price)
            affordable_amount = self._shares_to_adjusted_amount(affordable_shares, factor)
            if np.isfinite(affordable_amount):
                buy_amount = min(float(buy_amount), float(affordable_amount))

            # 按交易单位取整
            amount_unit = self.trade_exchange.get_amount_of_trade_unit(
                factor=factor, stock_id=sid,
                start_time=trade_start_time, end_time=trade_end_time,
            )
            if amount_unit is None or not np.isfinite(float(amount_unit)) or float(amount_unit) <= 0:
                # get_amount_of_trade_unit 返回 None 说明交易所无此股票的 trade_unit
                # 使用 lot_size (100股) 作为显式 fallback 并记录
                amount_unit = self._shares_to_adjusted_amount(self.lot_size, factor)
                logger.debug(
                    "[ScoreWeighted] %s 无 trade_unit 信息, 使用 lot_size=%s", sid, self.lot_size,
                )
            if np.isfinite(buy_amount) and amount_unit is not None and float(amount_unit) > 0:
                buy_amount = float(np.floor(buy_amount / float(amount_unit)) * float(amount_unit))

            if buy_amount > 0:
                buy_orders.append(
                    Order(sid, buy_amount, OrderDir.BUY, trade_start_time, trade_end_time)
                )

        self._diag_stats["buys"] = len(buy_orders)

        # 存储备选股（topk 之外、排名靠前的候选），供 inner_strategy TAIL_SUBSTITUTE 使用
        backup_depth = 15
        backup_start = self.topk
        backup_end = self.topk + backup_depth
        backup_sids = ranked.iloc[backup_start:backup_end].index.tolist()
        self._backup_candidates = [
            (sid, float(ranked[sid])) for sid in backup_sids
            if sid not in current_holdings
        ]

        if cur_dt is not None:
            logger.info(
                "[ScoreWeighted] date=%s holdings=%d sells=%d buys=%d "
                "ndrop_filtered=%d weight=%s threshold=%.4f",
                cur_dt, len(current_holdings), len(sell_orders), len(buy_orders),
                self._diag_stats["ndrop_filtered"], self.weight_method,
                self._diag_stats.get("threshold", 0.0),
            )

        return TradeDecisionWO(sell_orders + buy_orders, self)
