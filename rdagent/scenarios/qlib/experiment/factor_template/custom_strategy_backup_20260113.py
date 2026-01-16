"""
增强版TopkDropoutStrategy策略
支持分阶段止盈、低分清仓、动态权重分配等功能
"""

import logging

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class EnhancedTopkDropoutStrategy(TopkDropoutStrategy):
    """
    增强版TopkDropoutStrategy，支持：
    1. 止损机制：
       - 亏损达到10%立刻清仓
    2. 分阶段止盈：
       - 盈利15%抛出持仓份额的30%
       - 盈利超25%，再抛出持仓份额的30%（累计60%）
       - 盈利超过35%抛出全部持仓股票
    3. 备选列表清仓：
       - 最低评分阈值0.1
       - 持仓股票不在备选列表（评分前50名）中，直接清仓
    4. 动态权重分配：
       - 按评分分配权重，评分越高仓位越高
       - 最大仓位控制在90%
       - 最大持仓股票数量50只
    """
    
    def __init__(self, 
                 signal=None,
                 topk=50,
                 n_drop=5,
                 min_score=0.10,             # 最低评分阈值
                 max_position_ratio=0.90,     # 最大仓位比例
                 stop_loss=-0.10,             # 止损阈值：-10%
                 min_trade_price=0.5,
                 max_trade_price=5000.0,
                 max_single_order_value=5_000_000.0,
                 max_single_order_amount=5_000_000.0,
                 lot_size=100.0,
                 **kwargs):
        super().__init__(signal=signal, topk=topk, n_drop=n_drop, **kwargs)
        self.min_score = min_score
        self.max_position_ratio = max_position_ratio
        self.stop_loss = stop_loss
        self.min_trade_price = float(min_trade_price)
        self.max_trade_price = float(max_trade_price)
        self.max_single_order_value = float(max_single_order_value)
        self.max_single_order_amount = float(max_single_order_amount)
        self.lot_size = float(lot_size)
        self.entry_prices = {}  # 记录买入价格 {stock_id: price}，持仓期间永不删除
        self.entry_amounts = {}  # 记录初始持仓数量 {stock_id: amount}
        self._last_diag_date = None
        self._buy_skip_stats = {}
        self._warn_missing_entry_prices = set()
        
    def _get_current_price(self, stock_id, trade_step, direction):
        """获取当前股票的成交价格"""
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        return self.trade_exchange.get_deal_price(
            stock_id=stock_id,
            start_time=start_time,
            end_time=end_time,
            direction=direction
        )

    def _get_current_factor(self, stock_id, trade_step):
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        return self.trade_exchange.get_factor(stock_id=stock_id, start_time=start_time, end_time=end_time)

    def _get_daily_change(self, stock_id, trade_step):
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        try:
            return self.trade_exchange.get_quote_info(
                stock_id=stock_id,
                start_time=start_time,
                end_time=end_time,
                field="$change",
                method="ts_data_last",
            )
        except Exception:
            return None

    def _shares_to_adjusted_amount(self, shares: float, factor: float | None) -> float:
        # Qlib 的 Order.amount 语义是 adjusted amount。
        # 当 trade_w_adj_price=False 且 trade_unit 启用时：100股对应 adjusted_amount = trade_unit / factor。
        if shares is None:
            return 0.0
        if factor is None or not np.isfinite(factor) or factor <= 0:
            return float(shares)
        if getattr(self.trade_exchange, "trade_w_adj_price", False) or getattr(self.trade_exchange, "trade_unit", None) is None:
            return float(shares)
        return float(shares) / float(factor)

    def _adjusted_amount_to_shares(self, adjusted_amount: float, factor: float | None) -> float:
        if adjusted_amount is None:
            return 0.0
        if factor is None or not np.isfinite(factor) or factor <= 0:
            return float(adjusted_amount)
        if getattr(self.trade_exchange, "trade_w_adj_price", False) or getattr(self.trade_exchange, "trade_unit", None) is None:
            return float(adjusted_amount)
        return float(adjusted_amount) * float(factor)
    
    def _calculate_dynamic_weights(self, selected_scores):
        """
        根据评分计算动态权重
        评分越高，权重越大
        """
        if selected_scores is None or len(selected_scores) == 0:
            return pd.Series(dtype=float)
        
        # 确保评分是正数
        scores = selected_scores.copy()
        
        # 使用评分的平方来放大高分股票的权重
        weights = scores ** 2
        weights = weights / weights.sum()
        
        return weights

    def _normalize_signal_scores(self, all_pred_scores: pd.DataFrame | pd.Series, end_time):
        if all_pred_scores is None:
            return pd.Series(dtype=float)

        scores_obj = all_pred_scores
        if isinstance(scores_obj, pd.DataFrame):
            if "score" in scores_obj.columns:
                scores_obj = scores_obj["score"]
            else:
                return pd.Series(dtype=float)

        if not isinstance(scores_obj, pd.Series):
            return pd.Series(dtype=float)

        if isinstance(scores_obj.index, pd.MultiIndex):
            dt = None
            try:
                if "datetime" in scores_obj.index.names:
                    dt = pd.to_datetime(end_time)
                    scores_obj = scores_obj.xs(dt, level="datetime")
                else:
                    dt = pd.to_datetime(end_time)
                    scores_obj = scores_obj.xs(dt, level=0)
            except Exception:
                try:
                    dt_level = "datetime" if "datetime" in scores_obj.index.names else 0
                    last_dt = max(scores_obj.index.get_level_values(dt_level))
                    scores_obj = scores_obj.xs(last_dt, level=dt_level)
                except Exception:
                    return pd.Series(dtype=float)

        scores_obj = scores_obj.dropna()
        try:
            scores_obj.index = scores_obj.index.astype(str)
        except Exception:
            pass
        return scores_obj
    
    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_time = trade_start_time

        cur_dt = pd.Timestamp(trade_start_time).date() if trade_start_time is not None else None
        if cur_dt != self._last_diag_date:
            self._buy_skip_stats = {
                "skipped_already_holding": 0,
                "skipped_invalid_price": 0,
                "skipped_non_positive_target": 0,
                "skipped_zero_weight": 0,
                "skipped_reached_buy_limit": 0,
                "buy_orders_created": 0,
                "candidate_count": 0,
            }
            self._last_diag_date = cur_dt

        current_holdings = list(self.trade_position.get_stock_list())
        for stock_id in list(self.entry_prices.keys()):
            try:
                amt = float(self.trade_position.get_stock_amount(stock_id))
            except Exception:
                amt = None
            if amt is not None and amt <= 0:
                self.entry_prices.pop(stock_id, None)
                self.entry_amounts.pop(stock_id, None)

        for stock_id in current_holdings:
            if stock_id not in self.entry_prices and stock_id not in self._warn_missing_entry_prices:
                try:
                    amt = float(self.trade_position.get_stock_amount(stock_id))
                except Exception:
                    amt = None
                logger.warning(
                    "[StrategyDiag] holding exists but missing entry_prices. stock_id=%s amount=%s trade_time=%s",
                    stock_id,
                    amt,
                    trade_start_time,
                )
                self._warn_missing_entry_prices.add(stock_id)
        
        # 1. 获取所有股票的最新预测评分
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        all_pred_scores = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        all_pred_scores = self._normalize_signal_scores(all_pred_scores, pred_end_time)
        
        if all_pred_scores is None or all_pred_scores.empty:
            return TradeDecisionWO([], self)
        
        # 初始化卖出订单列表和当前持仓
        sell_orders = []
        current_holdings = self.trade_position.get_stock_list()
        
        # 2. 止损检查（优先级最高）
        # 亏损达到10%立刻清仓
        for stock_id in current_holdings:
            if stock_id not in self.entry_prices:
                continue  # 没有成本价无法计算收益率，跳过
            
            entry_price = self.entry_prices[stock_id]
            current_price = self._get_current_price(stock_id, trade_step, OrderDir.SELL)
            
            if current_price is None or current_price <= 0:
                continue
            
            return_rate = (current_price - entry_price) / entry_price
            
            # 止损：亏损达到10%立刻清仓
            if return_rate <= self.stop_loss:
                amount = self.trade_position.get_stock_amount(stock_id)
                sell_orders.append(
                    Order(stock_id, amount, OrderDir.SELL, trade_start_time, trade_end_time)
                )
                # 注意：不能在下单时就删除 entry_prices/entry_amounts。
                # 订单可能因为涨跌停/停牌/价格NaN等原因未成交，实际仍持仓。
                # 清理由每日开头的“持仓对账”统一处理：确认持仓量为0才清理。
        
        # 3. 止盈检查（分阶段止盈）
        for stock_id in current_holdings:
            if stock_id not in self.entry_prices:
                continue  # 没有成本价无法计算收益率，跳过
            
            entry_price = self.entry_prices[stock_id]
            current_price = self._get_current_price(stock_id, trade_step, OrderDir.SELL)
            
            if current_price is None or current_price <= 0:
                continue
            
            return_rate = (current_price - entry_price) / entry_price
            
            # 分阶段止盈逻辑（阈值调整为15%、25%、35%）
            if return_rate >= 0.35:
                # 盈利超过35%，抛出全部持仓
                amount = self.trade_position.get_stock_amount(stock_id)
                sell_orders.append(
                    Order(stock_id, amount, OrderDir.SELL, trade_start_time, trade_end_time)
                )
                # 注意：不能在下单时就删除 entry_prices/entry_amounts。
                # 订单可能因为涨跌停/停牌/价格NaN等原因未成交，实际仍持仓。
                # 清理由每日开头的"持仓对账"统一处理：确认持仓量为0才清理。
                    
            elif return_rate >= 0.25:
                # 盈利超过25%，再抛出持仓份额的30%（累计60%）
                current_amount = self.trade_position.get_stock_amount(stock_id)
                if stock_id in self.entry_amounts:
                    original_amount = self.entry_amounts[stock_id]
                    # 如果已经卖出过30%，当前持有量应该是70%
                    # 再卖出30%意味着卖出 0.3 * original_amount
                    sell_amount = 0.3 * original_amount
                    if sell_amount > current_amount:
                        sell_amount = current_amount
                else:
                    # 第一次达到25%盈利，卖出30%
                    sell_amount = 0.3 * current_amount
                    self.entry_amounts[stock_id] = current_amount  # 记录初始持仓
                
                if sell_amount > 0:
                    sell_orders.append(
                        Order(
                            stock_id,
                            sell_amount,
                            OrderDir.SELL,
                            trade_start_time,
                            trade_end_time,
                        )
                    )
                    if sell_amount >= current_amount:
                        # 如果卖出全部，entry_amounts 清理由每日开头的持仓对账统一处理
                        pass
                    
            elif return_rate >= 0.15:
                # 盈利15%，抛出持仓份额的30%
                current_amount = self.trade_position.get_stock_amount(stock_id)
                if stock_id not in self.entry_amounts:
                    # 第一次达到15%盈利，卖出30%
                    sell_amount = 0.3 * current_amount
                    self.entry_amounts[stock_id] = current_amount  # 记录初始持仓
                    
                    if sell_amount > 0:
                        sell_orders.append(
                            Order(
                                stock_id,
                                sell_amount,
                                OrderDir.SELL,
                                trade_start_time,
                                trade_end_time,
                            )
                        )
        
        # 4. 备选股票列表检查
        # 选择评分最高的topk只股票作为备选列表
        qualified_stocks = all_pred_scores[all_pred_scores >= self.min_score]
        if len(qualified_stocks) > 0:
            selected_stocks = qualified_stocks.nlargest(min(len(qualified_stocks), self.topk))
            candidate_list = set(selected_stocks.index.tolist())
        else:
            candidate_list = set()
        
        # 持仓股票不在备选列表中，直接清仓
        for stock_id in current_holdings:
            if stock_id in [order.stock_id for order in sell_orders]:
                continue  # 已经在卖出列表中
            
            if stock_id not in candidate_list:
                amount = self.trade_position.get_stock_amount(stock_id)
                sell_orders.append(
                    Order(stock_id, amount, OrderDir.SELL, trade_start_time, trade_end_time)
                )
                # 注意：不能在下单时就删除 entry_prices/entry_amounts。
                # 订单可能因为涨跌停/停牌/价格NaN等原因未成交，实际仍持仓。
                # 清理由每日开头的“持仓对账”统一处理：确认持仓量为0才清理。
        
        # 5. 买入逻辑
        # 筛选评分高于0.1的股票
        qualified_stocks = all_pred_scores[all_pred_scores >= self.min_score]
        
        # 如果没有股票满足评分阈值，保持空仓
        if len(qualified_stocks) == 0:
            if cur_dt is not None:
                logger.info(
                    "[StrategyDiag] date=%s no qualified stocks (min_score=%s). holdings=%d",
                    cur_dt,
                    self.min_score,
                    len(current_holdings),
                )
            return TradeDecisionWO(sell_orders, self)
        
        # 选择评分最高的topk只股票（如果数量不足，则选择所有符合条件的）
        selected_stocks = qualified_stocks.nlargest(min(len(qualified_stocks), self.topk))
        
        # 计算当前持仓数量（不包括已卖出的）
        current_holdings_set = set(current_holdings)
        already_selling = set([order.stock_id for order in sell_orders])
        current_holdings_after_sell = current_holdings_set - already_selling
        current_position_count = len(current_holdings_after_sell)
        
        # 如果已达到最大持仓数量，不买入
        if current_position_count >= self.topk:
            return TradeDecisionWO(sell_orders, self)
        
        # 6. 计算动态权重
        target_weights = self._calculate_dynamic_weights(selected_stocks)
        self._buy_skip_stats["candidate_count"] = int(len(target_weights))
        
        # 7. 计算可投资金额
        total_cash = self.trade_position.get_cash()
        
        # 最大仓位控制在90%
        investable_amount = total_cash * self.max_position_ratio
        
        # 计算还需要买入多少只股票
        stocks_to_buy = self.topk - current_position_count
        if stocks_to_buy <= 0:
            return TradeDecisionWO(sell_orders, self)
        
        # 8. 生成买入订单
        buy_orders = []
        bought_count = 0
        for stock_id, weight in target_weights.items():
            # 跳过已经持有的股票
            if stock_id in current_holdings_after_sell:
                self._buy_skip_stats["skipped_already_holding"] += 1
                continue
            
            # 达到最大买入数量，停止买入
            if bought_count >= stocks_to_buy:
                self._buy_skip_stats["skipped_reached_buy_limit"] += 1
                break

            if weight is None or float(weight) <= 0:
                self._buy_skip_stats["skipped_zero_weight"] += 1
                continue
            
            target_value = investable_amount * weight
            if target_value is None or float(target_value) <= 0:
                self._buy_skip_stats["skipped_non_positive_target"] += 1
                continue
            buy_price = self._get_current_price(stock_id, trade_step, OrderDir.BUY)

            if buy_price is None or (isinstance(buy_price, float) and np.isnan(buy_price)) or buy_price <= 0:
                self._buy_skip_stats["skipped_invalid_price"] += 1
                logger.warning(
                    "[StrategyDiag] skip buy due to invalid price. stock_id=%s price=%s trade_time=%s",
                    stock_id,
                    buy_price,
                    trade_start_time,
                )
                continue

            daily_chg = self._get_daily_change(stock_id, trade_step)
            if daily_chg is not None:
                try:
                    daily_chg_f = float(daily_chg)
                except Exception:
                    daily_chg_f = None
                if daily_chg_f is not None and np.isfinite(daily_chg_f) and abs(daily_chg_f) > 0.2:
                    self._buy_skip_stats["skipped_invalid_price"] += 1
                    logger.warning(
                        "[StrategyDiag] skip buy due to abnormal daily change. stock_id=%s change=%s trade_time=%s",
                        stock_id,
                        daily_chg_f,
                        trade_start_time,
                    )
                    continue
            
            factor = self._get_current_factor(stock_id, trade_step)
            capped_target_value = float(min(float(target_value), float(self.max_single_order_value)))
            target_shares = capped_target_value / float(buy_price)
            buy_amount = self._shares_to_adjusted_amount(target_shares, factor)

            try:
                available_cash = float(self.trade_position.get_cash())
            except Exception:
                available_cash = float(total_cash) if total_cash is not None else 0.0
            affordable_shares = (available_cash / float(buy_price)) if buy_price > 0 else 0.0
            affordable_amount = self._shares_to_adjusted_amount(affordable_shares, factor)
            if np.isfinite(affordable_amount):
                buy_amount = float(min(float(buy_amount), float(affordable_amount)))

            # 按交易单位取整：优先使用交易所的 trade_unit/factor
            try:
                amount_unit = self.trade_exchange.get_amount_of_trade_unit(
                    factor=factor,
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:
                amount_unit = None
            if amount_unit is None:
                # 退化：如果无法获取 amount_unit，则按策略参数 lot_size（通常100股）换算到 adjusted space
                amount_unit = self._shares_to_adjusted_amount(float(self.lot_size), factor)
            if np.isfinite(buy_amount) and amount_unit is not None and float(amount_unit) > 0:
                buy_amount = float(np.floor(buy_amount / float(amount_unit)) * float(amount_unit))

            if buy_amount > self.max_single_order_amount:
                buy_amount = float(np.floor(float(self.max_single_order_amount) / float(amount_unit)) * float(amount_unit))

            if buy_amount > 0:
                buy_orders.append(
                    Order(stock_id, buy_amount, OrderDir.BUY, trade_start_time, trade_end_time)
                )
                # 只在首次买入时记录成本价，后续加仓不更新
                if stock_id not in self.entry_prices:
                    self.entry_prices[stock_id] = buy_price
                self._buy_skip_stats["buy_orders_created"] += 1
                bought_count += 1
            else:
                self._buy_skip_stats["skipped_non_positive_target"] += 1
        
        # 9. 合并所有订单
        all_orders = sell_orders + buy_orders
        if cur_dt is not None:
            logger.info(
                "[StrategyDiag] date=%s holdings=%d sell_orders=%d buy_orders=%d stats=%s",
                cur_dt,
                len(current_holdings),
                len(sell_orders),
                len(buy_orders),
                self._buy_skip_stats,
            )
        return TradeDecisionWO(all_orders, self)
