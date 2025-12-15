# Backtest & Trading Assumptions (Current Qlib Templates)

This document captures the *current* backtest/trading assumptions implied by the Qlib template YAMLs used by RD-Agent.

## Strategy
- Qlib strategy: `TopkDropoutStrategy`
  - typical params: `topk`, `n_drop`
  - implies periodic rebalancing with partial turnover control.

## Account / Capital
- Initial capital is explicitly set in templates, e.g. `account: 100000000`.

## Transaction Costs
- Templates configure explicit costs:
  - `open_cost`
  - `close_cost`
  - `min_cost`

## Limit-Up/Limit-Down Constraint
- `limit_threshold` is set (e.g. 0.095) to model tradeability constraints near price limits.

## Execution Price
- `deal_price` is typically set to `close` (simplified daily execution assumption).

## Notes / Known Gaps
- T+1 and intraday microstructure are not explicitly parameterized in the current YAML templates.
- Stop-loss/take-profit rules described in some scenario texts are not implemented by default unless a custom strategy is added.
