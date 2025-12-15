# RAG Knowledge Base (Project)

This folder contains curated, distilled documents intended to be used as a Retrieval-Augmented Generation (RAG) knowledge base for RD-Agent + AIstock workflows.

Principles:
- The RAG corpus should prioritize **local system rules and ground-truth data schemas** over generic quant knowledge.
- Documents here are **distilled** from multiple sources under `docs/` and the current code/config templates.
- Prefer small, focused, stable documents ("knowledge cards") over copying large memos verbatim.

How to extend:
- When a new failure pattern appears in logs, add it to `05_runtime_failure_patterns.md` with:
  - symptom
  - root cause
  - minimal fix
  - prevention rule
- When a new dataset/field is added, update `02_data_assets_and_fields.md` and keep naming consistent with runtime DataFrame columns.

Source-of-truth priority order:
1) `01_system_hard_rules.md`
2) `02_data_assets_and_fields.md`
3) `03_factor_implementation_contract.md`
4) `04_backtest_and_trading_assumptions.md`
5) `05_runtime_failure_patterns.md`
