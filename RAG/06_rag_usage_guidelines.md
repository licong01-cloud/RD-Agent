# RAG Usage Guidelines (How to Retrieve & Inject)

## Goal
Increase the *effective iteration rate* by ensuring each LLM call sees the minimal authoritative context:
- hard rules
- field whitelist
- coding contract
- known failure patterns

## Retrieval Packs (recommended)
### Factor hypothesis / factor design
Retrieve:
- `01_system_hard_rules.md`
- `02_data_assets_and_fields.md`
- (optional) factor-family knowledge / examples

### Factor coding / auto-repair
Retrieve:
- `03_factor_implementation_contract.md`
- `05_runtime_failure_patterns.md`
- `02_data_assets_and_fields.md` (field whitelist)

### Result analysis / next-step suggestions
Retrieve:
- `04_backtest_and_trading_assumptions.md`
- `05_runtime_failure_patterns.md`

## Prompt Minimalism Principle
Even with RAG, keep a short hard prompt that:
- forces schema-based field usage
- enforces output format
- forbids hallucinated dependencies

RAG provides *facts and examples*; prompt enforces *behavior and structure*.
