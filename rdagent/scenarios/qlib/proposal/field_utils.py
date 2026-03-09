"""
Shared field tracking utilities for Qlib factor proposal/feedback stages.

Centralizes:
  1. STATIC_FACTOR_FIELDS -- complete set of 90 trackable data fields
  2. collect_used_fields()  -- extracts field usage from trace history
  3. format_field_usage_summary()  -- formats usage stats for prompt injection
  4. format_unused_field_whitelist() -- formats positive guidance whitelist
  5. normalize_variable_name()  -- strips _t suffix for accurate tracking
  6. validate_variables()  -- checks variable names against the known schema
"""

import logging
import re
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# STATIC_FACTOR_FIELDS: complete set of 90 trackable fields
# (excludes daily_pv: open, high, low, close, volume, amount, factor)
# Source: static_factors_schema.csv
# ============================================================

STATIC_FACTOR_FIELDS: Dict[str, List[str]] = {
    "db_": [  # 16 fields — daily_basic
        "db_close", "db_turnover_rate", "db_turnover_rate_f", "db_volume_ratio",
        "db_pe", "db_pe_ttm", "db_pb", "db_ps", "db_ps_ttm",
        "db_dv_ratio", "db_dv_ttm", "db_total_share", "db_float_share",
        "db_free_share", "db_total_mv", "db_circ_mv",
    ],
    "mf_": [  # 44 fields — moneyflow
        "mf_sm_buy_vol", "mf_sm_buy_amt", "mf_sm_sell_vol", "mf_sm_sell_amt",
        "mf_md_buy_vol", "mf_md_buy_amt", "mf_md_sell_vol", "mf_md_sell_amt",
        "mf_lg_buy_vol", "mf_lg_buy_amt", "mf_lg_sell_vol", "mf_lg_sell_amt",
        "mf_elg_buy_vol", "mf_elg_buy_amt", "mf_elg_sell_vol", "mf_elg_sell_amt",
        "mf_net_vol", "mf_net_amt",
        "mf_total_net_amt", "mf_total_net_vol",
        "mf_total_net_amt_ratio", "mf_total_net_vol_ratio",
        "mf_main_net_amt", "mf_main_net_vol",
        "mf_main_net_amt_ratio", "mf_main_net_vol_ratio",
        "mf_elg_net_amt", "mf_elg_net_vol",
        "mf_elg_net_amt_ratio", "mf_elg_net_vol_ratio",
        "mf_elg_share_in_main_amt", "mf_elg_share_in_main_vol",
        "mf_total_net_amt_5d", "mf_main_net_amt_5d", "mf_elg_net_amt_5d",
        "mf_total_net_amt_ratio_5d", "mf_main_net_amt_ratio_5d", "mf_elg_net_amt_ratio_5d",
        "mf_total_net_amt_20d", "mf_main_net_amt_20d", "mf_elg_net_amt_20d",
        "mf_total_net_amt_ratio_20d", "mf_main_net_amt_ratio_20d", "mf_elg_net_amt_ratio_20d",
    ],
    "bb_": [  # 15 fields — bak_basic
        "bb_pe_dyn", "bb_total_assets", "bb_liquid_assets", "bb_fixed_assets",
        "bb_reserved", "bb_reserved_pershare", "bb_eps", "bb_bvps",
        "bb_undp", "bb_per_undp", "bb_rev_yoy", "bb_profit_yoy",
        "bb_gpr", "bb_npr", "bb_holder_num",
    ],
    "cp_": [  # 9 fields — cyq_perf
        "cp_his_low", "cp_his_high", "cp_cost_5pct", "cp_cost_15pct",
        "cp_cost_50pct", "cp_cost_85pct", "cp_cost_95pct",
        "cp_weight_avg", "cp_winner_rate",
    ],
    "precomputed_": [  # 6 fields — precomputed factors
        "value_pe_inv", "value_pb_inv", "size_log_mv",
        "liquidity_turnover", "liquidity_vol_ratio", "PriceStrength_10D",
    ],
}

# Flat set for O(1) validation lookups
ALL_TRACKABLE_FIELDS: frozenset = frozenset(
    field for fields in STATIC_FACTOR_FIELDS.values() for field in fields
)

# Precomputed field names as a frozenset for fast membership checks
_PRECOMPUTED_FIELDS: frozenset = frozenset(STATIC_FACTOR_FIELDS["precomputed_"])

# Daily PV fields (always available, skip during validation)
_DAILY_PV_FIELDS: frozenset = frozenset(["open", "high", "low", "close", "volume", "amount", "factor"])

# Pattern to strip _t / _t1 / _t2 suffixes from LLM-generated variable names
# Does NOT match _ttm, _total, _turnover etc. (requires exactly _t followed by optional digits at end)
_SUFFIX_RE = re.compile(r"_t\d*$")


def normalize_variable_name(var_name: str) -> str:
    """Strip trailing _t / _t1 / _t2 suffix from a variable name.

    Examples:
        mf_net_amt_t  -> mf_net_amt
        db_pe_t1      -> db_pe
        db_pe_ttm     -> db_pe_ttm  (unchanged — _ttm is a real suffix)
        close_t       -> close
        db_pe         -> db_pe  (unchanged)
    """
    return _SUFFIX_RE.sub("", var_name)


def collect_used_fields(trace_hist: list, factor_only: bool = True) -> Dict[str, Set[str]]:
    """Scan trace history and return sets of used fields per prefix category.

    Args:
        trace_hist: list of (experiment, feedback) tuples
        factor_only: if True, skip non-factor experiment rounds

    Returns:
        dict mapping each prefix (including "precomputed_") to a set of
        *normalized* field names that were used in history.
    """
    prefix_fields: Dict[str, Set[str]] = {prefix: set() for prefix in STATIC_FACTOR_FIELDS}

    for experiment, feedback in trace_hist:
        if factor_only:
            action = getattr(getattr(experiment, "hypothesis", None), "action", "factor")
            if action != "factor":
                continue
        for task in getattr(experiment, "sub_tasks", []):
            variables = getattr(task, "variables", None)
            if not isinstance(variables, dict):
                continue
            for var_name in variables:
                normalized = normalize_variable_name(var_name)
                # Check prefixed categories (db_, mf_, bb_, cp_)
                matched = False
                for prefix in ("db_", "mf_", "bb_", "cp_"):
                    if normalized.startswith(prefix):
                        prefix_fields[prefix].add(normalized)
                        matched = True
                        break
                # Check precomputed (no common prefix — match by exact name)
                if not matched and normalized in _PRECOMPUTED_FIELDS:
                    prefix_fields["precomputed_"].add(normalized)

    return prefix_fields


def format_field_usage_summary(
    prefix_fields: Dict[str, Set[str]],
    n_rounds: int,
    language: str = "zh",
) -> str:
    """Format field usage statistics for prompt injection.

    Returns a multi-line summary string showing how many fields in each
    category have been used vs total available.
    """
    total = sum(len(f) for f in prefix_fields.values())

    if language == "zh":
        lines = ["\n\n## 历史字段使用统计（系统自动生成）"]
        lines.append(f"历史 {n_rounds} 轮已使用 {total} 个不同字段：")
        for prefix, fields in prefix_fields.items():
            label = f"{prefix}*" if prefix.endswith("_") else prefix
            available = len(STATIC_FACTOR_FIELDS[prefix])
            lines.append(
                f"  {label} 已使用 {len(fields)}/{available} 个: "
                f"{', '.join(sorted(fields)) if fields else '(无)'}"
            )
        lines.append("请优先探索上述使用数量为 0 或最少的类别。")
    else:
        lines = ["[Historical Field Usage — design factors using DIFFERENT fields from those listed below]"]
        for prefix, fields in prefix_fields.items():
            label = f"{prefix}*" if prefix.endswith("_") else prefix
            available = len(STATIC_FACTOR_FIELDS[prefix])
            lines.append(
                f"  {label} used {len(fields)}/{available}: "
                f"{', '.join(sorted(fields)) if fields else '(none)'}"
            )
        lines.append(f"Prioritize field groups with fewer used fields. Total used: {total}.")

    return "\n".join(lines)


def format_unused_field_whitelist(
    prefix_fields: Dict[str, Set[str]],
    language: str = "zh",
) -> str:
    """Format the UNUSED field whitelist — positive guidance for the LLM.

    Computes (all fields) - (used fields) per category and produces a
    compact list of unexplored fields the LLM can directly use.

    Returns a multi-line string, or "" if all fields have been used.
    """
    any_unused = False

    if language == "zh":
        lines = ["\n## 未使用字段白名单（优先从中选择）"]
    else:
        lines = ["\n## Unused Fields Whitelist (PRIORITIZE these for new factors)"]

    for prefix, all_fields in STATIC_FACTOR_FIELDS.items():
        used = prefix_fields.get(prefix, set())
        unused = sorted(set(all_fields) - used)
        if unused:
            any_unused = True
            label = f"{prefix}*" if prefix.endswith("_") else prefix
            if language == "zh":
                lines.append(f"  {label} 剩余 {len(unused)} 个未探索: {', '.join(unused)}")
            else:
                lines.append(f"  {label} {len(unused)} unexplored: {', '.join(unused)}")

    if not any_unused:
        return ""

    if language == "zh":
        lines.append("以上字段均为数据集中真实存在的字段，可直接在因子代码中使用。请优先从这些未探索字段中选择。")
    else:
        lines.append("All fields above are REAL fields in the dataset. Prioritize selecting from these unexplored fields.")

    return "\n".join(lines)


def validate_variables(
    variables: Dict[str, str],
    factor_name: str = "",
) -> Tuple[Dict[str, str], List[str]]:
    """Validate variable names against the known schema.

    Returns:
        (variables, warnings): The variables dict is returned UNCHANGED
        (no fields removed — to avoid breaking downstream). Warnings list
        any potentially hallucinated field names for logging.
    """
    warnings: List[str] = []

    for var_name in variables:
        normalized = normalize_variable_name(var_name)
        # Skip daily_pv fields (always available)
        if normalized in _DAILY_PV_FIELDS:
            continue
        # Only validate variables with a recognized data prefix
        has_data_prefix = any(normalized.startswith(p) for p in ("db_", "mf_", "bb_", "cp_"))
        if has_data_prefix and normalized not in ALL_TRACKABLE_FIELDS:
            warnings.append(
                f"[FieldValidation] Factor '{factor_name}': variable '{var_name}' "
                f"(normalized: '{normalized}') has a data prefix but is NOT in the "
                f"schema whitelist. This may be a hallucinated field name."
            )

    return variables, warnings
