import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd


_PHYSICAL_UNITS = {"元", "万元", "股", "万股"}


def _to_unix_path(p: Path) -> Path:
    s = str(p)
    if len(s) >= 3 and s[1:3] == ":/":
        drive = s[0].lower()
        return Path("/mnt") / drive / s[3:]
    return p


def _normalize_unit(u: str) -> str:
    u = (u or "").strip()
    return u if u in _PHYSICAL_UNITS else ""


def _write_schema(schema: dict[str, Any], out_dirs: list[Path], stem: str) -> None:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(schema["columns"]).to_csv(d / f"{stem}.csv", index=False, encoding="utf-8")


def _extract_sections(md_text: str) -> dict[str, list[str]]:
    # Minimal parser: map heading like "## 2. daily_basic_factors" to its lines.
    current = ""
    out: dict[str, list[str]] = {}
    for line in md_text.splitlines():
        m = re.match(r"^##\s+\d+\.\s*([^#]+?)\s*$", line.strip())
        if m:
            current = m.group(1).strip()
            out.setdefault(current, [])
            continue
        if current:
            out[current].append(line)
    return out


def _default_factor_defs() -> dict[str, list[dict[str, Any]]]:
    # Authoritative meaning/formula is documented in docs/rdagent_precomputed_factors_cn.md.
    # This table is a machine-readable mirror, and factors are marked as reference.
    return {
        "daily_basic_factors": [
            {
                "name": "value_pe_inv",
                "meaning_cn": "倒数市盈率（估值因子）。数值越大通常代表越‘便宜’。",
                "unit": "",
                "formula": "1/db_pe_ttm（优先）或 1/db_pe；分母为0视为缺失",
                "input_fields": ["db_pe_ttm", "db_pe"],
                "missing_policy": "db_pe_ttm/db_pe 为0或缺失 => NaN",
                "quality": "reference",
            },
            {
                "name": "value_pb_inv",
                "meaning_cn": "倒数市净率（估值因子）。数值越大通常代表越‘便宜’。",
                "unit": "",
                "formula": "1/db_pb；分母为0视为缺失",
                "input_fields": ["db_pb"],
                "missing_policy": "db_pb 为0或缺失 => NaN",
                "quality": "reference",
            },
            {
                "name": "size_log_mv",
                "meaning_cn": "市值对数（规模因子）。",
                "unit": "",
                "formula": "log(db_circ_mv 优先，否则 db_total_mv)，仅对 >0 取对数",
                "input_fields": ["db_circ_mv", "db_total_mv"],
                "missing_policy": "市值<=0 或缺失 => NaN",
                "quality": "reference",
            },
            {
                "name": "liquidity_turnover",
                "meaning_cn": "换手率（流动性因子）。",
                "unit": "",
                "formula": "db_turnover_rate",
                "input_fields": ["db_turnover_rate"],
                "missing_policy": "缺失 => NaN",
                "quality": "reference",
            },
            {
                "name": "liquidity_vol_ratio",
                "meaning_cn": "量比（流动性因子）。",
                "unit": "",
                "formula": "db_volume_ratio",
                "input_fields": ["db_volume_ratio"],
                "missing_policy": "缺失 => NaN",
                "quality": "reference",
            },
        ],
        "capital_flow_daily": [
            {
                "name": "mf_main_net_amt",
                "meaning_cn": "主力净流入金额（大单+超大单）。",
                "unit": "元",
                "formula": "(mf_elg_buy_amt+mf_lg_buy_amt)-(mf_elg_sell_amt+mf_lg_sell_amt)",
                "input_fields": [
                    "mf_elg_buy_amt",
                    "mf_lg_buy_amt",
                    "mf_elg_sell_amt",
                    "mf_lg_sell_amt",
                ],
                "missing_policy": "缺失字段按0处理（脚本逻辑）；若上游缺失将影响解释",
                "quality": "reference",
            },
            {
                "name": "mf_main_net_ratio",
                "meaning_cn": "主力净流入占当日成交额。",
                "unit": "",
                "formula": "mf_main_net_amt / amount（amount=0视为缺失）",
                "input_fields": ["mf_main_net_amt", "amount"],
                "missing_policy": "amount 为0或缺失 => NaN",
                "quality": "reference",
            },
            {
                "name": "mf_main_net_amt_5d",
                "meaning_cn": "5日滚动主力净流入金额。",
                "unit": "元",
                "formula": "sum_5d(mf_main_net_amt) by instrument",
                "input_fields": ["mf_main_net_amt"],
                "missing_policy": "滚动窗口内缺失按 pandas rolling 规则传播",
                "quality": "reference",
            },
            {
                "name": "mf_main_net_ratio_5d",
                "meaning_cn": "5日滚动主力净流入占比（对 ratio 做 rolling sum）。",
                "unit": "",
                "formula": "sum_5d(mf_main_net_ratio) by instrument",
                "input_fields": ["mf_main_net_ratio"],
                "missing_policy": "滚动窗口内缺失按 pandas rolling 规则传播",
                "quality": "reference",
            },
            {
                "name": "mf_main_net_mv_5d",
                "meaning_cn": "5日主力净流入 / 流通市值（如 daily_basic 可用）。",
                "unit": "",
                "formula": "mf_main_net_amt_5d / db_circ_mv（或 db_total_mv）",
                "input_fields": ["mf_main_net_amt_5d", "db_circ_mv", "db_total_mv"],
                "missing_policy": "市值为0或缺失 => NaN；若无 daily_basic 则不生成",
                "quality": "reference",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate factor schema CSV/JSON for precomputed factor sets (reference-only)."
    )
    parser.add_argument(
        "--factors-root",
        required=True,
        help="AIstock factors root, e.g. F:/Dev/AIstock/factors (Windows or WSL path).",
    )
    parser.add_argument(
        "--out-dir-governance",
        required=True,
        help="Governance output root, factor schemas will be placed under <out>/schemas/factors.",
    )
    parser.add_argument(
        "--md-doc",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "rdagent_precomputed_factors_cn.md"),
        help="Optional reference doc for humans (not strictly required by generator).",
    )

    args = parser.parse_args()

    factors_root = _to_unix_path(Path(args.factors_root)).resolve()
    out_dir_governance = _to_unix_path(Path(args.out_dir_governance)).resolve()

    md_doc_path = _to_unix_path(Path(args.md_doc)).resolve()

    md_doc = ""
    if md_doc_path.exists():
        md_doc = md_doc_path.read_text(encoding="utf-8")

    sections = _extract_sections(md_doc) if md_doc else {}

    defs = _default_factor_defs()

    for factor_set, cols in defs.items():
        enriched_cols: list[dict[str, Any]] = []
        for c in cols:
            cc = dict(c)
            cc["unit"] = _normalize_unit(str(cc.get("unit", "")))
            enriched_cols.append(cc)

        schema: dict[str, Any] = {
            "factor_set": factor_set,
            "index": {"type": "MultiIndex", "names": ["datetime", "instrument"]},
            "columns": enriched_cols,
            "quality": "reference",
            "doc_source": str(md_doc_path) if md_doc_path.exists() else "",
            "doc_section_hint": "".join((sections.get(factor_set, [])[:0])),
            "unit_policy": "unit is optional and should only contain physical units (e.g., 元/万元/股/万股). Ratios/percentages/PE/log should have empty unit and be inferred from meaning_cn.",
        }

        out_dirs = [
            (factors_root / factor_set / "metadata"),
            (out_dir_governance / "schemas" / "factors"),
        ]
        _write_schema(schema, out_dirs, stem=f"{factor_set}_schema")

    print("[SUCCESS] wrote factor schemas to:")
    print(" -", factors_root)
    print(" -", out_dir_governance / "schemas" / "factors")


if __name__ == "__main__":
    main()
