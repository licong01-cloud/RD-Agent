import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    root: str


def _load_calendar(provider_root: str) -> List[str]:
    cal_path = os.path.join(provider_root, "calendars", "day.txt")
    with open(cal_path, "r", encoding="utf-8") as f:
        cal = [x.strip() for x in f if x.strip()]
    if not cal:
        raise RuntimeError(f"Empty calendar: {cal_path}")
    return cal


def _calendar_index_map(cal: Sequence[str]) -> Dict[str, int]:
    return {d: i for i, d in enumerate(cal)}


def _find_inst_dir(provider_root: str, inst_code: str) -> str:
    """Try to locate instrument folder under features/.

    Supports common styles:
    - sh688981
    - sz000001
    - 000019.sz
    - 000019.SZ
    """

    feat_root = os.path.join(provider_root, "features")
    if not os.path.isdir(feat_root):
        raise RuntimeError(f"features dir not found: {feat_root}")

    code = inst_code.strip()
    if "." in code:
        # e.g. 000019.SZ
        c = code
        cand = [c, c.lower(), c.upper()]
    else:
        # e.g. SH688981 or sh688981
        c = code.upper()
        if len(c) >= 2 and c[:2] in {"SH", "SZ", "BJ"}:
            prefix = c[:2].lower()
            digits = c[2:]
            cand = [
                f"{prefix}{digits}",
                f"{digits}.{prefix}",
                f"{digits}.{prefix.upper()}",
            ]
        else:
            # only digits
            digits = c
            cand = [
                f"sh{digits}",
                f"sz{digits}",
                f"bj{digits}",
                f"{digits}.sh",
                f"{digits}.sz",
                f"{digits}.bj",
            ]

    for rel in cand:
        p = os.path.join(feat_root, rel)
        if os.path.isdir(p):
            return p

    # last resort: glob
    digits = "".join([ch for ch in code if ch.isdigit()])
    if digits:
        globs = [
            os.path.join(feat_root, f"*{digits}*"),
        ]
        for g in globs:
            hits = [h for h in glob.glob(g) if os.path.isdir(h)]
            if len(hits) == 1:
                return hits[0]

    raise FileNotFoundError(
        f"Cannot locate instrument folder for {inst_code!r} under {feat_root}. Tried: {cand}"
    )


def _list_fields(inst_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(inst_dir, "*.day.bin")))
    fields = [os.path.basename(p).replace(".day.bin", "") for p in files]
    return fields


def _read_bin_value(inst_dir: str, field: str, cal_idx: int) -> Tuple[Optional[int], Optional[float]]:
    path = os.path.join(inst_dir, f"{field}.day.bin")
    if not os.path.exists(path):
        return None, None
    a = np.fromfile(path, dtype="<f4")
    if len(a) < 2:
        return None, None
    start_idx = int(a[0])
    data = a[1:]
    pos = cal_idx - start_idx
    if pos < 0 or pos >= len(data):
        return start_idx, None
    return start_idx, float(data[pos])


def _is_nan(x: Optional[float]) -> bool:
    return x is not None and (isinstance(x, float) and np.isnan(x))


def _diff_kind(a: Optional[float], b: Optional[float], atol: float, rtol: float) -> str:
    if a is None and b is None:
        return "both_missing"
    if a is None and b is not None:
        return "only_B"
    if a is not None and b is None:
        return "only_A"
    assert a is not None and b is not None

    if _is_nan(a) and _is_nan(b):
        return "both_nan"
    if _is_nan(a) and not _is_nan(b):
        return "A_nan_B_value"
    if not _is_nan(a) and _is_nan(b):
        return "A_value_B_nan"

    # both numbers (not nan)
    if np.isclose(a, b, atol=atol, rtol=rtol):
        return "equal"
    return "numeric_diff"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two Qlib bin providers for one instrument over a date range. "
            "It compares all fields (union of both providers' available *.day.bin). "
            "Outputs summary + detailed diffs; optionally writes CSV/JSON."
        )
    )

    parser.add_argument(
        "--provider-a",
        default="/mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209",
        help="Provider A root (default: AIstock qlib_bin_20251209)",
    )
    parser.add_argument(
        "--provider-b",
        default="/mnt/f/dev/RD-Agent-main/qlib/qlib_bin/qlib_bin",
        help="Provider B root (default: RD-Agent repo qlib_bin)",
    )
    parser.add_argument(
        "--name-a",
        default="aistock_bin",
        help="Display name for provider A",
    )
    parser.add_argument(
        "--name-b",
        default="qlib_bin",
        help="Display name for provider B",
    )
    parser.add_argument(
        "--inst-a",
        default="688981.SH",
        help="Instrument code for provider A (default: 688981.SH)",
    )
    parser.add_argument(
        "--inst-b",
        default="SH688981",
        help="Instrument code for provider B (default: SH688981)",
    )
    parser.add_argument("--start", default="2025-08-20", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-09-02", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for numeric comparison (default: 1e-6)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for numeric comparison (default: 1e-6)",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional: write detailed diffs to CSV file path",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional: write summary JSON to file path",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="If set, output full table for every date × every field (no filtering by diffs).",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=0,
        help="Max number of diff rows to print to stdout. 0 means no limit (print all).",
    )

    args = parser.parse_args()

    prov_a = ProviderSpec(args.name_a, args.provider_a)
    prov_b = ProviderSpec(args.name_b, args.provider_b)

    # calendars
    cal_a = _load_calendar(prov_a.root)
    cal_b = _load_calendar(prov_b.root)
    idx_a = _calendar_index_map(cal_a)
    idx_b = _calendar_index_map(cal_b)

    # date range based on intersection of both calendars and user input
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    dates = []
    for d in pd.date_range(start, end, freq="D"):
        ds = d.strftime("%Y-%m-%d")
        if ds in idx_a and ds in idx_b:
            dates.append(ds)

    if not dates:
        raise RuntimeError(
            f"No overlapping trading dates between {args.start}~{args.end}. "
            f"(Maybe calendars differ or range has no trading days.)"
        )

    inst_dir_a = _find_inst_dir(prov_a.root, args.inst_a)
    inst_dir_b = _find_inst_dir(prov_b.root, args.inst_b)

    fields_a = _list_fields(inst_dir_a)
    fields_b = _list_fields(inst_dir_b)
    fields = sorted(set(fields_a) | set(fields_b))

    print("=" * 100)
    print("Compare Qlib bin providers")
    print("=" * 100)
    print(f"Provider A: {prov_a.name} -> {prov_a.root}")
    print(f"  inst_a={args.inst_a}  inst_dir_a={inst_dir_a}")
    print(f"  calendar_last={cal_a[-1]}  calendar_len={len(cal_a)}")
    print(f"  fields({len(fields_a)}): {fields_a}")
    print(f"Provider B: {prov_b.name} -> {prov_b.root}")
    print(f"  inst_b={args.inst_b}  inst_dir_b={inst_dir_b}")
    print(f"  calendar_last={cal_b[-1]}  calendar_len={len(cal_b)}")
    print(f"  fields({len(fields_b)}): {fields_b}")
    print("Range:")
    print(f"  requested: {args.start} ~ {args.end}")
    print(f"  overlap trading dates({len(dates)}): {dates[0]} ~ {dates[-1]}")
    print(f"Numeric compare: atol={args.atol} rtol={args.rtol}")
    print("=" * 100)

    rows = []
    for ds in dates:
        ia = idx_a[ds]
        ib = idx_b[ds]
        for f in fields:
            start_idx_a, va = _read_bin_value(inst_dir_a, f, ia)
            start_idx_b, vb = _read_bin_value(inst_dir_b, f, ib)
            kind = _diff_kind(va, vb, atol=args.atol, rtol=args.rtol)
            if not args.show_all and kind in {"equal", "both_nan", "both_missing"}:
                continue
            rows.append(
                {
                    "date": ds,
                    "field": f,
                    "A_value": va,
                    "B_value": vb,
                    "diff_kind": kind,
                    "A_start_idx": start_idx_a,
                    "B_start_idx": start_idx_b,
                    "A_has_field": f in fields_a,
                    "B_has_field": f in fields_b,
                    "A_inst_dir": os.path.basename(inst_dir_a),
                    "B_inst_dir": os.path.basename(inst_dir_b),
                }
            )

    df = pd.DataFrame(rows)

    print("\nSummary:")
    print(f"  total_fields_union={len(fields)}")
    print(f"  compared_dates={len(dates)}")
    if args.show_all:
        print(f"  rows_in_full_table={len(df)}")
    else:
        print(f"  diff_rows={len(df)}")

    if len(df) == 0:
        if args.show_all:
            print("  Empty table (unexpected).")
        else:
            print("  No diffs found (excluding equal/both_nan/both_missing).")
    else:
        print("\nDiff kind counts:")
        print(df["diff_kind"].value_counts().sort_index())

        print("\nTop fields (by row count in output):")
        print(df.groupby("field").size().sort_values(ascending=False).head(30))

        cols = [
            "date",
            "field",
            "diff_kind",
            "A_value",
            "B_value",
            "A_has_field",
            "B_has_field",
            "A_start_idx",
            "B_start_idx",
        ]
        max_print = int(args.max_print or 0)
        title = "Full table" if args.show_all else "Diffs"
        if max_print > 0:
            print(f"\n{title} (first {max_print} rows):")
            print(df[cols].head(max_print).to_string(index=False))
        else:
            print(f"\n{title} (all rows):")
            print(df[cols].to_string(index=False))

    if args.out_csv:
        out_csv = args.out_csv
        os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
        df.to_csv(out_csv, index=False)
        print(f"\n[Wrote] CSV diffs -> {out_csv}")

    if args.out_json:
        out_json = args.out_json
        os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
        summary = {
            "provider_a": {"name": prov_a.name, "root": prov_a.root, "inst": args.inst_a, "inst_dir": inst_dir_a},
            "provider_b": {"name": prov_b.name, "root": prov_b.root, "inst": args.inst_b, "inst_dir": inst_dir_b},
            "requested_range": {"start": args.start, "end": args.end},
            "overlap_trading_dates": {"count": len(dates), "first": dates[0], "last": dates[-1]},
            "calendar": {
                "a_last": cal_a[-1],
                "a_len": len(cal_a),
                "b_last": cal_b[-1],
                "b_len": len(cal_b),
            },
            "fields": {
                "a": fields_a,
                "b": fields_b,
                "union": fields,
            },
            "diff_rows": int(len(df)),
            "diff_kind_counts": df["diff_kind"].value_counts().to_dict() if len(df) else {},
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[Wrote] JSON summary -> {out_json}")


if __name__ == "__main__":
    main()
