import argparse
import json
from pathlib import Path

import pandas as pd


def _to_unix_path(p: Path) -> Path:
    s = str(p)
    if len(s) >= 3 and s[1:3] == ":/":
        drive = s[0].lower()
        return Path("/mnt") / drive / s[3:]
    return p


def _load_schema_columns(schema_path: Path) -> list[str]:
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    return [str(c.get("name")) for c in data.get("columns", []) if str(c.get("name", ""))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify factor outputs match factor schema columns.")
    parser.add_argument("--factors-root", required=True)
    parser.add_argument("--schema-dir", required=True, help="Directory containing *_schema.json")
    parser.add_argument(
        "--factor-sets",
        default="daily_basic_factors,capital_flow_daily",
        help="Comma-separated factor sets to verify.",
    )

    args = parser.parse_args()

    factors_root = _to_unix_path(Path(args.factors_root)).resolve()
    schema_dir = _to_unix_path(Path(args.schema_dir)).resolve()

    factor_sets = [s.strip() for s in str(args.factor_sets).split(",") if s.strip()]

    all_ok = True
    for fs in factor_sets:
        schema_path = schema_dir / f"{fs}_schema.json"
        out_pkl = factors_root / fs / "result.pkl"

        if not schema_path.exists():
            print(f"[FAIL] schema not found: {schema_path}")
            all_ok = False
            continue

        if not out_pkl.exists():
            print(f"[FAIL] result.pkl not found: {out_pkl}")
            all_ok = False
            continue

        schema_cols = _load_schema_columns(schema_path)
        df = pd.read_pickle(out_pkl)
        out_cols = [str(c) for c in df.columns]

        missing_in_output = sorted(set(schema_cols) - set(out_cols))
        extra_in_output = sorted(set(out_cols) - set(schema_cols))

        if missing_in_output or extra_in_output:
            print(f"[WARN] {fs}: schema/output mismatch")
            if missing_in_output:
                print("  - missing_in_output:", missing_in_output)
            if extra_in_output:
                print("  - extra_in_output:", extra_in_output)
            all_ok = False
        else:
            print(f"[OK] {fs}: schema columns match output columns ({len(out_cols)} cols)")

    if all_ok:
        print("[SUCCESS] all factor sets verified")
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
