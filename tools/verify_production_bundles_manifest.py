import argparse
import json
import subprocess
import sys
import sqlite3
from pathlib import Path


def _default_bundles_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "RDagentDB" / "production_bundles"


def _default_registry_sqlite() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "RDagentDB" / "registry.sqlite"


def _load_referenced_bundle_ids(registry_sqlite: Path) -> set[str]:
    if not registry_sqlite.exists():
        raise FileNotFoundError(f"registry.sqlite not found: {registry_sqlite}")
    con = sqlite3.connect(str(registry_sqlite))
    try:
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT DISTINCT asset_bundle_id
            FROM loops
            WHERE asset_bundle_id IS NOT NULL
              AND asset_bundle_id != ''
              AND (is_solidified = 1 OR is_solidified = '1')
            """
        ).fetchall()
        return {str(r[0]) for r in rows if r and r[0]}
    finally:
        con.close()


def _load_manifest(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _check_bundle(bundle_dir: Path, *, run_self_check: bool) -> tuple[bool, str]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return False, "manifest.json missing"

    try:
        manifest = _load_manifest(manifest_path)
    except Exception as e:
        return False, f"manifest.json parse error: {e}"

    if manifest.get("schema_version") != 1:
        return False, f"schema_version != 1 (got {manifest.get('schema_version')})"

    pa = manifest.get("primary_assets") or {}
    factor_rel = str(pa.get("factor_entry_relpath") or "").strip()
    model_rel = str(pa.get("model_weight_relpath") or "").strip()
    config_rel = str(pa.get("config_relpath") or "").strip()

    if not factor_rel:
        return False, "primary_assets.factor_entry_relpath empty"
    if not model_rel:
        return False, "primary_assets.model_weight_relpath empty"

    factor_path = bundle_dir / factor_rel
    model_path = bundle_dir / model_rel
    if not factor_path.exists():
        return False, f"factor entry not found: {factor_rel}"
    if not model_path.exists():
        return False, f"model weight not found: {model_rel}"

    if config_rel:
        config_path = bundle_dir / config_rel
        if not config_path.exists():
            return False, f"config not found: {config_rel}"

    if run_self_check:
        sc = bundle_dir / "self_check.py"
        if not sc.exists():
            return False, "self_check.py missing"
        try:
            proc = subprocess.run(
                [sys.executable, str(sc)],
                cwd=str(bundle_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode != 0:
                out = (proc.stdout or "").strip()
                return False, f"self_check.py failed rc={proc.returncode}: {out[-500:]}"
        except Exception as e:
            return False, f"self_check.py exec error: {e}"

    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles-dir", default=str(_default_bundles_dir()))
    ap.add_argument("--registry-sqlite", default=str(_default_registry_sqlite()))
    ap.add_argument(
        "--scope",
        choices=["referenced", "all"],
        default="referenced",
        help="referenced: only validate bundles referenced by loops.asset_bundle_id; all: validate every directory under bundles-dir",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--run-self-check", action="store_true")
    args = ap.parse_args()

    bundles_dir = Path(args.bundles_dir).expanduser().resolve()
    if not bundles_dir.exists():
        print(f"bundles dir not found: {bundles_dir}")
        return 2

    if args.scope == "referenced":
        referenced = _load_referenced_bundle_ids(Path(args.registry_sqlite).expanduser().resolve())
        bundle_dirs = [bundles_dir / bid for bid in sorted(referenced) if (bundles_dir / bid).is_dir()]
    else:
        bundle_dirs = [p for p in bundles_dir.iterdir() if p.is_dir()]
        bundle_dirs.sort(key=lambda p: p.name)

    if args.limit and args.limit > 0:
        bundle_dirs = bundle_dirs[: args.limit]

    total = len(bundle_dirs)
    ok_count = 0
    bad = []

    for i, b in enumerate(bundle_dirs, 1):
        ok, msg = _check_bundle(b, run_self_check=bool(args.run_self_check))
        if ok:
            ok_count += 1
        else:
            bad.append((b.name, msg))
        if (i % 50) == 0 or i == total:
            print(f"progress: {i}/{total} ok={ok_count} bad={len(bad)}")

    if bad:
        print("\nFAILED bundles:")
        for bid, msg in bad[:200]:
            print(f"- {bid}: {msg}")
        if len(bad) > 200:
            print(f"... and {len(bad) - 200} more")

    print(f"\nSUMMARY: total={total} ok={ok_count} bad={len(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
