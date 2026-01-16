import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_bundles_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "RDagentDB" / "production_bundles"


def _pick_best_config_name(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    prefer = [
        "conf_baseline.yaml",
        "conf.yaml",
        "config.yaml",
        "config.yml",
        "conf_baseline.yml",
        "conf.yml",
    ]
    lower_map = {c.lower(): c for c in candidates}
    for p in prefer:
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    return sorted(candidates)[0]


def _write_self_check(bundle_root: Path) -> None:
    script = """import json\nfrom pathlib import Path\n\n\ndef main() -> int:\n    root = Path(__file__).resolve().parent\n    m = root / 'manifest.json'\n    if not m.exists():\n        print('FAIL: manifest.json not found')\n        return 2\n\n    try:\n        manifest = json.loads(m.read_text(encoding='utf-8'))\n    except Exception as e:\n        print(f'FAIL: manifest.json parse error: {e}')\n        return 3\n\n    if manifest.get('schema_version') != 1:\n        print(f"FAIL: schema_version must be 1, got {manifest.get('schema_version')}")\n        return 4\n\n    pa = manifest.get('primary_assets') or {}\n    factor_rel = (pa.get('factor_entry_relpath') or '').strip()\n    model_rel = (pa.get('model_weight_relpath') or '').strip()\n    config_rel = (pa.get('config_relpath') or '').strip()\n\n    if not factor_rel:\n        print('FAIL: primary_assets.factor_entry_relpath is empty')\n        return 5\n    if not model_rel:\n        print('FAIL: primary_assets.model_weight_relpath is empty')\n        return 6\n\n    factor_path = (root / factor_rel).resolve()\n    model_path = (root / model_rel).resolve()\n\n    if not factor_path.exists():\n        print(f'FAIL: factor entry not found: {factor_rel}')\n        return 7\n    if not model_path.exists():\n        print(f'FAIL: model weight not found: {model_rel}')\n        return 8\n\n    if config_rel:\n        config_path = (root / config_rel).resolve()\n        if not config_path.exists():\n            print(f'FAIL: config not found: {config_rel}')\n            return 9\n\n    print('OK: bundle manifest & primary assets validated')\n    return 0\n\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n"""
    (bundle_root / "self_check.py").write_text(script, encoding="utf-8")


def _find_primary_files(bundle_dir: Path) -> tuple[str | None, str | None, str | None]:
    files = [p.name for p in bundle_dir.iterdir() if p.is_file()]

    factor = None
    if "factor.py" in files:
        factor = "factor.py"
    else:
        candidates = sorted([n for n in files if n.endswith("_factor.py")])
        if candidates:
            factor = candidates[0]

    model = None
    if "model.pkl" in files:
        model = "model.pkl"
    else:
        candidates = sorted([n for n in files if n.endswith("_model.pkl")])
        if candidates:
            model = candidates[0]

    yaml_candidates = [n for n in files if n.lower().endswith((".yaml", ".yml"))]
    config = _pick_best_config_name(yaml_candidates)

    return factor, model, config


def _write_manifest(bundle_dir: Path, *, factor: str, model: str, config: str | None) -> None:
    asset_bundle_id = bundle_dir.name
    manifest = {
        "schema_version": 1,
        "asset_bundle_id": asset_bundle_id,
        "generated_at_utc": _utc_now_iso(),
        "primary_workspace_id": "",
        "source_workspace_path": "",
        "log_dir": "",
        "log_uri": "",
        "primary_assets": {
            "factor_entry_relpath": factor,
            "model_weight_relpath": model,
            "config_relpath": config or "",
        },
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles-dir", default=str(_default_bundles_dir()))
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    bundles_dir = Path(args.bundles_dir).expanduser().resolve()
    if not bundles_dir.exists():
        print(f"bundles dir not found: {bundles_dir}")
        return 2

    bundle_dirs = [p for p in bundles_dir.iterdir() if p.is_dir()]
    bundle_dirs.sort(key=lambda p: p.name)
    if args.limit and args.limit > 0:
        bundle_dirs = bundle_dirs[: args.limit]

    total = len(bundle_dirs)
    fixed = 0
    skipped = 0
    failed = 0

    for i, b in enumerate(bundle_dirs, 1):
        m = b / "manifest.json"
        if m.exists() and (not args.overwrite):
            skipped += 1
            continue

        factor, model, config = _find_primary_files(b)
        if not factor or not model:
            failed += 1
            print(f"[FAIL] {b.name}: cannot determine factor/model (factor={factor}, model={model})")
            continue

        _write_manifest(b, factor=factor, model=model, config=config)
        _write_self_check(b)
        fixed += 1

        if (i % 50) == 0 or i == total:
            print(f"progress: {i}/{total} fixed={fixed} skipped={skipped} failed={failed}")

    print(f"SUMMARY: total={total} fixed={fixed} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
