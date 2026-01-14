import json
import shutil
import uuid
import sqlite3
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from rdagent.log import rdagent_logger as logger
from rdagent.utils.registry.sqlite_registry import get_registry


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    # candidates can be relpaths like workspaces/{workspace_id}/conf.yaml
    # prefer matching by basename but return original relpath
    lower_map = {Path(c).name.lower(): c for c in candidates}
    for p in prefer:
        key = p.lower()
        if key in lower_map:
            return lower_map[key]
    # fallback: first yaml/yml
    return sorted(candidates)[0]


def _write_bundle_manifest(
    *,
    bundle_root: Path,
    asset_bundle_id: str,
    task_run_id: str,
    loop_id: int,
    primary_workspace_id: str,
    source_workspace_path: str,
    factor_entry_relpath: str,
    model_weight_relpath: str,
    config_relpath: str | None,
    log_dir: str | None = None,
    log_uri: str | None = None,
) -> None:
    manifest = {
        "schema_version": 1,
        "asset_bundle_id": asset_bundle_id,
        "task_run_id": task_run_id,
        "loop_id": loop_id,
        "generated_at_utc": _utc_now_iso(),
        "primary_workspace_id": primary_workspace_id,
        "source_workspace_path": source_workspace_path,
        "log_dir": log_dir or "",
        "log_uri": log_uri or "",
        "primary_assets": {
            "factor_entry_relpath": factor_entry_relpath,
            "model_weight_relpath": model_weight_relpath,
            "config_relpath": config_relpath or "",
        },
    }

    (bundle_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_bundle_self_check(bundle_root: Path) -> None:
    script = """import json\nfrom pathlib import Path\n\n\ndef main() -> int:\n    root = Path(__file__).resolve().parent\n    m = root / 'manifest.json'\n    if not m.exists():\n        print('FAIL: manifest.json not found')\n        return 2\n\n    try:\n        manifest = json.loads(m.read_text(encoding='utf-8'))\n    except Exception as e:\n        print(f'FAIL: manifest.json parse error: {e}')\n        return 3\n\n    if manifest.get('schema_version') != 1:\n        print(f"FAIL: schema_version must be 1, got {manifest.get('schema_version')}")\n        return 4\n\n    pa = manifest.get('primary_assets') or {}\n    factor_rel = (pa.get('factor_entry_relpath') or '').strip()\n    model_rel = (pa.get('model_weight_relpath') or '').strip()\n    config_rel = (pa.get('config_relpath') or '').strip()\n\n    if not factor_rel:\n        print('FAIL: primary_assets.factor_entry_relpath is empty')\n        return 5\n    if not model_rel:\n        print('FAIL: primary_assets.model_weight_relpath is empty')\n        return 6\n\n    factor_path = (root / factor_rel).resolve()\n    model_path = (root / model_rel).resolve()\n\n    if not factor_path.exists():\n        print(f'FAIL: factor entry not found: {factor_rel}')\n        return 7\n    if not model_path.exists():\n        print(f'FAIL: model weight not found: {model_rel}')\n        return 8\n\n    if config_rel:\n        config_path = (root / config_rel).resolve()\n        if not config_path.exists():\n            print(f'FAIL: config not found: {config_rel}')\n            return 9\n\n    print('OK: bundle manifest & primary assets validated')\n    return 0\n\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n"""

    (bundle_root / "self_check.py").write_text(script, encoding="utf-8")


def _safe_rglob(root: Path, pattern: str) -> list[Path]:
    out: list[Path] = []
    try:
        for p in root.rglob(pattern):
            try:
                _ = p.is_file()
            except Exception:
                continue
            out.append(p)
    except Exception:
        return []
    return out


def _safe_glob(root: Path, pattern: str) -> list[Path]:
    try:
        return list(root.glob(pattern))
    except Exception:
        return []


def _to_native_path(p_str: str) -> Path:
    """Convert path between WSL and Windows format based on current OS."""
    if not p_str:
        return Path()
    is_windows = os.name == "nt"
    # Normalize slashes first
    p_str = p_str.replace("/", os.sep).replace("\\", os.sep)
    
    if is_windows:
        # Handle WSL path: /mnt/f/... -> F:\...
        # Case-insensitive check for /mnt/
        lower_p = p_str.lower()
        if lower_p.startswith(f"{os.sep}mnt{os.sep}"):
            parts = p_str.split(os.sep)
            if len(parts) >= 3:
                drive = parts[2].upper()
                return Path(f"{drive}:\\") / Path(*parts[3:])
        # Handle already Windows path
        if len(p_str) > 1 and p_str[1] == ":":
            return Path(p_str)
    else:
        # Handle Windows path in WSL: F:\... -> /mnt/f/...
        if len(p_str) > 1 and p_str[1] == ":":
            drive = p_str[0].lower()
            rel = p_str[3:].replace("\\", "/")
            return Path(f"/mnt/{drive}") / rel
            
    return Path(p_str)


def _load_alpha_meta(db_path: Path) -> dict[str, dict[str, Any]]:
    """Helper to load Alpha factor metadata from external catalogs."""
    alpha_all_path = db_path.parent / "aistock" / "alpha_all_meta.json"
    alpha_map = {}
    if alpha_all_path.exists():
        try:
            data = json.loads(alpha_all_path.read_text(encoding="utf-8"))
            factors = data.get("factors") or []
            for f in factors:
                name = f.get("name")
                if name:
                    alpha_map[name] = f
        except Exception:
            pass
    return alpha_map


def solidify_loop_assets(task_run_id: str, loop_id: int, db_path: Path | None = None) -> str | None:
    """Extract core assets (YAML, PY, PKL) from workspaces of a loop and store them in a bundle.
    
    This implementation strictly follows Phase 3 solidification specs:
    1. Identifies all successful workspaces for the loop.
    2. Creates a unique asset_bundle_id (UUID).
    3. Copies only execution logic (config, code, weights), excluding backtest data.
    4. Persists structured factor/model metadata into the database.
    """
    reg = get_registry()
    if db_path:
        from rdagent.utils.registry.sqlite_registry import SQLiteRegistry, RegistryConfig
        reg = SQLiteRegistry(RegistryConfig(db_path=db_path))

    # 1. Get loop info and check if already solidified
    # Note: We assume the caller has ensured has_result=1
    
    # 2. Get all workspaces for this loop (authoritative after backfill: registry.sqlite)
    conn = sqlite3.connect(str(reg.config.db_path))
    conn.row_factory = sqlite3.Row
    try:
        ws_rows = conn.execute(
            """
            SELECT workspace_id, workspace_path, experiment_type, workspace_role
            FROM workspaces
            WHERE task_run_id = ? AND loop_id = ?
              AND (status IS NULL OR status IN ('finished', 'present'))
            """,
            (task_run_id, loop_id),
        ).fetchall()
    finally:
        conn.close()

    if not ws_rows:
        logger.warning(f"No workspaces found for loop {task_run_id}/{loop_id}")
        return None

    # 如果没有任何 experiment_workspace，但有 workspace，则放宽限制，将第一个 workspace 视为 experiment_workspace
    has_exp_ws = any(ws["workspace_role"] == "experiment_workspace" for ws in ws_rows)
    if not has_exp_ws and ws_rows:
        logger.info(f"No experiment_workspace found for loop {task_run_id}/{loop_id}, falling back to all workspaces.")
    # 在这种情况下，我们不再强制要求 experiment_workspace，而是处理所有可用的 workspace
        # 在这种情况下，我们不再强制要求 experiment_workspace，而是处理所有可用的 workspace
    
    asset_bundle_id = str(uuid.uuid4())
    repo_root = reg.config.db_path.parent.parent
    bundle_root = repo_root / "RDagentDB" / "production_bundles" / asset_bundle_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    workspaces_root = bundle_root / "workspaces"
    workspaces_root.mkdir(parents=True, exist_ok=True)

    # 简化目录结构：所有重要资产尽可能放在 bundle_root 根目录
    # 冲突时使用 workspace_id 作为前缀

    # Load Alpha metadata for enrichment
    alpha_map = _load_alpha_meta(reg.config.db_path)

    success_count = 0
    
    # 获取所有的 workspace，按照优先级排序：如果有 experiment_workspace 优先处理
    # 这样如果有冲突，主要实验的结果文件会保留原始名称
    ws_rows = sorted(ws_rows, key=lambda x: 0 if x["workspace_role"] == "experiment_workspace" else 1)

    primary_ws_id: str | None = None
    primary_ws_path_raw: str | None = None
    primary_factor_entry_rel: str | None = None
    primary_model_weight_rel: str | None = None
    primary_yaml_candidates: list[str] = []
    preferred_factor_ws_id: str | None = None
    preferred_model_ws_id: str | None = None
    factor_entry_candidates: list[str] = []
    factor_entry_preferred: str | None = None
    
    for ws in ws_rows:
        ws_id = ws["workspace_id"]
        ws_path_raw = ws["workspace_path"]
        ws_path = _to_native_path(ws_path_raw)

        ws_role = ws["workspace_role"]
        if preferred_factor_ws_id is None and ws_role == "experiment_workspace":
            preferred_factor_ws_id = ws_id
        if preferred_model_ws_id is None and ws_role == "experiment_workspace":
            preferred_model_ws_id = ws_id

        if primary_ws_id is None:
            primary_ws_id = ws_id
            primary_ws_path_raw = ws_path_raw
        
        if not ws_path.exists():
            logger.warning(f"Workspace path not found: {ws_path} (raw: {ws_path_raw})")
            continue

        logger.info(f"Scanning workspace {ws_id} at {ws_path}...")

        ws_bundle_root = workspaces_root / str(ws_id)
        ws_bundle_root.mkdir(parents=True, exist_ok=True)

        # A. Copy YAML configs (Keep them in bundle root for easier discovery)
        yaml_count = 0
        for yaml_file in _safe_glob(ws_path, "*.yaml"):
            target_name = yaml_file.name
            if (ws_bundle_root / target_name).exists():
                target_name = f"{ws_id}_{yaml_file.name}"
            try:
                shutil.copy2(yaml_file, ws_bundle_root / target_name)
                yaml_count += 1
                success_count += 1
                if ws_id == primary_ws_id:
                    primary_yaml_candidates.append(f"workspaces/{ws_id}/{target_name}")
            except Exception as e:
                logger.warning(f"Failed to copy yaml {yaml_file}: {e}")
        
        for yml_file in _safe_glob(ws_path, "*.yml"):
            target_name = yml_file.name
            if (ws_bundle_root / target_name).exists():
                target_name = f"{ws_id}_{yml_file.name}"
            try:
                shutil.copy2(yml_file, ws_bundle_root / target_name)
                yaml_count += 1
                success_count += 1
                if ws_id == primary_ws_id:
                    primary_yaml_candidates.append(f"workspaces/{ws_id}/{target_name}")
            except Exception as e:
                logger.warning(f"Failed to copy yml {yml_file}: {e}")

        # B. Copy Python implementation code and Model weights
        # 仅按需导出：所有 factor.py + model.pkl（或 mlruns 下 params.pkl 映射为 model.pkl）。
        py_count = 0
        pkl_count = 0

        # 1) factor entry python files: 支持多种命名（factor.py / *_factor.py / 名称包含 factor 的轻量 py）
        skip_dirs = {"mlruns", ".git", "__pycache__", "node_modules", ".venv"}
        factor_py_paths: list[Path] = []
        factor_py_paths.extend(_safe_rglob(ws_path, "factor.py"))
        factor_py_paths.extend(_safe_rglob(ws_path, "*_factor.py"))

        # fallback: any *factor*.py (keep it lightweight)
        if not factor_py_paths:
            for p in _safe_rglob(ws_path, "*.py"):
                if any(part in skip_dirs for part in p.parts):
                    continue
                name = p.name.lower()
                if "factor" not in name:
                    continue
                try:
                    if p.stat().st_size > 2 * 1024 * 1024:
                        continue
                except Exception:
                    continue
                factor_py_paths.append(p)

        # 去重 + 稳定排序（优先浅层）
        uniq: dict[str, Path] = {}
        for p in factor_py_paths:
            uniq[str(p.resolve())] = p
        factor_py_paths = sorted(uniq.values(), key=lambda p: (len(p.parts), str(p)))

        # If there is no factor code at all, DO NOT fallback to precomputed factors parquet for solidification.
        # Real-time inference in AIstock must compute factors from miniQMT + DB (get_history_window), not replay cached parquet.

        copied_factor_files: list[str] = []
        for f_path in factor_py_paths:
            if any(part in skip_dirs for part in f_path.parts):
                continue
            target_name = f_path.name
            if (ws_bundle_root / target_name).exists():
                target_name = f"{ws_id}_{f_path.name}"
            try:
                shutil.copy2(f_path, ws_bundle_root / target_name)
                py_count += 1
                success_count += 1
                logger.info(f"  [Factor] Flattened: {f_path.name} -> {target_name}")
                copied_factor_files.append(target_name)
                factor_entry_candidates.append(f"workspaces/{ws_id}/{target_name}")
                if ws_role == "experiment_workspace" and factor_entry_preferred is None:
                    factor_entry_preferred = f"workspaces/{ws_id}/{target_name}"
            except Exception as e:
                logger.warning(f"Failed to copy factor {f_path}: {e}")

        # 2) model weights: 优先 workspace 内 model.pkl；否则尝试 mlruns/**/params.pkl
        model_candidates: list[Path] = []
        model_candidates.extend([p for p in _safe_rglob(ws_path, "model.pkl") if "mlruns" not in p.parts])
        if not model_candidates:
            mlruns_dir = ws_path / "mlruns"
            if mlruns_dir.exists():
                model_candidates.extend(_safe_rglob(mlruns_dir, "params.pkl"))

        if model_candidates:
            # 选择较浅层的文件（通常是主产物），减少误选巨大历史文件
            model_candidates.sort(key=lambda p: (len(p.parts), p.stat().st_size if p.exists() else 0))
            source_file = model_candidates[0]
            target_name = "model.pkl"
            try:
                shutil.copy2(source_file, ws_bundle_root / target_name)
                pkl_count += 1
                success_count += 1
                logger.info(f"  [Weight] Flattened: {source_file.name} -> {target_name}")
                if primary_model_weight_rel is None:
                    primary_model_weight_rel = f"workspaces/{ws_id}/{target_name}"
                else:
                    if preferred_model_ws_id and ws_id == preferred_model_ws_id:
                        primary_model_weight_rel = f"workspaces/{ws_id}/{target_name}"
            except Exception as e:
                logger.warning(f"Failed to copy model weight {source_file}: {e}")

        if copied_factor_files:
            entry_name = "factor_entry.py"
            entry_path = ws_bundle_root / entry_name
            ordered = sorted(set(copied_factor_files))
            payload = "\n".join(
                [
                    "from __future__ import annotations",
                    "import importlib.util",
                    "from pathlib import Path",
                    "from typing import Any, Callable",
                    "import pandas as pd",
                    "",
                    "_WS_DIR = Path(__file__).resolve().parent",
                    f"_FACTOR_FILES = {ordered!r}",
                    "",
                    "def _load_module(file_name: str):",
                    "    p = (_WS_DIR / file_name).resolve()",
                    "    module_name = f'rd_ws_factor_{p.stem}'",
                    "    spec = importlib.util.spec_from_file_location(module_name, str(p))",
                    "    if spec is None or spec.loader is None:",
                    "        raise RuntimeError(f'cannot create spec for {p}')",
                    "    m = importlib.util.module_from_spec(spec)",
                    "    spec.loader.exec_module(m)",
                    "    return m",
                    "",
                    "def _resolve_compute(mod: Any) -> Callable[[pd.DataFrame], Any]:",
                    "    for attr in dir(mod):",
                    "        if attr.startswith('factor_'):",
                    "            fn = getattr(mod, attr)",
                    "            if callable(fn):",
                    "                return fn",
                    "    if hasattr(mod, 'Factor'):",
                    "        obj = mod.Factor()",
                    "        if hasattr(obj, 'compute') and callable(getattr(obj, 'compute')):",
                    "            return obj.compute",
                    "    if hasattr(mod, 'compute') and callable(getattr(mod, 'compute')):",
                    "        return getattr(mod, 'compute')",
                    "    raise RuntimeError('factor module missing callable entry')",
                    "",
                    "def compute(df_history: pd.DataFrame) -> pd.DataFrame:",
                    "    frames: list[pd.DataFrame] = []",
                    "    for file_name in _FACTOR_FILES:",
                    "        mod = _load_module(file_name)",
                    "        fn = _resolve_compute(mod)",
                    "        out = fn(df_history)",
                    "        if out is None:",
                    "            continue",
                    "        if isinstance(out, pd.Series):",
                    "            out = out.to_frame(name=file_name)",
                    "        if not isinstance(out, pd.DataFrame):",
                    "            raise RuntimeError(f'factor output must be DataFrame/Series, got {type(out)} from {file_name}')",
                    "        out = out.copy()",
                    "        frames.append(out)",
                    "    if not frames:",
                    "        raise RuntimeError('no factor outputs produced')",
                    "    return pd.concat(frames, axis=1)",
                    "",
                ]
            )
            try:
                entry_path.write_text(payload, encoding="utf-8")
                if ws_role == "experiment_workspace" and primary_factor_entry_rel is None:
                    primary_factor_entry_rel = f"workspaces/{ws_id}/{entry_name}"
                if ws_id == primary_ws_id and primary_factor_entry_rel is None:
                    primary_factor_entry_rel = f"workspaces/{ws_id}/{entry_name}"
            except Exception as e:
                logger.warning(f"Failed to write factor_entry.py for workspace {ws_id}: {e}")

        elif (ws_bundle_root / "conf_baseline.yaml").exists() or (ws_bundle_root / "conf.yaml").exists():
            entry_name = "factor_entry.py"
            entry_path = ws_bundle_root / entry_name
            payload = "\n".join(
                [
                    "from __future__ import annotations",
                    "",
                    "import numpy as np",
                    "import pandas as pd",
                    "",
                    "",
                    "def _resolve_level_name(df: pd.DataFrame, prefer: str) -> str | None:",
                    "    if not isinstance(df.index, pd.MultiIndex):",
                    "        return None",
                    "    if prefer in df.index.names:",
                    "        return prefer",
                    "    for cand in ('instrument', 'symbol', 'code'):",
                    "        if cand in df.index.names:",
                    "            return cand",
                    "    return None",
                    "",
                    "",
                    "def _rolling_residual_last(s: pd.Series, window: int) -> pd.Series:",
                    "    x = np.arange(window, dtype=float)",
                    "    x_mean = float(x.mean())",
                    "    x_var = float(((x - x_mean) ** 2).mean())",
                    "    x_last = float(x[-1])",
                    "",
                    "    def _fn(y: np.ndarray) -> float:",
                    "        y = y.astype(float)",
                    "        y_mean = float(y.mean())",
                    "        xy_mean = float((x * y).mean())",
                    "        cov = xy_mean - x_mean * y_mean",
                    "        slope = cov / (x_var + 1e-12)",
                    "        intercept = y_mean - slope * x_mean",
                    "        y_hat_last = slope * x_last + intercept",
                    "        return float(y[-1] - y_hat_last)",
                    "",
                    "    return s.rolling(window, min_periods=window).apply(_fn, raw=True)",
                    "",
                    "",
                    "def _rolling_rsquare(s: pd.Series, window: int) -> pd.Series:",
                    "    x = np.arange(window, dtype=float)",
                    "    x_mean = float(x.mean())",
                    "    x_var = float(((x - x_mean) ** 2).mean())",
                    "",
                    "    def _fn(y: np.ndarray) -> float:",
                    "        y = y.astype(float)",
                    "        y_mean = float(y.mean())",
                    "        xy_mean = float((x * y).mean())",
                    "        cov = xy_mean - x_mean * y_mean",
                    "        slope = cov / (x_var + 1e-12)",
                    "        intercept = y_mean - slope * x_mean",
                    "        y_hat = slope * x + intercept",
                    "        ss_res = float(((y - y_hat) ** 2).sum())",
                    "        ss_tot = float(((y - y_mean) ** 2).sum())",
                    "        if ss_tot <= 1e-12:",
                    "            return 0.0",
                    "        return float(1.0 - ss_res / ss_tot)",
                    "",
                    "    return s.rolling(window, min_periods=window).apply(_fn, raw=True)",
                    "",
                    "",
                    "def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:",
                    "    return a.rolling(window, min_periods=window).corr(b)",
                    "",
                    "",
                    "def compute(df_history: pd.DataFrame) -> pd.DataFrame:",
                    "    if df_history is None or df_history.empty:",
                    "        raise ValueError('df_history is empty')",
                    "",
                    "    inst_level = _resolve_level_name(df_history, 'instrument')",
                    "    if not inst_level:",
                    "        raise RuntimeError('df_history must be a MultiIndex DataFrame with instrument level')",
                    "",
                    "    df = df_history.copy()",
                    "    df.columns = [str(c).lower() for c in df.columns]",
                    "    need = {'open', 'high', 'low', 'close', 'volume'}",
                    "    missing = sorted([c for c in need if c not in df.columns])",
                    "    if missing:",
                    "        raise RuntimeError(f'df_history missing fields: {missing}')",
                    "",
                    "    def _calc_one(g: pd.DataFrame) -> pd.DataFrame:",
                    "        o = g['open'].astype(float)",
                    "        h = g['high'].astype(float)",
                    "        low = g['low'].astype(float)",
                    "        c = g['close'].astype(float)",
                    "        v = g['volume'].astype(float)",
                    "",
                    "        c_ref1 = c.shift(1)",
                    "        v_ref1 = v.shift(1)",
                    "        ret = c / (c_ref1 + 1e-12) - 1.0",
                    "        abs_ret_vol = ret.abs() * v",
                    "",
                    "        out = pd.DataFrame(index=g.index)",
                    "        out['RESI5'] = _rolling_residual_last(c, 5) / (c + 1e-12)",
                    "        out['WVMA5'] = abs_ret_vol.rolling(5, min_periods=5).std() / (abs_ret_vol.rolling(5, min_periods=5).mean() + 1e-12)",
                    "        out['RSQR5'] = _rolling_rsquare(c, 5)",
                    "        out['KLEN'] = (h - low) / (o + 1e-12)",
                    "        out['RSQR10'] = _rolling_rsquare(c, 10)",
                    "        out['CORR5'] = _rolling_corr(c, np.log(v + 1.0), 5)",
                    "        out['CORD5'] = _rolling_corr(c / (c_ref1 + 1e-12), np.log(v / (v_ref1 + 1e-12) + 1.0), 5)",
                    "        out['CORR10'] = _rolling_corr(c, np.log(v + 1.0), 10)",
                    "        out['ROC60'] = c.shift(60) / (c + 1e-12)",
                    "        out['RESI10'] = _rolling_residual_last(c, 10) / (c + 1e-12)",
                    "        out['VSTD5'] = v.rolling(5, min_periods=5).std() / (v + 1e-12)",
                    "        out['RSQR60'] = _rolling_rsquare(c, 60)",
                    "        out['CORR60'] = _rolling_corr(c, np.log(v + 1.0), 60)",
                    "        out['WVMA60'] = abs_ret_vol.rolling(60, min_periods=60).std() / (abs_ret_vol.rolling(60, min_periods=60).mean() + 1e-12)",
                    "        out['STD5'] = c.rolling(5, min_periods=5).std() / (c + 1e-12)",
                    "        out['RSQR20'] = _rolling_rsquare(c, 20)",
                    "        out['CORD60'] = _rolling_corr(c / (c_ref1 + 1e-12), np.log(v / (v_ref1 + 1e-12) + 1.0), 60)",
                    "        out['CORD10'] = _rolling_corr(c / (c_ref1 + 1e-12), np.log(v / (v_ref1 + 1e-12) + 1.0), 10)",
                    "        out['CORR20'] = _rolling_corr(c, np.log(v + 1.0), 20)",
                    "        out['KLOW'] = (np.minimum(o, c) - low) / (o + 1e-12)",
                    "        return out",
                    "",
                    "    features = df.groupby(level=inst_level, group_keys=False).apply(_calc_one)",
                    "    cols = ['RESI5','WVMA5','RSQR5','KLEN','RSQR10','CORR5','CORD5','CORR10','ROC60','RESI10','VSTD5','RSQR60','CORR60','WVMA60','STD5','RSQR20','CORD60','CORD10','CORR20','KLOW']",
                    "    return features.reindex(columns=cols)",
                    "",
                ]
            )
            try:
                entry_path.write_text(payload, encoding="utf-8")
                success_count += 1
                if ws_role == "experiment_workspace" and primary_factor_entry_rel is None:
                    primary_factor_entry_rel = f"workspaces/{ws_id}/{entry_name}"
                if ws_id == primary_ws_id and primary_factor_entry_rel is None:
                    primary_factor_entry_rel = f"workspaces/{ws_id}/{entry_name}"
            except Exception as e:
                logger.warning(f"Failed to write real-time Alpha158 factor_entry.py for workspace {ws_id}: {e}")

        logger.info(f"Workspace {ws_id} summary: {yaml_count} yamls, {py_count} factor.py files, {pkl_count} model weights.")

        # D. Persist Factor Registry
        factor_meta_path = ws_path / "factor_meta.json"
        factor_perf_path = ws_path / "factor_perf.json"
        
        if factor_meta_path.exists():
            try:
                meta = json.loads(factor_meta_path.read_text(encoding="utf-8"))
                perf = {}
                if factor_perf_path.exists():
                    perf = json.loads(factor_perf_path.read_text(encoding="utf-8"))
                
                factors = meta.get("factors") or []
                for f in factors:
                    f_name = f.get("name")
                    if not f_name:
                        continue
                    
                    # Extract this specific factor's performance from perf_json if possible
                    f_perf = {}
                    if perf:
                        # Best effort: find factor in perf list
                        for p_item in perf.get("factors") or []:
                            if p_item.get("name") == f_name:
                                f_perf = p_item
                                break
                        # Also include combination metrics if it's the main combination
                        if not f_perf and perf.get("combinations"):
                            f_perf = {"_combo_metrics": perf["combinations"][0]}

                    # REQ-FACTOR-P3-001: Enrich with Alpha metadata if it's an Alpha factor
                    expression = f.get("expression") or f.get("formula_hint") or ""
                    if not expression and f_name in alpha_map:
                        expression = alpha_map[f_name].get("expression") or ""

                    reg.upsert_factor_registry(
                        factor_name=f_name,
                        expression=expression,
                        performance_json=f_perf,
                        asset_bundle_id=asset_bundle_id,
                        workspace_id=ws_id,
                        task_run_id=task_run_id,
                        loop_id=loop_id
                    )
            except Exception as e:
                logger.warning(f"Failed to persist factor registry for {ws_id}: {e}")

        success_count += 1

    if success_count > 0:
        # Multi-factor support: generate an aggregated factor entry that executes all factor.py implementations
        # in this bundle and concatenates their outputs.
        # NOTE: This file becomes the manifest primary factor entry.
        def _write_aggregate_factor_entry(*, bundle_root: Path, factor_files: list[str], preferred: str | None) -> str:
            ordered: list[str] = []
            if preferred and preferred in factor_files:
                ordered.append(preferred)
            for f in sorted(set(factor_files)):
                if f == preferred:
                    continue
                ordered.append(f)

            entry_name = "factor_entry.py"
            entry_path = bundle_root / entry_name
            payload = "\n".join(
                [
                    "from __future__ import annotations",
                    "import importlib.util",
                    "from pathlib import Path",
                    "from typing import Any, Callable, Optional",
                    "import pandas as pd",
                    "",
                    "_BUNDLE_DIR = Path(__file__).resolve().parent",
                    f"_FACTOR_FILES = {ordered!r}",
                    "",
                    "def _load_module_from_file(file_name: str):",
                    "    p = (_BUNDLE_DIR / file_name).resolve()",
                    "    if not p.exists():",
                    "        raise FileNotFoundError(f'factor file not found: {file_name}')",
                    "    module_name = f'rd_bundle_factor_{p.stem}'",
                    "    spec = importlib.util.spec_from_file_location(module_name, str(p))",
                    "    if spec is None or spec.loader is None:",
                    "        raise RuntimeError(f'cannot create spec for {p}')",
                    "    m = importlib.util.module_from_spec(spec)",
                    "    spec.loader.exec_module(m)",
                    "    return m",
                    "",
                    "def _resolve_compute(mod: Any) -> Callable[[pd.DataFrame], Any]:",
                    "    # Prefer standardized wrapper if present",
                    "    for attr in dir(mod):",
                    "        if attr.startswith('factor_'):",
                    "            fn = getattr(mod, attr)",
                    "            if callable(fn):",
                    "                return fn",
                    "    if hasattr(mod, 'Factor'):",
                    "        obj = mod.Factor()",
                    "        if hasattr(obj, 'compute') and callable(getattr(obj, 'compute')):",
                    "            return obj.compute",
                    "    raise RuntimeError('factor module missing callable entry')",
                    "",
                    "def compute(df_history: pd.DataFrame) -> pd.DataFrame:",
                    "    frames: list[pd.DataFrame] = []",
                    "    for file_name in _FACTOR_FILES:",
                    "        mod = _load_module_from_file(file_name)",
                    "        fn = _resolve_compute(mod)",
                    "        out = fn(df_history)",
                    "        if out is None:",
                    "            continue",
                    "        if isinstance(out, pd.Series):",
                    "            out = out.to_frame(name=file_name)",
                    "        if not isinstance(out, pd.DataFrame):",
                    "            raise RuntimeError(f'factor output must be DataFrame/Series, got {type(out)} from {file_name}')",
                    "        # Keep original feature names to preserve model compatibility; de-duplicate only if needed.",
                    "        out = out.copy()",
                    "        cols = out.columns.astype(str).tolist()",
                    "        seen: dict[str, int] = {}",
                    "        new_cols: list[str] = []",
                    "        for c in cols:",
                    "            n = seen.get(c, 0)",
                    "            if n == 0:",
                    "                new_cols.append(c)",
                    "            else:",
                    "                new_cols.append(f'{c}__dup{n}')",
                    "            seen[c] = n + 1",
                    "        out.columns = new_cols",
                    "        frames.append(out)",
                    "    if not frames:",
                    "        raise RuntimeError('no factor outputs produced')",
                    "    return pd.concat(frames, axis=1)",
                    "",
                ]
            )
            entry_path.write_text(payload, encoding="utf-8")
            return entry_name

        # legacy aggregate entry is no longer used; per-workspace factor_entry.py is generated above.

        # Write manifest.json & self-check for AIstock. This is the hard contract.
        if primary_ws_id is None or primary_ws_path_raw is None:
            logger.warning("No primary workspace resolved; skip manifest generation")
        else:
            if not primary_factor_entry_rel:
                # Best-effort fallback: accept any workspace factor_entry.py
                any_entry = sorted([p for p in (workspaces_root).rglob("factor_entry.py")])
                if any_entry:
                    primary_factor_entry_rel = str(any_entry[0].relative_to(bundle_root)).replace("\\", "/")

            if not primary_model_weight_rel:
                any_model = sorted([p for p in (workspaces_root).rglob("model.pkl")])
                if any_model:
                    primary_model_weight_rel = str(any_model[0].relative_to(bundle_root)).replace("\\", "/")

            config_name = _pick_best_config_name(primary_yaml_candidates)

            if primary_factor_entry_rel and primary_model_weight_rel:
                factor_ok = (bundle_root / primary_factor_entry_rel).exists()
                model_ok = (bundle_root / primary_model_weight_rel).exists()
                if factor_ok and model_ok:
                    _write_bundle_manifest(
                        bundle_root=bundle_root,
                        asset_bundle_id=asset_bundle_id,
                        task_run_id=task_run_id,
                        loop_id=loop_id,
                        primary_workspace_id=primary_ws_id,
                        source_workspace_path=str(primary_ws_path_raw),
                        factor_entry_relpath=primary_factor_entry_rel,
                        model_weight_relpath=primary_model_weight_rel,
                        config_relpath=config_name,
                    )
                    _write_bundle_self_check(bundle_root)
                else:
                    logger.warning(
                        f"Skip solidification: missing primary assets in bundle. factor_ok={factor_ok}, model_ok={model_ok}"
                    )
                    if bundle_root.exists():
                        shutil.rmtree(bundle_root)
                    return None
            else:
                logger.warning(
                    f"Skip solidification: unresolved primary assets. factor_entry_rel={primary_factor_entry_rel}, model_weight_rel={primary_model_weight_rel}"
                )
                if bundle_root.exists():
                    shutil.rmtree(bundle_root)
                return None

        # Update loop status
        reg.upsert_loop(
            task_run_id=task_run_id,
            loop_id=loop_id,
            action=None, # Keep existing
            status="success",
            asset_bundle_id=asset_bundle_id,
            is_solidified=True,
            sync_status="pending"
        )
        logger.info(f"Loop {task_run_id}/{loop_id} solidified successfully with bundle {asset_bundle_id}")
        return asset_bundle_id
    else:
        # Cleanup empty bundle
        if bundle_root.exists():
            shutil.rmtree(bundle_root)
        return None
