from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess
import zipfile
import io
import os
from datetime import datetime, timezone
import sqlite3
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rdagent.utils.registry.sqlite_registry import get_registry


def _load_json(path: Path) -> Any | None:
    try:
        import json

        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_app() -> FastAPI:
    app = FastAPI(title="RD-Agent Results API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    reg = get_registry()

    def _db_path() -> Path:
        return Path(reg.config.db_path)

    def _repo_root() -> Path:
        return _db_path().parent.parent

    def _bundles_root() -> Path:
        return _repo_root() / "RDagentDB" / "production_bundles"

    def _log_root() -> Path:
        return _repo_root() / "log"

    def _workspace_root() -> Path:
        return _repo_root() / "git_ignore_folder" / "RD-Agent_workspace"

    def _open_registry_db() -> sqlite3.Connection:
        conn = sqlite3.connect(str(_db_path()))
        conn.row_factory = sqlite3.Row
        return conn

    _hex32_re = re.compile(r"\b[0-9a-f]{32}\b", flags=re.IGNORECASE)
    _ws_re = re.compile(r"RD-Agent_workspace/([0-9a-f]{32})", flags=re.IGNORECASE)

    def _find_task_run_id_in_registry(*, task_id: str) -> str | None:
        tid = str(task_id).strip()
        if not tid:
            return None
        with _open_registry_db() as conn:
            row = conn.execute(
                """
                SELECT task_run_id
                FROM task_runs
                WHERE log_trace_path LIKE ? OR log_trace_path LIKE ?
                ORDER BY updated_at_utc DESC
                LIMIT 1
                """,
                (f"%/log/{tid}%", f"%\\log\\{tid}%"),
            ).fetchone()
            if row is None:
                return None
            v = row["task_run_id"]
            return str(v).lower() if v else None

    def _list_workspace_ids_from_registry(*, task_run_id: str) -> list[str]:
        trid = str(task_run_id).strip().lower()
        if not trid:
            return []
        out: list[str] = []
        with _open_registry_db() as conn:
            cur = conn.execute(
                """
                SELECT workspace_id
                FROM loops
                WHERE task_run_id = ? AND workspace_id IS NOT NULL AND workspace_id != ''
                ORDER BY loop_id DESC
                LIMIT 10
                """,
                (trid,),
            )
            rows = cur.fetchall() or []
            for r in rows:
                ws = (r["workspace_id"] if isinstance(r, sqlite3.Row) else r[0])
                if ws:
                    out.append(str(ws).lower())
        seen = set()
        uniq: list[str] = []
        for ws in out:
            if ws in seen:
                continue
            seen.add(ws)
            uniq.append(ws)
        return uniq

    def _iter_recent_log_tasks(*, limit: int) -> list[dict[str, Any]]:
        root = _log_root()
        if not root.exists():
            return []

        items: list[tuple[float, str]] = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            items.append((mtime, entry.name))

        items.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for mtime, name in items[: int(limit)]:
            out.append({"task_id": name, "updated_at_utc": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()})
        return out

    def _extract_task_run_id_from_log_dir(*, task_id: str) -> str | None:
        log_dir = _log_root() / str(task_id)
        if not log_dir.exists() or not log_dir.is_dir():
            return None

        candidates: set[str] = set()
        max_files = 200
        max_read_bytes = 256 * 1024
        scanned = 0
        for p in log_dir.rglob("*"):
            if not p.is_file():
                continue
            scanned += 1
            if scanned > max_files:
                break
            # 只读前 2MB，避免扫爆内存
            try:
                bs = p.read_bytes()[:max_read_bytes]
            except OSError:
                continue

            try:
                text = bs.decode("utf-8", errors="ignore")
            except Exception:
                continue
            for m in _hex32_re.finditer(text):
                candidates.add(m.group(0).lower())

        if not candidates:
            return None

        with _open_registry_db() as conn:
            for h in candidates:
                row = conn.execute(
                    "SELECT 1 FROM loops WHERE task_run_id = ? LIMIT 1",
                    (h,),
                ).fetchone()
                if row is not None:
                    return h
        return None

    def _extract_workspace_ids_from_log_dir(*, task_id: str) -> list[str]:
        log_dir = _log_root() / str(task_id)
        if not log_dir.exists() or not log_dir.is_dir():
            return []

        found: set[str] = set()
        max_files = 200
        max_read_bytes = 256 * 1024
        scanned = 0
        for p in log_dir.rglob("*"):
            if not p.is_file():
                continue
            scanned += 1
            if scanned > max_files:
                break
            try:
                bs = p.read_bytes()[:max_read_bytes]
            except OSError:
                continue
            try:
                text = bs.decode("utf-8", errors="ignore")
            except Exception:
                continue
            for m in _ws_re.finditer(text):
                found.add(m.group(1).lower())

        # 过滤掉不存在的 workspace
        ws_root = _workspace_root()
        out: list[str] = []
        for ws in sorted(found):
            if (ws_root / ws).exists():
                out.append(ws)
        return out

    def _get_latest_tasks(*, limit: int) -> list[dict[str, Any]]:
        # AIstock 的 task_id 来自 log/<时间戳目录名>，这里直接以 log 为权威来源
        return _iter_recent_log_tasks(limit=int(limit))

    def _resolve_bundle_for_task(*, task_id: str) -> dict[str, Any]:
        tid = str(task_id).strip()
        if not tid:
            raise HTTPException(status_code=400, detail="task_id 为空")

        # 兼容两种输入：
        # - 32位 hex：视为 task_run_id
        # - 时间戳目录名：先从 log 中解析出 task_run_id
        task_run_id = tid.lower()
        if not _hex32_re.fullmatch(task_run_id):
            parsed = _find_task_run_id_in_registry(task_id=tid)
            if not parsed:
                parsed = _extract_task_run_id_from_log_dir(task_id=tid)
            if not parsed:
                # 兜底：部分 RD-Agent 版本/运行方式可能不会把 task_run_id 写入 log_trace_path，
                # 或者 log 里难以抽取 task_run_id。
                # 只要能从 log 中抽取出 workspace_id，就允许构建一个基于 workspace 的 task 视图，
                # 以支持 AIstock 同步与资产下载。
                ws_ids = _extract_workspace_ids_from_log_dir(task_id=tid)
                if ws_ids:
                    wd = _workspace_root() / ws_ids[0]
                    return {
                        "task_id": tid,
                        "task_run_id": "",
                        "loop_id": 0,
                        "asset_bundle_id": "",
                        "bundle_dir": None,
                        "bundle_manifest": None,
                        "workspace_id": ws_ids[0],
                        "workspace_dir": wd if wd.exists() else None,
                    }
                raise HTTPException(status_code=404, detail=f"task_id not found in logs/registry: {tid}")
            task_run_id = parsed

        with _open_registry_db() as conn:
            row = conn.execute(
                """
                SELECT task_run_id, loop_id, asset_bundle_id
                FROM loops
                WHERE task_run_id = ?
                ORDER BY loop_id DESC
                LIMIT 1
                """,
                (task_run_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"task_run_id not found in registry: {task_run_id}")

            asset_bundle_id = row["asset_bundle_id"]
            loop_id = int(row["loop_id"])

            bundle_dir: Path | None = None
            bundle_manifest: dict[str, Any] | None = None
            primary_workspace_id: str | None = None
            workspace_dir: Path | None = None

            if asset_bundle_id:
                bd = _bundles_root() / str(asset_bundle_id)
                if bd.exists():
                    bm = _load_json(bd / "manifest.json")
                    if isinstance(bm, dict):
                        bundle_dir = bd
                        bundle_manifest = bm
                        pw = (
                            bm.get("primary_workspace_id")
                            or bm.get("primary_workspace")
                            or bm.get("workspace_id")
                        )
                        if pw:
                            primary_workspace_id = str(pw)
                            wd = bd / "workspaces" / str(pw)
                            if wd.exists():
                                workspace_dir = wd

            # 若 bundle 未就绪（asset_bundle_id 为空或目录缺失），回退到 log 中抽取的 workspace
            if workspace_dir is None:
                ws_ids = _list_workspace_ids_from_registry(task_run_id=task_run_id)
                if not ws_ids:
                    ws_ids = _extract_workspace_ids_from_log_dir(task_id=tid)
                if ws_ids:
                    primary_workspace_id = ws_ids[0]
                    wd = _workspace_root() / ws_ids[0]
                    if wd.exists():
                        workspace_dir = wd

            return {
                "task_id": tid,
                "task_run_id": task_run_id,
                "loop_id": loop_id,
                "asset_bundle_id": str(asset_bundle_id) if asset_bundle_id else "",
                "bundle_dir": bundle_dir,
                "bundle_manifest": bundle_manifest,
                "workspace_id": str(primary_workspace_id) if primary_workspace_id else "",
                "workspace_dir": workspace_dir,
            }

    def _list_workspace_assets(*, workspace_dir: Path) -> list[dict[str, Any]]:
        allow_ext = {
            ".py",
            ".yaml",
            ".yml",
            ".json",
            ".csv",
            ".pkl",
            ".pt",
            ".txt",
            ".md",
            ".png",
        }
        max_bytes = 200 * 1024 * 1024

        out: list[dict[str, Any]] = []
        for p in workspace_dir.iterdir():
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix not in allow_ext:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                continue
            out.append({"name": p.name, "size": int(size)})

        out.sort(key=lambda x: (x["name"] != "factor_entry.py", x["name"]))
        return out

    def _pick_primary_assets(*, ws_id: str, files: list[dict[str, Any]]) -> dict[str, str]:
        names = [f["name"] for f in files]

        def rel(n: str) -> str:
            return f"workspaces/{ws_id}/{n}"

        factor = "factor_entry.py" if "factor_entry.py" in names else ""

        model = ""
        for cand in ("model.pkl", "model.pt", "model.pth"):
            if cand in names:
                model = cand
                break

        config = ""
        for cand in ("conf_baseline.yaml", "conf_baseline.yml", "conf_baseline_factors_model.yaml", "conf_sota_factors_model.yaml"):
            if cand in names:
                config = cand
                break
        if not config:
            for n in names:
                if n.lower().endswith((".yaml", ".yml")):
                    config = n
                    break

        return {
            "factor_entry_relpath": rel(factor) if factor else "",
            "model_weight_relpath": rel(model) if model else "",
            "config_relpath": rel(config) if config else "",
        }

    def _build_task_manifest(task_id: str) -> dict[str, Any]:
        info = _resolve_bundle_for_task(task_id=task_id)
        ws_id = info["workspace_id"]

        ws_dir = info.get("workspace_dir")
        assets: list[dict[str, Any]] = []
        primary_assets: dict[str, str] = {
            "factor_entry_relpath": "",
            "model_weight_relpath": "",
            "config_relpath": "",
        }

        if ws_id and isinstance(ws_dir, Path) and ws_dir.exists():
            files = _list_workspace_assets(workspace_dir=ws_dir)
            assets = [{"relpath": f"workspaces/{ws_id}/{f['name']}", "size": f["size"]} for f in files]
            primary_assets = _pick_primary_assets(ws_id=ws_id, files=files)

        return {
            "schema_version": 1,
            "task_id": info["task_id"],
            "task_run_id": info["task_run_id"],
            "loop_id": info["loop_id"],
            "asset_bundle_id": info["asset_bundle_id"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_assets": primary_assets,
            "assets": assets,
            "source": {
                "bundle_manifest": info.get("bundle_manifest")
            },
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tasks/latest")
    def tasks_latest(limit: int = 20) -> dict[str, Any]:
        items = _get_latest_tasks(limit=int(limit))
        return {"ok": True, "count": len(items), "tasks": items}

    @app.get("/tasks/{task_id}/summary")
    def task_summary(task_id: str) -> dict[str, Any]:
        info = _resolve_bundle_for_task(task_id=task_id)
        return {
            "ok": True,
            "task_id": info["task_id"],
            "task_run_id": info["task_run_id"],
            "loop_id": info["loop_id"],
            "asset_bundle_id": info["asset_bundle_id"],
            "workspace_id": info["workspace_id"],
            "bundle_manifest": info.get("bundle_manifest"),
        }

    @app.get("/tasks/{task_id}")
    def task_manifest(task_id: str) -> dict[str, Any]:
        return _build_task_manifest(task_id)

    @app.get("/tasks/{task_id}/assets")
    def task_asset_download(task_id: str, relpath: str = Query(...)) -> StreamingResponse:
        info = _resolve_bundle_for_task(task_id=task_id)
        rp = str(relpath or "").strip().replace("\\", "/")
        while rp.startswith("/"):
            rp = rp[1:]
        if not rp:
            raise HTTPException(status_code=400, detail="relpath 为空")

        # 优先从 bundle_dir 下取；否则回退 workspace_dir
        base_dir: Path | None = info.get("bundle_dir") or None
        if base_dir is None:
            ws_dir = info.get("workspace_dir")
            if isinstance(ws_dir, Path):
                base_dir = ws_dir

        if base_dir is None:
            raise HTTPException(status_code=404, detail="no bundle/workspace available for this task")

        # 兼容 relpath = workspaces/<ws_id>/xxx
        if rp.startswith("workspaces/"):
            parts = rp.split("/", 2)
            if len(parts) >= 3:
                rp = parts[2]

        target = (base_dir / rp).resolve()
        if not str(target).startswith(str(base_dir.resolve())):
            raise HTTPException(status_code=400, detail="invalid relpath")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"asset not found: {relpath}")

        def iter_file():
            with target.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            iter_file(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={target.name}"},
        )

    @app.get("/catalog/factors")
    def get_factor_catalog() -> Any:
        db_path = reg.config.db_path
        catalog_path = db_path.parent / "aistock" / "factor_catalog.json"
        payload = _load_json(catalog_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="factor_catalog.json not found")
        return payload

    @app.get("/catalog/strategies")
    def get_strategy_catalog() -> Any:
        db_path = reg.config.db_path
        catalog_path = db_path.parent / "aistock" / "strategy_catalog.json"
        payload = _load_json(catalog_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="strategy_catalog.json not found")
        return payload

    @app.get("/catalog/loops")
    def get_loop_catalog() -> Any:
        db_path = reg.config.db_path
        catalog_path = db_path.parent / "aistock" / "loop_catalog.json"
        payload = _load_json(catalog_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="loop_catalog.json not found")
        return payload

    @app.get("/catalog/models")
    def get_model_catalog() -> Any:
        """Return model_catalog.json for AIstock, if present.

        行为与其它 /catalog/* 路由一致：
        - 从 registry.sqlite 所在目录旁的 RDagentDB/aistock/model_catalog.json 读取；
        - 若文件不存在或无法解析，则返回 404。
        """

        db_path = reg.config.db_path
        catalog_path = db_path.parent / "aistock" / "model_catalog.json"
        payload = _load_json(catalog_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="model_catalog.json not found")
        return payload

    @app.get("/alpha158/meta")
    def get_alpha158_meta() -> Any:
        db_path = reg.config.db_path
        meta_path = db_path.parent / "aistock" / "alpha158_meta.json"
        payload = _load_json(meta_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="alpha158_meta.json not found")
        return payload

    @app.get("/alpha360/meta")
    def get_alpha360_meta() -> Any:
        db_path = reg.config.db_path
        meta_path = db_path.parent / "aistock" / "alpha360_meta.json"
        payload = _load_json(meta_path)
        if payload is None:
            raise HTTPException(status_code=404, detail="alpha360_meta.json not found")
        return payload

    @app.get("/catalog/incremental")
    def get_incremental_catalog(
        last_sync_time: str | None = Query(None, description="ISO-8601 timestamp of last sync"),
        limit: int = 100
    ) -> Any:
        """Fetch incremental loop and factor metadata since last_sync_time."""
        db_path = reg.config.db_path
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            # 1. Fetch solidified loops
            query = "SELECT * FROM loops WHERE is_solidified = 1"
            params = []
            if last_sync_time:
                # Use updated_at_utc for reliable incremental sync
                query += " AND updated_at_utc > ?"
                params.append(last_sync_time)
            
            query += " ORDER BY updated_at_utc ASC LIMIT ?"
            params.append(limit)
            
            loops = conn.execute(query, params).fetchall()
            
            results = []
            for l in loops:
                l_dict = dict(l)
                # Fetch associated factors from factor_registry
                factors = conn.execute(
                    "SELECT * FROM factor_registry WHERE asset_bundle_id = ?",
                    (l["asset_bundle_id"],)
                ).fetchall()
                l_dict["factors"] = [dict(f) for f in factors]
                results.append(l_dict)
                
            return {
                "version": "v1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "count": len(results),
                "loops": results
            }
        finally:
            conn.close()

    @app.get("/artifacts/bundle/{asset_bundle_id}")
    def download_bundle(asset_bundle_id: str) -> StreamingResponse:
        """Stream a zipped asset bundle for the given asset_bundle_id."""
        db_path = reg.config.db_path
        repo_root = db_path.parent.parent
        bundle_dir = repo_root / "RDagentDB" / "production_bundles" / asset_bundle_id
        
        if not bundle_dir.exists():
            raise HTTPException(status_code=404, detail=f"Bundle {asset_bundle_id} not found")

        def iter_zip():
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(bundle_dir):
                    for file in files:
                        file_path = Path(root) / file
                        zf.write(file_path, file_path.relative_to(bundle_dir))
            
            buffer.seek(0)
            yield from buffer

        return StreamingResponse(
            iter_zip(),
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": f"attachment; filename=bundle_{asset_bundle_id}.zip"}
        )

    @app.post("/ops/sync-confirm")
    def sync_confirm(payload: dict[str, str]) -> dict[str, str]:
        """AIstock confirms successful sync of an asset bundle."""
        asset_bundle_id = payload.get("asset_bundle_id")
        status = payload.get("status")
        
        if not asset_bundle_id or status != "synced":
            raise HTTPException(status_code=400, detail="Invalid payload")
            
        db_path = reg.config.db_path
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE loops SET sync_status = 'synced' WHERE asset_bundle_id = ?",
                (asset_bundle_id,)
            )
            conn.commit()
            return {"status": "ok", "message": f"Bundle {asset_bundle_id} marked as synced"}
        finally:
            conn.close()

    @app.post("/ops/materialize-and-refresh")
    def materialize_and_refresh() -> dict[str, Any]:
        """Trigger offline materialization for pending loops and refresh AIstock catalogs.

        This Ops endpoint is intended to be called from AIstock backend running on the
        same machine or trusted network. It performs the following steps synchronously:

        1. Run tools/backfill_registry_artifacts.py --mode materialize-pending (提取指标与元数据)
        2. Run tools/backfill_registry_artifacts.py --mode solidify-all (固化物理资产，支持全量 Alpha158 与扁平化结构)
        3. Regenerate factor/strategy/model/loop catalogs under RDagentDB/aistock/ (指纹去重)
        """

        db_path = reg.config.db_path
        root = db_path.parent
        project_root = Path(__file__).resolve().parents[2]
        tools_dir = project_root / "tools"

        def _run(cmd: list[str]) -> tuple[int, str]:
            try:
                # 使用 sys.executable 确保环境一致
                import sys
                
                # 如果是 Windows 环境，subprocess.run 默认支持
                proc = subprocess.run(
                    [sys.executable] + cmd if not cmd[0].startswith("python") else cmd,
                    cwd=str(project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                return proc.returncode, proc.stdout
            except Exception as e:
                return 1, f"failed to run {' '.join(cmd)}: {e}"

        results: list[dict[str, Any]] = []

        # 1) materialize-pending (Generate missing factor_meta.json/factor_perf.json)
        backfill_script = tools_dir / "backfill_registry_artifacts.py"
        code, out = _run([str(backfill_script), "--db", str(db_path), "--mode", "materialize-pending"])
        results.append({"step": "materialize_pending", "returncode": code, "output": out})

        # 2) solidify-all (Copy assets to flattened bundles and fill factor_registry)
        # 这里会触发最新的 solidification.py 逻辑，支持全量 Alpha158 提取和扁平化目录
        code, out = _run([str(backfill_script), "--db", str(db_path), "--mode", "solidify-all"])
        results.append({"step": "solidify_all", "returncode": code, "output": out})

        # 3) refresh catalogs (基于指纹去重的导出)
        catalog_scripts = [
            ("export_factor", tools_dir / "export_aistock_factor_catalog.py"),
            ("export_strategy", tools_dir / "export_aistock_strategy_catalog.py"),
            ("export_model", tools_dir / "export_aistock_model_catalog.py"),
            ("export_loop", tools_dir / "export_aistock_loop_catalog.py"),
        ]

        aistock_dir = root / "aistock"
        aistock_dir.mkdir(parents=True, exist_ok=True)
        alpha_metas = list(aistock_dir.glob("alpha*_meta.json"))

        for name, script in catalog_scripts:
            out_file = aistock_dir / f"{name.split('_')[1]}_catalog.json"
            cmd = [str(script), "--registry-sqlite", str(db_path), "--output", str(out_file)]
            if name == "export_factor":
                for am in alpha_metas:
                    cmd.extend(["--alpha-meta", str(am)])
            
            code, out = _run(cmd)
            results.append({"step": name, "returncode": code, "output": out})

        overall_ok = all(r.get("returncode") == 0 for r in results)
        if not overall_ok:
            raise HTTPException(status_code=500, detail={"steps": results})

        return {"status": "ok", "steps": results}

    @app.get("/factors/{name}")
    def get_factor(name: str) -> Any:
        # 简单实现：在 factor_catalog 中按 name 查一条
        db_path = reg.config.db_path
        catalog_path = db_path.parent / "aistock" / "factor_catalog.json"
        payload = _load_json(catalog_path)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=404, detail="factor_catalog.json not found")
        factors = payload.get("factors") or []
        if not isinstance(factors, list):
            raise HTTPException(status_code=404, detail="invalid factor_catalog format")
        for f in factors:
            if isinstance(f, dict) and f.get("name") == name:
                return f
        raise HTTPException(status_code=404, detail="factor not found")

    @app.get("/loops/{task_run_id}/{loop_id}/artifacts")
    def get_loop_artifacts(task_run_id: str, loop_id: int) -> Any:
        # 直接查询 registry.artifacts / artifact_files 表，返回指定 loop 的汇总视图
        conn = None
        try:
            import sqlite3

            db_path = reg.config.db_path
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            cur = conn.execute(
                """
                SELECT a.artifact_id, a.artifact_type, a.name, a.status, a.primary_flag,
                       a.entry_path, a.summary_json,
                       f.file_id, f.path AS file_path, f.sha256, f.size_bytes, f.mtime_utc, f.kind
                FROM artifacts AS a
                LEFT JOIN artifact_files AS f ON a.artifact_id = f.artifact_id
                WHERE a.task_run_id = ? AND a.loop_id = ?
                ORDER BY a.artifact_type, a.name
                """,
                (task_run_id, int(loop_id)),
            )
            rows = cur.fetchall()
        except Exception as e:  # pragma: no cover - best effort
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=f"registry query failed: {e}")

        artifacts: dict[str, dict[str, Any]] = {}
        for r in rows:
            aid = str(r["artifact_id"])
            art = artifacts.get(aid)
            if art is None:
                art = {
                    "artifact_id": aid,
                    "artifact_type": r["artifact_type"],
                    "name": r["name"],
                    "status": r["status"],
                    "primary_flag": bool(r["primary_flag"]),
                    "entry_path": r["entry_path"],
                    "summary": {},
                    "files": [],
                }
                try:
                    import json

                    summary_json = r["summary_json"]
                    if isinstance(summary_json, str) and summary_json:
                        art["summary"] = json.loads(summary_json)
                except Exception:
                    pass
                artifacts[aid] = art

            file_id = r["file_id"]
            if file_id is None:
                continue
            art["files"].append(
                {
                    "file_id": file_id,
                    "path": r["file_path"],
                    "sha256": r["sha256"],
                    "size_bytes": r["size_bytes"],
                    "mtime_utc": r["mtime_utc"],
                    "kind": r["kind"],
                }
            )

        return {"task_run_id": task_run_id, "loop_id": loop_id, "artifacts": list(artifacts.values())}

    return app
