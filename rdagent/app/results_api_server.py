from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess
import zipfile
import io
import os
from datetime import datetime, timezone

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

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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
