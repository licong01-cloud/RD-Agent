import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


WS_RE = re.compile(r"[^\s\"']*RD-Agent_workspace/[0-9a-f]{32}")
CONF_RE = re.compile(r"conf_[A-Za-z0-9_\.\-]+\.ya?ml")


@dataclass
class FileProbe:
    path: str
    exists: bool
    size: int | None = None
    mtime: float | None = None
    sha256: str | None = None


def _iter_log_files(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    files: list[Path] = []
    for p in log_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".log", ".txt", ".out", ".err"}:
            files.append(p)
    if not files:
        for p in log_dir.rglob("*"):
            if p.is_file() and p.stat().st_size > 0:
                files.append(p)
    files.sort(key=lambda x: (x.stat().st_mtime, str(x)))
    return files


def _read_text_best_effort(p: Path, max_bytes: int = 2_000_000) -> str:
    try:
        data = p.read_bytes()
    except Exception:
        return ""
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin1"):
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            continue
    return ""


def _sha256_file(p: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            remain = max_bytes
            while remain > 0:
                chunk = f.read(min(1024 * 1024, remain))
                if not chunk:
                    break
                h.update(chunk)
                remain -= len(chunk)
    return h.hexdigest()


def probe_file(path: Path, fast: bool = False) -> FileProbe:
    if not path.exists():
        return FileProbe(path=str(path), exists=False)
    st = path.stat()
    sha = None
    try:
        sha = _sha256_file(path, max_bytes=20 * 1024 * 1024 if fast else None)
    except Exception:
        sha = None
    return FileProbe(path=str(path), exists=True, size=st.st_size, mtime=st.st_mtime, sha256=sha)


def parquet_quick_stats(parquet_path: Path) -> dict[str, Any]:
    # Prefer pyarrow to avoid requiring pandas in diagnostic environments.
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(parquet_path)
        schema = pf.schema_arrow
        col_names = [str(n) for n in schema.names]
        num_rows = pf.metadata.num_rows if pf.metadata is not None else None
        num_cols = len(col_names)

        head_hash = None
        try:
            # Read a small head for fingerprinting; keep it cheap.
            table = pf.read_row_groups([0], columns=schema.names) if pf.num_row_groups > 0 else pf.read(columns=schema.names)
            head = table.slice(0, min(200, table.num_rows))
            head_hash = hashlib.sha256(head.to_pylist().__repr__().encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            head_hash = None

        shape = (num_rows, num_cols) if num_rows is not None else (None, num_cols)
        return {
            "shape": shape,
            "first_cols": col_names[:10],
            "head_hash": head_hash,
            "engine": "pyarrow",
        }
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(parquet_path)
        cols = [str(c) for c in list(df.columns)[:10]]
        head_hash = hashlib.sha256(pd.util.hash_pandas_object(df.head(200), index=True).values.tobytes()).hexdigest()
        return {
            "shape": tuple(df.shape),
            "first_cols": cols,
            "head_hash": head_hash,
            "engine": "pandas",
        }
    except Exception as e:
        return {"error": str(e)}


def scan_logs(log_dir: Path) -> dict[str, Any]:
    files = _iter_log_files(log_dir)
    ws_hits: dict[str, int] = {}
    conf_hits: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    error_lines: list[str] = []

    for f in files:
        text = _read_text_best_effort(f)
        if not text:
            continue
        wss = set(m.group(0).strip('"\'') for m in WS_RE.finditer(text))
        confs = set(m.group(0).strip('"\'') for m in CONF_RE.finditer(text))

        for w in wss:
            ws_hits[w] = ws_hits.get(w, 0) + 1
        for c in confs:
            conf_hits[c] = conf_hits.get(c, 0) + 1
        for w in wss:
            for c in confs:
                pairs[(w, c)] = pairs.get((w, c), 0) + 1

        for line in text.splitlines():
            low = line.lower()
            if any(k in low for k in ("traceback", "error", "exception", "factoremptyerror", "all nan", "empty dataframe")):
                s = line.strip()
                if s and len(error_lines) < 200:
                    error_lines.append(s)

    top_ws = sorted(ws_hits.items(), key=lambda x: (-x[1], x[0]))[:50]
    top_conf = sorted(conf_hits.items(), key=lambda x: (-x[1], x[0]))[:50]
    top_pairs = sorted(pairs.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:50]

    return {
        "log_dir": str(log_dir),
        "log_files": [str(p) for p in files[:200]],
        "workspaces_top": [{"workspace": w, "count": n} for w, n in top_ws],
        "confs_top": [{"conf": c, "count": n} for c, n in top_conf],
        "pairs_top": [{"workspace": w, "conf": c, "count": n} for (w, c), n in top_pairs],
        "error_lines_sample": error_lines[:80],
    }


def analyze_workspace(ws: Path, fast_hash: bool = False) -> dict[str, Any]:
    key_files = [
        "conf.yaml",
        "conf_baseline.yaml",
        "conf_combined_factors_dynamic.yaml",
        "conf_combined_factors.yaml",
        "conf_combined_factors_sota_model.yaml",
        "combined_factors_df.parquet",
        "qlib_res.csv",
        "ret.pkl",
    ]

    out: dict[str, Any] = {
        "workspace": str(ws),
        "exists": ws.exists(),
    }
    if not ws.exists():
        return out

    probes: list[dict[str, Any]] = []
    for rel in key_files:
        probes.append(probe_file(ws / rel, fast=fast_hash).__dict__)
    out["key_files"] = probes

    mlruns = ws / "mlruns"
    out["mlruns"] = {
        "path": str(mlruns),
        "exists": mlruns.exists(),
    }
    if mlruns.exists():
        try:
            out["mlruns"]["file_count"] = sum(1 for _ in mlruns.rglob("*") if _.is_file())
        except Exception:
            pass

    pq = ws / "combined_factors_df.parquet"
    if pq.exists():
        out["parquet_quick"] = parquet_quick_stats(pq)

    return out


def compare_workspaces(a: Path, b: Path) -> dict[str, Any]:
    a_info = analyze_workspace(a)
    b_info = analyze_workspace(b)
    diffs: list[dict[str, Any]] = []

    def _index(probes: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        m: dict[str, dict[str, Any]] = {}
        for p in probes:
            m[str(p.get("path"))] = p
        return m

    a_map = _index(a_info.get("key_files", []))
    b_map = _index(b_info.get("key_files", []))

    for rel in (
        "combined_factors_df.parquet",
        "qlib_res.csv",
        "ret.pkl",
        "conf_baseline.yaml",
        "conf_combined_factors_dynamic.yaml",
        "conf_combined_factors.yaml",
    ):
        pa = a / rel
        pb = b / rel
        da = a_map.get(str(pa), {})
        db = b_map.get(str(pb), {})
        diffs.append(
            {
                "file": rel,
                "a_exists": da.get("exists"),
                "b_exists": db.get("exists"),
                "a_size": da.get("size"),
                "b_size": db.get("size"),
                "a_sha256": da.get("sha256"),
                "b_sha256": db.get("sha256"),
                "same_sha256": (da.get("sha256") is not None and da.get("sha256") == db.get("sha256")),
            }
        )

    return {
        "workspace_a": a_info,
        "workspace_b": b_info,
        "diffs": diffs,
    }


def _guess_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--loop1-log", required=False)
    p.add_argument("--loop3-log", required=False)
    p.add_argument("--ws1", required=True)
    p.add_argument("--ws3", required=True)
    p.add_argument("--out", required=False)
    p.add_argument("--fast-hash", action="store_true")
    args = p.parse_args(argv)

    ws1 = Path(args.ws1)
    ws3 = Path(args.ws3)

    result: dict[str, Any] = {
        "ws_compare": compare_workspaces(ws1, ws3),
    }

    if args.loop1_log:
        result["loop1_log_scan"] = scan_logs(Path(args.loop1_log))
    if args.loop3_log:
        result["loop3_log_scan"] = scan_logs(Path(args.loop3_log))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
