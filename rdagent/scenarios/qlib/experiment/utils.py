import gc
import json
import os
import random
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq
import tables
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.log import rdagent_logger as logger

# 限制 pandas 警告
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


def _cleanup_memory() -> None:
    """清理内存，释放 pandas 和 Qlib 缓存"""
    try:
        gc.collect()
        # 清理 pandas 缓存
        pd.options.mode.chained_assignment = None
    except Exception:
        pass


def _to_unix_path(p: Path) -> Path:
    s = str(p)
    # Windows path like F:\... or F:/...
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        # Convert backslashes to forward slashes for consistency if needed
        tail = s[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}{tail}")
    return p


def _candidate_schema_paths_for_file(p: Path) -> list[Path]:
    # Prefer schema colocated with the data file, then snapshot metadata/schemas,
    # then governance folder schemas.
    stem = p.stem

    cands: list[Path] = []

    # 1) colocated with the file
    cands.append(p.with_name(f"{stem}_schema.csv"))
    cands.append(p.with_name(f"{stem}_schema.json"))

    # 2) snapshot metadata/schemas (walk up a few levels)
    cur = p.parent
    for _ in range(6):
        cands.append(cur / "metadata" / "schemas" / f"{stem}_schema.csv")
        cands.append(cur / "metadata" / "schemas" / f"{stem}_schema.json")
        cur = cur.parent

    # 3) governance schemas (env override, else default)
    gov_root = os.environ.get("AISTOCK_DATA_GOVERNANCE_DIR", "") or os.environ.get("AIstock_DATA_GOVERNANCE_DIR", "")
    if gov_root:
        gov = Path(gov_root)
        cands.append(gov / "schemas" / f"{stem}_schema.csv")
        cands.append(gov / "schemas" / f"{stem}_schema.json")
        cands.append(gov / "schemas" / "factors" / f"{stem}_schema.csv")
        cands.append(gov / "schemas" / "factors" / f"{stem}_schema.json")
        
        # Also try Unix-style mapping if we are in a mixed environment
        gov_unix = _to_unix_path(gov)
        if str(gov_unix) != str(gov):
            cands.append(gov_unix / "schemas" / f"{stem}_schema.csv")
            cands.append(gov_unix / "schemas" / f"{stem}_schema.json")
            cands.append(gov_unix / "schemas" / "factors" / f"{stem}_schema.csv")
            cands.append(gov_unix / "schemas" / "factors" / f"{stem}_schema.json")

    # 4) common default governance folder (both Windows and Unix styles)
    default_gov_win = Path("F:/Dev/AIstock/data_governance")
    cands.append(default_gov_win / "schemas" / f"{stem}_schema.csv")
    cands.append(default_gov_win / "schemas" / f"{stem}_schema.json")
    
    default_gov_unix = Path("/mnt/f/Dev/AIstock/data_governance")
    cands.append(default_gov_unix / "schemas" / f"{stem}_schema.csv")
    cands.append(default_gov_unix / "schemas" / f"{stem}_schema.json")
    cands.append(default_gov_unix / "schemas" / "factors" / f"{stem}_schema.csv")
    cands.append(default_gov_unix / "schemas" / "factors" / f"{stem}_schema.json")

    # De-dup while preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for x in cands:
        xs = str(x)
        if xs in seen:
            continue
        seen.add(xs)
        out.append(x)
    return out


def _is_nan(val: Any) -> bool:
    """Check if a value is NaN, None, or the string 'nan'."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    s = str(val).lower().strip()
    return s == "" or s == "nan"


def _load_schema_preview(schema_path: Path, data_path: Optional[Path] = None, max_rows: int = 60) -> str:
    try:
        h5_dtypes: dict[str, str] = {}
        if data_path and data_path.suffix.lower() == ".h5":
            h5_dtypes = _try_get_h5_columns_dtypes(data_path)

        if schema_path.suffix.lower() == ".csv":
            df = pd.read_csv(schema_path)
            lines = []
            for _, row in df.head(max_rows).iterrows():
                col = str(row.get("name", ""))
                dtype = row.get("dtype")
                meaning = (
                    row.get("meaning")
                    if "meaning" in df.columns
                    else row.get("meaning_cn")
                    if "meaning_cn" in df.columns
                    else row.get("meaning_zh")
                )
                meaning_str = "" if _is_nan(meaning) else str(meaning)
                unit = row.get("unit")
                unit_str = "" if _is_nan(unit) else str(unit)
                if col:
                    if _is_nan(dtype) and col in h5_dtypes:
                        dtype = h5_dtypes[col]
                    
                    if not _is_nan(dtype):
                        line = f"- {col} ({dtype})"
                    else:
                        line = f"- {col}"
                    
                    if unit_str:
                        line += f" [{unit_str}]"
                    if meaning_str:
                        line += f": {meaning_str}"
                    lines.append(line)
            return "\n".join(lines) + ("\n..." if len(df) > max_rows else "")

        if schema_path.suffix.lower() == ".json":
            obj = json.loads(schema_path.read_text(encoding="utf-8"))
            cols_list = obj.get("columns", []) if isinstance(obj, dict) else []
            lines = []
            for item in cols_list[:max_rows]:
                if not isinstance(item, dict):
                    continue
                col = str(item.get("name", ""))
                dtype = item.get("dtype")
                meaning = item.get("meaning") or item.get("meaning_cn") or item.get("meaning_zh")
                meaning_str = "" if _is_nan(meaning) else str(meaning)
                unit = item.get("unit")
                unit_str = "" if _is_nan(unit) else str(unit)
                if col:
                    if _is_nan(dtype) and col in h5_dtypes:
                        dtype = h5_dtypes[col]

                    if not _is_nan(dtype):
                        line = f"- {col} ({dtype})"
                    else:
                        line = f"- {col}"
                    
                    if unit_str:
                        line += f" [{unit_str}]"
                    if meaning_str:
                        line += f": {meaning_str}"
                    lines.append(line)
            return "\n".join(lines) + ("\n..." if isinstance(cols_list, list) and len(cols_list) > max_rows else "")
    except Exception:
        return ""

    return ""


def _try_get_h5_columns_dtypes(p: Path, key: str = "data") -> dict[str, str]:
    # Avoid loading full HDF5; best-effort extract columns and dtypes.
    cols: list[str] = []
    try:
        with pd.HDFStore(str(p), mode="r") as store:
            h5_key = f"/{key}" if f"/{key}" in store.keys() else key
            if h5_key not in store:
                keys = store.keys()
                if not keys:
                    return {}
                h5_key = keys[0]

            storer = store.get_storer(h5_key)
            axes = getattr(storer, "axes", None)
            if axes and len(axes) >= 1:
                try:
                    cols = [str(c) for c in list(axes[0])]
                except Exception:
                    pass
            if not cols:
                non_index_axes = getattr(storer, "non_index_axes", None)
                if non_index_axes:
                    try:
                        cols = [str(c) for c in list(non_index_axes[0][1])]
                    except Exception:
                        pass

            try:
                if getattr(storer, "is_table", False):
                    df_head = store.select(h5_key, start=0, stop=1)
                else:
                    df_head = pd.read_hdf(p, key=h5_key.strip("/"), start=0, stop=1)
                
                if df_head is not None:
                    # Filter out "nan" strings and real NaNs from column names and types
                    cleaned_dtypes = {}
                    for col in df_head.columns:
                        c_str = str(col)
                        if _is_nan(c_str):
                            continue
                        dt_str = str(df_head[col].dtype)
                        cleaned_dtypes[c_str] = dt_str if not _is_nan(dt_str) else ""
                    return cleaned_dtypes
            except Exception:
                if cols:
                    return {str(c): "" for c in cols if not _is_nan(c)}
    except Exception:
        # Fallback to PyTables for column extraction if pandas metadata extraction fails
        try:
            import tables
            h5 = tables.open_file(str(p), mode="r")
            try:
                # Standard pandas-saved HDF5 layout
                if f"/{key}" in h5:
                    node = h5.get_node(f"/{key}")
                    if hasattr(node, "axis0"):
                        cols = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in node.axis0[:]]
                    elif hasattr(node, "block0_items"):
                        cols = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in node.block0_items[:]]
            finally:
                h5.close()
        except Exception:
            pass
    
    return {c: "" for c in cols} if cols else {}


def generate_data_folder_from_qlib() -> None:
    template_path = Path(__file__).parent / "factor_data_template"
    # Run the Qlib backtest locally in the current conda environment instead of Docker
    result = subprocess.run(
        ["python", "generate.py"],
        cwd=str(template_path),
        capture_output=True,
        text=True,
    )
    execute_log = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to run generate.py to prepare Qlib factor data. Please check the following log:\n" + execute_log
        )

    assert (Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5").exists(), (
        "daily_pv_all.h5 is not generated. It means rdagent/scenarios/qlib/experiment/factor_data_template/generate.py is not executed correctly. Please check the log: \n"
        + execute_log
    )
    assert (Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5").exists(), (
        "daily_pv_debug.h5 is not generated. It means rdagent/scenarios/qlib/experiment/factor_data_template/generate.py is not executed correctly. Please check the log: \n"
        + execute_log
    )

    Path(FACTOR_COSTEER_SETTINGS.data_folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "README.md",
    )

    repo_root = Path(__file__).resolve().parents[4]
    repo_static_parquet = repo_root / "git_ignore_folder" / "factor_implementation_source_data" / "static_factors.parquet"
    repo_static_schema_csv = repo_static_parquet.with_name("static_factors_schema.csv")
    repo_static_schema_json = repo_static_parquet.with_name("static_factors_schema.json")

    factors_root = os.environ.get("AISTOCK_FACTORS_ROOT", "") or os.environ.get("AIstock_FACTORS_ROOT", "")
    if factors_root:
        factors_root_p = _to_unix_path(Path(factors_root)).resolve()
    else:
        factors_root_p = Path("/mnt/f/Dev/AIstock/factors")
    aistock_static_candidates = [
        factors_root_p / "combined_static_factors.parquet",
        factors_root_p / "static_factors.parquet",
    ]
    aistock_static_parquet = next((p for p in aistock_static_candidates if p.exists()), None)

    if repo_static_parquet.exists():
        shutil.copy(
            repo_static_parquet,
            Path(FACTOR_COSTEER_SETTINGS.data_folder) / "static_factors.parquet",
        )
        if repo_static_schema_csv.exists():
            shutil.copy(
                repo_static_schema_csv,
                Path(FACTOR_COSTEER_SETTINGS.data_folder) / "static_factors_schema.csv",
            )
        if repo_static_schema_json.exists():
            shutil.copy(
                repo_static_schema_json,
                Path(FACTOR_COSTEER_SETTINGS.data_folder) / "static_factors_schema.json",
            )
    elif aistock_static_parquet is not None:
        shutil.copy(
            aistock_static_parquet,
            Path(FACTOR_COSTEER_SETTINGS.data_folder) / "static_factors.parquet",
        )

    Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "README.md",
    )

    if repo_static_parquet.exists():
        shutil.copy(
            repo_static_parquet,
            Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "static_factors.parquet",
        )
        if repo_static_schema_csv.exists():
            shutil.copy(
                repo_static_schema_csv,
                Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "static_factors_schema.csv",
            )
        if repo_static_schema_json.exists():
            shutil.copy(
                repo_static_schema_json,
                Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "static_factors_schema.json",
            )
    elif aistock_static_parquet is not None:
        shutil.copy(
            aistock_static_parquet,
            Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "static_factors.parquet",
        )

    # Optional: copy governance schemas into the data folder so LLMs/agents can access
    # consistent field meanings/units without needing external paths.
    gov_dir = Path(os.environ.get("AISTOCK_DATA_GOVERNANCE_DIR", "") or "/mnt/f/Dev/AIstock/data_governance")
    gov_schemas = gov_dir / "schemas"
    if gov_schemas.exists():
        try:
            shutil.copytree(
                gov_schemas,
                Path(FACTOR_COSTEER_SETTINGS.data_folder) / "schemas",
                dirs_exist_ok=True,
            )
        except Exception as e:
            print(f"Failed to copy schemas to data_folder: {e}")
        try:
            shutil.copytree(
                gov_schemas,
                Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "schemas",
                dirs_exist_ok=True,
            )
        except Exception as e:
            print(f"Failed to copy schemas to data_folder_debug: {e}")


def _try_get_h5_index_samples(p: Path, n: int = 5) -> list[str]:
    """Try to get n index samples from H5 file."""
    try:
        # Try reading with pandas first for structured HDF5
        df_head = pd.read_hdf(p, key="data", start=0, stop=n)
        if isinstance(df_head.index, pd.MultiIndex):
            result = [str(idx) for idx in df_head.index.tolist()]
        else:
            result = [str(idx) for idx in df_head.index[:n].tolist()]
        # 清理 DataFrame 和内存
        del df_head
        _cleanup_memory()
        return result
    except ValueError as e:
        # Handle corrupted MultiIndex data
        if "inconsistent state" in str(e):
            logger.debug(f"H5 file contains corrupted MultiIndex data: {p}")
            logger.debug(f"  Error: {e}")
            # Try to read with tables directly to bypass pandas validation
            try:
                with tables.open_file(str(p), mode="r") as h5:
                    if "/data/axis1" in h5:
                        samples = h5.root.data.axis1[:n]
                        result = [s.decode() if isinstance(s, bytes) else str(s) for s in samples]
                        _cleanup_memory()
                        return result
            except Exception:
                pass
        logger.debug(f"Failed to read H5 index samples with pandas: {p}")
    except Exception as e:
        logger.debug(f"Failed to read H5 index samples with pandas: {p}")
        logger.debug(f"  Error: {e}")
        try:
            with tables.open_file(str(p), mode="r") as h5:
                # Common pandas HDF5 structure
                if "/data/axis1_level1" in h5:
                    samples = h5.root.data.axis1_level1[:n]
                    return [s.decode() if isinstance(s, bytes) else str(s) for s in samples]
                if "/data/table" in h5:
                    # Table format might have different structure
                    table = h5.root.data.table
                    # Try to find instrument-like column
                    instr_col = None
                    for col in table.colnames:
                        if "instrument" in col.lower() or "axis1" in col.lower():
                            instr_col = col
                            break
                    if instr_col:
                        return [row[instr_col].decode() if isinstance(row[instr_col], bytes) else str(row[instr_col]) 
                                for row in table.read(0, n)]
        except Exception as e2:
            logger.debug(f"Failed to read H5 index samples with tables: {e2}")
    return []


def get_file_desc(p: Path, variable_list: list[str] = [], exclude_columns: set[str] | None = None) -> str:
    """
    Get the description of a file based on its type.

    Parameters
    ----------
    p : Path
        The path of the file.
    variable_list : list[str]
        Optional list of relevant variables to highlight.
    exclude_columns : set[str] | None
        Column names to exclude from the description (used to avoid
        duplicating fields already described by other data files).

    Returns
    -------
    str
        The description of the file.
    """
    p = Path(p)

    JJ_TPL = Environment(undefined=StrictUndefined).from_string(
        """
# {{file_name}}

## File Type
{{type_desc}}

## Content Overview
{{content}}
"""
    )

    if p.name.endswith(".h5"):
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)

        df_info = "### Data Structure\n"
        df_info += "- Index: MultiIndex with levels ['datetime', 'instrument'] (convention)\n"
        index_samples = _try_get_h5_index_samples(p)
        if index_samples:
            df_info += f"- Index Samples (datetime, instrument): {index_samples}\n"

        df_info += "\n### Columns\n"
        columns = _try_get_h5_columns_dtypes(p)
        grouped_columns: dict[str, list[str]] = {}

        for col in columns:
            if _is_nan(col):
                continue
            if col.startswith("$"):
                prefix = col.split("_")[0] if "_" in col else col
                grouped_columns.setdefault(prefix, []).append(col)
            else:
                grouped_columns.setdefault("other", []).append(col)

        if variable_list:
            df_info += "#### Relevant Columns:\n"
            relevant_cols = [col for col in variable_list if col in columns and not _is_nan(col)]
            relevant_line = ", ".join(f"{col}: {columns[col]}" if not _is_nan(columns.get(col)) else f"{col}" for col in relevant_cols)
            df_info += (relevant_line if relevant_line else "None found") + "\n"
        else:
            df_info += "#### All Columns:\n"
            grouped_items = list(grouped_columns.items())
            random.shuffle(grouped_items)
            for prefix, cols_to_show in grouped_items:
                header = "Other Columns" if prefix == "other" else f"{prefix} Related Columns"
                df_info += f"\n#### {header}:\n"
                random.shuffle(cols_to_show)
                line = ", ".join(f"{col}: {columns[col]}" if not _is_nan(columns.get(col)) else f"{col}" for col in cols_to_show)
                df_info += line + "\n"

        # Attach schema (preferred) if available.
        schema_preview = ""
        for sp in _candidate_schema_paths_for_file(p):
            if sp.exists():
                schema_preview = _load_schema_preview(sp, data_path=p)
                if schema_preview:
                    df_info += f"\n### Schema (from {sp.name})\n" + schema_preview + "\n"
                    break

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="HDF5 Data File",
            content=df_info,
        )

    elif p.name.endswith(".pkl"):
        content = "### File Info\n"
        file_size_mb = p.stat().st_size / (1024 * 1024)
        content += f"- Size: {file_size_mb:.2f} MB\n"

        schema_preview = ""
        for sp in _candidate_schema_paths_for_file(p):
            if sp.exists():
                schema_preview = _load_schema_preview(sp, data_path=p)
                if schema_preview:
                    content += f"\n### Schema (from {sp.name})\n" + schema_preview + "\n"
                    break

        if not schema_preview:
            content += "\n### Notes\n- No schema found near this pickle file.\n"

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="Pickle Data File",
            content=content,
        )

    elif p.name.endswith(".parquet"):
        file_size_mb = p.stat().st_size / (1024 * 1024)
        df_info = "### File Info\n"
        df_info += f"- Size: {file_size_mb:.2f} MB\n"

        df_info += "\n### Data Structure\n"
        df_info += "- Index: MultiIndex with levels ['datetime', 'instrument'] (convention)\n"
        try:
            df_head = pd.read_parquet(p)
            if not df_head.empty:
                index_samples = [str(idx) for idx in df_head.index[:5].tolist()]
                df_info += f"- Index Samples (datetime, instrument): {index_samples}\n"
        except Exception:
            pass

        if p.name == "static_factors.parquet":
            df_info += "\n### Intended Usage\n"
            df_info += "- This is a joinable static factor table (e.g., fundamentals/liquidity/flow fields)\n"
            df_info += "- Typical index convention: (datetime, instrument)\n"
            df_info += "- Common column prefixes: db_ (daily_basic), mf_ (moneyflow), ae_ (precomputed factors)\n"

        schema_preview = ""
        for sp in _candidate_schema_paths_for_file(p):
            if sp.exists():
                schema_preview = _load_schema_preview(sp, data_path=p)
                if schema_preview:
                    df_info += f"\n### Schema (from {sp.name})\n" + schema_preview + "\n"
                    break

        if not schema_preview:
            try:
                pf = pq.ParquetFile(p)
                arrow_schema = pf.schema_arrow
                names = [arrow_schema.names[i] for i in range(len(arrow_schema.names))]
                if exclude_columns:
                    unique_names = [n for n in names if n not in exclude_columns]
                    df_info += f"\n### Columns (from parquet metadata, {len(names) - len(unique_names)} columns shared with H5 files omitted)\n"
                    names = unique_names
                else:
                    df_info += "\n### Columns (from parquet metadata)\n"
                shown = names[:120]
                df_info += ", ".join(shown) + ("\n..." if len(names) > 120 else "\n")
            except Exception as e:
                logger.debug(f"Failed to read parquet schema: {e}")
                df_info += "\n### Columns\n"
                df_info += "- Unable to read parquet schema (pyarrow not available or file unreadable).\n"

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="Parquet Data File",
            content=df_info,
        )

    elif p.name.endswith(".md"):
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="Markdown Documentation",
                content=content,
            )

    elif p.name.endswith(".csv"):
        try:
            if p.name.endswith("_schema.csv"):
                content = "### Schema Preview\n" + _load_schema_preview(p)
            else:
                df = pd.read_csv(p)
                content = ""
                content += "\n## Content Overview\n"
                content += f"- rows: {len(df)}\n"
                content += f"- columns: {list(df.columns)}\n"
                if "name" in df.columns:
                    content += "\n### Preview\n"
                    content += _load_schema_preview(p, max_rows=80)
                else:
                    content += "\n### Preview\n"
                    content += df.head(40).to_string(index=False) + "\n"
        except Exception as e:
            content = "\n## Content Overview\n"
            content += f"- failed to read csv: {repr(e)}\n"
        return JJ_TPL.render(
            file_name=p.name,
            type_desc="CSV Data File",
            content=content,
        )

    elif p.name.endswith(".json"):
        try:
            # Check if it's a schema file itself
            if p.name.endswith("_schema.json"):
                content = "### Schema Preview\n" + _load_schema_preview(p)
            else:
                obj = json.loads(p.read_text(encoding="utf-8"))
                content = "\n## Content Overview\n"
                if isinstance(obj, dict) and "columns" in obj:
                    cols_count = len(obj["columns"]) if isinstance(obj["columns"], list) else "N/A"
                    content += f"- columns entries: {cols_count}\n"
                    content += "\n### Preview\n"
                    content += _load_schema_preview(p, max_rows=80)
                else:
                    content += "\n### Preview\n"
                    content += str(obj)[:4000] + ("\n...\n" if len(str(obj)) > 4000 else "\n")
        except Exception as e:
            content = "\n## Content Overview\n"
            content += f"- failed to read json: {repr(e)}\n"
        return JJ_TPL.render(
            file_name=p.name,
            type_desc="JSON Data File",
            content=content,
        )

    else:
        raise NotImplementedError(
            f"file type {p.name} is not supported. Please implement its description function.",
        )


def get_data_folder_intro(fname_reg: str = ".*", flags: int = 0, variable_mapping: Optional[dict[str, list[str]]] = None) -> str:
    """
    Directly get the info of the data folder.
    It is for preparing prompting message.

    Parameters
    ----------
    fname_reg : str
        a regular expression to filter the file name.

    flags: str
        flags for re.match

    Returns
    -------
        str
            The description of the data folder.
    """

    debug_root = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
    needs_regen = (
        not Path(FACTOR_COSTEER_SETTINGS.data_folder).exists()
        or not debug_root.exists()
        or not (debug_root / "daily_pv.h5").exists()
        or not (debug_root / "static_factors.parquet").exists()
    )
    if needs_regen:
        # FIXME: (xiao) I think this is writing in a hard-coded way.
        # get data folder intro does not imply that we are generating the data folder.
        generate_data_folder_from_qlib()
    # Skip standalone schema files (*_schema.csv, *_schema.json) to avoid
    # duplicating field information that is already attached to the
    # corresponding data file (H5/parquet) via _candidate_schema_paths_for_file().
    _SCHEMA_SUFFIX_RE = re.compile(r"^.+_schema\.(csv|json)$", re.IGNORECASE)

    data_dir = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
    all_files = sorted(p for p in data_dir.iterdir() if p.is_file())

    # Two-pass approach: first process H5 files to collect their column names,
    # then process parquet files with those columns excluded to avoid duplication.
    h5_described_columns: set[str] = set()
    content_l = []

    # Pass 1: H5 and non-parquet files
    for p in all_files:
        if _SCHEMA_SUFFIX_RE.match(p.name):
            continue
        if p.suffix.lower() == ".parquet":
            continue
        if re.match(fname_reg, p.name, flags) is None:
            continue
        if variable_mapping:
            content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
        else:
            content_l.append(get_file_desc(p))
        # Collect H5 column names (strip $ prefix) for dedup
        if p.suffix.lower() == ".h5":
            cols = _try_get_h5_columns_dtypes(p)
            for col in cols:
                h5_described_columns.add(col.lstrip("$"))

    # Pass 2: Parquet files with H5-described columns excluded
    for p in all_files:
        if _SCHEMA_SUFFIX_RE.match(p.name):
            continue
        if p.suffix.lower() != ".parquet":
            continue
        if re.match(fname_reg, p.name, flags) is None:
            continue
        if variable_mapping:
            content_l.append(get_file_desc(p, variable_mapping.get(p.stem, []), exclude_columns=h5_described_columns or None))
        else:
            content_l.append(get_file_desc(p, exclude_columns=h5_described_columns or None))

    return "\n----------------- file splitter -------------\n".join(content_l)
