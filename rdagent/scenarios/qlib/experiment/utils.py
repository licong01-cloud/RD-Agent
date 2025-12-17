import random
import re
import shutil
import subprocess
import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS


def generate_data_folder_from_qlib():
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

    aistock_static_parquet = Path("/mnt/f/Dev/AIstock/factors/combined_static_factors.parquet")

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
    elif aistock_static_parquet.exists():
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
    elif aistock_static_parquet.exists():
        shutil.copy(
            aistock_static_parquet,
            Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "static_factors.parquet",
        )


def get_file_desc(p: Path, variable_list=[]) -> str:
    """
    Get the description of a file based on its type.

    Parameters
    ----------
    p : Path
        The path of the file.

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
        df = pd.read_hdf(p)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)

        df_info = "### Data Structure\n"
        df_info += (
            f"- Index: MultiIndex with levels {df.index.names}\n"
            if isinstance(df.index, pd.MultiIndex)
            else f"- Index: {df.index.name}\n"
        )

        df_info += "\n### Columns\n"
        columns = df.dtypes.to_dict()
        grouped_columns = {}

        for col in columns:
            if col.startswith("$"):
                prefix = col.split("_")[0] if "_" in col else col
                grouped_columns.setdefault(prefix, []).append(col)
            else:
                grouped_columns.setdefault("other", []).append(col)

        if variable_list:
            df_info += "#### Relevant Columns:\n"
            relevant_line = ", ".join(f"{col}: {columns[col]}" for col in variable_list if col in columns)
            df_info += relevant_line + "\n"
        else:
            df_info += "#### All Columns:\n"
            grouped_items = list(grouped_columns.items())
            random.shuffle(grouped_items)
            for prefix, cols in grouped_items:
                header = "Other Columns" if prefix == "other" else f"{prefix} Related Columns"
                df_info += f"\n#### {header}:\n"
                random.shuffle(cols)
                line = ", ".join(f"{col}: {columns[col]}" for col in cols)
                df_info += line + "\n"

        if "REPORT_PERIOD" in df.columns:
            one_instrument = df.index.get_level_values("instrument")[0]
            df_on_one_instrument = df.loc[pd.IndexSlice[:, one_instrument], ["REPORT_PERIOD"]]
            df_info += "\n### Sample Data\n"
            df_info += f"Showing data for instrument {one_instrument}:\n"
            df_info += str(df_on_one_instrument.head(5))

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="HDF5 Data File",
            content=df_info,
        )

    elif p.name.endswith(".parquet"):
        file_size_mb = p.stat().st_size / (1024 * 1024)
        content = "### File Info\n"
        content += f"- Size: {file_size_mb:.2f} MB\n"

        if p.name == "static_factors.parquet":
            content += "\n### Intended Usage\n"
            content += "- This is a joinable static factor table (e.g., fundamentals/liquidity/flow fields)\n"
            content += "- Typical index convention: (datetime, instrument)\n"
            content += "- Common column prefixes: db_ (daily_basic), mf_ (moneyflow), ae_ (precomputed factors)\n"

        schema_csv = p.with_name(f"{p.stem}_schema.csv")
        schema_json = p.with_name(f"{p.stem}_schema.json")

        schema_loaded = False
        if schema_csv.exists():
            try:
                schema_df = pd.read_csv(schema_csv)
                content += "\n### Schema (from .csv)\n"
                for _, row in schema_df.head(60).iterrows():
                    col = str(row.get("name", ""))
                    dtype = str(row.get("dtype", ""))
                    meaning = (
                        row.get("meaning")
                        if "meaning" in schema_df.columns
                        else row.get("meaning_cn")
                        if "meaning_cn" in schema_df.columns
                        else row.get("meaning_zh")
                    )
                    meaning = "" if meaning is None else str(meaning)
                    if col:
                        line = f"- {col} ({dtype})"
                        if meaning and meaning != "nan":
                            line += f": {meaning}"
                        content += line + "\n"
                schema_loaded = True
            except Exception:
                schema_loaded = False
        elif schema_json.exists():
            try:
                schema_obj = json.loads(schema_json.read_text(encoding="utf-8"))
                cols = schema_obj.get("columns", []) if isinstance(schema_obj, dict) else []
                content += "\n### Schema (from .json)\n"
                for item in cols[:60]:
                    if not isinstance(item, dict):
                        continue
                    col = str(item.get("name", ""))
                    dtype = str(item.get("dtype", ""))
                    meaning = item.get("meaning") or item.get("meaning_cn") or item.get("meaning_zh") or ""
                    meaning = str(meaning)
                    if col:
                        line = f"- {col} ({dtype})"
                        if meaning:
                            line += f": {meaning}"
                        content += line + "\n"
                schema_loaded = True
            except Exception:
                schema_loaded = False

        if not schema_loaded:
            try:
                import pyarrow.parquet as pq  # type: ignore[import-not-found]

                pf = pq.ParquetFile(p)
                arrow_schema = pf.schema_arrow
                names = [arrow_schema.names[i] for i in range(len(arrow_schema.names))]
                content += "\n### Columns (from parquet metadata)\n"
                shown = names[:120]
                content += ", ".join(shown) + ("\n..." if len(names) > 120 else "\n")
            except Exception:
                content += "\n### Columns\n"
                content += "- Unable to read parquet schema (pyarrow not available or file unreadable).\n"

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="Parquet Data File",
            content=content,
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
            df = pd.read_csv(p)
            content = ""
            content += f"\n## Content Overview\n"
            content += f"- rows: {len(df)}\n"
            content += f"- columns: {list(df.columns)}\n"
            if "name" in df.columns:
                shown = df.head(80)
                content += "\n### Preview (first 80 rows)\n"
                for _, row in shown.iterrows():
                    col = str(row.get("name", ""))
                    dtype = str(row.get("dtype", ""))
                    meaning = (
                        row.get("meaning")
                        if "meaning" in df.columns
                        else row.get("meaning_cn")
                        if "meaning_cn" in df.columns
                        else row.get("meaning_zh")
                    )
                    meaning = "" if meaning is None else str(meaning)
                    if col:
                        line = f"- {col} ({dtype})"
                        if meaning and meaning != "nan":
                            line += f": {meaning}"
                        content += line + "\n"
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
            obj = json.loads(p.read_text(encoding="utf-8"))
            content = "\n## Content Overview\n"
            if isinstance(obj, dict) and "columns" in obj:
                cols = obj.get("columns", [])
                content += f"- columns entries: {len(cols) if isinstance(cols, list) else 'N/A'}\n"
                if isinstance(cols, list):
                    content += "\n### Preview (first 80 columns)\n"
                    for item in cols[:80]:
                        if not isinstance(item, dict):
                            continue
                        col = str(item.get("name", ""))
                        dtype = str(item.get("dtype", ""))
                        meaning = item.get("meaning") or item.get("meaning_cn") or item.get("meaning_zh") or ""
                        meaning = str(meaning)
                        if col:
                            line = f"- {col} ({dtype})"
                            if meaning:
                                line += f": {meaning}"
                            content += line + "\n"
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


def get_data_folder_intro(fname_reg: str = ".*", flags=0, variable_mapping=None) -> str:
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

    if (
        not Path(FACTOR_COSTEER_SETTINGS.data_folder).exists()
        or not Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).exists()
    ):
        # FIXME: (xiao) I think this is writing in a hard-coded way.
        # get data folder intro does not imply that we are generating the data folder.
        generate_data_folder_from_qlib()
    content_l = []
    for p in Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- file splitter -------------\n".join(content_l)
