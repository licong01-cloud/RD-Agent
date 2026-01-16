import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Hit:
    category: str
    path: str
    token: str
    snippet: str


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".pkl", ".log", ".txt"}:
            yield p


def _read_as_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")


TOKENS: dict[str, str] = {
    "qlib_dataset_not_supported": "NotImplementedError: This type of input is not supported",
    "pandas_multiindex_not_impl": "NotImplementedError: isna is not defined for MultiIndex",
    "runtime_shape_mismatch": "size mismatch",
    "runtime_matmul_mismatch": "mat1 and mat2 shapes cannot be multiplied",
    "runtime_expected_dim": "expected",
    "torch_einsum": "einsum():",
    "ts_window_params": "num_timesteps",
    "ts_step_len": "step_len",
    "handler_alpha158": "Alpha158",
    "handler_keyword": "handler",
    "empty_shape": "shape: (0,",
}

_RE_SPACE = re.compile(r"\s+")


FAILURE_CONTEXT_NEEDLES = [
    "traceback",
    "error - ",
    "exception has been raised",
    "failed to run",
    "raise ",
    "notimplementederror",
    "runtimeerror",
    "valueerror",
    "keyerror",
]


def _has_any(s: str, needles: list[str]) -> bool:
    return any(n in s for n in needles)


def _classify(token: str, snippet: str) -> str:
    """Classify a hit into model↔dataset related buckets.

    NOTE: This is heuristic-based; the goal is high-level triage.
    """

    s = snippet.lower()

    if token == "pandas_multiindex_not_impl":
        return "pandas_multiindex_notimplemented"

    if token == "qlib_dataset_not_supported" or "this type of input is not supported" in s:
        if _has_any(s, ["qlib/data/dataset", "dataset/__init__.py", "_get_row_col", "__getitem__"]):
            return "qlib_dataset_indexing_unsupported"
        return "qlib_not_supported_other"

    # Model input shape related (typical for 2D/3D mismatch)
    if token in {"runtime_shape_mismatch", "runtime_matmul_mismatch", "torch_einsum"}:
        return "model_input_shape_mismatch"
    if token == "runtime_expected_dim":
        # only treat as shape mismatch if batch/feature/timestep context exists
        if _has_any(s, ["num_features", "num_timesteps", "(batch", "batch_size", "3d", "2d"]):
            return "model_input_shape_mismatch"
        return "runtime_expected_other"

    # TimeSeries window params related
    if token in {"ts_window_params", "ts_step_len"}:
        return "ts_window_params_related"

    # Handler or empty data
    if token in {"handler_alpha158", "handler_keyword", "empty_shape"}:
        return "handler_or_empty_data_related"

    return "unknown"


def _is_failure_context(snippet: str) -> bool:
    s = snippet.lower()
    return any(n in s for n in FAILURE_CONTEXT_NEEDLES)


def _extract_hits(text: str, path: Path, max_hits_per_file: int = 20) -> list[Hit]:
    hits: list[Hit] = []
    lower = text.lower()

    for token, needle in TOKENS.items():
        needle_lower = needle.lower()
        if needle_lower not in lower:
            continue

        idx = 0
        while True:
            pos = lower.find(needle_lower, idx)
            if pos < 0:
                break
            start = max(0, pos - 500)
            end = min(len(text), pos + 800)
            snippet = text[start:end].replace("\x00", "").replace("\n", " ")
            snippet = _RE_SPACE.sub(" ", snippet).strip()

            # Avoid inflated counts from prompt/config text: only keep hits that appear in failure contexts.
            if not _is_failure_context(snippet):
                idx = pos + len(needle_lower)
                continue

            cat = _classify(token, snippet)
            hits.append(Hit(cat, str(path), token, snippet))
            idx = pos + len(needle_lower)
            if len(hits) >= max_hits_per_file:
                break

    return hits


def _render_md(counts: Counter, token_counts: dict[str, Counter], samples: dict[str, list[Hit]], root: Path) -> str:
    lines: list[str] = []
    lines.append("# 模型↔Dataset 相关错误统计（失败上下文过滤，近似）\n")
    lines.append(f"日志扫描目录：`{root}`\n")
    lines.append(
        "说明：log 目录内多为二进制 `.pkl`。为避免把 prompts/模板里的关键词当成错误，本报告仅统计出现在失败上下文（Traceback/ERROR/Exception/Failed to run 等）附近的命中，再基于关键字归类（用于定位高频根因，**并非 100% 精确**）。\n"
    )

    if not counts:
        lines.append("未发现 NotImplementedError。\n")
        return "\n".join(lines)

    lines.append("## 分类统计（按出现次数降序）\n")
    lines.append("| Category | Count | Notes |")
    lines.append("| --- | ---: | --- |")

    notes = {
        "qlib_dataset_indexing_unsupported": "Qlib Dataset 在 _get_row_col 等路径上遇到不支持的 idx/切片类型（最典型的 724 问题来源）",
        "qlib_not_supported_other": "出现 'This type of input is not supported'，但片段中缺少明确 dataset/_get_row_col 线索",
        "model_input_shape_mismatch": "模型输入维度/shape 不匹配（常见于 2D/3D 错配、einsum/matmul mismatch）",
        "ts_window_params_related": "出现 step_len/num_timesteps，可能与时序窗口参数/配置相关",
        "handler_or_empty_data_related": "出现 Alpha158/handler/empty/shape(0,*) 等，可能是数据为空或 handler 问题间接触发",
        "pandas_multiindex_notimplemented": "pandas 的 MultiIndex NotImplementedError（噪声项，与模型↔dataset 关系弱，应剔除后再看 qlib 相关问题）",
        "runtime_expected_other": "包含 'expected' 但缺少明显 shape 上下文",
        "unknown": "无法从片段关键字判断",
    }

    for cat, c in counts.most_common():
        lines.append(f"| `{cat}` | {c} | {notes.get(cat, '')} |")

    lines.append("\n## 触发 token 统计（用于校验分类覆盖面）\n")
    lines.append("| Category | Token | Count |")
    lines.append("| --- | --- | ---: |")
    for cat, c in counts.most_common():
        tc = token_counts.get(cat)
        if not tc:
            continue
        for token, cnt in tc.most_common():
            lines.append(f"| `{cat}` | `{token}` | {cnt} |")

    lines.append("\n## 每类样例（最多 3 条）\n")
    for cat, _ in counts.most_common():
        lines.append(f"### `{cat}`\n")
        for h in samples.get(cat, [])[:3]:
            lines.append(f"- **File**: `{h.path}`")
            lines.append(f"  - **Token**: `{h.token}`")
            lines.append(f"  - **Snippet**: `{h.snippet}`")
        lines.append("")

    return "\n".join(lines).replace("\x00", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and classify model↔dataset related failures by scanning RD-Agent log directory for multiple error tokens."
    )
    parser.add_argument("--log-root", default=str(Path(__file__).resolve().parents[1] / "log"), help="Log directory to scan")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "model_dataset_failure_breakdown.md"),
        help="Output markdown path",
    )
    args = parser.parse_args()

    root = Path(args.log_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    token_counts: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list[Hit]] = defaultdict(list)

    for p in _iter_files(root):
        try:
            text = _read_as_text(p)
        except Exception:
            continue

        hits = _extract_hits(text, p)
        # De-duplicate per file: only count one occurrence per (category, token) per file
        seen: set[tuple[str, str]] = set()
        for h in hits:
            key = (h.category, h.token)
            if key in seen:
                continue
            seen.add(key)
            counts[h.category] += 1
            token_counts[h.category][h.token] += 1
            if len(samples[h.category]) < 10:
                samples[h.category].append(h)

    out_path.write_text(_render_md(counts, token_counts, samples, root), encoding="utf-8")
    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Categories: {len(counts)}")


if __name__ == "__main__":
    main()
