import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Hit:
    error_type: str
    path: str
    snippet: str


_ERROR_TYPE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b"
)


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".pkl", ".log", ".txt", ".json", ".yaml", ".yml", ".md"}:
            yield p


def _extract_hits_from_text(text: str, path: Path, max_snippets_per_file: int = 5) -> list[Hit]:
    hits: list[Hit] = []
    # Prefer explicit traceback blocks
    if "Traceback" in text:
        for m in _ERROR_TYPE_RE.finditer(text):
            et = m.group(1)
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 200)
            snippet = text[start:end].replace("\x00", "").replace("\n", " ")
            hits.append(Hit(et, str(path), snippet))
            if len(hits) >= max_snippets_per_file:
                break
        if not hits:
            hits.append(Hit("Traceback", str(path), "Traceback present but no explicit *Error/*Exception token found"))
        return hits

    # Otherwise search for error tokens
    for m in _ERROR_TYPE_RE.finditer(text):
        et = m.group(1)
        start = max(0, m.start() - 200)
        end = min(len(text), m.end() + 200)
        snippet = text[start:end].replace("\x00", "").replace("\n", " ")
        hits.append(Hit(et, str(path), snippet))
        if len(hits) >= max_snippets_per_file:
            break

    return hits


def _read_as_text(path: Path) -> str:
    # Logs in this repo are often pickle/binary; we do a best-effort bytes->text decode.
    data = path.read_bytes()
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")


def _render_markdown(counts: Counter, samples: dict[str, list[Hit]], root: Path) -> str:
    lines: list[str] = []
    lines.append(f"# RD-Agent log 错误汇总\n")
    lines.append(f"日志扫描目录：`{root}`\n")
    lines.append("说明：log 目录内多为 `.pkl`（pickle）二进制日志，本脚本不会反序列化（避免依赖缺失导致 import 失败），而是通过对二进制内容做 best-effort 解码并提取 `*Error/*Exception/Traceback` 关键词进行统计。\n")

    if not counts:
        lines.append("未在可扫描文件中发现明显的 `*Error/*Exception/Traceback` 关键词。\n")
        return "\n".join(lines)

    lines.append("## 错误类型统计（按出现次数降序）\n")
    lines.append("| ErrorType | Count | SampleFiles |")
    lines.append("| --- | ---: | --- |")
    for et, c in counts.most_common():
        sample_files = ", ".join({Path(h.path).name for h in samples.get(et, [])} or [])
        lines.append(f"| `{et}` | {c} | {sample_files} |")

    lines.append("\n## 典型片段（每类最多 3 条）\n")
    for et, _ in counts.most_common():
        lines.append(f"### `{et}`\n")
        for h in samples.get(et, [])[:3]:
            lines.append(f"- **File**: `{h.path}`")
            snippet = (h.snippet or "").replace("\x00", "")
            snippet = re.sub(r"\s+", " ", snippet).strip()
            lines.append(f"  - **Snippet**: `{snippet}`")
        lines.append("")

    out = "\n".join(lines)
    return out.replace("\x00", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RD-Agent log errors by scanning log directory for *Error/*Exception tokens.")
    parser.add_argument("--log-root", default=str(Path(__file__).resolve().parents[1] / "log"), help="Log directory to scan")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "log_error_summary.md"), help="Output markdown path")
    args = parser.parse_args()

    root = Path(args.log_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    samples: dict[str, list[Hit]] = defaultdict(list)

    for p in _iter_files(root):
        try:
            text = _read_as_text(p)
        except Exception:
            continue

        hits = _extract_hits_from_text(text, p)
        for h in hits:
            counts[h.error_type] += 1
            if len(samples[h.error_type]) < 10:
                samples[h.error_type].append(h)

    out_path.write_text(_render_markdown(counts, samples, root), encoding="utf-8")

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Error types: {len(counts)}")


if __name__ == "__main__":
    main()
