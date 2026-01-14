import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_qlib_on_path(qlib_root: Path) -> None:
    """确保可以导入 qlib。

    优先使用已安装的 qlib（pip 安装的版本通常自带正确的版本元数据），
    只有在 import 失败时才将本地源码 qlib_root 加入 sys.path 作为降级方案，
    避免直接从源码目录导入时触发 setuptools_scm 版本探测错误。
    """

    try:
        import qlib  # type: ignore  # noqa: F401

        # 若已能正常导入，则直接返回
        return
    except Exception:
        pass

    if str(qlib_root) not in sys.path:
        sys.path.insert(0, str(qlib_root))


def _export_alpha158(qlib_root: Path, output: Path) -> int:
    _ensure_qlib_on_path(qlib_root)

    from qlib.contrib.data.loader import Alpha158DL  # type: ignore

    # 修正：不传递自定义 conf，使用 Alpha158DL 默认的 158 个因子配置
    fields, names = Alpha158DL.get_feature_config()

    factors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expr, name in zip(fields, names):
        if not isinstance(name, str) or not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        factors.append(
            {
                "name": name,
                "expression": str(expr),
                "source": "qlib_alpha158",
                "region": "cn",
                "tags": ["alpha158"],
            }
        )

    payload = {
        "version": "v1",
        "generated_at_utc": None,
        "source": "qlib_alpha158",
        "factors": factors,
        "config_path": "qlib.contrib.data.handler.Alpha158",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"Alpha158 因子个数: {len(factors)}")
    print(f"已写入: {output}")
    return len(factors)


def _export_alpha360(qlib_root: Path, output: Path) -> int:
    _ensure_qlib_on_path(qlib_root)

    from qlib.contrib.data.loader import Alpha360DL  # type: ignore

    fields, names = Alpha360DL.get_feature_config()

    factors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expr, name in zip(fields, names):
        if not isinstance(name, str) or not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        factors.append(
            {
                "name": name,
                "expression": str(expr),
                "source": "qlib_alpha360",
                "region": "cn",
                "tags": ["alpha360"],
            }
        )

    payload = {
        "version": "v1",
        "generated_at_utc": None,
        "source": "qlib_alpha360",
        "factors": factors,
        "config_path": "qlib.contrib.data.handler.Alpha360",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"Alpha360 因子个数: {len(factors)}")
    print(f"已写入: {output}")
    return len(factors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 qlib 源码导出 Alpha158 / Alpha360 全量因子 meta JSON (qlib_alpha158 / qlib_alpha360)",
    )
    parser.add_argument(
        "--qlib-root",
        default="qlib-main",
        help="本地 qlib 源码根目录（包含 qlib/ 子目录），默认: qlib-main",
    )
    parser.add_argument(
        "--alpha158-output",
        default="RDagentDB/aistock/alpha158_full_meta.json",
        help="Alpha158 meta 输出路径，默认: RDagentDB/aistock/alpha158_full_meta.json",
    )
    parser.add_argument(
        "--alpha360-output",
        default="RDagentDB/aistock/alpha360_meta.json",
        help="Alpha360 meta 输出路径，默认: RDagentDB/aistock/alpha360_meta.json",
    )
    parser.add_argument(
        "--combined-output",
        default="RDagentDB/aistock/alpha_all_meta.json",
        help=(
            "Alpha158+Alpha360 合并 meta 输出路径，默认: "
            "RDagentDB/aistock/alpha_all_meta.json（方便作为 --alpha-meta 直接使用）"
        ),
    )
    args = parser.parse_args()

    qlib_root = Path(args.qlib_root)
    if not qlib_root.exists():
        raise SystemExit(f"qlib 根目录不存在: {qlib_root}")

    alpha158_out = Path(args.alpha158_output)
    alpha360_out = Path(args.alpha360_output)
    combined_out = Path(args.combined_output)

    n158 = _export_alpha158(qlib_root, alpha158_out)
    n360 = _export_alpha360(qlib_root, alpha360_out)
    # 生成合并 meta，供 exporter 作为单一 external catalog 使用
    try:
        data158 = json.loads(alpha158_out.read_text(encoding="utf-8"))
        data360 = json.loads(alpha360_out.read_text(encoding="utf-8"))
        f158 = [f for f in (data158.get("factors") or []) if isinstance(f, dict)]
        f360 = [f for f in (data360.get("factors") or []) if isinstance(f, dict)]
        merged = {f.get("name"): f for f in f158}
        for f in f360:
            name = f.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name in merged:
                # 保留第一个定义，避免覆盖
                continue
            merged[name] = f

        payload = {
            "version": "v1",
            "generated_at_utc": None,
            "source": "qlib_alpha_all",
            "factors": list(merged.values()),
            "sources": {
                "alpha158_meta": str(alpha158_out),
                "alpha360_meta": str(alpha360_out),
            },
        }
        combined_out.parent.mkdir(parents=True, exist_ok=True)
        combined_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print("合并因子个数 (Alpha158+Alpha360 去重后):", len(payload["factors"]))
        print("已写入合并 meta:", combined_out)
    except Exception as e:  # noqa: F841
        print("警告: 生成合并 alpha_all_meta.json 失败，可单独使用 alpha158/alpha360 meta 文件。")

    print("导出完成: Alpha158 =", n158, ", Alpha360 =", n360)


if __name__ == "__main__":
    main()
