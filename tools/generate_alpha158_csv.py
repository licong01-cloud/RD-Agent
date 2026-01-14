import argparse
import csv
import sys
import types
from pathlib import Path
from typing import Any, Iterable, List

from export_alpha158_meta import _load_yaml, _extract_alpha158_factors


FIELD_MAP: dict[str, str] = {
    "$close": "收盘价",
    "$open": "开盘价",
    "$high": "最高价",
    "$low": "最低价",
    "$vwap": "成交均价",
    "$volume": "成交量",
    "$amount": "成交额",
}


def _translate_field(token: str) -> str:
    return FIELD_MAP.get(token, token)


def _ensure_dependencies_stub() -> None:
    """注入最小的依赖 stub，避免导入 qlib 时因环境缺少包而报错。"""

    # 1. setuptools_scm stub
    if "setuptools_scm" not in sys.modules:
        mod = types.ModuleType("setuptools_scm")
        def get_version(*args: Any, **kwargs: Any) -> str:
            return "0.0.0"
        mod.get_version = get_version  # type: ignore
        sys.modules["setuptools_scm"] = mod

    # 2. pydantic_settings stub
    if "pydantic_settings" not in sys.modules:
        mod = types.ModuleType("pydantic_settings")
        class BaseSettings:
            def __init__(self, *args, **kwargs): pass
        class SettingsConfigDict:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return self
        mod.BaseSettings = BaseSettings  # type: ignore
        mod.SettingsConfigDict = SettingsConfigDict  # type: ignore
        sys.modules["pydantic_settings"] = mod

    # 3. pydantic stub
    if "pydantic" not in sys.modules:
        mod = types.ModuleType("pydantic")
        class BaseModel:
            def __init__(self, *args, **kwargs): pass
        class Field:
            def __init__(self, *args, **kwargs): pass
        mod.BaseModel = BaseModel  # type: ignore
        mod.Field = Field  # type: ignore
        sys.modules["pydantic"] = mod

    # 4. loguru stub
    if "loguru" not in sys.modules:
        mod = types.ModuleType("loguru")
        class Logger:
            def info(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def debug(self, *args, **kwargs): pass
        mod.logger = Logger()  # type: ignore
        sys.modules["loguru"] = mod

    # 5. joblib stub
    if "joblib" not in sys.modules:
        mod = types.ModuleType("joblib")
        mod.__package__ = "joblib"
        mod.__path__ = []
        def Parallel(*args, **kwargs):
            class _Parallel:
                def __call__(self, iterable): return list(iterable)
            return _Parallel()
        def delayed(func):
            return func
        mod.Parallel = Parallel  # type: ignore
        mod.delayed = delayed  # type: ignore
        sys.modules["joblib"] = mod

        # 增加子模块以防 qlib 内部 import joblib.xxx
        pb = types.ModuleType("joblib._parallel_backends")
        class MultiprocessingBackend:
            pass
        pb.MultiprocessingBackend = MultiprocessingBackend  # type: ignore
        sys.modules["joblib._parallel_backends"] = pb

    # 6. redis_lock stub
    if "redis_lock" not in sys.modules:
        mod = types.ModuleType("redis_lock")
        class Lock:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args, **kwargs): pass
        mod.Lock = Lock  # type: ignore
        sys.modules["redis_lock"] = mod

    # 7. redis stub
    if "redis" not in sys.modules:
        mod = types.ModuleType("redis")
        class Redis:
            def __init__(self, *args, **kwargs): pass
        mod.Redis = Redis  # type: ignore
        sys.modules["redis"] = mod

    # 8. fire stub
    if "fire" not in sys.modules:
        mod = types.ModuleType("fire")
        def Fire(*args, **kwargs): pass
        mod.Fire = Fire  # type: ignore
        sys.modules["fire"] = mod

    # 9. scipy stub
    if "scipy" not in sys.modules:
        mod = types.ModuleType("scipy")
        sys.modules["scipy"] = mod
        stats = types.ModuleType("scipy.stats")
        def percentileofscore(*args, **kwargs): return 0.0
        stats.percentileofscore = percentileofscore  # type: ignore
        sys.modules["scipy.stats"] = stats

    # 10. matplotlib stub
    if "matplotlib" not in sys.modules:
        mod = types.ModuleType("matplotlib")
        sys.modules["matplotlib"] = mod
        plt = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib.pyplot"] = plt

    # 11. qlib._libs stubs
    if "qlib.data._libs" not in sys.modules:
        mod = types.ModuleType("qlib.data._libs")
        mod.__package__ = "qlib.data"
        mod.__path__ = []
        sys.modules["qlib.data._libs"] = mod
        
        rolling = types.ModuleType("qlib.data._libs.rolling")
        # 补全 qlib 导入时可能需要的函数名
        for func in ["rolling_slope", "rolling_rsquared", "rolling_rsquare", "rolling_resi"]:
            setattr(rolling, func, lambda *args, **kwargs: 0.0)
        sys.modules["qlib.data._libs.rolling"] = rolling
        
        expand = types.ModuleType("qlib.data._libs.expand")
        sys.modules["qlib.data._libs.expand"] = expand

        expanding = types.ModuleType("qlib.data._libs.expanding")
        for func in ["expanding_slope", "expanding_rsquared", "expanding_rsquare", "expanding_resi"]:
            setattr(expanding, func, lambda *args, **kwargs: 0.0)
        sys.modules["qlib.data._libs.expanding"] = expanding

    # 12. qlib_internal_missing_stub
    # 如果后续还有报错，我们可以在这里动态增加更多 stub


def _describe_func(name: str, args: list[str]) -> str:
    """根据常见 qlib 运算符生成中文描述（简要模板）。"""
    lname = name.lower()
    # 注意：args 里通常是字符串表达的字段/数字
    if lname == "ref" and len(args) >= 2:
        return f"{_translate_field(args[0])}向前回看{args[1]}期的值"
    if lname == "mean" and len(args) >= 2:
        return f"过去{args[1]}期{_translate_field(args[0])}的均值"
    if lname == "std" and len(args) >= 2:
        return f"过去{args[1]}期{_translate_field(args[0])}的标准差"
    if lname == "delta" and len(args) >= 2:
        return f"{_translate_field(args[0])}在{args[1]}期的差分（当前减去{args[1]}期前）"
    if lname == "resi" and len(args) >= 2:
        return f"{_translate_field(args[0])}相对于其{args[1]}期均值的残差"
    if lname == "rank" and len(args) >= 1:
        return f"{_translate_field(args[0])}在截面上的排序值"
    if lname == "tsrank" and len(args) >= 2:
        return f"{_translate_field(args[0])}在时间窗口{args[1]}期内的时间序列排序值"
    # 未覆盖的函数，退化为原样
    arg_str = ", ".join(args)
    return f"{name}({arg_str})"


def _split_once(expr: str, op: str) -> tuple[str, str] | None:
    idx = expr.find(op)
    if idx <= 0:
        return None
    left = expr[:idx].strip()
    right = expr[idx + len(op) :].strip()
    if not left or not right:
        return None
    return left, right


def _generate_formula_hint(expression: str) -> str:
    """基于简化规则，将 qlib 表达式转成一段中文说明。"""
    expr = expression.strip()

    # 处理常见的二元运算（只拆最外层一次）
    for op, tmpl in [
        ("/", "{left}与{right}的比值"),
        ("-", "{left}减去{right}"),
        ("+", "{left}与{right}之和"),
    ]:
        parts = _split_once(expr, op)
        if parts is not None:
            left_raw, right_raw = parts
            left = _generate_formula_hint(left_raw)
            right = _generate_formula_hint(right_raw)
            return tmpl.format(left=left, right=right)

    # 简单函数调用：Name(arg1, arg2, ...)
    if "(" in expr and expr.endswith(")"):
        name, rest = expr.split("(", 1)
        args_str = rest[:-1]  # 去掉最后一个 ')'
        args = [a.strip() for a in args_str.split(",") if a.strip()]
        return _describe_func(name.strip(), args)

    # 纯字段
    if expr in FIELD_MAP:
        return _translate_field(expr)

    # 兜底
    return f"表达式：{expr}"


def _load_alpha158_from_yaml(conf_yaml: Path) -> list[dict[str, Any]]:
    """优先尝试从 YAML 中的 alpha158_config 提取因子定义。

    当给定的 YAML 不包含 alpha158_config（例如 benchmark 示例 workflow），将返回空列表，
    外层逻辑会自动退回到从 qlib Alpha158DL 源码获取默认因子列表。
    """

    conf = _load_yaml(conf_yaml)
    return _extract_alpha158_factors(conf)


def _load_alpha158_from_loader() -> list[dict[str, Any]]:
    """通过解析 qlib-main 源码获取 Alpha158 因子定义。
    
    由于导入 qlib 存在大量编译依赖（.so/.pyd）且无法通过 stub 完全绕过，
    改为直接模拟 qlib/contrib/data/loader.py 源码中的 get_feature_config 逻辑。
    """
    factors: list[dict[str, Any]] = []
    
    # 1. kbar 因子 (names: KMID, KLEN, ...)
    kbar_fields = [
        "($close-$open)/$open",
        "($high-$low)/$open",
        "($close-$open)/($high-$low+1e-12)",
        "($high-Greater($open, $close))/$open",
        "($high-Greater($open, $close))/($high-$low+1e-12)",
        "(Less($open, $close)-$low)/$open",
        "(Less($open, $close)-$low)/($high-$low+1e-12)",
        "(2*$close-$high-$low)/$open",
        "(2*$close-$high-$low)/($high-$low+1e-12)",
    ]
    kbar_names = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
    for f, n in zip(kbar_fields, kbar_names):
        factors.append({"name": n, "expression": f})

    # 2. price 因子 (names: OPEN0-4, HIGH0-4, ...)
    # 默认 windows: range(5), feature: ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
    price_features = ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
    for field in price_features:
        f_lower = field.lower()
        for d in range(5):
            name = f"{field}{d}"
            expr = f"Ref(${{f_lower}}, {{d}})/$close" if d != 0 else f"${{f_lower}}/$close"
            factors.append({"name": name, "expression": expr})

    # 3. volume 因子 (names: VOLUME0-4)
    # 默认 windows: range(5)
    for d in range(5):
        name = f"VOLUME{d}"
        expr = f"Ref($volume, {{d}})/($volume+1e-12)" if d != 0 else f"$volume/($volume+1e-12)"
        factors.append({"name": name, "expression": expr})

    # 提取 rolling 因子 (Alpha158DL.get_feature_config 默认开启了全部 29 种算子)
    # 默认 windows: [5, 10, 20, 30, 60]
    windows = [5, 10, 20, 30, 60]
    
    # 算子列表及对应的表达式模板
    rolling_ops = [
        ("ROC", "Ref($close, {d})/$close"),
        ("MA", "Mean($close, {d})/$close"),
        ("STD", "Std($close, {d})/$close"),
        ("BETA", "Slope($close, {d})/$close"),
        ("RSQR", "Rsquare($close, {d})"),
        ("RESI", "Resi($close, {d})/$close"),
        ("MAX", "Max($high, {d})/$close"),
        ("LOW", "Min($low, {d})/$close"),
        ("QTLU", "Quantile($close, {d}, 0.8)/$close"),
        ("QTLD", "Quantile($close, {d}, 0.2)/$close"),
        ("RANK", "Rank($close, {d})"),
        ("RSV", "($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)"),
        ("IMAX", "IdxMax($high, {d})/{d}"),
        ("IMIN", "IdxMin($low, {d})/{d}"),
        ("IMXD", "(IdxMax($high, {d})-IdxMin($low, {d}))/{d}"),
        ("CORR", "Corr($close, Log($volume+1), {d})"),
        ("CORD", "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})"),
        ("CNTP", "Mean($close>Ref($close, 1), {d})"),
        ("CNTN", "Mean($close<Ref($close, 1), {d})"),
        ("CNTD", "Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})"),
        ("SUMP", "Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"),
        ("SUMN", "Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"),
        ("SUMD", "(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"),
        ("VMA", "Mean($volume, {d})/($volume+1e-12)"),
        ("VSTD", "Std($volume, {d})/($volume+1e-12)"),
        ("WVMA", "Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"),
        ("VSUMP", "Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"),
        ("VSUMN", "Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"),
        ("VSUMD", "(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"),
    ]

    for op, expr_tmpl in rolling_ops:
        for d in windows:
            # 修正某些算子的 name 差异
            name_prefix = "MIN" if op == "LOW" else op
            factors.append({
                "name": f"{name_prefix}{d}",
                "expression": expr_tmpl.format(d=d)
            })

    # 标注来源
    for f in factors:
        f.update({"source": "qlib_alpha158", "region": "cn", "tags": ["alpha158"]})

    return factors


def _iter_rows(factors: Iterable[dict[str, Any]]):
    for item in factors:
        name = str(item.get("name", "")).strip()
        expr = str(item.get("expression", "")).strip()
        if not name or not expr:
            continue
        hint = _generate_formula_hint(expr)
        desc = f"{name}：{hint}"
        yield name, expr, desc


def run(conf_yaml: Path, output_csv: Path) -> None:
    # 先尝试从 YAML 中提取 alpha158_config，如果该 YAML 没有内嵌 alpha158_config（如 benchmark workflow），
    # 则回退到直接使用 qlib Alpha158DL 的默认特征配置生成全部因子列表。
    factors = _load_alpha158_from_yaml(conf_yaml)
    if not factors:
        factors = _load_alpha158_from_loader()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 带 BOM，方便 Windows / Excel 正常显示中文
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "expression", "description_cn"])
        for row in _iter_rows(factors):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 Qlib Alpha158 配置中提取因子名称和表达式，并生成附带中文说明的 CSV 文件。"
        )
    )
    parser.add_argument(
        "--conf-yaml",
        required=True,
        help=(
            "包含 alpha158_config 的 Qlib YAML 配置路径，例如 qlib-main/examples/benchmarks/"
            "LightGBM/workflow_config_lightgbm_Alpha158.yaml"
        ),
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="输出 CSV 路径（将以 UTF-8 带 BOM 编码写入，便于 Windows/Excel 打开）",
    )
    args = parser.parse_args()

    conf_yaml = Path(args.conf_yaml)
    if not conf_yaml.exists():
        raise SystemExit(f"YAML config not found: {conf_yaml}")

    output_csv = Path(args.output_csv)
    run(conf_yaml=conf_yaml, output_csv=output_csv)


if __name__ == "__main__":
    main()
