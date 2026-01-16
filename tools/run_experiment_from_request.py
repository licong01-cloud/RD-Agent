from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Add project root to sys.path to allow running as a script
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rdagent.core.experiment_request_schema import (
    ExperimentRequest,
    FactorRef,
    FactorSetConfig,
    ObjectiveConfig,
    StrategyConfig,
    TrainTestConfig,
    ModelConfig,
    DatasetConfig,
)


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"ExperimentRequest JSON root must be an object: {path}")
    return data


def _parse_factor_set(payload: dict[str, Any] | None) -> FactorSetConfig | None:
    if not payload:
        return None
    factors_payload = payload.get("factors") or []
    factors: list[FactorRef] = []
    if isinstance(factors_payload, list):
        for f in factors_payload:
            if not isinstance(f, dict):
                continue
            name = f.get("name")
            if not isinstance(name, str) or not name:
                continue
            factors.append(
                FactorRef(
                    name=name,
                    source=f.get("source"),
                    version=f.get("version"),
                )
            )
    tags = payload.get("tags") or []
    extra = payload.get("extra") or {}
    return FactorSetConfig(
        factors=factors,
        tags=tags if isinstance(tags, list) else [],
        extra=extra if isinstance(extra, dict) else {},
    )


def _parse_strategy(payload: dict[str, Any] | None) -> StrategyConfig | None:
    if not payload:
        return None
    strategy_type = payload.get("strategy_type")
    if not isinstance(strategy_type, str) or not strategy_type:
        return None
    return StrategyConfig(
        strategy_type=strategy_type,
        name=payload.get("name"),
        description=payload.get("description"),
        risk_model=payload.get("risk_model") or {},
        trading_rule=payload.get("trading_rule") or {},
        benchmark=payload.get("benchmark"),
        prompt_template_id=payload.get("prompt_template_id"),
        extra=payload.get("extra") or {},
    )


def _parse_train_test(payload: dict[str, Any] | None) -> TrainTestConfig | None:
    if not payload:
        return None
    return TrainTestConfig(
        train_start=payload.get("train_start"),
        train_end=payload.get("train_end"),
        valid_start=payload.get("valid_start"),
        valid_end=payload.get("valid_end"),
        test_start=payload.get("test_start"),
        test_end=payload.get("test_end"),
        dataset_extra=payload.get("dataset_extra") or {},
    )


def _parse_objective(payload: dict[str, Any] | None) -> ObjectiveConfig:
    if not payload:
        return ObjectiveConfig()
    primary_metric = payload.get("primary_metric") or "ic"
    direction = payload.get("direction") or "maximize"
    return ObjectiveConfig(
        primary_metric=str(primary_metric),
        direction="minimize" if direction == "minimize" else "maximize",
        constraints=payload.get("constraints") or {},
    )


def _parse_model(payload: dict[str, Any] | None) -> ModelConfig | None:
    if not payload:
        return None
    return ModelConfig(
        model_type=payload.get("model_type"),
        model_class=payload.get("model_class"),
        kwargs=payload.get("kwargs") or {},
        extra=payload.get("extra") or {},
    )


def _parse_dataset(payload: dict[str, Any] | None) -> DatasetConfig | None:
    if not payload:
        return None
    return DatasetConfig(
        class_name=payload.get("class_name", "DatasetH"),
        handler_class=payload.get("handler_class", "Alpha158"),
        handler_kwargs=payload.get("handler_kwargs") or {},
        segments=payload.get("segments") or {},
        extra=payload.get("extra") or {},
    )


def _parse_experiment_request(obj: dict[str, Any]) -> ExperimentRequest:
    factor_set = _parse_factor_set(obj.get("factor_set"))
    strategy = _parse_strategy(obj.get("strategy"))
    train_test = _parse_train_test(obj.get("train_test"))
    objective = _parse_objective(obj.get("objective"))
    model = _parse_model(obj.get("model"))
    dataset = _parse_dataset(obj.get("dataset"))

    kind = obj.get("kind") or "research"
    if kind not in ("research", "benchmark", "ab_test"):
        kind = "research"

    return ExperimentRequest(
        kind=kind,  # type: ignore[arg-type]
        strategy=strategy,
        factor_set=factor_set,
        train_test=train_test,
        objective=objective,
        model=model,
        dataset=dataset,
        name=obj.get("name"),
        description=obj.get("description"),
        user_tags=obj.get("user_tags") or [],
        extra=obj.get("extra") or {},
    )


def _build_qlib_yaml_payload(req: ExperimentRequest) -> dict[str, Any]:
    """Mapping from ExperimentRequest to a detailed Qlib-style YAML tree.

    Supports:
    - data_handler_config (historical compatibility and times)
    - port_analysis_config
    - task.model (from req.model)
    - task.dataset (from req.dataset)
    """

    # 1. 基础时间与数据配置
    data_cfg: dict[str, Any] = {}
    segments: dict[str, Any] = {}
    if req.train_test is not None:
        if req.train_test.train_start and req.train_test.train_end:
            segments["train"] = [req.train_test.train_start, req.train_test.train_end]
        if req.train_test.valid_start and req.train_test.valid_end:
            segments["valid"] = [req.train_test.valid_start, req.train_test.valid_end]
        if req.train_test.test_start and req.train_test.test_end:
            segments["test"] = [req.train_test.test_start, req.train_test.test_end]
        
        # 兼容旧版 data_handler_config 风格的 start/end time
        data_cfg = dict(req.train_test.dataset_extra)
        for s, times in segments.items():
            data_cfg.setdefault(s, {})["start_time"] = times[0]
            data_cfg.setdefault(s, {})["end_time"] = times[1]

    # 2. 策略配置
    portfolio_cfg: dict[str, Any] = {}
    backtest_cfg: dict[str, Any] = {}
    if req.strategy is not None:
        portfolio_cfg = dict(req.strategy.trading_rule)
        if req.strategy.benchmark:
            backtest_cfg["benchmark"] = req.strategy.benchmark

    # 3. 模型配置 (ModelConfig -> Qlib model)
    model_cfg: dict[str, Any] = {}
    if req.model:
        if req.model.model_class:
            model_cfg["class"] = req.model.model_class
        elif req.model.model_type:
            # 自动映射常见简称
            type_map = {
                "LGBModel": "qlib.contrib.model.gbdt.LGBModel",
                "XGBModel": "qlib.contrib.model.gbdt.XGBModel",
            }
            model_cfg["class"] = type_map.get(req.model.model_type, req.model.model_type)
        
        model_cfg["kwargs"] = dict(req.model.kwargs)
        model_cfg.update(req.model.extra)

    # 4. 数据集配置 (DatasetConfig -> Qlib dataset)
    dataset_cfg: dict[str, Any] = {}
    if req.dataset:
        dataset_cfg["class"] = req.dataset.class_name or "DatasetH"
        dataset_cfg["kwargs"] = {
            "handler": {
                "class": req.dataset.handler_class or "Alpha158",
                "kwargs": dict(req.dataset.handler_kwargs)
            },
            "segments": req.dataset.segments or segments
        }
        dataset_cfg.update(req.dataset.extra)
    elif req.train_test:
        # 降级：若没有 explicit dataset config，尝试从 train_test.dataset_extra 恢复
        dataset_cfg = dict(req.train_test.dataset_extra)
        if "segments" not in dataset_cfg and segments:
            dataset_cfg["segments"] = segments

    yaml_tree: dict[str, Any] = {
        "data_handler_config": data_cfg,
        "port_analysis_config": {
            "strategy": portfolio_cfg,
            "backtest": backtest_cfg,
        },
        "task": {
            "model": model_cfg,
            "dataset": dataset_cfg,
        },
    }
    return yaml_tree


def run(input_json: Path, output_dir: Path) -> None:
    raw = _load_request(input_json)
    req = _parse_experiment_request(raw)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 保存归一化后的 ExperimentRequest（便于调试和对账）。
    normalized_path = output_dir / "experiment_request.normalized.json"
    normalized_path.write_text(json.dumps(asdict(req), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 2) 生成一个最小可用的 Qlib YAML。
    yaml_tree = _build_qlib_yaml_payload(req)
    yaml_path = output_dir / "experiment_qlib_config.yaml"
    try:
        import yaml

        yaml_path.write_text(yaml.safe_dump(yaml_tree, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Failed to write YAML: {e!s}")

    print(f"Normalized ExperimentRequest written to: {normalized_path}")
    print(f"Qlib YAML config written to: {yaml_path}")
    print("\nYou can now wire this YAML into your existing RD-Agent/Qlib workflow (e.g., fin_quant).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load an ExperimentRequest JSON file, validate/normalize it using "
            "rdagent.core.experiment_request_schema, and emit a minimal Qlib YAML config."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ExperimentRequest JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write normalized ExperimentRequest and Qlib YAML.",
    )

    args = parser.parse_args()

    input_json = Path(args.input)
    if not input_json.exists():
        raise SystemExit(f"Input JSON not found: {input_json}")

    output_dir = Path(args.output_dir)
    run(input_json=input_json, output_dir=output_dir)


if __name__ == "__main__":
    main()
