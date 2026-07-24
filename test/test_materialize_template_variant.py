from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.materialize_template_variant import MaterializationError, main, materialize


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "app_tpl/all/v4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("version", "horizon", "label"),
    [("v4-5d", 5, "Ref($close, -6)"), ("v4-10d", 10, "Ref($close, -11)")],
)
def test_fixed_horizon_variant_changes_only_labels_and_manifest(tmp_path, version, horizon, label):
    output = tmp_path / version
    manifest = materialize(
        spec_path=ROOT / f"app_tpl/all/{version}/variant.json",
        base_dir=BASE,
        output_dir=output,
    )

    assert manifest["version"] == version
    assert manifest["base_version"] == "v4"
    assert manifest["label_horizon"] == horizon
    assert manifest["label_expression"] == label
    assert len(manifest["changed_files"]) == 8
    assert manifest["files"]

    base_files = {path.relative_to(BASE).as_posix(): path for path in BASE.rglob("*") if path.is_file()}
    output_files = {
        path.relative_to(output).as_posix(): path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(output_files) == set(base_files) - {"manifest.json"}
    for relative, output_path in output_files.items():
        if relative in manifest["changed_files"]:
            text = output_path.read_text(encoding="utf-8")
            assert label in text
            assert "Ref($close, -2)" not in text
        else:
            assert _sha256(output_path) == _sha256(base_files[relative])

    for required in (
        "rdagent/scenarios/qlib/experiment/factor_template/benchmark_sh000300.parquet",
        "rdagent/scenarios/qlib/experiment/model_template/benchmark_sh000300.parquet",
        "rdagent/scenarios/qlib/experiment/factor_template/minute_execution_contract.py",
        "rdagent/scenarios/qlib/experiment/model_template/minute_execution_contract.py",
        "rdagent/scenarios/qlib/experiment/factor_template/tail_twap_strategy.py",
        "rdagent/scenarios/qlib/experiment/model_template/tail_twap_strategy.py",
        "rdagent/scenarios/qlib/experiment/factor_template/tail_twap_v24_strategy.py",
        "rdagent/scenarios/qlib/experiment/model_template/tail_twap_v24_strategy.py",
        "rdagent/scenarios/qlib/experiment/factor_template/tail_twap_v25_strategy.py",
        "rdagent/scenarios/qlib/experiment/model_template/tail_twap_v25_strategy.py",
    ):
        assert (output / required).is_file()


def test_materialization_is_reproducible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = materialize(
        spec_path=ROOT / "app_tpl/all/v4-5d/variant.json",
        base_dir=BASE,
        output_dir=first,
    )
    second_manifest = materialize(
        spec_path=ROOT / "app_tpl/all/v4-5d/variant.json",
        base_dir=BASE,
        output_dir=second,
    )
    assert first_manifest == second_manifest
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_materializer_refuses_repo_output_and_existing_output(tmp_path):
    with pytest.raises(MaterializationError, match="outside repository"):
        materialize(
            spec_path=ROOT / "app_tpl/all/v4-5d/variant.json",
            base_dir=BASE,
            output_dir=ROOT / "mlruns_compare",
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(MaterializationError, match="already exists"):
        materialize(
            spec_path=ROOT / "app_tpl/all/v4-5d/variant.json",
            base_dir=BASE,
            output_dir=existing,
        )


def test_variant_specs_are_the_only_committed_derived_template_files():
    for version in ("v4-5d", "v4-10d"):
        files = [path.relative_to(ROOT / f"app_tpl/all/{version}").as_posix() for path in (ROOT / f"app_tpl/all/{version}").rglob("*") if path.is_file()]
        assert files == ["variant.json"]
        spec = json.loads((ROOT / f"app_tpl/all/{version}/variant.json").read_text(encoding="utf-8"))
        assert spec["base_version"] == "v4"


def test_materializer_cli_returns_input_error_code(tmp_path, capsys):
    output = tmp_path / "output"
    code = main(
        [
            "--spec",
            str(ROOT / "app_tpl/all/v4-5d/variant.json"),
            "--base-dir",
            str(tmp_path / "missing-base"),
            "--output-dir",
            str(output),
        ]
    )
    assert code == 2
    assert "base template is incomplete" in capsys.readouterr().err


def test_replacement_mismatch_leaves_no_partial_output(tmp_path):
    spec = json.loads((ROOT / "app_tpl/all/v4-5d/variant.json").read_text(encoding="utf-8"))
    spec["expected_replacement_count"] = 999
    spec_path = tmp_path / "bad-variant.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(MaterializationError, match="replacement count mismatch"):
        materialize(spec_path=spec_path, base_dir=BASE, output_dir=output)
    assert not output.exists()
