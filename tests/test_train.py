"""Mocked workflow boundaries and tiny data; no historical training or Test reads."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np
import pandas as pd
import pytest

import src.cross_validation as cv_module
import src.train as train_module
from src.cross_validation import CrossValidationConfig, CrossValidationResult
from src.data import DataSplit, PreparedData
from src.evaluation import calculate_metrics
from src.model import ModelConfig, TrainedModel
from src.train import TrainConfig, TrainResult, load_config, main, run_train


@pytest.fixture
def payload(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    return {
        "data_dir": str(data_dir),
        "model": {"iterations": 10, "depth": 2, "learning_rate": 0.1,
                  "l2_leaf_reg": 3.0, "random_seed": 42, "loss_function": "Logloss",
                  "thread_count": 1, "verbose": False},
        "cross_validation": {"enabled": True, "n_splits": 2, "shuffle": True,
                             "random_state": 42},
    }


def write_config(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_json_creates_existing_config_types(tmp_path, payload):
    config = load_config(write_config(tmp_path, payload))
    assert isinstance(config, TrainConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.cross_validation, CrossValidationConfig)
    assert config.data_dir == Path(payload["data_dir"]).resolve()
    assert asdict(config.model) == payload["model"]
    assert asdict(config.cross_validation) == payload["cross_validation"]


def test_relative_data_path_is_project_relative_not_cwd_or_config_directory(tmp_path, payload, monkeypatch):
    root = tmp_path / "project"
    data_dir = root / "data" / "processed" / "v4"
    data_dir.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(train_module, "_PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)
    payload["data_dir"] = "data/processed/v4"
    config = load_config(write_config(tmp_path, payload))
    assert config.data_dir == data_dir.resolve()


@pytest.mark.parametrize("section, key", [
    (None, "modle"), ("model", "early_stopping_rounds"), ("cross_validation", "stratified"),
])
def test_unknown_fields_are_rejected(tmp_path, payload, section, key):
    target = payload if section is None else payload[section]
    target[key] = 1
    with pytest.raises(ValueError, match=f"[Uu]nknown.*{key}"):
        load_config(write_config(tmp_path, payload))


@pytest.mark.parametrize("section, key", [
    (None, "data_dir"), (None, "model"), (None, "cross_validation"),
    ("model", "iterations"), ("model", "verbose"), ("cross_validation", "n_splits"),
])
def test_missing_fields_are_rejected(tmp_path, payload, section, key):
    target = payload if section is None else payload[section]
    target.pop(key)
    with pytest.raises(ValueError, match=f"[Mm]issing.*{key}"):
        load_config(write_config(tmp_path, payload))


@pytest.mark.parametrize("content", ["{bad", "[]", "null", "true"])
def test_invalid_json_or_root_is_rejected(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="JSON|object"):
        load_config(path)


def test_invalid_json_encoding_is_reported(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_config(path)


@pytest.mark.parametrize("section", ["model", "cross_validation"])
@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_sections_must_be_objects(tmp_path, payload, section, value):
    payload[section] = value
    with pytest.raises(ValueError, match=f"{section}.*object"):
        load_config(write_config(tmp_path, payload))


@pytest.mark.parametrize("section, field, value", [
    ("model", "iterations", 0), ("model", "depth", 17),
    ("model", "learning_rate", "0.1"), ("model", "learning_rate", float("nan")),
    ("model", "thread_count", 0), ("model", "verbose", -1),
    ("cross_validation", "n_splits", 1), ("cross_validation", "enabled", "false"),
    ("cross_validation", "shuffle", 1), ("cross_validation", "random_state", -1),
])
def test_existing_parameter_validation_is_used(tmp_path, payload, section, field, value):
    payload[section][field] = value
    with pytest.raises(ValueError, match=field):
        load_config(write_config(tmp_path, payload))


@pytest.mark.parametrize("value", [None, 1, [], "", "   "])
def test_data_dir_must_be_nonempty_string(tmp_path, payload, value):
    payload["data_dir"] = value
    with pytest.raises(ValueError, match="data_dir"):
        load_config(write_config(tmp_path, payload))


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.json"):
        load_config(tmp_path / "missing.json")


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_data_path_must_be_existing_directory(tmp_path, payload, kind):
    path = tmp_path / "dataset"
    if kind == "file":
        path.write_text("not a directory")
    payload["data_dir"] = str(path)
    with pytest.raises(ValueError, match="data_dir.*directory"):
        load_config(write_config(tmp_path, payload))


@pytest.fixture
def workflow(tmp_path):
    def split(ids):
        return DataSplit(
            X=pd.DataFrame({"income": [1., 2., 3., 4.], "job": ["a", "b", "a", "b"]}),
            y=pd.Series([0, 0, 1, 1], name="TARGET"),
            ids=pd.Series(ids, name="SK_ID_CURR"),
        )

    train, validation = split([10, 11, 12, 13]), split([20, 21, 22, 23])
    development = DataSplit(
        X=pd.concat([train.X, validation.X], ignore_index=True),
        y=pd.concat([train.y, validation.y], ignore_index=True),
        ids=pd.concat([train.ids, validation.ids], ignore_index=True),
    )
    data = PreparedData(train, validation, development, ["income", "job"], ["job"],
                        {"pipeline_version": "synthetic"})
    config = TrainConfig(tmp_path, ModelConfig(iterations=10, depth=2, learning_rate=0.1,
                                              thread_count=1, verbose=False),
                         CrossValidationConfig(n_splits=2))
    values = {"roc_auc": 0.9, "average_precision": 0.85,
              "log_loss": 0.2, "brier_score": 0.1}
    spread = {"roc_auc": 0.02, "average_precision": 0.03,
              "log_loss": 0.01, "brier_score": 0.005}
    fold_metrics = pd.DataFrame([
        {"fold": fold, **{key: value + sign * spread[key] for key, value in values.items()},
         "duration_seconds": 0.1}
        for fold, sign in [(1, -1), (2, 1)]
    ])
    cv_result = CrossValidationResult(
        fold_metrics, pd.DataFrame({"mean": values,
                                    "std": {key: value * np.sqrt(2) for key, value in spread.items()},
                                    "ddof": 1}).rename_axis("metric"),
        pd.DataFrame({"SK_ID_CURR": development.ids, "fold": [1, 2] * 4,
                      "target": development.y, "probability": 0.5}), 0.2,
    )
    return SimpleNamespace(data=data, config=config, cv_result=cv_result,
                           trained=TrainedModel(sentinel.model, 0.25),
                           probabilities=np.array([0.1, 0.4, 0.35, 0.8]))


@pytest.fixture
def wired_workflow(workflow, monkeypatch):
    events = []
    monkeypatch.setattr(train_module, "save_experiment", Mock(return_value="mock-mlflow-run"))

    def load(path):
        events.append("load")
        assert path == workflow.config.data_dir
        return workflow.data

    def cv(development, config, cv_config, *, categorical_features):
        events.append("cv")
        assert development is workflow.data.development
        assert config == workflow.config.model and cv_config == workflow.config.cross_validation
        assert categorical_features == ["job"]
        return workflow.cv_result

    def train(split, config, *, categorical_features):
        events.append("train")
        assert split is workflow.data.train
        assert config == workflow.config.model and categorical_features == ["job"]
        return workflow.trained

    def predict(model, X, *, categorical_features):
        events.append("predict")
        assert model is workflow.trained.model
        assert X is workflow.data.validation.X and categorical_features == ["job"]
        return workflow.probabilities

    def metrics(y, probabilities):
        events.append("metrics")
        assert y is workflow.data.validation.y
        assert probabilities is workflow.probabilities
        return calculate_metrics(y, probabilities)

    for name, function in [("load_data", load), ("cross_validate", cv), ("train_model", train),
                           ("predict_probabilities", predict), ("calculate_metrics", metrics)]:
        monkeypatch.setattr(train_module, name, Mock(side_effect=function))
    workflow.events = events
    return workflow


def test_workflow_order_split_selection_and_result(wired_workflow):
    fixture = wired_workflow
    result = run_train(fixture.config)
    assert fixture.events == ["load", "cv", "train", "predict", "metrics"]
    assert isinstance(result, TrainResult)
    assert result.trained_model is fixture.trained
    assert result.cv_result is fixture.cv_result
    assert result.config is fixture.config
    assert result.validation_metrics == pytest.approx({
        "roc_auc": 0.75, "average_precision": 5 / 6,
        "log_loss": 0.47228795380917615, "brier_score": 0.158125})
    assert result.dataset_info == {"train_rows": 4, "validation_rows": 4,
                                   "development_rows": 8, "feature_count": 2,
                                   "categorical_count": 1}
    assert np.isfinite(result.duration_seconds) and result.duration_seconds >= 0


def test_disabled_cv_still_trains_candidate_once(wired_workflow, monkeypatch):
    fixture = wired_workflow
    config = replace(fixture.config, cross_validation=CrossValidationConfig(enabled=False))
    monkeypatch.setattr(train_module, "cross_validate", cv_module.cross_validate)
    monkeypatch.setattr(cv_module, "train_model", Mock(side_effect=AssertionError("Disabled CV trained a fold")))
    result = run_train(config)
    assert result.cv_result.fold_metrics.empty and result.cv_result.oof.empty
    assert result.trained_model is fixture.trained
    assert fixture.events == ["load", "train", "predict", "metrics"]
    train_module.train_model.assert_called_once()


def test_workflow_has_no_test_access(wired_workflow):
    fixture = wired_workflow

    class GuardedData(PreparedData):
        @property
        def test(self):
            raise AssertionError("Workflow accessed Test")

    fixture.data = GuardedData(**vars(fixture.data))
    run_train(fixture.config)
    assert fixture.events == ["load", "cv", "train", "predict", "metrics"]


def test_cli_runs_workflow_and_prints_actual_report(tmp_path, payload, wired_workflow, capsys):
    fixture = wired_workflow
    payload["data_dir"] = str(fixture.config.data_dir)
    assert main(["--config", str(write_config(tmp_path, payload))]) == 0
    output = capsys.readouterr().out
    assert "STATUS: SUCCESS" in output
    assert "CREDIT SCORING" in output and "VALIDATION (EXPLORATORY)" in output
    assert "Folds: 2" in output and "0.750000" in output and "0.833333" in output
    cv_section, validation_section = output.split("VALIDATION (EXPLORATORY)")
    assert "ROC-AUC: 0.900000 ± 0.028284" in cv_section and "ddof=1" in cv_section
    assert "ROC-AUC: 0.750000" in validation_section
    for label, value in [("Train:", "4"), ("Validation:", "4"), ("Development:", "8"),
                          ("Features:", "2"), ("Categorical:", "1")]:
        assert next(line for line in output.splitlines() if line.startswith(label)).split() == [label, value]
    assert "Iterations: 10" in output and "Depth: 2" in output
    assert "0.250" in output
    assert "SK_ID_CURR" not in output and "probability" not in output
    assert fixture.events == ["load", "cv", "train", "predict", "metrics"]


def test_cli_displays_disabled_cv(tmp_path, payload, wired_workflow, monkeypatch, capsys):
    fixture = wired_workflow
    payload["data_dir"] = str(fixture.config.data_dir)
    payload["cross_validation"]["enabled"] = False
    monkeypatch.setattr(train_module, "cross_validate", cv_module.cross_validate)
    assert main(["--config", str(write_config(tmp_path, payload))]) == 0
    assert "Cross-Validation: disabled" in capsys.readouterr().out


def test_cli_reports_config_error_without_training(tmp_path, capsys, monkeypatch):
    run = Mock(side_effect=AssertionError("Invalid config reached training"))
    monkeypatch.setattr(train_module, "run_train", run)
    assert main(["--config", str(tmp_path / "missing.json")]) == 1
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err and "missing.json" in captured.err
    assert "SUCCESS" not in captured.out
    run.assert_not_called()


def test_cli_reports_workflow_failure(tmp_path, payload, monkeypatch, capsys):
    monkeypatch.setattr(train_module, "run_train", Mock(side_effect=ValueError("invalid dataset")))
    assert main(["--config", str(write_config(tmp_path, payload))]) == 1
    captured = capsys.readouterr()
    assert "invalid dataset" in captured.err and "SUCCESS" not in captured.out


def test_cli_requires_config_argument():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_module_cli_help_and_missing_file_do_not_train(tmp_path):
    root = Path(train_module.__file__).resolve().parents[1]
    help_result = subprocess.run([sys.executable, "-m", "src.train", "--help"], cwd=root,
                                 capture_output=True, text=True, check=False)
    assert help_result.returncode == 0 and "--config" in help_result.stdout
    error_result = subprocess.run([sys.executable, "-m", "src.train", "--config",
                                  str(tmp_path / "missing.json")], cwd=root,
                                 capture_output=True, text=True, check=False)
    assert error_result.returncode == 1 and "ERROR:" in error_result.stderr
