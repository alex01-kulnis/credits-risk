"""Real native-model round trips in temporary SQLite MLflow; no HTTP server or real data."""

import hashlib
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from mlflow import MlflowClient

import src.tracking as tracking_module
import src.train as train_module
from src.cross_validation import CrossValidationConfig, cross_validate
from src.data import load_data
from src.evaluation import calculate_metrics
from src.model import ModelConfig, predict_probabilities, train_model
from src.tracking import TrackingConfig, save_experiment
from src.train import TrainConfig, TrainResult, load_config, main


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlflow-store")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        client = MlflowClient(tracking_uri=uri)
        client.search_experiments()
    return SimpleNamespace(root=root, uri=uri, client=client)


@pytest.fixture
def case(tmp_path, store, monkeypatch):
    monkeypatch.chdir(store.root)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    ids = np.arange(1001, 1065, dtype=np.int64)
    target = np.tile([0, 1], 32)
    target[-8:] = -1  # Synthetic held-out targets must never enter training/evaluation.
    features = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": target, "income": np.linspace(1., 10., 64),
        "segment": (["employed", "student", "self_employed"] * 22)[:64],
    })
    splits = pd.DataFrame({"SK_ID_CURR": ids,
                           "split": ["train"] * 40 + ["validation"] * 16 + ["test"] * 8})
    schema = {
        "pipeline_version": "tracking-fixture", "row_count": 64, "feature_count": 2,
        "id_column": "SK_ID_CURR", "target_column": "TARGET",
        "column_order": ["SK_ID_CURR", "TARGET", "income", "segment"],
        "feature_order": ["income", "segment"], "categorical_features": ["segment"],
        "dtypes": {"SK_ID_CURR": "int64", "TARGET": "int64", "income": "float64", "segment": "str"},
        "split_counts": {"train": 40, "validation": 16, "test": 8},
    }
    features.sample(frac=1, random_state=7).to_parquet(data_dir / "features.parquet", index=False)
    splits.to_parquet(data_dir / "splits.parquet", index=False)
    (data_dir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    config = TrainConfig(
        data_dir, ModelConfig(iterations=10, depth=2, thread_count=1, verbose=False),
        CrossValidationConfig(n_splits=2),
        tracking=TrackingConfig(experiment_name=f"tracking-{uuid4().hex}", tracking_uri=store.uri),
    )
    started = perf_counter()
    data = load_data(data_dir)
    cv = cross_validate(data.development, config.model, config.cross_validation,
                        categorical_features=data.categorical_features)
    trained = train_model(data.train, config.model, categorical_features=data.categorical_features)
    probabilities = predict_probabilities(trained.model, data.validation.X,
                                            categorical_features=data.categorical_features)
    result = TrainResult(
        trained, cv, calculate_metrics(data.validation.y, probabilities), config,
        perf_counter() - started,
        {"train_rows": 40, "validation_rows": 16, "development_rows": 56,
         "feature_count": 2, "categorical_count": 1},
    )
    return SimpleNamespace(result=result, data=data, client=store.client, store=store,
                           download_dir=tmp_path / "download")


def download(case, run_id, artifact):
    case.download_dir.mkdir(exist_ok=True)
    return Path(case.client.download_artifacts(run_id, artifact, str(case.download_dir)))


def artifact_names(client, run_id, path=""):
    names = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            names.update(artifact_names(client, run_id, item.path))
        else:
            names.add(item.path)
    return names


def failed_run(case):
    experiment = case.client.get_experiment_by_name(case.result.config.tracking.experiment_name)
    runs = case.client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags.get("serialization.verified") != "true"
    return runs[0]


def config_file(case):
    payload = asdict(case.result.config)
    payload["data_dir"] = str(case.result.config.data_dir)
    path = case.result.config.data_dir.parent / "train.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_each_save_creates_new_finished_run_without_global_state_changes(case):
    uri_before, active_before = mlflow.get_tracking_uri(), mlflow.active_run()
    first = save_experiment(case.result, case.data.validation.X.head(5))
    second = save_experiment(case.result, case.data.validation.X.head(5))
    assert first != second
    assert case.client.get_run(first).info.status == "FINISHED"
    assert case.client.get_run(second).info.status == "FINISHED"
    assert mlflow.get_tracking_uri() == uri_before and mlflow.active_run() is active_before
    assert case.client.get_experiment(case.client.get_run(first).info.experiment_id).name == case.result.config.tracking.experiment_name


def test_parameters_and_actual_metrics_are_logged(case):
    run_id = save_experiment(case.result, case.data.validation.X.head(5))
    run = case.client.get_run(run_id)
    for key, value in asdict(case.result.config.model).items():
        assert run.data.params[key] == str(value)
    for key, value in asdict(case.result.config.cross_validation).items():
        assert run.data.params[f"cv.{key}"] == str(value)
    assert run.data.params["actual_tree_count"] == "10"
    assert run.data.params["feature_count"] == "2"
    assert run.data.params["categorical_count"] == "1"
    assert run.data.params["cv.std_ddof"] == "1"
    for key, value in case.result.validation_metrics.items():
        assert run.data.metrics[f"validation.{key}"] == pytest.approx(value)
    for _, row in case.result.cv_result.fold_metrics.iterrows():
        for key in ["roc_auc", "average_precision", "log_loss", "brier_score", "duration_seconds"]:
            assert run.data.metrics[f"cv.fold_{int(row['fold'])}.{key}"] == pytest.approx(row[key])
    for key, row in case.result.cv_result.summary.iterrows():
        assert run.data.metrics[f"cv.mean.{key}"] == pytest.approx(row["mean"])
        assert run.data.metrics[f"cv.std.{key}"] == pytest.approx(row["std"])
    assert run.data.metrics["cv.duration_seconds"] == case.result.cv_result.duration_seconds
    assert run.data.metrics["training.duration_seconds"] == case.result.trained_model.duration_seconds
    assert run.data.metrics["experiment.duration_seconds"] == case.result.duration_seconds
    assert run.data.metrics["tracking.duration_seconds"] >= 0
    assert run.data.tags["validation.role"] == "exploratory"
    assert run.data.tags["serialization.verified"] == "true"


def test_artifacts_metadata_and_native_model_round_trip(case, monkeypatch):
    def forbidden_parquet_read(*args, **kwargs):
        raise AssertionError("Tracking decoded Parquet instead of hashing bytes")
    monkeypatch.setattr(pd, "read_parquet", forbidden_parquet_read)
    run_id = save_experiment(case.result, case.data.validation.X.head(5))
    assert artifact_names(case.client, run_id) == {
        "model/model.cbm", "config/train.json", "schema/schema.json",
        "cross_validation/fold_metrics.csv", "cross_validation/summary.csv",
        "cross_validation/oof_predictions.csv", "evaluation/validation_metrics.json",
        "metadata/dataset.json", "metadata/environment.json",
    }
    model = CatBoostClassifier()
    model.load_model(str(download(case, run_id, "model/model.cbm")))
    assert model.tree_count_ == 10 and model.feature_names_ == ["income", "segment"]
    before = predict_probabilities(case.result.trained_model.model, case.data.validation.X,
                                   categorical_features=["segment"])
    after = predict_probabilities(model, case.data.validation.X, categorical_features=["segment"])
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)
    schema_path = case.result.config.data_dir / "schema.json"
    assert download(case, run_id, "schema/schema.json").read_bytes() == schema_path.read_bytes()
    saved_config = json.loads(download(case, run_id, "config/train.json").read_text())
    assert saved_config["model"] == asdict(case.result.config.model)
    assert saved_config["tracking"]["tracking_uri"] == case.store.uri
    assert saved_config["data_dir"] == str(case.result.config.data_dir)
    metadata = json.loads(download(case, run_id, "metadata/dataset.json").read_text())
    digests = {}
    for name in ["features.parquet", "splits.parquet", "schema.json"]:
        digest = hashlib.sha256((case.result.config.data_dir / name).read_bytes()).hexdigest()
        assert metadata["files"][name]["sha256"] == digest
        digests[name] = digest
    assert metadata["dataset_id"] == hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    assert metadata["pipeline_version"] == "tracking-fixture"
    environment = json.loads(download(case, run_id, "metadata/environment.json").read_text())
    assert environment["python"] == platform.python_version()
    for package in ["catboost", "scikit-learn", "mlflow", "numpy", "pandas", "pyarrow"]:
        assert environment["packages"][package] == version(package)
    assert "git_sha" in environment
    pd.testing.assert_frame_equal(pd.read_csv(download(case, run_id, "cross_validation/fold_metrics.csv")),
                                  case.result.cv_result.fold_metrics)
    pd.testing.assert_frame_equal(pd.read_csv(download(case, run_id, "cross_validation/summary.csv"), index_col="metric"),
                                  case.result.cv_result.summary)
    pd.testing.assert_frame_equal(pd.read_csv(download(case, run_id, "cross_validation/oof_predictions.csv")),
                                  case.result.cv_result.oof)
    assert json.loads(download(case, run_id, "evaluation/validation_metrics.json").read_text()) == case.result.validation_metrics


def test_disabled_cv_has_no_fake_cv_metrics_or_csv_artifacts(case):
    config = replace(case.result.config, cross_validation=CrossValidationConfig(enabled=False))
    cv = cross_validate(case.data.development, config.model, config.cross_validation,
                        categorical_features=["segment"])
    result = replace(case.result, config=config, cv_result=cv)
    run_id = save_experiment(result, case.data.validation.X.head(5))
    run = case.client.get_run(run_id)
    assert run.info.status == "FINISHED" and run.data.params["cv.enabled"] == "False"
    assert not any(key.startswith("cv.") for key in run.data.metrics)
    assert not any(path.startswith("cross_validation/") for path in artifact_names(case.client, run_id))


def test_environment_uri_overrides_json_without_mutating_config(case, monkeypatch):
    tracking = replace(case.result.config.tracking, tracking_uri="http://127.0.0.1:1")
    result = replace(case.result, config=replace(case.result.config, tracking=tracking))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", case.store.uri)
    run_id = save_experiment(result, case.data.validation.X.head(5))
    assert case.client.get_run(run_id).info.status == "FINISHED"
    config = json.loads(download(case, run_id, "config/train.json").read_text())
    assert config["tracking"]["tracking_uri"] == case.store.uri
    assert result.config.tracking.tracking_uri == "http://127.0.0.1:1"


@pytest.mark.parametrize("field, value", [
    ("experiment_name", ""), ("experiment_name", "   "), ("experiment_name", None),
    ("tracking_uri", ""), ("tracking_uri", 1), ("tracking_uri", None),
])
def test_invalid_tracking_config_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        TrackingConfig(**{field: value})


def test_tracking_json_validation_and_legacy_defaults(case):
    path = config_file(case)
    assert load_config(path).tracking == case.result.config.tracking
    payload = json.loads(path.read_text())
    payload["tracking"]["unknown"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown.*unknown"):
        load_config(path)
    payload["tracking"] = {"experiment_name": "test"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Missing.*tracking_uri"):
        load_config(path)
    payload.pop("tracking")
    path.write_text(json.dumps(payload))
    assert load_config(path).tracking == TrackingConfig()


@pytest.mark.parametrize("operation", ["log_batch", "log_artifacts", "download_artifacts"])
def test_required_mlflow_operation_failure_marks_run_failed(case, monkeypatch, operation):
    def fail(*args, **kwargs):
        raise OSError(f"injected {operation} failure")
    monkeypatch.setattr(MlflowClient, operation, fail)
    with pytest.raises(RuntimeError, match="MLflow"):
        save_experiment(case.result, case.data.validation.X.head(5))
    failed_run(case)


@pytest.mark.parametrize("fault", ["trees", "features", "probabilities"])
def test_round_trip_failure_marks_run_failed(case, monkeypatch, fault):
    original_load = CatBoostClassifier.load_model
    original_predict = CatBoostClassifier.predict_proba

    def broken_load(model, *args, **kwargs):
        value = original_load(model, *args, **kwargs)
        model._loaded_for_verification = True
        if fault == "trees":
            model.shrink(ntree_end=model.tree_count_ - 1)
        elif fault == "features":
            model.set_feature_names(list(reversed(model.feature_names_)))
        return value

    def broken_predict(model, *args, **kwargs):
        probabilities = original_predict(model, *args, **kwargs)
        if fault == "probabilities" and getattr(model, "_loaded_for_verification", False):
            probabilities = probabilities.copy()
            probabilities[:, 1] += 1e-5
        return probabilities

    monkeypatch.setattr(CatBoostClassifier, "load_model", broken_load)
    monkeypatch.setattr(CatBoostClassifier, "predict_proba", broken_predict)
    with pytest.raises(RuntimeError, match="serialization|Serialization"):
        save_experiment(case.result, case.data.validation.X.head(5))
    failed_run(case)


def test_missing_required_dataset_file_marks_run_failed(case):
    (case.result.config.data_dir / "features.parquet").unlink()
    with pytest.raises(RuntimeError, match="features.parquet"):
        save_experiment(case.result, case.data.validation.X.head(5))
    failed_run(case)


def test_unavailable_server_has_clear_error_without_finished_run(case, monkeypatch):
    def offline(*args, **kwargs):
        raise ConnectionError("server unavailable")
    monkeypatch.setattr(MlflowClient, "get_experiment_by_name", offline)
    with pytest.raises(RuntimeError, match="MLflow.*server unavailable"):
        save_experiment(case.result, case.data.validation.X.head(5))


def test_cli_persists_run_with_validation_control_rows(case, monkeypatch, capsys):
    real_save = train_module.save_experiment
    calls = []

    def record_save(result, validation_features):
        pd.testing.assert_frame_equal(validation_features, case.data.validation.X.head(32))
        calls.append(result)
        return real_save(result, validation_features)

    monkeypatch.setattr(train_module, "save_experiment", record_save)
    assert main(["--config", str(config_file(case))]) == 0
    output = capsys.readouterr().out
    run_id = next(line.split(": ", 1)[1] for line in output.splitlines() if line.startswith("MLflow Run ID: "))
    assert len(calls) == 1 and calls[0].run_id == run_id
    assert case.client.get_run(run_id).info.status == "FINISHED"
    assert "STATUS: SUCCESS" in output and "VALIDATION (EXPLORATORY)" in output


def test_cli_does_not_report_success_when_required_save_fails(case, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("required upload failed")
    monkeypatch.setattr(MlflowClient, "log_artifacts", fail)
    assert main(["--config", str(config_file(case))]) == 1
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err and "MLflow" in captured.err
    assert "STATUS: SUCCESS" not in captured.out and "MLflow Run ID:" not in captured.out
    failed_run(case)
