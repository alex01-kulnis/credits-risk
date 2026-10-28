"""Persist completed experiments and verify the native model downloaded from MLflow."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

from .model import predict_probabilities

if TYPE_CHECKING:
    from .train import TrainResult


@dataclass(frozen=True)
class TrackingConfig:
    """Explicit client settings; MLFLOW_TRACKING_URI overrides the URI at save time."""

    experiment_name: str = "credits-risk-v4"
    tracking_uri: str = "http://127.0.0.1:5050"

    def __post_init__(self) -> None:
        for name in ("experiment_name", "tracking_uri"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _environment() -> dict:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = None
    return {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "catboost",
                "scikit-learn",
                "mlflow",
                "numpy",
                "pandas",
                "pyarrow",
            )
        },
        "git_sha": git_sha,
        "experiment_duration_scope": "load_data, CV, candidate fit and Validation; tracking time is separate",
    }


def _write_artifacts(result: TrainResult, root: Path, uri: str) -> tuple[dict, str]:
    data_dir = result.config.data_dir
    schema = json.loads((data_dir / "schema.json").read_text(encoding="utf-8"))
    model = result.trained_model.model
    categories = [model.feature_names_[i] for i in model.get_cat_feature_indices()]
    if schema["feature_order"] != model.feature_names_ or set(
        schema["categorical_features"]
    ) != set(categories):
        raise ValueError("Model feature names/categories do not match schema.json")

    files = {}
    for name in ("features.parquet", "splits.parquet", "schema.json"):
        path = data_dir / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[name] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    digests = {name: entry["sha256"] for name, entry in files.items()}
    dataset_id = hashlib.sha256(
        json.dumps(digests, sort_keys=True).encode()
    ).hexdigest()

    (root / "model").mkdir(parents=True)
    model.save_model(str(root / "model/model.cbm"), format="cbm")
    config = asdict(result.config)
    config["data_dir"] = str(data_dir)
    config["tracking"]["tracking_uri"] = uri
    _write_json(root / "config/train.json", config)
    (root / "schema").mkdir()
    shutil.copy2(data_dir / "schema.json", root / "schema/schema.json")
    if result.config.cross_validation.enabled:
        (root / "cross_validation").mkdir()
        result.cv_result.fold_metrics.to_csv(
            root / "cross_validation/fold_metrics.csv", index=False
        )
        result.cv_result.summary.to_csv(
            root / "cross_validation/summary.csv", index_label="metric"
        )
        result.cv_result.oof.to_csv(
            root / "cross_validation/oof_predictions.csv", index=False
        )
    _write_json(root / "evaluation/validation_metrics.json", result.validation_metrics)
    _write_json(
        root / "metadata/dataset.json",
        {
            "dataset_id": dataset_id,
            "pipeline_version": schema["pipeline_version"],
            "files": files,
            "dataset_info": result.dataset_info,
        },
    )
    _write_json(root / "metadata/environment.json", _environment())
    return schema, dataset_id


def _log_values(client: MlflowClient, run_id: str, result: TrainResult) -> None:
    params = {
        **asdict(result.config.model),
        **{
            f"cv.{key}": value
            for key, value in asdict(result.config.cross_validation).items()
        },
        "actual_tree_count": result.trained_model.model.tree_count_,
        "feature_count": result.dataset_info["feature_count"],
        "categorical_count": result.dataset_info["categorical_count"],
        "cv.actual_n_splits": len(result.cv_result.fold_metrics),
    }
    metrics = {
        f"validation.{key}": value for key, value in result.validation_metrics.items()
    }
    metrics["training.duration_seconds"] = result.trained_model.duration_seconds
    metrics["experiment.duration_seconds"] = result.duration_seconds
    if result.config.cross_validation.enabled:
        params["cv.std_ddof"] = int(result.cv_result.summary["ddof"].iloc[0])
        metrics["cv.duration_seconds"] = result.cv_result.duration_seconds
        for _, row in result.cv_result.fold_metrics.iterrows():
            for key, value in row.items():
                if key != "fold":
                    metrics[f"cv.fold_{int(row['fold'])}.{key}"] = float(value)
        for key, row in result.cv_result.summary.iterrows():
            metrics[f"cv.mean.{key}"] = float(row["mean"])
            metrics[f"cv.std.{key}"] = float(row["std"])
    timestamp = int(time() * 1000)
    client.log_batch(
        run_id,
        params=[Param(key, str(value)) for key, value in params.items()],
        metrics=[
            Metric(key, float(value), timestamp, 0) for key, value in metrics.items()
        ],
        synchronous=True,
    )


def _verify_model(
    client: MlflowClient,
    run_id: str,
    result: TrainResult,
    features: pd.DataFrame,
    schema: dict,
    destination: Path,
) -> float:
    sample = features.head(32).copy()
    if sample.empty:
        raise ValueError("Serialization check needs nonempty Validation features")
    destination.mkdir()
    path = client.download_artifacts(run_id, "model/model.cbm", str(destination))
    loaded = CatBoostClassifier()
    loaded.load_model(path)
    original = result.trained_model.model
    if loaded.tree_count_ != original.tree_count_:
        raise ValueError("Serialization check failed: tree count changed")
    if loaded.feature_names_ != original.feature_names_:
        raise ValueError("Serialization check failed: feature names/order changed")
    before = predict_probabilities(
        original, sample, categorical_features=schema["categorical_features"]
    )
    after = predict_probabilities(
        loaded, sample, categorical_features=schema["categorical_features"]
    )
    error = float(np.max(np.abs(before - after)))
    if not np.allclose(before, after, rtol=0, atol=1e-12, equal_nan=False):
        raise ValueError(
            f"Serialization check failed: probabilities differ (max abs error={error})"
        )
    return error


def save_experiment(result: TrainResult, validation_features: pd.DataFrame) -> str:
    """Save a fresh Run and verify its downloaded CBM on at most 32 Validation rows.

    Caller must supply features from the historical Validation split, in schema
    order. Parquet files are hashed as bytes and never decoded or uploaded here.
    Returns the Run ID only after all writes/checks succeed. On failure, marks
    an existing Run FAILED and raises RuntimeError; no global MLflow state changes.
    """
    started = perf_counter()
    settings = result.config.tracking
    uri = os.environ.get("MLFLOW_TRACKING_URI", settings.tracking_uri)
    if not uri.strip():
        raise RuntimeError("MLflow MLFLOW_TRACKING_URI must be nonempty")
    client = None
    run_id = None
    try:
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(settings.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(settings.experiment_name)
        )
        run_id = client.create_run(
            experiment_id,
            tags={
                "project": "credits_risk",
                "validation.role": "exploratory",
                "serialization.verified": "false",
            },
        ).info.run_id
        with TemporaryDirectory(prefix="credits-risk-tracking-") as workspace:
            root = Path(workspace) / "artifacts"
            schema, dataset_id = _write_artifacts(result, root, uri)
            _log_values(client, run_id, result)
            client.log_artifacts(run_id, str(root))
            error = _verify_model(
                client,
                run_id,
                result,
                validation_features,
                schema,
                Path(workspace) / "download",
            )
            client.log_batch(
                run_id,
                tags=[
                    RunTag("dataset_id", dataset_id),
                    RunTag("serialization.verified", "true"),
                    RunTag("serialization.max_abs_error", str(error)),
                    RunTag(
                        "serialization.checked_rows",
                        str(min(len(validation_features), 32)),
                    ),
                ],
                metrics=[
                    Metric(
                        "tracking.duration_seconds",
                        perf_counter() - started,
                        int(time() * 1000),
                        0,
                    )
                ],
                synchronous=True,
            )
        client.set_terminated(run_id, status="FINISHED")
        return run_id
    except Exception as exc:
        status_error = ""
        if client is not None and run_id is not None:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception as failure:
                status_error = f"; could not record FAILED status: {failure}"
        raise RuntimeError(
            f"MLflow save failed at {uri} (run_id={run_id}): {exc}{status_error}"
        ) from exc
