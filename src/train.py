"""JSON-configured experiment: Development CV, Train candidate, Validation report."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from time import perf_counter

from .cross_validation import (
    CrossValidationConfig,
    CrossValidationResult,
    cross_validate,
)
from .data import load_data
from .evaluation import calculate_metrics
from .model import ModelConfig, TrainedModel, predict_probabilities, train_model
from .tracking import TrackingConfig, save_experiment

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Average Precision",
    "log_loss": "LogLoss",
    "brier_score": "Brier Score",
}


@dataclass(frozen=True)
class TrainConfig:
    """Experiment settings; relative data paths always use the project root."""

    data_dir: Path
    model: ModelConfig
    cross_validation: CrossValidationConfig
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def __post_init__(self) -> None:
        directory = Path(self.data_dir)
        if not directory.is_absolute():
            directory = _PROJECT_ROOT / directory
        directory = directory.resolve()
        if not directory.is_dir():
            raise ValueError(f"data_dir must be an existing directory: {directory}")
        object.__setattr__(self, "data_dir", directory)


@dataclass
class TrainResult:
    """One candidate, CV output and exploratory Validation metrics; no dataset copy."""

    trained_model: TrainedModel
    cv_result: CrossValidationResult
    validation_metrics: dict[str, float]
    config: TrainConfig
    duration_seconds: float
    dataset_info: dict[str, int]
    run_id: str | None = None


def run_train(config: TrainConfig) -> TrainResult:
    """Compare parameters with Development CV, then fit a separate Train candidate.

    Validation belongs to Development and was used in historical model selection;
    its candidate metrics are exploratory. No Test features or targets are used.
    """
    started = perf_counter()
    data = load_data(config.data_dir)

    cv_result = cross_validate(
        data.development,
        config.model,
        config.cross_validation,
        categorical_features=data.categorical_features,
    )
    trained = train_model(
        data.train, config.model, categorical_features=data.categorical_features
    )
    probabilities = predict_probabilities(
        trained.model, data.validation.X, categorical_features=data.categorical_features
    )
    validation_metrics = calculate_metrics(data.validation.y, probabilities)

    result = TrainResult(
        trained_model=trained,
        cv_result=cv_result,
        validation_metrics=validation_metrics,
        config=config,
        duration_seconds=perf_counter() - started,
        dataset_info={
            "train_rows": len(data.train.X),
            "validation_rows": len(data.validation.X),
            "development_rows": len(data.development.X),
            "feature_count": len(data.feature_names),
            "categorical_count": len(data.categorical_features),
        },
    )
    result.run_id = save_experiment(result, data.validation.X.head(32))
    return result


def _require_fields(
    value: object, expected: set[str], section: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a JSON object")
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"Unknown fields in {section}: {', '.join(sorted(unknown))}")
    missing = expected - set(value)
    if missing:
        raise ValueError(f"Missing fields in {section}: {', '.join(sorted(missing))}")
    return value


def load_config(path: Path) -> TrainConfig:
    """Read strict JSON: every root/model/CV field must be explicitly specified."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON configuration {path}: {exc}") from exc
    if isinstance(raw, dict):
        raw.setdefault("tracking", asdict(TrackingConfig()))
    raw = _require_fields(
        raw, {"data_dir", "model", "cross_validation", "tracking"}, "config"
    )
    model = _require_fields(
        raw["model"], {field.name for field in fields(ModelConfig)}, "model"
    )
    cv = _require_fields(
        raw["cross_validation"],
        {field.name for field in fields(CrossValidationConfig)},
        "cross_validation",
    )
    tracking = _require_fields(
        raw["tracking"], {item.name for item in fields(TrackingConfig)}, "tracking"
    )
    if not isinstance(raw["data_dir"], str) or not raw["data_dir"].strip():
        raise ValueError("data_dir must be a nonempty string")
    return TrainConfig(
        data_dir=Path(raw["data_dir"]),
        model=ModelConfig(**model),
        cross_validation=CrossValidationConfig(**cv),
        tracking=TrackingConfig(**tracking),
    )


def _print_report(result: TrainResult) -> None:
    print("=" * 41)
    print("CREDIT SCORING — TRAINING")
    print("=" * 41)
    print("\nDATASET")
    for key, label in (
        ("train_rows", "Train:"),
        ("validation_rows", "Validation:"),
        ("development_rows", "Development:"),
        ("feature_count", "Features:"),
        ("categorical_count", "Categorical:"),
    ):
        print(f"{label:<14}{result.dataset_info[key]}")

    model = result.config.model
    print("\nMODEL")
    print(f"Iterations: {model.iterations}")
    print(f"Depth: {model.depth}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"L2 leaf regularization: {model.l2_leaf_reg}")
    print(f"Random seed: {model.random_seed}")
    print(f"Loss function: {model.loss_function}")
    print(f"Threads: {model.thread_count}")
    print(f"Verbose: {model.verbose}")

    print("\nCROSS-VALIDATION")
    if result.config.cross_validation.enabled:
        print(f"Folds: {len(result.cv_result.fold_metrics)}")
        ddof = int(result.cv_result.summary["ddof"].iloc[0])
        print(f"Mean ± std (ddof={ddof})")
        for key, label in _METRIC_LABELS.items():
            row = result.cv_result.summary.loc[key]
            print(f"{label}: {row['mean']:.6f} ± {row['std']:.6f}")
    else:
        print("Cross-Validation: disabled")

    print("\nVALIDATION (EXPLORATORY)")
    print("Validation belongs to Development; use CV to compare hyperparameters.")
    for key, label in _METRIC_LABELS.items():
        print(f"{label}: {result.validation_metrics[key]:.6f}")
    print("\nTRAINING")
    print(f"Candidate duration: {result.trained_model.duration_seconds:.3f} seconds")
    print(f"Total duration: {result.duration_seconds:.3f} seconds")
    print(f"MLflow Run ID: {result.run_id}")
    print("\nSTATUS: SUCCESS")
    print("=" * 41)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Development CV and report exploratory Validation metrics."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="JSON training configuration"
    )
    args = parser.parse_args(argv)
    try:
        result = run_train(load_config(args.config))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
