"""Stratified CV using the shared training and probability functions."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from .data import DataSplit
from .evaluation import calculate_metrics
from .model import ModelConfig, predict_probabilities, train_model

_METRIC_COLUMNS = ["roc_auc", "average_precision", "log_loss", "brier_score"]
_OOF_COLUMNS = ["SK_ID_CURR", "fold", "target", "probability"]


@dataclass(frozen=True)
class CrossValidationConfig:
    """Sequential stratified folds; random_state is ignored when shuffle=False."""

    enabled: bool = True
    n_splits: int = 3
    shuffle: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        for name in ("enabled", "shuffle"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if type(self.n_splits) is not int or self.n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32)")


@dataclass
class CrossValidationResult:
    """Per-fold metrics/times, metric mean/std (ddof=1), and OOF in input row order.

    Folds are numbered from 1. OOF has a fresh RangeIndex and retains original
    IDs and targets. No fold models are retained. duration_seconds measures
    the whole CV call; fold durations measure only train_model's fit().
    """

    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    oof: pd.DataFrame
    duration_seconds: float


def _empty_cv_result() -> CrossValidationResult:
    return CrossValidationResult(
        fold_metrics=pd.DataFrame(
            columns=["fold", *_METRIC_COLUMNS, "duration_seconds"]
        ),
        summary=pd.DataFrame(columns=["mean", "std", "ddof"]).rename_axis("metric"),
        oof=pd.DataFrame(columns=_OOF_COLUMNS),
        duration_seconds=0.0,
    )


def _validate_development(development: DataSplit, n_splits: int) -> None:
    if development.X.empty:
        raise ValueError(
            "Development must contain rows and features; data cannot be empty"
        )
    if len(development.X) != len(development.y) or len(development.X) != len(
        development.ids
    ):
        raise ValueError("Development X, y and ids must have equal lengths")
    if not development.X.index.equals(
        development.y.index
    ) or not development.X.index.equals(development.ids.index):
        raise ValueError("Development X, y and ids indexes must align in row order")
    if development.ids.isna().any() or not development.ids.is_unique:
        raise ValueError("Development ids must be unique and contain no null values")
    if development.y.isna().any() or set(development.y.unique()) != {0, 1}:
        raise ValueError(
            "Development TARGET must contain both classes 0 and 1, with no nulls"
        )
    minority_count = int(development.y.value_counts().min())
    if minority_count < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds minority class count={minority_count}"
        )


def _make_fold_split(development: DataSplit, indices: NDArray[np.int64]) -> DataSplit:
    return DataSplit(
        X=development.X.iloc[indices].copy(),
        y=development.y.iloc[indices].copy(),
        ids=development.ids.iloc[indices].copy(),
    )


def cross_validate(
    development: DataSplit,
    model_config: ModelConfig,
    cv_config: CrossValidationConfig,
    *,
    categorical_features: list[str],
) -> CrossValidationResult:
    """Evaluate fixed-iteration models on Development, never accessing Test.

    Each ID is validated exactly once by StratifiedKFold. Mean/std summarize
    fold metrics (not pooled OOF metrics); std uses sample ddof=1, also recorded
    in summary. Disabled CV returns empty tables without validation or training.
    """
    if not cv_config.enabled:
        return _empty_cv_result()

    started = perf_counter()
    _validate_development(development, cv_config.n_splits)
    splitter = StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=cv_config.random_state if cv_config.shuffle else None,
    )
    fold_records = []
    oof_predictions = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(development.X, development.y), start=1
    ):
        train = _make_fold_split(development, train_idx)
        validation = _make_fold_split(development, valid_idx)
        trained = train_model(
            train, model_config, categorical_features=categorical_features
        )
        probabilities = predict_probabilities(
            trained.model, validation.X, categorical_features=categorical_features
        )
        metrics = calculate_metrics(validation.y, probabilities)
        fold_records.append(
            {
                "fold": fold_number,
                **metrics,
                "duration_seconds": trained.duration_seconds,
            }
        )
        oof_predictions.append(
            pd.DataFrame(
                {
                    "SK_ID_CURR": validation.ids.to_numpy(),
                    "fold": fold_number,
                    "target": validation.y.to_numpy(),
                    "probability": probabilities,
                },
                index=valid_idx,
            )
        )

    fold_metrics = pd.DataFrame(fold_records)
    scores = fold_metrics[_METRIC_COLUMNS]
    summary = pd.DataFrame(
        {"mean": scores.mean(), "std": scores.std(ddof=1), "ddof": 1}
    ).rename_axis("metric")
    # valid_idx contains original row positions, independent of DataFrame labels.
    oof = pd.concat(oof_predictions).sort_index().reset_index(drop=True)
    return CrossValidationResult(
        fold_metrics=fold_metrics,
        summary=summary,
        oof=oof,
        duration_seconds=perf_counter() - started,
    )
