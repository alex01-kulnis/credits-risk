"""Fixed-iteration CatBoost training and binary probability prediction."""

from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from numpy.typing import NDArray

from .data import DataSplit


@dataclass(frozen=True)
class ModelConfig:
    """CPU binary-classification parameters; construct a new config to change them."""

    iterations: int = 1200
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    random_seed: int = 42
    loss_function: Literal["Logloss"] = "Logloss"
    thread_count: int = -1
    verbose: int | bool = 100

    def __post_init__(self) -> None:
        for name, minimum in (("iterations", 1), ("depth", 1), ("random_seed", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.depth > 16:
            raise ValueError("depth must be in [1, 16] for CPU training")
        if type(self.thread_count) is not int or (
            self.thread_count != -1 and self.thread_count < 1
        ):
            raise ValueError("thread_count must be -1 or a positive integer")
        if type(self.verbose) is not bool and (
            type(self.verbose) is not int or self.verbose <= 0
        ):
            raise ValueError("verbose must be a boolean or a positive integer")
        for name in ("learning_rate", "l2_leaf_reg"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if self.loss_function != "Logloss":
            raise ValueError("loss_function must be Logloss for binary TARGET")


@dataclass
class TrainedModel:
    model: CatBoostClassifier
    duration_seconds: float


def _validate_features(X: pd.DataFrame, categorical_features: list[str]) -> None:
    if X.empty:
        raise ValueError("X must contain rows and columns; empty data cannot be used")
    if not X.columns.is_unique:
        raise ValueError("X column names must be unique")
    if any(not isinstance(name, str) or not name for name in X.columns):
        raise ValueError("X column names must be nonempty strings")
    if "TARGET" in X or "SK_ID_CURR" in X:
        raise ValueError("TARGET and SK_ID_CURR cannot be model features")
    if not isinstance(categorical_features, list) or any(
        not isinstance(name, str) for name in categorical_features
    ):
        raise ValueError("categorical_features must be a list of column names")
    if len(set(categorical_features)) != len(categorical_features):
        raise ValueError("categorical_features must contain unique names")
    missing = set(categorical_features) - set(X.columns)
    if missing:
        raise ValueError(f"categorical_features missing from X: {sorted(missing)}")


def train_model(
    train: DataSplit, config: ModelConfig, *, categorical_features: list[str]
) -> TrainedModel:
    """Fit exactly config.iterations on Train only; duration measures fit(), in seconds.

    X, y and ids must have equal lengths and identical indexes in row order.
    Labels must contain both 0 and 1. Inputs are validated before creating Pool
    and are never modified. Row, label and feature contract violations raise ValueError.
    """
    if len(train.X) != len(train.y) or len(train.X) != len(train.ids):
        raise ValueError("X, y and ids must have equal lengths")
    if not train.X.index.equals(train.y.index) or not train.X.index.equals(
        train.ids.index
    ):
        raise ValueError("X, y and ids indexes must align in the same row order")
    if train.ids.isna().any() or not train.ids.is_unique:
        raise ValueError("ids must be unique and contain no null values")
    _validate_features(train.X, categorical_features)
    if train.y.isna().any() or set(train.y.unique()) != {0, 1}:
        raise ValueError("TARGET must contain both classes 0 and 1, with no nulls")

    pool = Pool(train.X, label=train.y, cat_features=categorical_features)
    model = CatBoostClassifier(
        **asdict(config),
        task_type="CPU",
        use_best_model=False,
        allow_writing_files=False,
    )
    started = perf_counter()
    model.fit(pool)
    return TrainedModel(model=model, duration_seconds=perf_counter() - started)


def predict_probabilities(
    model: CatBoostClassifier, X: pd.DataFrame, *, categorical_features: list[str]
) -> NDArray[np.float64]:
    """Return P(TARGET=1) from a fitted model, preserving X row order.

    The model must have classes [0, 1]. X must match training column order
    and categorical feature names. Invalid model/schema raises ValueError.
    """
    if not model.is_fitted():
        raise ValueError("model must be fitted before predicting probabilities")
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("model must have exactly TARGET classes 0 and 1")
    _validate_features(X, categorical_features)
    if X.columns.tolist() != model.feature_names_:
        raise ValueError("X columns and order must match the trained model features")
    expected_categories = {
        model.feature_names_[i] for i in model.get_cat_feature_indices()
    }
    if set(categorical_features) != expected_categories:
        raise ValueError("categorical_features must match the trained model")
    return model.predict_proba(Pool(X, cat_features=categorical_features))[:, 1]
