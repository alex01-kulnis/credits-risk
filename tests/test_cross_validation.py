"""Real, small CV runs with explicit one-thread, quiet training."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import src.cross_validation as cv_module
import src.model as model_module
from src.cross_validation import CrossValidationConfig, cross_validate
from src.data import DataSplit
from src.evaluation import calculate_metrics
from src.model import ModelConfig


@pytest.fixture
def development() -> DataSplit:
    rng = np.random.default_rng(2026)
    index = pd.Index(rng.permutation(range(1000, 1072)), name="original_row")
    return DataSplit(
        X=pd.DataFrame({"income": rng.normal(5000, 1000, 72),
                        "segment": ["employed", "student", "self_employed"] * 24}, index=index),
        y=pd.Series(rng.permutation([0, 0, 0, 1] * 18), index=index, name="TARGET"),
        ids=pd.Series(rng.permutation(range(70001, 70073)), index=index, name="SK_ID_CURR"),
    )


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(iterations=10, depth=3, thread_count=1, verbose=False)


def test_real_folds_delegate_training_and_preserve_oof_identity(development, model_config, monkeypatch):
    assert cv_module.train_model is model_module.train_model
    real_train, real_predict = cv_module.train_model, cv_module.predict_probabilities
    training_calls, prediction_calls = [], []

    def record_training(train, config, *, categorical_features):
        assert config is model_config
        assert categorical_features == ["segment"]
        trained = real_train(train, config, categorical_features=categorical_features)
        training_calls.append({"ids": train.ids.tolist(), "trees": trained.model.tree_count_,
                               "duration": trained.duration_seconds, "model_id": id(trained.model)})
        return trained

    def record_prediction(model, X, *, categorical_features):
        assert id(model) == training_calls[-1]["model_id"]
        probabilities = real_predict(model, X, categorical_features=categorical_features)
        prediction_calls.append({"ids": development.ids.loc[X.index].tolist(),
                                 "probabilities": probabilities.copy()})
        return probabilities

    monkeypatch.setattr(cv_module, "train_model", record_training)
    monkeypatch.setattr(cv_module, "predict_probabilities", record_prediction)
    result = cross_validate(development, model_config, CrossValidationConfig(),
                            categorical_features=["segment"])
    assert len(training_calls) == len(prediction_calls) == 3
    assert result.fold_metrics["fold"].tolist() == [1, 2, 3]
    assert result.oof.columns.tolist() == ["SK_ID_CURR", "fold", "target", "probability"]
    assert result.oof["SK_ID_CURR"].tolist() == development.ids.tolist()
    assert result.oof["target"].tolist() == development.y.tolist()
    assert result.oof["SK_ID_CURR"].is_unique
    assert len(result.oof) == 72
    assert result.oof["probability"].between(0, 1).all()
    assert np.isfinite(result.oof["probability"]).all()
    all_ids = set(development.ids)
    for fold, (train_call, predict_call) in enumerate(zip(training_calls, prediction_calls), start=1):
        oof_fold = result.oof.loc[result.oof["fold"] == fold]
        assert len(train_call["ids"]) == 48 and len(predict_call["ids"]) == 24
        assert set(train_call["ids"]).isdisjoint(predict_call["ids"])
        assert set(train_call["ids"]) | set(predict_call["ids"]) == all_ids
        assert set(oof_fold["SK_ID_CURR"]) == set(predict_call["ids"])
        assert oof_fold["target"].value_counts().to_dict() == {0: 18, 1: 6}
        assert train_call["trees"] == 10
        fold_row = result.fold_metrics.loc[result.fold_metrics["fold"] == fold].iloc[0]
        assert fold_row["duration_seconds"] == train_call["duration"]
        expected = calculate_metrics(oof_fold["target"], oof_fold["probability"])
        for metric, value in expected.items():
            assert fold_row[metric] == pytest.approx(value)
        by_id = oof_fold.set_index("SK_ID_CURR").loc[predict_call["ids"], "probability"]
        np.testing.assert_allclose(by_id, predict_call["probabilities"])
    assert result.duration_seconds >= result.fold_metrics["duration_seconds"].sum()
    assert result.summary.index.name == "metric"
    assert result.summary.columns.tolist() == ["mean", "std", "ddof"]
    for metric in ["roc_auc", "average_precision", "log_loss", "brier_score"]:
        values = result.fold_metrics[metric].to_numpy()
        assert result.summary.loc[metric, "mean"] == pytest.approx(values.mean())
        assert result.summary.loc[metric, "std"] == pytest.approx(values.std(ddof=1))
        assert result.summary.loc[metric, "ddof"] == 1


def test_same_seed_reproduces_folds_and_predictions(development, model_config):
    config = CrossValidationConfig(random_state=17)
    first = cross_validate(development, model_config, config, categorical_features=["segment"])
    second = cross_validate(development, model_config, config, categorical_features=["segment"])
    pd.testing.assert_frame_equal(first.oof, second.oof)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.fold_metrics.drop(columns="duration_seconds"),
                                  second.fold_metrics.drop(columns="duration_seconds"))
    expected_folds = np.zeros(72, dtype=int)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    for number, (_, valid_idx) in enumerate(splitter.split(development.X, development.y), start=1):
        expected_folds[valid_idx] = number
    np.testing.assert_array_equal(first.oof["fold"], expected_folds)


def test_shuffle_false_ignores_random_state(development, model_config):
    first = cross_validate(development, model_config, CrossValidationConfig(shuffle=False),
                           categorical_features=["segment"])
    second = cross_validate(development, model_config, CrossValidationConfig(shuffle=False, random_state=99),
                            categorical_features=["segment"])
    pd.testing.assert_frame_equal(first.oof, second.oof)


def test_minority_class_count_equal_to_n_splits_is_allowed(development, model_config):
    development = replace(development, y=pd.Series([1] * 3 + [0] * 69,
                                                   index=development.y.index, name="TARGET"))
    result = cross_validate(development, model_config, CrossValidationConfig(),
                            categorical_features=["segment"])
    assert result.oof.groupby("fold")["target"].sum().tolist() == [1, 1, 1]


@pytest.mark.parametrize("n_splits", [2, 4])
def test_fold_count_is_configurable_without_categorical_features(development, model_config, n_splits):
    development = replace(development, X=development.X[["income"]])
    result = cross_validate(development, model_config, CrossValidationConfig(n_splits=n_splits),
                            categorical_features=[])
    assert result.fold_metrics["fold"].tolist() == list(range(1, n_splits + 1))
    assert len(result.oof) == 72 and result.oof["SK_ID_CURR"].is_unique


def test_cv_leaves_inputs_unchanged(development, model_config):
    X, y, ids = development.X.copy(deep=True), development.y.copy(deep=True), development.ids.copy(deep=True)
    categories = ["segment"]
    cross_validate(development, model_config, CrossValidationConfig(), categorical_features=categories)
    pd.testing.assert_frame_equal(development.X, X)
    pd.testing.assert_series_equal(development.y, y)
    pd.testing.assert_series_equal(development.ids, ids)
    assert categories == ["segment"]


def test_disabled_cv_returns_empty_result_without_training(development, model_config, monkeypatch):
    def forbidden_training(*args, **kwargs):
        raise AssertionError("Disabled CV attempted training")
    monkeypatch.setattr(cv_module, "train_model", forbidden_training)
    empty = DataSplit(development.X.iloc[:0], development.y.iloc[:0], development.ids.iloc[:0])
    result = cross_validate(empty, model_config, CrossValidationConfig(enabled=False),
                            categorical_features=["segment"])
    assert result.fold_metrics.empty and result.summary.empty and result.oof.empty
    assert result.duration_seconds == 0.0
    assert result.fold_metrics.columns.tolist() == [
        "fold", "roc_auc", "average_precision", "log_loss", "brier_score", "duration_seconds"]
    assert result.summary.columns.tolist() == ["mean", "std", "ddof"]
    assert result.oof.columns.tolist() == ["SK_ID_CURR", "fold", "target", "probability"]


@pytest.mark.parametrize("changes", [
    {"enabled": 1}, {"enabled": "false"}, {"shuffle": 0}, {"shuffle": None},
    {"n_splits": 1}, {"n_splits": 0}, {"n_splits": 2.5}, {"n_splits": True},
    {"random_state": -1}, {"random_state": 2**32}, {"random_state": True},
    {"random_state": None}, {"random_state": "42"},
])
def test_invalid_cv_config_is_rejected(changes):
    with pytest.raises(ValueError, match=next(iter(changes))):
        CrossValidationConfig(**changes)


def test_cv_config_is_frozen():
    config = CrossValidationConfig()
    with pytest.raises(FrozenInstanceError):
        config.n_splits = 1


@pytest.mark.parametrize("change, message", [
    (lambda s: replace(s, y=s.y.iloc[:-1]), "length|rows"),
    (lambda s: replace(s, ids=s.ids.iloc[:-1]), "length|rows"),
    (lambda s: replace(s, y=s.y.iloc[::-1]), "index|align"),
    (lambda s: replace(s, ids=s.ids.iloc[::-1]), "index|align"),
    (lambda s: replace(s, ids=s.ids * 0), "unique|duplicate"),
    (lambda s: replace(s, ids=s.ids.astype("Int64").mask(s.ids == s.ids.iloc[0])), "null|missing"),
    (lambda s: replace(s, y=s.y * 0), "TARGET|classes"),
    (lambda s: replace(s, y=s.y * 2), "TARGET|classes"),
    (lambda s: replace(s, y=s.y.astype("Int64").mask(s.y == 1)), "TARGET|classes"),
    (lambda s: replace(s, X=s.X.iloc[:0], y=s.y.iloc[:0], ids=s.ids.iloc[:0]), "empty|rows"),
    (lambda s: replace(s, y=pd.Series([1, 1] + [0] * 70, index=s.y.index)), "n_splits|minority|class"),
])
def test_invalid_development_is_rejected_before_training(development, model_config, monkeypatch, change, message):
    def forbidden_training(*args, **kwargs):
        raise AssertionError("Invalid Development reached training")
    monkeypatch.setattr(cv_module, "train_model", forbidden_training)
    with pytest.raises(ValueError, match=message):
        cross_validate(change(development), model_config, CrossValidationConfig(),
                       categorical_features=["segment"])
