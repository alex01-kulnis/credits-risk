"""Exercise real CatBoost training on tiny data; never load historical data."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool

import src.model as model_module
from src.data import DataSplit
from src.model import ModelConfig, predict_probabilities, train_model


@pytest.fixture
def train() -> DataSplit:
    index = pd.Index(range(100, 160), name="row")
    return DataSplit(
        X=pd.DataFrame({
            "income": np.linspace(1000., 9000., 60),
            "segment": ["employed", "student", "self_employed"] * 20,
        }, index=index),
        y=pd.Series([0, 1, 0, 1, 0, 1] * 10, index=index, name="TARGET"),
        ids=pd.Series(range(10001, 10061), index=index, name="SK_ID_CURR"),
    )


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(iterations=20, depth=3, learning_rate=0.1, random_seed=42,
                       thread_count=1, verbose=False)


def test_trains_requested_trees_and_records_duration(train, config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = train_model(train, config, categorical_features=["segment"])
    assert isinstance(result.model, CatBoostClassifier)
    assert result.model.is_fitted()
    assert result.model.tree_count_ == 20
    assert np.isfinite(result.duration_seconds) and result.duration_seconds >= 0
    assert result.model.get_cat_feature_indices() == [1]
    parameters = result.model.get_params()
    assert parameters["loss_function"] == "Logloss"
    assert parameters["depth"] == 3
    assert parameters["learning_rate"] == 0.1
    assert parameters["l2_leaf_reg"] == 3.0
    assert parameters["random_seed"] == 42
    assert parameters["allow_writing_files"] is False
    assert parameters["use_best_model"] is False
    assert "early_stopping_rounds" not in parameters
    assert "od_type" not in parameters
    assert result.model.get_evals_result().keys() == {"learn"}
    assert list(tmp_path.iterdir()) == []


def test_probabilities_are_one_dimensional_positive_class_values(train, config):
    result = train_model(train, config, categorical_features=["segment"])
    probabilities = predict_probabilities(result.model, train.X, categorical_features=["segment"])
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (60,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    expected = result.model.predict_proba(Pool(train.X, cat_features=["segment"]))[:, 1]
    np.testing.assert_allclose(probabilities, expected, rtol=1e-12, atol=1e-12)
    assert result.model.classes_.tolist() == [0, 1]


def test_repeated_training_is_reproducible(train, config):
    first = train_model(train, config, categorical_features=["segment"])
    second = train_model(train, config, categorical_features=["segment"])
    np.testing.assert_allclose(
        predict_probabilities(first.model, train.X, categorical_features=["segment"]),
        predict_probabilities(second.model, train.X, categorical_features=["segment"]),
        rtol=1e-10, atol=1e-12,
    )


def test_training_and_prediction_leave_inputs_unchanged(train, config):
    original_X, original_y, original_ids = train.X.copy(deep=True), train.y.copy(deep=True), train.ids.copy(deep=True)
    categories = ["segment"]
    result = train_model(train, config, categorical_features=categories)
    predict_probabilities(result.model, train.X, categorical_features=categories)
    pd.testing.assert_frame_equal(train.X, original_X)
    pd.testing.assert_series_equal(train.y, original_y)
    pd.testing.assert_series_equal(train.ids, original_ids)
    assert categories == ["segment"]


def test_numeric_only_features_are_supported(train, config):
    numeric = replace(train, X=train.X[["income"]])
    result = train_model(numeric, config, categorical_features=[])
    assert result.model.tree_count_ == 20
    assert predict_probabilities(result.model, numeric.X, categorical_features=[]).shape == (60,)


def test_categorical_only_data_can_be_learned(config):
    split = DataSplit(
        X=pd.DataFrame({"segment": ["employed", "student"] * 20}),
        y=pd.Series([0, 1] * 20, name="TARGET"),
        ids=pd.Series(range(40), name="SK_ID_CURR"),
    )
    result = train_model(split, config, categorical_features=["segment"])
    probabilities = predict_probabilities(result.model, split.X.iloc[:2], categorical_features=["segment"])
    assert probabilities[1] > probabilities[0]


@pytest.mark.parametrize("changes", [
    {"iterations": 0}, {"iterations": -1}, {"iterations": 10.5}, {"iterations": True},
    {"depth": 0}, {"depth": 17}, {"depth": 3.5}, {"depth": False},
    {"learning_rate": 0}, {"learning_rate": -0.1}, {"learning_rate": float("nan")},
    {"learning_rate": float("inf")}, {"learning_rate": "0.1"}, {"learning_rate": True},
    {"l2_leaf_reg": -1}, {"l2_leaf_reg": float("nan")}, {"l2_leaf_reg": float("inf")},
    {"l2_leaf_reg": "3"}, {"l2_leaf_reg": False},
    {"random_seed": -1}, {"random_seed": 1.5}, {"random_seed": True},
    {"thread_count": 0}, {"thread_count": -2}, {"thread_count": 1.5}, {"thread_count": True},
    {"loss_function": "MultiClass"},
    {"verbose": 0}, {"verbose": -1}, {"verbose": 1.5},
    {"verbose": "100"}, {"verbose": None},
])
def test_invalid_config_is_rejected(changes):
    field = next(iter(changes))
    with pytest.raises(ValueError, match=field):
        ModelConfig(**changes)


def test_config_cannot_be_mutated_after_validation(config):
    with pytest.raises(FrozenInstanceError):
        config.iterations = 0


def test_valid_boundary_config_trains(train):
    config = ModelConfig(iterations=10, depth=1, learning_rate=1., l2_leaf_reg=0., random_seed=0,
                         thread_count=1, verbose=False)
    result = train_model(train, config, categorical_features=["segment"])
    assert result.model.tree_count_ == 10


@pytest.mark.parametrize("change, message", [
    (lambda s: replace(s, y=s.y.iloc[:-1]), "length|rows"),
    (lambda s: replace(s, ids=s.ids.iloc[:-1]), "length|rows"),
    (lambda s: replace(s, y=s.y.iloc[::-1]), "index|align"),
    (lambda s: replace(s, ids=s.ids.iloc[::-1]), "index|align"),
    (lambda s: replace(s, ids=pd.Series([1] * 60, index=s.ids.index)), "unique|duplicate"),
    (lambda s: replace(s, ids=s.ids.astype("Int64").mask(s.ids == 10001)), "null|missing"),
    (lambda s: replace(s, y=s.y * 0), "TARGET|classes"),
    (lambda s: replace(s, y=s.y * 2), "TARGET|classes"),
    (lambda s: replace(s, y=s.y.astype("Int64").mask(s.y == 1)), "TARGET|classes"),
    (lambda s: replace(s, X=s.X.iloc[:0], y=s.y.iloc[:0], ids=s.ids.iloc[:0]), "empty|rows"),
    (lambda s: replace(s, X=s.X.assign(TARGET=s.y)), "TARGET"),
    (lambda s: replace(s, X=s.X.assign(SK_ID_CURR=s.ids)), "SK_ID_CURR"),
    (lambda s: replace(s, X=s.X.set_axis(["income", "income"], axis=1)), "unique|duplicate"),
    (lambda s: replace(s, X=s.X.set_axis(["income", 1], axis=1)), "names|strings"),
])
def test_invalid_training_data_is_rejected_before_pool(train, config, monkeypatch, change, message):
    def forbidden_pool(*args, **kwargs):
        raise AssertionError("Invalid input reached CatBoost Pool")
    monkeypatch.setattr(model_module, "Pool", forbidden_pool)
    with pytest.raises(ValueError, match=message):
        train_model(change(train), config, categorical_features=["segment"])


@pytest.mark.parametrize("categories", [["unknown"], ["segment", "segment"], [1]])
def test_invalid_categories_are_rejected_before_pool(train, config, monkeypatch, categories):
    def forbidden_pool(*args, **kwargs):
        raise AssertionError("Invalid categorical features reached CatBoost Pool")
    monkeypatch.setattr(model_module, "Pool", forbidden_pool)
    with pytest.raises(ValueError, match="categorical_features"):
        train_model(train, config, categorical_features=categories)


@pytest.mark.parametrize("change, categories", [
    (lambda X: X[["segment", "income"]], ["segment"]),
    (lambda X: X[["income"]], []),
    (lambda X: X.assign(extra=1), ["segment"]),
    (lambda X: X, []),
])
def test_prediction_schema_must_match_training(train, config, change, categories):
    result = train_model(train, config, categorical_features=["segment"])
    with pytest.raises(ValueError, match="features|columns|categorical"):
        predict_probabilities(result.model, change(train.X), categorical_features=categories)


def test_prediction_rejects_unfitted_model(train):
    with pytest.raises(ValueError, match="fitted|trained"):
        predict_probabilities(CatBoostClassifier(thread_count=1, verbose=False), train.X,
                              categorical_features=["segment"])


@pytest.mark.parametrize("labels, loss", [([0, 2], "Logloss"), ([0, 1, 2], "MultiClass")])
def test_prediction_rejects_models_with_other_target_classes(train, labels, loss):
    model = CatBoostClassifier(iterations=10, depth=2, thread_count=1, loss_function=loss,
                               allow_writing_files=False, verbose=False)
    y = np.resize(labels, len(train.X))
    model.fit(Pool(train.X, label=y, cat_features=["segment"]))
    with pytest.raises(ValueError, match="classes|TARGET"):
        predict_probabilities(model, train.X, categorical_features=["segment"])


def test_default_training_resources_and_hyperparameters():
    config = ModelConfig()
    assert config.thread_count == -1
    assert config.verbose == 100
    assert (config.iterations, config.depth, config.learning_rate,
            config.l2_leaf_reg, config.random_seed, config.loss_function) == (
                1200, 6, 0.05, 3.0, 42, "Logloss")


@pytest.mark.parametrize("verbose", [False, True, 1, 100])
def test_valid_verbosity_settings(verbose):
    assert ModelConfig(thread_count=1, verbose=verbose).verbose == verbose


def test_logging_period_is_forwarded_to_catboost(train, config, capsys):
    result = train_model(train, replace(config, iterations=10, verbose=5),
                         categorical_features=["segment"])
    output = capsys.readouterr().out
    assert "0:" in output and "5:" in output and "9:" in output
    assert result.model.get_params()["verbose"] == 5


def test_verbose_false_is_quiet(train, config, capsys):
    train_model(train, config, categorical_features=["segment"])
    assert capsys.readouterr().out == ""
