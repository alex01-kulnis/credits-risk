"""Known metric values and input validation, independent of any model."""

from math import log

import numpy as np
import pandas as pd
import pytest

from src.evaluation import calculate_metrics


def test_metrics_match_hand_calculated_values():
    result = calculate_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert set(result) == {"roc_auc", "average_precision", "log_loss", "brier_score"}
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["log_loss"] == pytest.approx(-sum(map(log, [0.9, 0.6, 0.35, 0.8])) / 4)
    assert result["brier_score"] == pytest.approx(0.158125)
    assert all(isinstance(value, float) for value in result.values())


def test_tied_predictions_and_boundary_probabilities():
    tied = calculate_metrics([0, 0, 1, 1], [0.5] * 4)
    assert tied == pytest.approx({"roc_auc": 0.5, "average_precision": 0.5,
                                 "log_loss": log(2), "brier_score": 0.25})
    perfect = calculate_metrics([0, 0, 1, 1], [0., 0., 1., 1.])
    assert perfect["roc_auc"] == 1.0
    assert perfect["average_precision"] == 1.0
    assert 0 <= perfect["log_loss"] < 1e-12
    assert perfect["brier_score"] == 0.0


def test_metrics_do_not_modify_inputs_or_align_pandas_indexes():
    target = pd.Series([0, 0, 1, 1], index=[40, 30, 20, 10])
    probabilities = pd.Series([0.1, 0.4, 0.35, 0.8], index=[1, 2, 3, 4])
    original_target, original_probabilities = target.copy(), probabilities.copy()
    assert calculate_metrics(target, probabilities)["roc_auc"] == pytest.approx(0.75)
    pd.testing.assert_series_equal(target, original_target)
    pd.testing.assert_series_equal(probabilities, original_probabilities)


@pytest.mark.parametrize("target, probabilities, message", [
    ([0, 1], [0.5], "length|size"),
    ([], [], "empty"),
    ([[0], [1]], [0.1, 0.9], "one-dimensional|1D"),
    ([0, 1], [[0.1], [0.9]], "one-dimensional|1D"),
    (1, [0.1], "one-dimensional|1D"),
    ([0, 1], [0.1, np.nan], "finite|NaN"),
    ([0, 1], [0.1, np.inf], "finite|NaN"),
    ([0, 1], [-np.inf, 0.9], "finite|NaN"),
    ([0, np.nan], [0.1, 0.9], "finite|NaN"),
    ([0, np.inf], [0.1, 0.9], "finite|NaN"),
    ([0, 1], [-0.1, 0.9], "range|\\[0, 1\\]"),
    ([0, 1], [0.1, 1.1], "range|\\[0, 1\\]"),
    ([0, 0], [0.1, 0.2], "classes|binary"),
    ([1, 1], [0.8, 0.9], "classes|binary"),
    ([0, 2], [0.1, 0.9], "classes|binary"),
    ([-1, 1], [0.1, 0.9], "classes|binary"),
    ([0, 0.5, 1], [0.1, 0.5, 0.9], "classes|binary"),
])
def test_invalid_metric_input_is_rejected(target, probabilities, message):
    with pytest.raises(ValueError, match=message):
        calculate_metrics(target, probabilities)
