"""Threshold-independent metrics for binary TARGET probabilities."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def calculate_metrics(y_true: ArrayLike, probabilities: ArrayLike) -> dict[str, float]:
    """Score equally sized 1D arrays positionally; no threshold is applied.

    Both TARGET classes 0 and 1 must be present. Probabilities must be finite
    and lie in [0, 1]. Invalid inputs raise ValueError. LogLoss uses sklearn's
    clipping at the floating-point boundaries, so exact 0 and 1 are accepted.
    """
    target = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if target.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional arrays")
    if target.size != scores.size:
        raise ValueError("y_true and probabilities must have equal lengths")
    if target.size == 0:
        raise ValueError("y_true and probabilities cannot be empty")
    if not np.isfinite(target).all() or not np.isfinite(scores).all():
        raise ValueError("y_true and probabilities must contain only finite values")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("y_true must contain both binary classes 0 and 1")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return {
        "roc_auc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
        "log_loss": float(log_loss(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
    }
