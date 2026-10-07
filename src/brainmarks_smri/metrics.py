"""Evaluation metrics and bootstrap confidence intervals."""

from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor


def auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """AUROC from class indices and class probabilities (n, n_classes); macro one-vs-rest for
    multiclass. NaN if a class is missing from `labels`."""
    n_classes = probabilities.shape[1]
    if len(np.unique(labels)) < n_classes:
        return np.nan
    if n_classes == 2:
        return float(roc_auc_score(labels, probabilities[:, 1]))
    return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))


def pearson_r(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.corrcoef(targets, predictions)[0, 1])


def dice(predicted: Tensor, target: Tensor) -> Tensor:
    """Dice per channel of boolean masks (C, ...). An empty prediction of an empty target is 1."""
    predicted = predicted.flatten(1)
    target = target.flatten(1)
    overlap = (predicted & target).sum(dim=1)
    total = predicted.sum(dim=1) + target.sum(dim=1)
    return torch.where(total > 0, 2 * overlap / total.clamp_min(1), 1.0)


def bootstrap_ci(
    metric: Callable[..., float],
    *arrays: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> list[float]:
    """95% percentile interval of `metric(*arrays)` over resamples of the rows. NaN values (e.g.
    an AUROC on a resample that misses a class) are ignored."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    values = []
    for _ in range(n_bootstrap):
        ids = rng.integers(0, n, n)
        values.append(metric(*[array[ids] for array in arrays]))
    return np.nanpercentile(values, [2.5, 97.5]).tolist()
