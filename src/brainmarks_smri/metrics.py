"""Evaluation metrics and bootstrap confidence intervals."""

from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import rankdata
from torch import Tensor


def auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """AUROC from class indices and class probabilities (n, n_classes); macro one-vs-rest for
    multiclass. NaN if a class is missing from `labels`."""
    n_classes = probabilities.shape[1]
    if len(np.unique(labels)) < n_classes:
        return np.nan
    # binary: the positive class only; multiclass: each class against the rest
    if n_classes == 2:
        positive_classes = [1]
    else:
        positive_classes = list(range(n_classes))
    # from ranks (ties averaged), as sklearn's roc_auc_score but faster for the bootstrap
    values = []
    for class_index in positive_classes:
        positive = labels == class_index
        n_positive = positive.sum()
        n_negative = len(labels) - n_positive
        ranks = rankdata(probabilities[:, class_index])
        # Mann-Whitney U of the positives, as a share of all positive-negative pairs
        u_statistic = ranks[positive].sum() - n_positive * (n_positive + 1) / 2
        values.append(u_statistic / (n_positive * n_negative))
    return float(np.mean(values))


def pearson_r(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.corrcoef(targets, predictions)[0, 1])


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Coefficient of determination, as sklearn's r2_score."""
    residual = ((predictions - targets) ** 2).sum()
    total = ((targets - targets.mean()) ** 2).sum()
    return float(1 - residual / total)


def dice(predicted: Tensor, target: Tensor) -> float:
    """Dice of two boolean masks. An empty prediction of an empty target is 1."""
    overlap = (predicted & target).sum()
    total = predicted.sum() + target.sum()
    if total == 0:
        return 1.0
    return float(2 * overlap / total)


def voxel_auroc(probabilities: Tensor, target: Tensor) -> float:
    """AUROC over voxels, from ranks (ties not merged). NaN without positives."""
    probabilities = probabilities.flatten()
    target = target.flatten()
    n_positive = int(target.sum())
    n_negative = len(target) - n_positive
    if n_positive == 0:
        return np.nan
    ranks = torch.empty(len(target), dtype=torch.float64, device=target.device)
    order = probabilities.argsort()
    ranks[order] = torch.arange(1, len(target) + 1, dtype=torch.float64, device=target.device)
    positive_rank_sum = float(ranks[target].sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def average_precision(probabilities: Tensor, target: Tensor) -> float:
    """Average precision over voxels (ties not merged). NaN without positives."""
    order = probabilities.flatten().argsort(descending=True)
    sorted_target = target.flatten()[order].float()
    n_positive = float(sorted_target.sum())
    if n_positive == 0:
        return np.nan
    ranks = torch.arange(1, len(sorted_target) + 1, device=sorted_target.device)
    precision = sorted_target.cumsum(dim=0) / ranks
    return float((precision * sorted_target).sum()) / n_positive


def bootstrap_score(
    metric: Callable[..., float],
    *arrays: np.ndarray,
    strata: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """`metric(*arrays)`, and its values over resamples of the rows, drawn within each stratum
    if `strata` is given."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    if strata is None:
        strata = np.zeros(n)
    members = [np.flatnonzero(strata == stratum) for stratum in np.unique(strata)]
    values = []
    for _ in range(n_bootstrap):
        ids = np.concatenate([rng.choice(indices, len(indices)) for indices in members])
        values.append(metric(*[array[ids] for array in arrays]))
    return metric(*arrays), np.array(values)


def bootstrap_ci(
    metric: Callable[..., float],
    *arrays: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> list[float]:
    """Percentile interval of `metric(*arrays)` over resamples of the rows, ignoring NaNs."""
    _, values = bootstrap_score(metric, *arrays, n_bootstrap=n_bootstrap, seed=seed)
    tail = 100 * (1 - confidence) / 2
    return np.nanpercentile(values, [tail, 100 - tail]).tolist()
