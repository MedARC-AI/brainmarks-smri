"""Multi-output L2 logistic regression in torch, fit with full-batch L-BFGS."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, GroupShuffleSplit
from torch import Tensor

from brainmarks_smri.metrics import dice


class TorchLogisticRegressionCV:
    """Like sklearn's `LogisticRegressionCV`, but scored by Dice, which also picks the threshold."""

    def __init__(
        self,
        alphas: tuple[float, ...] = (1e1, 1e2, 1e3, 1e4, 1e5),
        thresholds: Tensor = torch.logspace(-3, -0.1, 30),
        cv: BaseCrossValidator | BaseShuffleSplit = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=0
        ),
        # rows without a positive target are subsampled to about this many per positive row
        max_negative_ratio: float | None = 10.0,
        max_iter: int = 1000,
        seed: int = 0,
    ):
        self.alphas = alphas
        self.thresholds = thresholds
        self.cv = cv
        self.max_negative_ratio = max_negative_ratio
        self.max_iter = max_iter
        self.seed = seed

    def fit(
        self, features: Tensor, targets: Tensor, groups: np.ndarray | None = None
    ) -> "TorchLogisticRegressionCV":
        """`targets` is (n, n_outputs) bool; outputs share the features and the threshold."""
        fit_kwargs = dict(
            max_negative_ratio=self.max_negative_ratio, max_iter=self.max_iter, seed=self.seed
        )
        splits = list(self.cv.split(np.zeros(len(features)), groups=groups))
        # Dice per (split, alpha, threshold), pooled over the split's validation outputs
        cv_dice = torch.zeros(len(splits), len(self.alphas), len(self.thresholds))
        for split_id, (train_ids, val_ids) in enumerate(splits):
            train_ids = torch.as_tensor(train_ids, device=features.device)
            val_ids = torch.as_tensor(val_ids, device=features.device)
            for alpha_id, alpha in enumerate(self.alphas):
                coef, intercept = fit_logistic(
                    features[train_ids], targets[train_ids], alpha, **fit_kwargs
                )
                val_logits = features[val_ids].float() @ coef + intercept
                val_probabilities = torch.sigmoid(val_logits)
                for threshold_id, threshold in enumerate(self.thresholds):
                    predicted = val_probabilities >= threshold
                    cv_dice[split_id, alpha_id, threshold_id] = dice(predicted, targets[val_ids])
        self.cv_dice_ = cv_dice.mean(dim=0)  # (alphas, thresholds)

        best = np.unravel_index(int(self.cv_dice_.argmax()), self.cv_dice_.shape)
        alpha_id, threshold_id = int(best[0]), int(best[1])
        self.alpha_ = self.alphas[alpha_id]
        self.threshold_ = float(self.thresholds[threshold_id])
        self.coef_, self.intercept_ = fit_logistic(features, targets, self.alpha_, **fit_kwargs)
        return self

    def decision_function(self, features: Tensor) -> Tensor:
        return features.float() @ self.coef_ + self.intercept_

    def predict_proba(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.decision_function(features))

    def predict(self, features: Tensor) -> Tensor:
        return self.predict_proba(features) >= self.threshold_


def fit_logistic(
    features: Tensor,
    targets: Tensor,
    alpha: float,
    max_negative_ratio: float | None = None,
    max_iter: int = 1000,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """L2 logistic regression on standardized features, one binary problem per target column.

    Returns weights and intercepts that apply to the raw features.
    """
    if max_negative_ratio is not None:
        # keep every row with a positive target, and about max_negative_ratio negatives per positive
        positive = targets.any(dim=1)
        n_positive = positive.sum()
        n_negative = (~positive).sum().clamp_min(1)
        negative_fraction = max_negative_ratio * n_positive / n_negative
        generator = torch.Generator().manual_seed(seed)
        random = torch.rand(len(positive), generator=generator).to(features.device)
        keep = positive | (random < negative_fraction)
        features = features[keep]
        targets = targets[keep]

    features = features.float()
    targets = targets.float()
    mean = features.mean(dim=0)
    std = features.std(dim=0, correction=0).clamp_min(1e-6)
    features = (features - mean) / std

    n, d = features.shape
    n_outputs = targets.shape[1]
    coef = torch.zeros(d, n_outputs, device=features.device)
    intercept = torch.logit(targets.mean(dim=0).clamp(1e-6, 1 - 1e-6))
    coef.requires_grad_(True)
    intercept.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [coef, intercept], max_iter=max_iter, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = features @ coef + intercept
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + alpha * coef.square().sum() / (n * n_outputs)
        loss.backward()
        return loss

    optimizer.step(closure)
    # l-bfgs stops on its own gradient and step tolerances; exhausting the budget means neither met
    n_iter = optimizer.state[coef]["n_iter"]
    assert n_iter < max_iter, f"l-bfgs used all {max_iter} iterations without converging"

    # fold the standardization into the weights: (x - mean) / std @ w + b = x @ w' + b'
    coef = coef.detach()
    raw_coef = coef / std[:, None]
    raw_intercept = intercept.detach() - (mean / std) @ coef
    return raw_coef, raw_intercept
