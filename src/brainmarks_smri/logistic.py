"""Multi-output L2 logistic regression in torch, fit with full-batch L-BFGS."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor

from brainmarks_smri.metrics import dice


class TorchLogisticRegressionCV:
    """Like sklearn's `LogisticRegressionCV`, but scored by Dice, which also picks the threshold."""

    def __init__(
        self,
        alphas: tuple[float, ...] = (1e1, 1e2, 1e3, 1e4, 1e5),
        thresholds: Tensor = torch.logspace(-3, -0.1, 30),
        cv=GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0),
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
        self, features: Tensor, targets: Tensor, groups: np.ndarray
    ) -> "TorchLogisticRegressionCV":
        """`targets` is (n, n_outputs) bool; outputs share the features and the threshold."""
        splits = list(self.cv.split(np.zeros(len(groups)), groups=groups))
        # Dice per (split, alpha, threshold), pooled over the split's validation outputs
        cv_dice = torch.zeros(len(splits), len(self.alphas), len(self.thresholds))
        for split_id, (train_ids, val_ids) in enumerate(splits):
            train_ids = torch.as_tensor(train_ids, device=features.device)
            val_ids = torch.as_tensor(val_ids, device=features.device)
            for alpha_id, alpha in enumerate(self.alphas):
                self.fit_alpha(features[train_ids], targets[train_ids], alpha)
                val_probabilities = torch.sigmoid(self.decision_function(features[val_ids]))
                for threshold_id, threshold in enumerate(self.thresholds):
                    predicted = val_probabilities >= threshold
                    cv_dice[split_id, alpha_id, threshold_id] = dice(predicted, targets[val_ids])
        self.cv_dice_ = cv_dice.mean(dim=0)  # (alphas, thresholds)

        best = np.unravel_index(int(self.cv_dice_.argmax()), self.cv_dice_.shape)
        alpha_id, threshold_id = int(best[0]), int(best[1])
        self.alpha_ = self.alphas[alpha_id]
        self.threshold_ = float(self.thresholds[threshold_id])
        self.fit_alpha(features, targets, self.alpha_)
        return self

    def fit_alpha(self, features: Tensor, targets: Tensor, alpha: float) -> None:
        """Fit with one penalty: subsample negative rows, standardize, L-BFGS."""
        if self.max_negative_ratio is not None:
            positive = targets.any(dim=1)
            n_positive = positive.sum()
            n_negative = (~positive).sum().clamp_min(1)
            negative_fraction = self.max_negative_ratio * n_positive / n_negative
            generator = torch.Generator().manual_seed(self.seed)
            random = torch.rand(len(positive), generator=generator).to(features.device)
            keep = positive | (random < negative_fraction)
            features = features[keep]
            targets = targets[keep]

        features = features.float()
        self.mean_ = features.mean(dim=0)
        self.std_ = features.std(dim=0, correction=0).clamp_min(1e-6)
        features = (features - self.mean_) / self.std_
        self.coef_, self.intercept_ = fit_logistic(features, targets.float(), alpha, self.max_iter)

    def decision_function(self, features: Tensor) -> Tensor:
        features = (features.float() - self.mean_) / self.std_
        return features @ self.coef_ + self.intercept_

    def predict(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.decision_function(features)) >= self.threshold_


def fit_logistic(
    features: Tensor, targets: Tensor, alpha: float, max_iter: int = 1000
) -> tuple[Tensor, Tensor]:
    """L2 logistic regression with L-BFGS, one binary problem per target column."""
    n, d = features.shape
    n_outputs = targets.shape[1]
    coef = torch.zeros(d, n_outputs, device=features.device, dtype=features.dtype)
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
    return coef.detach(), intercept.detach()
