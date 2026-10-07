"""L2 logistic regression for many binary outputs, fit with full-batch L-BFGS in torch.

Used by the segmentation probe, where each patch embedding predicts the labels of the voxels in
its patch.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from brainmarks_smri.metrics import dice


class LogisticRegressionVal:
    """Like sklearn's `LogisticRegressionCV`, with two differences: the penalty is chosen on a
    validation set rather than by k-fold, and the score is Dice of thresholded probabilities, which
    also picks a decision threshold per channel.

    Targets are bool (n, n_outputs, n_channels): one binary problem per output and channel, sharing
    the features. All outputs of a channel share its threshold (for segmentation, the outputs are
    the voxels of a patch). Features are standardized. Training rows with no positive target are
    subsampled to about `max_negative_ratio` per row with a positive.
    """

    def __init__(
        self,
        alphas: tuple[float, ...] = (1e1, 1e2, 1e3, 1e4, 1e5),
        thresholds: Tensor = torch.logspace(-3, -0.1, 30),
        max_negative_ratio: float | None = 10.0,
        max_iter: int = 1000,
        seed: int = 0,
    ):
        self.alphas = alphas
        self.thresholds = thresholds
        self.max_negative_ratio = max_negative_ratio
        self.max_iter = max_iter
        self.seed = seed

    def fit(
        self, features: Tensor, targets: Tensor, val_features: Tensor, val_targets: Tensor
    ) -> "LogisticRegressionVal":
        n_channels = targets.shape[2]
        self.n_channels_ = n_channels
        device = features.device

        if self.max_negative_ratio is not None:
            positive = targets.flatten(1).any(dim=1)
            negative_fraction = (
                self.max_negative_ratio * positive.sum() / (~positive).sum().clamp_min(1)
            )
            generator = torch.Generator().manual_seed(self.seed)
            random = torch.rand(len(positive), generator=generator).to(device)
            keep = positive | (random < negative_fraction)
            features = features[keep]
            targets = targets[keep]

        features = features.float()
        self.mean_ = features.mean(dim=0)
        self.std_ = features.std(dim=0, correction=0).clamp_min(1e-6)
        features = (features - self.mean_) / self.std_
        targets = targets.flatten(1).float()

        # Dice per (alpha, channel, threshold), pooled over all validation voxels
        val_targets = val_targets.reshape(-1, n_channels).T  # (C, n_val * n_outputs)
        self.val_dice_ = torch.zeros(len(self.alphas), n_channels, len(self.thresholds))
        fits = []
        for alpha_id, alpha in enumerate(self.alphas):
            coef, intercept = fit_logistic(features, targets, alpha, self.max_iter)
            fits.append((coef, intercept))
            # score this fit through decision_function
            self.coef_, self.intercept_ = coef, intercept
            val_probabilities = torch.sigmoid(self.decision_function(val_features))
            val_probabilities = val_probabilities.reshape(-1, n_channels).T
            for threshold_id, threshold in enumerate(self.thresholds):
                predicted = val_probabilities >= threshold
                self.val_dice_[alpha_id, :, threshold_id] = dice(predicted, val_targets).cpu()

        # one penalty for all channels (best mean over channels), one threshold per channel
        best_dice_per_alpha = self.val_dice_.max(dim=2).values.mean(dim=1)
        alpha_id = int(best_dice_per_alpha.argmax())
        self.alpha_ = self.alphas[alpha_id]
        self.thresholds_ = self.thresholds[self.val_dice_[alpha_id].argmax(dim=1)]
        self.coef_, self.intercept_ = fits[alpha_id]
        return self

    def decision_function(self, features: Tensor) -> Tensor:
        """Logits, (n, n_outputs, n_channels)."""
        features = (features.float() - self.mean_) / self.std_
        logits = features @ self.coef_ + self.intercept_
        return logits.reshape(len(features), -1, self.n_channels_)

    def predict(self, features: Tensor) -> Tensor:
        probabilities = torch.sigmoid(self.decision_function(features))
        return probabilities >= self.thresholds_.to(probabilities.device)


def fit_logistic(
    features: Tensor, targets: Tensor, alpha: float, max_iter: int = 1000
) -> tuple[Tensor, Tensor]:
    """L2 penalized logistic regression fit with L-BFGS. Each target column is a separate binary
    problem sharing the same features."""
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
