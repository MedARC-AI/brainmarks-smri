"""Linear probes on frozen embeddings. Each probe embeds the task's samples, fits on train,
predicts eval, and scores with bootstrap CIs.

Classification and regression probe the global embedding with sklearn, tuned by CV inside the
training set. Segmentation probes the dense embedding: each patch predicts the voxels inside it
(one sigmoid per voxel and channel), fit with full-batch L-BFGS; the penalty and thresholds are
chosen on a holdout of the training participants. Predictions are scored on the image grid.

The task dataset's `transform` must be the model's transform.
"""

from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from brainmarks_smri.tasks import Task

SEED = 0
N_BOOTSTRAP = 1000
LOGISTIC_CS = np.logspace(-4, 4, 9)
RIDGE_ALPHAS = np.logspace(-2, 6, 9)
SEGMENTATION_ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
SEGMENTATION_THRESHOLDS = torch.logspace(-3, -0.1, 30)
HOLDOUT_FRACTION = 0.2
MAX_NEGATIVE_RATIO = 10.0
# Logit for voxels without a prediction (dropped tokens, outside the model input): sigmoid ~ 2e-9.
BACKGROUND_LOGIT = -20.0


def probe_classification(model, task: Task, batch_size: int, num_workers: int) -> dict[str, Any]:
    samples = task.dataset.samples
    class_index = {value: ii for ii, value in enumerate(task.classes)}
    train_labels = np.array(
        [class_index[value] for value in samples[task.target].iloc[task.train_ids]]
    )
    eval_labels = np.array(
        [class_index[value] for value in samples[task.target].iloc[task.eval_ids]]
    )
    # predict_proba columns are the classes seen in training
    assert len(np.unique(train_labels)) == len(task.classes), (
        f"{task.name}: a class is missing from train"
    )

    train_features = np.stack(
        [
            output["global_embedding"].cpu().numpy()
            for output, _ in embed(model, task, task.train_ids, batch_size, num_workers)
        ]
    )
    eval_features = np.stack(
        [
            output["global_embedding"].cpu().numpy()
            for output, _ in embed(model, task, task.eval_ids, batch_size, num_workers)
        ]
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=LOGISTIC_CS,
            l1_ratios=(0.0,),
            scoring="neg_log_loss",
            # so that argmax is a sensible decision for balanced accuracy
            class_weight="balanced",
            max_iter=1000,
            use_legacy_attributes=False,
        ),
    )
    classifier.fit(train_features, train_labels)
    probabilities = classifier.predict_proba(eval_features)
    predictions = probabilities.argmax(axis=1)

    def auroc(ids: np.ndarray) -> float:
        if len(task.classes) == 2:
            return roc_auc_score(eval_labels[ids], probabilities[ids, 1])
        return roc_auc_score(
            eval_labels[ids], probabilities[ids], multi_class="ovr", average="macro"
        )

    bootstrap_auroc = []
    bootstrap_balanced_accuracy = []
    rng = np.random.default_rng(SEED)
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, len(eval_labels), len(eval_labels))
        # auroc is undefined when a resample misses a class
        if len(np.unique(eval_labels[ids])) < len(task.classes):
            continue
        bootstrap_auroc.append(auroc(ids))
        bootstrap_balanced_accuracy.append(
            balanced_accuracy_score(eval_labels[ids], predictions[ids])
        )
    metrics = {
        "auroc": float(auroc(np.arange(len(eval_labels)))),
        "auroc_ci": np.percentile(bootstrap_auroc, [2.5, 97.5]).tolist(),
        "balanced_accuracy": float(balanced_accuracy_score(eval_labels, predictions)),
        "balanced_accuracy_ci": np.percentile(bootstrap_balanced_accuracy, [2.5, 97.5]).tolist(),
    }
    subjects = samples.iloc[task.eval_ids][["participant_id", "session_id", task.target]].to_dict(
        "records"
    )
    for subject, subject_probabilities in zip(subjects, probabilities):
        subject["probabilities"] = dict(zip(task.classes, subject_probabilities.tolist()))
    return {
        "classes": task.classes,
        "n_train": len(task.train_ids),
        "n_eval": len(task.eval_ids),
        "hyperparameters": {"C": float(classifier[-1].C_)},
        "metrics": metrics,
        "subjects": subjects,
    }


def probe_regression(model, task: Task, batch_size: int, num_workers: int) -> dict[str, Any]:
    samples = task.dataset.samples
    train_targets = samples[task.target].iloc[task.train_ids].to_numpy(dtype=float)
    eval_targets = samples[task.target].iloc[task.eval_ids].to_numpy(dtype=float)

    train_features = np.stack(
        [
            output["global_embedding"].cpu().numpy()
            for output, _ in embed(model, task, task.train_ids, batch_size, num_workers)
        ]
    )
    eval_features = np.stack(
        [
            output["global_embedding"].cpu().numpy()
            for output, _ in embed(model, task, task.eval_ids, batch_size, num_workers)
        ]
    )
    regressor = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    regressor.fit(train_features, train_targets)
    predictions = regressor.predict(eval_features)

    bootstrap = {"mae": [], "r": [], "r2": []}
    rng = np.random.default_rng(SEED)
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, len(eval_targets), len(eval_targets))
        bootstrap["mae"].append(mean_absolute_error(eval_targets[ids], predictions[ids]))
        bootstrap["r"].append(np.corrcoef(eval_targets[ids], predictions[ids])[0, 1])
        bootstrap["r2"].append(r2_score(eval_targets[ids], predictions[ids]))
    metrics = {
        "mae": float(mean_absolute_error(eval_targets, predictions)),
        "mae_ci": np.percentile(bootstrap["mae"], [2.5, 97.5]).tolist(),
        "r": float(np.corrcoef(eval_targets, predictions)[0, 1]),
        "r_ci": np.percentile(bootstrap["r"], [2.5, 97.5]).tolist(),
        "r2": float(r2_score(eval_targets, predictions)),
        "r2_ci": np.percentile(bootstrap["r2"], [2.5, 97.5]).tolist(),
    }
    subjects = samples.iloc[task.eval_ids][["participant_id", "session_id", task.target]].to_dict(
        "records"
    )
    for subject, prediction in zip(subjects, predictions):
        subject["prediction"] = float(prediction)
    return {
        "n_train": len(task.train_ids),
        "n_eval": len(task.eval_ids),
        "hyperparameters": {"alpha": float(regressor[-1].alpha_)},
        "metrics": metrics,
        "subjects": subjects,
    }


def embed(
    model,
    task: Task,
    ids: np.ndarray,
    batch_size: int,
    num_workers: int,
    return_dense: bool = False,
):
    """Yield `(embedding output, targets)` for each sample."""
    loader = DataLoader(
        Subset(task.dataset, ids),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list,
        # fork, so the model's transform is not pickled into the workers
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    for batch in loader:
        with torch.inference_mode():
            outputs = model.forward_embeddings(
                [image_input for image_input, _ in batch], return_dense
            )
        yield from zip(outputs, [targets for _, targets in batch])


def probe_segmentation(model, task: Task, batch_size: int, num_workers: int) -> dict[str, Any]:
    samples = task.dataset.samples
    device = next(model.parameters()).device
    channel_names = list(task.channels)
    n_channels = len(channel_names)
    patch_size = tuple(model.patch_size)

    def target_channels(label_image: nib.Nifti1Image) -> Tensor:
        """(C, X, Y, Z) bool on the label's grid."""
        label = torch.as_tensor(np.asanyarray(label_image.dataobj).astype(np.int64), device=device)
        return torch.stack(
            [
                torch.isin(label, torch.tensor(values, device=device))
                for values in task.channels.values()
            ]
        )

    # Hold out some training participants to choose the penalty and thresholds.
    participants = np.sort(samples["participant_id"].iloc[task.train_ids].unique())
    n_holdout = round(HOLDOUT_FRACTION * len(participants))
    assert 0 < n_holdout < len(participants), f"{task.name}: too few participants for a holdout"
    holdout_participants = np.random.default_rng(SEED).choice(
        participants, n_holdout, replace=False
    )
    is_holdout = (
        samples["participant_id"].iloc[task.train_ids].isin(holdout_participants).to_numpy()
    )

    # Features of the kept patches, and their voxel labels resampled onto the model input grid.
    patches = {}
    for part, ids in [
        ("fit", task.train_ids[~is_holdout]),
        ("holdout", task.train_ids[is_holdout]),
    ]:
        features, labels = [], []
        for output, targets in embed(model, task, ids, batch_size, num_workers, return_dense=True):
            dense = output["dense_embedding"]  # (X, Y, Z, D)
            keep = (
                output["dense_mask"]
                if output["dense_mask"] is not None
                else torch.ones(dense.shape[:3], dtype=torch.bool)
            )
            input_shape = tuple(g * p for g, p in zip(dense.shape[:3], patch_size))
            label_image = targets[task.target]
            # input voxel -> label voxel
            matrix = np.linalg.inv(label_image.affine) @ output["dense_affine"]
            label = (
                resample(target_channels(label_image).float(), matrix, input_shape, "nearest") > 0.5
            )
            label = patchify(label.permute(1, 2, 3, 0), patch_size)  # (X, Y, Z, n_voxels, C)
            features.append(dense[keep.to(device)].half().cpu())
            labels.append(label[keep.to(device)].flatten(1).cpu())  # (n, n_voxels * C)
        patches[part] = (torch.cat(features), torch.cat(labels))

    features, labels = patches["fit"]
    # all patches with a positive voxel, and about MAX_NEGATIVE_RATIO negatives per positive
    positive = labels.any(dim=1)
    negative_fraction = MAX_NEGATIVE_RATIO * positive.sum() / (~positive).sum().clamp_min(1)
    keep = positive | (
        torch.rand(len(positive), generator=torch.Generator().manual_seed(SEED)) < negative_fraction
    )
    features = features[keep].to(device).float()
    labels = labels[keep].to(device).float()
    mean = features.mean(dim=0)
    std = features.std(dim=0, correction=0).clamp_min(1e-6)
    features = (features - mean) / std

    # Holdout Dice per (alpha, channel, threshold), pooled over the holdout patch voxels.
    holdout_features = (patches["holdout"][0].to(device).float() - mean) / std
    holdout_labels = patches["holdout"][1].to(device).reshape(-1, n_channels)
    fits = []
    holdout_dice = torch.zeros(len(SEGMENTATION_ALPHAS), n_channels, len(SEGMENTATION_THRESHOLDS))
    for alpha_id, alpha in enumerate(SEGMENTATION_ALPHAS):
        coef, intercept = fit_logistic(features, labels, alpha)
        fits.append((coef, intercept))
        probabilities = torch.sigmoid(holdout_features @ coef + intercept).reshape(-1, n_channels)
        for threshold_id, threshold in enumerate(SEGMENTATION_THRESHOLDS):
            predicted = probabilities >= threshold
            overlap = (predicted & holdout_labels).sum(dim=0)
            total = predicted.sum(dim=0) + holdout_labels.sum(dim=0)
            holdout_dice[alpha_id, :, threshold_id] = (2 * overlap / total.clamp_min(1)).cpu()
    # one penalty for all channels, one threshold per channel
    alpha_id = holdout_dice.max(dim=2).values.mean(dim=1).argmax().item()
    channel_thresholds = SEGMENTATION_THRESHOLDS[holdout_dice[alpha_id].argmax(dim=1)].to(device)
    coef, intercept = fits[alpha_id]
    del patches, features, labels, holdout_features, holdout_labels, fits

    # Eval: logits of every patch, resampled onto the label's grid; Dice per channel.
    subject_dice = []
    for output, targets in embed(
        model, task, task.eval_ids, batch_size, num_workers, return_dense=True
    ):
        dense = (output["dense_embedding"].to(device).float() - mean) / std
        logits = (dense @ coef + intercept).reshape(*dense.shape[:3], -1, n_channels)
        if output["dense_mask"] is not None:
            logits[~output["dense_mask"].to(device)] = BACKGROUND_LOGIT
        logits = unpatchify(logits, patch_size).permute(3, 0, 1, 2)  # (C, Xi, Yi, Zi)
        label_image = targets[task.target]
        # label voxel -> input voxel; shifted so that outside the input grid is background
        matrix = np.linalg.inv(output["dense_affine"]) @ label_image.affine
        logits = (
            resample(logits - BACKGROUND_LOGIT, matrix, label_image.shape, "bilinear")
            + BACKGROUND_LOGIT
        )
        predicted = (torch.sigmoid(logits) >= channel_thresholds[:, None, None, None]).flatten(1)
        target = target_channels(label_image).flatten(1)
        overlap = (predicted & target).sum(dim=1)
        total = predicted.sum(dim=1) + target.sum(dim=1)
        # empty prediction of an empty target counts as 1
        subject_dice.append(
            torch.where(total > 0, 2 * overlap / total.clamp_min(1), 1.0).cpu().numpy()
        )
    subject_dice = np.stack(subject_dice)  # (n_eval, C)

    metrics = {}
    rng = np.random.default_rng(SEED)
    bootstrap_ids = rng.integers(0, len(subject_dice), (N_BOOTSTRAP, len(subject_dice)))
    columns = {name: subject_dice[:, ii] for ii, name in enumerate(channel_names)}
    columns["mean"] = subject_dice.mean(axis=1)
    for name, dice in columns.items():
        metrics[f"dice_{name}"] = float(dice.mean())
        metrics[f"dice_{name}_ci"] = np.percentile(
            dice[bootstrap_ids].mean(axis=1), [2.5, 97.5]
        ).tolist()
    subjects = samples.iloc[task.eval_ids][["participant_id", "session_id"]].to_dict("records")
    for subject, dice in zip(subjects, subject_dice):
        subject["dice"] = dict(zip(channel_names, dice.tolist()))
    return {
        "channels": task.channels,
        "n_train": int((~is_holdout).sum()),
        "n_holdout": int(is_holdout.sum()),
        "n_eval": len(task.eval_ids),
        "hyperparameters": {
            "alpha": SEGMENTATION_ALPHAS[alpha_id],
            "thresholds": dict(zip(channel_names, channel_thresholds.tolist())),
            "holdout_dice": holdout_dice.max(dim=2).values.tolist(),  # (alphas, channels)
        },
        "metrics": metrics,
        "subjects": subjects,
    }


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


def resample(volume: Tensor, matrix: np.ndarray, output_shape: tuple, mode: str) -> Tensor:
    """Sample `volume` (C, X, Y, Z) at `matrix @ [i, j, k, 1]` for each output voxel; 0 outside."""
    axes = [torch.arange(n, device=volume.device, dtype=torch.float32) for n in output_shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    matrix = torch.as_tensor(matrix, device=volume.device, dtype=torch.float32)
    coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
    # grid_sample wants (x, y, z) = (last, middle, first) axis, normalized to [-1, 1]
    sizes = torch.tensor(volume.shape[1:], device=volume.device, dtype=torch.float32)
    grid = (2 * coords / (sizes - 1).clamp_min(1) - 1).flip(-1)[None]
    return F.grid_sample(volume[None], grid, mode=mode, padding_mode="zeros", align_corners=True)[0]


def patchify(volume: Tensor, patch_size: tuple) -> Tensor:
    """(X*px, Y*py, Z*pz, C) -> (X, Y, Z, px*py*pz, C)."""
    px, py, pz = patch_size
    xi, yi, zi, c = volume.shape
    volume = volume.reshape(xi // px, px, yi // py, py, zi // pz, pz, c)
    return volume.permute(0, 2, 4, 1, 3, 5, 6).reshape(
        xi // px, yi // py, zi // pz, px * py * pz, c
    )


def unpatchify(patches: Tensor, patch_size: tuple) -> Tensor:
    """(X, Y, Z, px*py*pz, C) -> (X*px, Y*py, Z*pz, C)."""
    px, py, pz = patch_size
    x, y, z, _, c = patches.shape
    patches = patches.reshape(x, y, z, px, py, pz, c)
    return patches.permute(0, 3, 1, 4, 2, 5, 6).reshape(x * px, y * py, z * pz, c)
