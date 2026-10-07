"""Linear probes on frozen embeddings: embed the task's samples, fit on train, score eval.

The task dataset's `transform` must be the model's transform.
"""

from collections.abc import Iterator
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from brainmarks_smri.logistic import TorchLogisticRegressionCV
from brainmarks_smri.metrics import auroc, bootstrap_ci, dice, pearson_r
from brainmarks_smri.models.base import EmbeddingOutput, Model
from brainmarks_smri.tasks import Task

SEED = 0
LOGISTIC_CS = np.logspace(-4, 4, 9)
RIDGE_ALPHAS = np.logspace(-2, 6, 9)
# Logit for voxels without a prediction (dropped tokens, outside the model input): sigmoid ~ 2e-9.
BACKGROUND_LOGIT = -20.0


def probe_classification(
    model: Model, task: Task, batch_size: int, num_workers: int
) -> dict[str, Any]:
    samples = task.dataset.samples
    class_index = {value: ii for ii, value in enumerate(task.classes)}
    train_values = samples[task.target].iloc[task.train_ids]
    eval_values = samples[task.target].iloc[task.eval_ids]
    train_labels = np.array([class_index[value] for value in train_values])
    eval_labels = np.array([class_index[value] for value in eval_values])
    # predict_proba columns are the classes seen in training
    n_train_classes = len(np.unique(train_labels))
    assert n_train_classes == len(task.classes), f"{task.name}: a class is missing from train"

    train_features = global_embeddings(model, task, task.train_ids, batch_size, num_workers)
    eval_features = global_embeddings(model, task, task.eval_ids, batch_size, num_workers)
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

    metrics = {
        "auroc": auroc(eval_labels, probabilities),
        "auroc_ci": bootstrap_ci(auroc, eval_labels, probabilities, seed=SEED),
        "balanced_accuracy": float(balanced_accuracy_score(eval_labels, predictions)),
        "balanced_accuracy_ci": bootstrap_ci(
            balanced_accuracy_score, eval_labels, predictions, seed=SEED
        ),
    }
    eval_samples = samples.iloc[task.eval_ids][["participant_id", "session_id", task.target]]
    eval_samples = eval_samples.to_dict("records")
    for record, record_probabilities in zip(eval_samples, probabilities):
        record["probabilities"] = dict(zip(task.classes, record_probabilities.tolist()))
    return {
        "classes": task.classes,
        "n_train": len(task.train_ids),
        "n_eval": len(task.eval_ids),
        "hyperparameters": {"C": float(classifier[-1].C_)},
        "metrics": metrics,
        "eval_samples": eval_samples,
    }


def probe_regression(model: Model, task: Task, batch_size: int, num_workers: int) -> dict[str, Any]:
    samples = task.dataset.samples
    train_targets = samples[task.target].iloc[task.train_ids].to_numpy(dtype=float)
    eval_targets = samples[task.target].iloc[task.eval_ids].to_numpy(dtype=float)

    train_features = global_embeddings(model, task, task.train_ids, batch_size, num_workers)
    eval_features = global_embeddings(model, task, task.eval_ids, batch_size, num_workers)
    regressor = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    regressor.fit(train_features, train_targets)
    predictions = regressor.predict(eval_features)

    metrics = {
        "mae": float(mean_absolute_error(eval_targets, predictions)),
        "mae_ci": bootstrap_ci(mean_absolute_error, eval_targets, predictions, seed=SEED),
        "r": pearson_r(eval_targets, predictions),
        "r_ci": bootstrap_ci(pearson_r, eval_targets, predictions, seed=SEED),
        "r2": float(r2_score(eval_targets, predictions)),
        "r2_ci": bootstrap_ci(r2_score, eval_targets, predictions, seed=SEED),
    }
    eval_samples = samples.iloc[task.eval_ids][["participant_id", "session_id", task.target]]
    eval_samples = eval_samples.to_dict("records")
    for record, prediction in zip(eval_samples, predictions):
        record["prediction"] = float(prediction)
    return {
        "n_train": len(task.train_ids),
        "n_eval": len(task.eval_ids),
        "hyperparameters": {"alpha": float(regressor[-1].alpha_)},
        "metrics": metrics,
        "eval_samples": eval_samples,
    }


def probe_segmentation(
    model: Model, task: Task, batch_size: int, num_workers: int
) -> dict[str, Any]:
    samples = task.dataset.samples
    device = next(model.parameters()).device

    features, labels, groups = segmentation_patches(
        model, task, task.train_ids, batch_size, num_workers
    )
    # the penalty and threshold are tuned on held-out training participants
    classifier = TorchLogisticRegressionCV()
    classifier.fit(features.to(device), labels.to(device), groups)

    subject_dice = segmentation_dice(model, task, classifier, batch_size, num_workers)
    metrics = {
        "dice": float(subject_dice.mean()),
        "dice_ci": bootstrap_ci(np.mean, subject_dice, seed=SEED),
    }
    eval_samples = samples.iloc[task.eval_ids][["participant_id", "session_id"]]
    eval_samples = eval_samples.to_dict("records")
    for record, record_dice in zip(eval_samples, subject_dice):
        record["dice"] = float(record_dice)
    return {
        "label_values": task.label_values,
        "n_train": len(task.train_ids),
        "n_eval": len(task.eval_ids),
        "hyperparameters": {
            "alpha": classifier.alpha_,
            "threshold": classifier.threshold_,
            # best CV Dice over thresholds, per alpha
            "cv_dice": classifier.cv_dice_.max(dim=1).values.tolist(),
        },
        "metrics": metrics,
        "eval_samples": eval_samples,
    }


def compute_embeddings(
    model: Model,
    task: Task,
    ids: np.ndarray,
    batch_size: int,
    num_workers: int,
    return_dense: bool = False,
) -> Iterator[tuple[EmbeddingOutput, dict[str, Any]]]:
    """Yield `(embedding output, targets)` for each sample of `task.dataset` in `ids`."""
    loader = DataLoader(
        Subset(task.dataset, ids),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list,
        # fork, so the model's transform is not pickled into the workers
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    for batch in loader:
        image_inputs = [image_input for image_input, _ in batch]
        targets = [sample_targets for _, sample_targets in batch]
        with torch.inference_mode():
            outputs = model.forward_embeddings(image_inputs, return_dense)
        yield from zip(outputs, targets)


def global_embeddings(
    model: Model, task: Task, ids: np.ndarray, batch_size: int, num_workers: int
) -> np.ndarray:
    """(n, D) global embeddings."""
    embeddings = []
    for output, _ in compute_embeddings(model, task, ids, batch_size, num_workers):
        embeddings.append(output["global_embedding"].cpu().numpy())
    return np.stack(embeddings)


def segmentation_patches(
    model: Model, task: Task, ids: np.ndarray, batch_size: int, num_workers: int
) -> tuple[Tensor, Tensor, np.ndarray]:
    """Features of the tokens the model kept, their patch voxel labels, and their participants."""
    all_features = []
    all_labels = []
    all_groups = []
    px, py, pz = model.patch_size
    embeddings = compute_embeddings(model, task, ids, batch_size, num_workers, return_dense=True)
    for output, targets in embeddings:
        dense = output["dense_embedding"]  # (X, Y, Z, D)
        grid_shape = dense.shape[:3]
        input_shape = (grid_shape[0] * px, grid_shape[1] * py, grid_shape[2] * pz)
        if output["dense_mask"] is not None:
            kept = output["dense_mask"].to(dense.device)
        else:
            kept = torch.ones(grid_shape, dtype=torch.bool, device=dense.device)

        label_image = targets[task.target]
        mask = label_mask(label_image, task.label_values, dense.device)
        input_to_label = np.linalg.inv(label_image.affine) @ output["dense_affine"]
        mask = resample(mask[None].float(), input_to_label, input_shape, "nearest")[0] > 0.5
        patch_labels = rearrange(
            mask, "(x px) (y py) (z pz) -> x y z (px py pz)", px=px, py=py, pz=pz
        )

        all_features.append(dense[kept].half().cpu())  # (n_kept, D)
        all_labels.append(patch_labels[kept].cpu())  # (n_kept, n_patch_voxels)
        all_groups.extend([targets["participant_id"]] * int(kept.sum()))
    return torch.cat(all_features), torch.cat(all_labels), np.array(all_groups)


def segmentation_dice(
    model: Model,
    task: Task,
    classifier: TorchLogisticRegressionCV,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Dice per eval subject, scored on the label image grid."""
    subject_dice = []
    px, py, pz = model.patch_size
    embeddings = compute_embeddings(
        model, task, task.eval_ids, batch_size, num_workers, return_dense=True
    )
    for output, targets in embeddings:
        dense = output["dense_embedding"]  # (X, Y, Z, D)
        grid_shape = dense.shape[:3]
        logits = classifier.decision_function(dense.reshape(-1, dense.shape[-1]))
        logits = logits.reshape(*grid_shape, -1)  # (X, Y, Z, n_patch_voxels)
        if output["dense_mask"] is not None:
            # dropped tokens predict background
            logits[~output["dense_mask"].to(logits.device)] = BACKGROUND_LOGIT
        logits = rearrange(logits, "x y z (px py pz) -> (x px) (y py) (z pz)", px=px, py=py, pz=pz)

        label_image = targets[task.target]
        label_to_input = np.linalg.inv(output["dense_affine"]) @ label_image.affine
        # shifted so that voxels outside the model input grid get BACKGROUND_LOGIT
        shifted = logits[None] - BACKGROUND_LOGIT
        logits = resample(shifted, label_to_input, label_image.shape, "bilinear")[0]
        logits = logits + BACKGROUND_LOGIT

        predicted = torch.sigmoid(logits) >= classifier.threshold_
        target = label_mask(label_image, task.label_values, logits.device)
        subject_dice.append(dice(predicted, target))
    return np.array(subject_dice)


def label_mask(
    label_image: nib.Nifti1Image, label_values: list[int], device: torch.device
) -> Tensor:
    """Bool mask of the voxels with one of `label_values`."""
    label = np.asanyarray(label_image.dataobj).astype(np.int64)
    label = torch.as_tensor(label, device=device)
    return torch.isin(label, torch.tensor(label_values, device=device))


def resample(volume: Tensor, matrix: np.ndarray, output_shape: tuple, mode: str) -> Tensor:
    """Sample `volume` (C, X, Y, Z) at `matrix @ [i, j, k, 1]` for each output voxel; 0 outside."""
    axes = [torch.arange(n, device=volume.device, dtype=torch.float32) for n in output_shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    matrix = torch.as_tensor(matrix, device=volume.device, dtype=torch.float32)
    coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
    # grid_sample wants (x, y, z) = (last, middle, first) axis, normalized to [-1, 1]
    sizes = torch.tensor(volume.shape[1:], device=volume.device, dtype=torch.float32)
    coords = 2 * coords / (sizes - 1).clamp_min(1) - 1
    grid = coords.flip(-1)[None]
    resampled = F.grid_sample(
        volume[None], grid, mode=mode, padding_mode="zeros", align_corners=True
    )
    return resampled[0]
