"""Linear probes on frozen embeddings: embed the task's samples, fit on train, score eval.

The task dataset's `transform` must be the model's transform (after the task's, if any).
"""

import logging
import math
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from brainmarks_smri.logistic import TorchLogisticRegressionCV
from brainmarks_smri.metrics import (
    auroc,
    average_precision,
    bootstrap_ci,
    dice,
    pearson_r,
    voxel_auroc,
)
from brainmarks_smri.models.base import EmbeddingOutput, Model
from brainmarks_smri.tasks import Task

SEED = 0
LOGISTIC_CS = np.logspace(-4, 4, 9)
RIDGE_ALPHAS = np.logspace(-2, 6, 9)
SEGMENTATION_ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
SEGMENTATION_THRESHOLDS = torch.logspace(-3, -0.1, 30)
# fraction of training participants held out to tune the segmentation penalty and threshold
HOLDOUT_FRACTION = 0.2
# inner CV of the k-fold segmentation probe, over participants
INNER_FOLDS = 5
MAX_NEGATIVE_RATIO = 10.0
# Logit for voxels without a prediction (dropped tokens, outside the model input): sigmoid ~ 2e-9.
BACKGROUND_LOGIT = -20.0

logger = logging.getLogger(__name__)


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
    fit_start = time.perf_counter()
    classifier.fit(train_features, train_labels)
    logger.info(f"fit logistic regression probe in {time.perf_counter() - fit_start:.0f}s")
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
    fit_start = time.perf_counter()
    regressor.fit(train_features, train_targets)
    logger.info(f"fit ridge regression probe in {time.perf_counter() - fit_start:.0f}s")
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
    classifier = TorchLogisticRegressionCV(
        alphas=SEGMENTATION_ALPHAS,
        thresholds=SEGMENTATION_THRESHOLDS,
        cv=GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=SEED),
        max_negative_ratio=MAX_NEGATIVE_RATIO,
        seed=SEED,
    )
    fit_start = time.perf_counter()
    classifier.fit(features.to(device), labels.to(device), groups)
    logger.info(f"fit segmentation probe in {time.perf_counter() - fit_start:.0f}s")

    subject_metrics = {"dice": [], "voxel_auroc": [], "average_precision": []}
    predictions = segmentation_predictions(
        model, task, task.eval_ids, classifier, batch_size, num_workers
    )
    for probabilities, target, brain in predictions:
        predicted = probabilities >= classifier.threshold_
        subject_metrics["dice"].append(dice(predicted, target))
        # ranking metrics within the brain, where the background is not trivial
        brain_probabilities = probabilities[brain]
        brain_target = target[brain]
        subject_metrics["voxel_auroc"].append(voxel_auroc(brain_probabilities, brain_target))
        subject_metrics["average_precision"].append(
            average_precision(brain_probabilities, brain_target)
        )
    metrics = {}
    for name, values in subject_metrics.items():
        values = np.array(values)
        metrics[name] = float(np.nanmean(values))
        metrics[f"{name}_ci"] = bootstrap_ci(np.nanmean, values, seed=SEED)

    eval_samples = samples.iloc[task.eval_ids][["participant_id", "session_id"]]
    eval_samples = eval_samples.to_dict("records")
    for ii, record in enumerate(eval_samples):
        for name, values in subject_metrics.items():
            record[name] = float(values[ii])
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


def probe_classification_kfold(
    model: Model, task: Task, batch_size: int, num_workers: int
) -> dict[str, Any]:
    samples = task.dataset.samples
    class_index = {value: ii for ii, value in enumerate(task.classes)}
    values = samples[task.target].iloc[task.train_ids]
    labels = np.array([class_index[value] for value in values])
    n_folds = int(task.folds.max()) + 1

    features = global_embeddings(model, task, task.train_ids, batch_size, num_workers)
    probabilities = np.zeros((len(labels), len(task.classes)))
    fold_Cs = []
    fit_start = time.perf_counter()
    for fold in range(n_folds):
        in_fold = task.folds == fold
        # predict_proba columns are the classes seen in training
        n_train_classes = len(np.unique(labels[~in_fold]))
        assert n_train_classes == len(task.classes), f"{task.name}: a class is missing from fold"
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
        classifier.fit(features[~in_fold], labels[~in_fold])
        probabilities[in_fold] = classifier.predict_proba(features[in_fold])
        fold_Cs.append(float(classifier[-1].C_))
    logger.info(
        f"fit {n_folds} logistic regression probes in {time.perf_counter() - fit_start:.0f}s"
    )
    predictions = probabilities.argmax(axis=1)

    metrics = {
        "auroc": auroc(labels, probabilities),
        "auroc_ci": bootstrap_ci(auroc, labels, probabilities, seed=SEED),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "balanced_accuracy_ci": bootstrap_ci(
            balanced_accuracy_score, labels, predictions, seed=SEED
        ),
    }
    eval_samples = samples.iloc[task.train_ids][["participant_id", "session_id", task.target]]
    eval_samples = eval_samples.to_dict("records")
    for record, fold, record_probabilities in zip(eval_samples, task.folds, probabilities):
        record["fold"] = int(fold)
        record["probabilities"] = dict(zip(task.classes, record_probabilities.tolist()))
    return {
        "classes": task.classes,
        "n_samples": len(task.train_ids),
        "n_folds": n_folds,
        "hyperparameters": {"C": fold_Cs},
        "metrics": metrics,
        "eval_samples": eval_samples,
    }


def probe_segmentation_kfold(
    model: Model, task: Task, batch_size: int, num_workers: int
) -> dict[str, Any]:
    samples = task.dataset.samples
    device = next(model.parameters()).device
    participants = samples["participant_id"].iloc[task.train_ids].to_numpy()
    n_folds = int(task.folds.max()) + 1

    features, labels, groups = segmentation_patches(
        model, task, task.train_ids, batch_size, num_workers
    )
    features = features.to(device)
    labels = labels.to(device)

    subject_metrics = {
        "dice": np.zeros(len(task.train_ids)),
        "voxel_auroc": np.zeros(len(task.train_ids)),
        "average_precision": np.zeros(len(task.train_ids)),
    }
    fold_hyperparameters = {"alpha": [], "threshold": [], "cv_dice": []}
    for fold in range(n_folds):
        in_fold = task.folds == fold
        train_rows = ~np.isin(groups, participants[in_fold])
        train_rows_device = torch.as_tensor(train_rows, device=device)
        classifier = TorchLogisticRegressionCV(
            alphas=SEGMENTATION_ALPHAS,
            thresholds=SEGMENTATION_THRESHOLDS,
            cv=GroupKFold(n_splits=INNER_FOLDS),
            max_negative_ratio=MAX_NEGATIVE_RATIO,
            seed=SEED,
        )
        fit_start = time.perf_counter()
        classifier.fit(features[train_rows_device], labels[train_rows_device], groups[train_rows])
        logger.info(
            f"fold {fold}: fit segmentation probe in {time.perf_counter() - fit_start:.0f}s"
        )
        fold_hyperparameters["alpha"].append(classifier.alpha_)
        fold_hyperparameters["threshold"].append(classifier.threshold_)
        # best CV Dice over thresholds, per alpha
        fold_hyperparameters["cv_dice"].append(classifier.cv_dice_.max(dim=1).values.tolist())

        predictions = segmentation_predictions(
            model, task, task.train_ids[in_fold], classifier, batch_size, num_workers
        )
        for position, (probabilities, target, brain) in zip(np.flatnonzero(in_fold), predictions):
            predicted = probabilities >= classifier.threshold_
            subject_metrics["dice"][position] = dice(predicted, target)
            # ranking metrics within the brain, where the background is not trivial
            brain_probabilities = probabilities[brain]
            brain_target = target[brain]
            subject_metrics["voxel_auroc"][position] = voxel_auroc(
                brain_probabilities, brain_target
            )
            subject_metrics["average_precision"][position] = average_precision(
                brain_probabilities, brain_target
            )
    metrics = {}
    for name, values in subject_metrics.items():
        metrics[name] = float(np.nanmean(values))
        metrics[f"{name}_ci"] = bootstrap_ci(np.nanmean, values, seed=SEED)

    eval_samples = samples.iloc[task.train_ids][["participant_id", "session_id"]]
    eval_samples = eval_samples.to_dict("records")
    for ii, record in enumerate(eval_samples):
        record["fold"] = int(task.folds[ii])
        for name, values in subject_metrics.items():
            record[name] = float(values[ii])
    return {
        "label_values": task.label_values,
        "n_samples": len(task.train_ids),
        "n_folds": n_folds,
        "hyperparameters": fold_hyperparameters,
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
    log_every: int = 20,
) -> Iterator[tuple[EmbeddingOutput, dict[str, Any]]]:
    """Yield `(embedding output, targets)` for each sample of `task.dataset` in `ids`."""
    # smaller batches for small passes, so that every worker gets one
    batch_size = min(batch_size, math.ceil(len(ids) / max(num_workers, 1)))
    loader = DataLoader(
        Subset(task.dataset, ids),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list,
        # fork, so the model's transform is not pickled into the workers
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    n_done = 0
    loader_seconds = 0.0
    model_seconds = 0.0
    start = time.perf_counter()
    loader_start = time.perf_counter()
    for batch_id, batch in enumerate(loader):
        loader_seconds += time.perf_counter() - loader_start
        image_inputs = [image_input for image_input, _ in batch]
        targets = [sample_targets for _, sample_targets in batch]
        batch_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.forward_embeddings(image_inputs, return_dense)
        model_seconds += time.perf_counter() - batch_start
        n_done += len(batch)
        if (batch_id + 1) % log_every == 0 or batch_id + 1 == len(loader):
            elapsed = time.perf_counter() - start
            # the rest of the elapsed time is the caller's, between yields
            logger.info(
                f"embedded {n_done}/{len(ids)} in {elapsed:.0f}s "
                f"({n_done / elapsed:.2f} samples/s, data {loader_seconds:.0f}s, "
                f"model {model_seconds:.0f}s)"
            )
        yield from zip(outputs, targets)
        loader_start = time.perf_counter()


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

        mask = targets[task.target].to(dense.device)
        input_to_label = np.linalg.inv(targets["affine"]) @ output["dense_affine"]
        mask = resample(mask[None].float(), input_to_label, input_shape, "nearest")[0] > 0.5
        patch_labels = rearrange(
            mask, "(x px) (y py) (z pz) -> x y z (px py pz)", px=px, py=py, pz=pz
        )

        all_features.append(dense[kept].half().cpu())  # (n_kept, D)
        all_labels.append(patch_labels[kept].cpu())  # (n_kept, n_patch_voxels)
        all_groups.extend([targets["participant_id"]] * int(kept.sum()))
    return torch.cat(all_features), torch.cat(all_labels), np.array(all_groups)


def segmentation_predictions(
    model: Model,
    task: Task,
    ids: np.ndarray,
    classifier: TorchLogisticRegressionCV,
    batch_size: int,
    num_workers: int,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Yield `(probabilities, target, brain mask)` on the label image grid for each sample."""
    px, py, pz = model.patch_size
    embeddings = compute_embeddings(model, task, ids, batch_size, num_workers, return_dense=True)
    for output, targets in embeddings:
        dense = output["dense_embedding"]  # (X, Y, Z, D)
        grid_shape = dense.shape[:3]
        logits = classifier.decision_function(dense.reshape(-1, dense.shape[-1]))
        logits = logits.reshape(*grid_shape, -1)  # (X, Y, Z, n_patch_voxels)
        if output["dense_mask"] is not None:
            # dropped tokens predict background
            logits[~output["dense_mask"].to(logits.device)] = BACKGROUND_LOGIT
        logits = rearrange(logits, "x y z (px py pz) -> (x px) (y py) (z pz)", px=px, py=py, pz=pz)

        target = targets[task.target].to(logits.device)
        label_to_input = np.linalg.inv(output["dense_affine"]) @ targets["affine"]
        # shifted so that voxels outside the model input grid get BACKGROUND_LOGIT
        shifted = logits[None] - BACKGROUND_LOGIT
        logits = resample(shifted, label_to_input, target.shape, "bilinear")[0]
        logits = logits + BACKGROUND_LOGIT
        probabilities = torch.sigmoid(logits)

        brain = targets["brain_mask"].to(logits.device)
        yield probabilities, target, brain


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
