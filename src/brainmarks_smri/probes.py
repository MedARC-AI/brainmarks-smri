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
    ids = np.concatenate([task.train_ids, task.eval_ids])
    embeddings, targets = global_embeddings(model, task, ids, batch_size, num_workers)
    class_index = {value: ii for ii, value in enumerate(task.classes)}
    labels = np.array([class_index[sample_targets[task.target]] for sample_targets in targets])
    n_train = len(task.train_ids)
    train_labels = labels[:n_train]
    eval_labels = labels[n_train:]
    # predict_proba columns are the classes seen in training
    n_train_classes = len(np.unique(train_labels))
    assert n_train_classes == len(task.classes), f"{task.name}: a class is missing from train"

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
    classifier.fit(embeddings[:n_train], train_labels)
    logger.info(f"fit logistic regression probe in {time.perf_counter() - fit_start:.0f}s")
    probabilities = classifier.predict_proba(embeddings[n_train:])
    predictions = probabilities.argmax(axis=1)

    metrics = {
        "auroc": auroc(eval_labels, probabilities),
        "auroc_ci": bootstrap_ci(auroc, eval_labels, probabilities, seed=SEED),
        "balanced_accuracy": float(balanced_accuracy_score(eval_labels, predictions)),
        "balanced_accuracy_ci": bootstrap_ci(
            balanced_accuracy_score, eval_labels, predictions, seed=SEED
        ),
    }
    eval_samples = []
    for sample_targets, sample_probabilities in zip(targets[n_train:], probabilities):
        record = {key: sample_targets[key] for key in ("participant_id", "session_id", task.target)}
        record["probabilities"] = dict(zip(task.classes, sample_probabilities.tolist()))
        eval_samples.append(record)
    return {
        "classes": task.classes,
        "n_train": len(task.train_ids),
        "n_eval": len(task.eval_ids),
        "hyperparameters": {"C": float(classifier[-1].C_)},
        "metrics": metrics,
        "eval_samples": eval_samples,
    }


def probe_regression(model: Model, task: Task, batch_size: int, num_workers: int) -> dict[str, Any]:
    ids = np.concatenate([task.train_ids, task.eval_ids])
    embeddings, targets = global_embeddings(model, task, ids, batch_size, num_workers)
    values = np.array([sample_targets[task.target] for sample_targets in targets], dtype=float)
    n_train = len(task.train_ids)
    eval_values = values[n_train:]

    regressor = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    fit_start = time.perf_counter()
    regressor.fit(embeddings[:n_train], values[:n_train])
    logger.info(f"fit ridge regression probe in {time.perf_counter() - fit_start:.0f}s")
    predictions = regressor.predict(embeddings[n_train:])

    metrics = {
        "mae": float(mean_absolute_error(eval_values, predictions)),
        "mae_ci": bootstrap_ci(mean_absolute_error, eval_values, predictions, seed=SEED),
        "r": pearson_r(eval_values, predictions),
        "r_ci": bootstrap_ci(pearson_r, eval_values, predictions, seed=SEED),
        "r2": float(r2_score(eval_values, predictions)),
        "r2_ci": bootstrap_ci(r2_score, eval_values, predictions, seed=SEED),
    }
    eval_samples = []
    for sample_targets, prediction in zip(targets[n_train:], predictions):
        record = {key: sample_targets[key] for key in ("participant_id", "session_id", task.target)}
        record["prediction"] = float(prediction)
        eval_samples.append(record)
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
    device = next(model.parameters()).device
    ids = np.concatenate([task.train_ids, task.eval_ids])
    features, labels, groups, samples = segmentation_embeddings(
        model, task, ids, batch_size, num_workers
    )
    n_train = len(task.train_ids)
    # train samples come first
    n_train_rows = samples[n_train - 1]["token_rows"].stop

    classifier = TorchLogisticRegressionCV(
        alphas=SEGMENTATION_ALPHAS,
        thresholds=SEGMENTATION_THRESHOLDS,
        cv=GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=SEED),
        max_negative_ratio=MAX_NEGATIVE_RATIO,
        seed=SEED,
    )
    fit_start = time.perf_counter()
    classifier.fit(
        features[:n_train_rows].to(device),
        labels[:n_train_rows].to(device),
        groups[:n_train_rows],
    )
    logger.info(f"fit segmentation probe in {time.perf_counter() - fit_start:.0f}s")

    subject_metrics = {"dice": [], "voxel_auroc": [], "average_precision": []}
    for sample in samples[n_train:]:
        sample_features = features[sample["token_rows"]].to(device)
        probabilities = segmentation_probabilities(
            classifier, sample_features, sample, model.patch_size
        )
        target = sample[task.target].to(device)
        brain = sample["brain_mask"].to(device)
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

    eval_samples = []
    for ii, sample in enumerate(samples[n_train:]):
        record = {key: sample[key] for key in ("participant_id", "session_id")}
        for name, values in subject_metrics.items():
            record[name] = float(values[ii])
        eval_samples.append(record)
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
    embeddings, targets = global_embeddings(model, task, task.train_ids, batch_size, num_workers)
    class_index = {value: ii for ii, value in enumerate(task.classes)}
    labels = np.array([class_index[sample_targets[task.target]] for sample_targets in targets])
    n_folds = int(task.folds.max()) + 1

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
        classifier.fit(embeddings[~in_fold], labels[~in_fold])
        probabilities[in_fold] = classifier.predict_proba(embeddings[in_fold])
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
    eval_samples = []
    for sample_targets, fold, sample_probabilities in zip(targets, task.folds, probabilities):
        record = {key: sample_targets[key] for key in ("participant_id", "session_id", task.target)}
        record["fold"] = int(fold)
        record["probabilities"] = dict(zip(task.classes, sample_probabilities.tolist()))
        eval_samples.append(record)
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
    device = next(model.parameters()).device
    features, labels, groups, samples = segmentation_embeddings(
        model, task, task.train_ids, batch_size, num_workers
    )
    features = features.to(device)
    labels = labels.to(device)
    participants = np.array([sample["participant_id"] for sample in samples])
    n_folds = int(task.folds.max()) + 1

    subject_metrics = {
        "dice": np.zeros(len(samples)),
        "voxel_auroc": np.zeros(len(samples)),
        "average_precision": np.zeros(len(samples)),
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

        for position in np.flatnonzero(in_fold):
            sample = samples[position]
            probabilities = segmentation_probabilities(
                classifier, features[sample["token_rows"]], sample, model.patch_size
            )
            target = sample[task.target].to(device)
            brain = sample["brain_mask"].to(device)
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

    eval_samples = []
    for ii, sample in enumerate(samples):
        record = {key: sample[key] for key in ("participant_id", "session_id")}
        record["fold"] = int(task.folds[ii])
        for name, values in subject_metrics.items():
            record[name] = float(values[ii])
        eval_samples.append(record)
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
        # GPU work is async; wait for it so that it counts as model time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """(n, D) global embeddings, and the targets of each sample."""
    embeddings = []
    targets = []
    for output, sample_targets in compute_embeddings(model, task, ids, batch_size, num_workers):
        embeddings.append(output["global_embedding"].cpu().numpy())
        targets.append(sample_targets)
    return np.stack(embeddings), targets


def segmentation_embeddings(
    model: Model, task: Task, ids: np.ndarray, batch_size: int, num_workers: int
) -> tuple[Tensor, Tensor, np.ndarray, list[dict[str, Any]]]:
    """Features of the tokens the model kept, their patch voxel labels and participants, and
    each sample's targets with its `token_rows`, `dense_mask` and `dense_affine`."""
    all_features = []
    all_labels = []
    all_groups = []
    samples = []
    n_rows = 0
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

        n_kept = int(kept.sum())
        all_features.append(dense[kept].half().cpu())  # (n_kept, D)
        all_labels.append(patch_labels[kept].cpu())  # (n_kept, n_patch_voxels)
        all_groups.extend([targets["participant_id"]] * n_kept)
        # copies out of the loader's shared memory, which these would otherwise hold
        targets[task.target] = targets[task.target].clone()
        targets["brain_mask"] = targets["brain_mask"].clone()
        targets["token_rows"] = slice(n_rows, n_rows + n_kept)
        targets["dense_mask"] = kept.cpu()
        targets["dense_affine"] = output["dense_affine"]
        samples.append(targets)
        n_rows += n_kept
    return torch.cat(all_features), torch.cat(all_labels), np.array(all_groups), samples


def segmentation_probabilities(
    classifier: TorchLogisticRegressionCV,
    features: Tensor,
    sample: dict[str, Any],
    patch_size: tuple[int, int, int],
) -> Tensor:
    """Probabilities on the label image grid of one sample, from its kept token features."""
    px, py, pz = patch_size
    kept = sample["dense_mask"].to(features.device)
    n_patch_voxels = px * py * pz
    # dropped tokens predict background
    logits = torch.full(
        (*kept.shape, n_patch_voxels), BACKGROUND_LOGIT, device=features.device
    )  # (X, Y, Z, n_patch_voxels)
    logits[kept] = classifier.decision_function(features)
    logits = rearrange(logits, "x y z (px py pz) -> (x px) (y py) (z pz)", px=px, py=py, pz=pz)

    label_to_input = np.linalg.inv(sample["dense_affine"]) @ sample["affine"]
    label_shape = sample["brain_mask"].shape
    # shifted so that voxels outside the model input grid get BACKGROUND_LOGIT
    shifted = logits[None] - BACKGROUND_LOGIT
    logits = resample(shifted, label_to_input, label_shape, "bilinear")[0]
    logits = logits + BACKGROUND_LOGIT
    return torch.sigmoid(logits)


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
