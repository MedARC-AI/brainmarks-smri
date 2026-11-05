"""Benchmark scores of one model: task scores with bootstrap resamples, and their means over
groups of tasks.

Task scores are 0 at chance and at most 1. The resamples have a fixed seed and are drawn over the
samples sorted by participant and session, so they are paired across models. This assumes every
model is scored on the same samples per task, which holds while the task definitions are frozen.
"""

import numpy as np
import pandas as pd

from brainmarks_smri.metrics import auroc, bootstrap_score, r2_score
from brainmarks_smri.tasks import TASK_GROUPS

N_BOOTSTRAP = 1000
SEED = 0


def sorted_samples(result: dict) -> list[dict]:
    return sorted(result["eval_samples"], key=lambda s: (s["participant_id"], s["session_id"]))


def score_auroc(result: dict) -> tuple[float, np.ndarray]:
    """2 AUROC - 1, resampled within each class."""
    samples = sorted_samples(result)
    labels = np.array([result["classes"].index(sample[result["target"]]) for sample in samples])
    # stored in the order of the classes
    probabilities = np.array([list(sample["probabilities"].values()) for sample in samples])
    point, boot = bootstrap_score(
        auroc,
        labels,
        probabilities,
        strata=labels,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
    return 2 * point - 1, 2 * boot - 1


def score_r2(result: dict) -> tuple[float, np.ndarray]:
    samples = sorted_samples(result)
    targets = np.array([sample[result["target"]] for sample in samples], dtype=float)
    predictions = np.array([sample["prediction"] for sample in samples], dtype=float)
    return bootstrap_score(r2_score, targets, predictions, n_bootstrap=N_BOOTSTRAP, seed=SEED)


def score_dice(result: dict) -> tuple[float, np.ndarray]:
    """Mean per-subject Dice."""
    dice = np.array([sample["dice"] for sample in sorted_samples(result)], dtype=float)
    return bootstrap_score(np.nanmean, dice, n_bootstrap=N_BOOTSTRAP, seed=SEED)


def score_ap_log_lift(result: dict) -> tuple[float, np.ndarray]:
    """Mean per-subject AP relative to chance (the label prevalence), on a log scale."""
    samples = sorted_samples(result)
    average_precision = np.array([sample["average_precision"] for sample in samples])
    prevalence = np.array([sample["prevalence"] for sample in samples])
    log_lift = np.log(average_precision / prevalence) / np.log(1 / prevalence)
    return bootstrap_score(np.nanmean, log_lift, n_bootstrap=N_BOOTSTRAP, seed=SEED)


SCORE_BY_TYPE = {"classification": score_auroc, "regression": score_r2, "segmentation": score_dice}
# Tiny structures: Dice is too noisy, raw AP too small to count in the mean.
SCORE_BY_TASK = {"fomo26_task4_nerve": score_ap_log_lift, "fomo26_task4_vessel": score_ap_log_lift}


def aggregate_scores(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score, 95% CI and sample count per task and per group of tasks (rows), and the bootstrap
    resamples (columns: tasks and groups)."""
    subsets = {result["subset"] for result in results}
    assert len(subsets) == 1, f"results from more than one subset: {subsets}"
    rows = {}
    boot = pd.DataFrame()
    for result in sorted(results, key=lambda result: result["task"]):
        task = result["task"]
        score = SCORE_BY_TASK.get(task, SCORE_BY_TYPE[result["type"]])
        point, task_boot = score(result)
        low, high = np.percentile(task_boot, [2.5, 97.5])
        n_samples = len(result["eval_samples"])
        rows[task] = {"score": float(point), "low": low, "high": high, "n_samples": n_samples}
        boot[task] = task_boot

    types = {result["task"]: result["type"] for result in results}
    tasks = list(rows)
    groups = {
        "global": [task for task in tasks if types[task] in ("classification", "regression")],
        "segmentation": [task for task in tasks if types[task] == "segmentation"],
    }
    # only complete: a mean over part of the core tasks is not the core score
    for group in ["core", "open"]:
        if set(TASK_GROUPS[group]) <= set(tasks):
            groups[group] = TASK_GROUPS[group]
    groups["all"] = tasks
    for group, members in groups.items():
        if not members:
            continue
        boot[group] = boot[members].mean(axis=1)
        low, high = np.percentile(boot[group], [2.5, 97.5])
        rows[group] = {
            "score": float(np.mean([rows[task]["score"] for task in members])),
            "low": low,
            "high": high,
            "n_samples": sum(rows[task]["n_samples"] for task in members),
        }
    return pd.DataFrame.from_dict(rows, orient="index"), boot
