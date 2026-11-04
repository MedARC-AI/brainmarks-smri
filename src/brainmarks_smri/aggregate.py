"""Benchmark score: the mean per-task score over the core tasks, with paired bootstrap CIs.

Reads `<results>/<name>/<task>.json` as written by `run.py`, and writes `<results>/aggregate.json`.

python -m brainmarks_smri.aggregate output
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brainmarks_smri.metrics import auroc
from brainmarks_smri.tasks import TASK_GROUPS, expand_tasks, list_tasks

# Per-task scores, 0 at chance and at most 1.
SCORE_BY_TYPE = {"classification": "auroc", "regression": "r2", "segmentation": "dice"}
# Tiny structures: AP relative to chance (the label prevalence), on a log scale.
SCORE_BY_TASK = {"fomo26_task4_nerve": "ap_log_lift", "fomo26_task4_vessel": "ap_log_lift"}
N_BOOTSTRAP = 1000
SEED = 0
# Keys of a classification or regression sample other than its target.
NON_TARGET_KEYS = {"participant_id", "session_id", "fold", "probabilities", "prediction"}


def load_results(folder: Path) -> dict[str, dict[str, dict]]:
    """Results by name (the subfolder) and task."""
    results = {}
    for path in sorted(folder.glob("*/*.json")):
        result = json.loads(path.read_text())
        results.setdefault(path.parent.name, {})[result["task"]] = result
    return results


def sample_values(result: dict) -> tuple[str, list, np.ndarray | None, np.ndarray]:
    """Score rule, sample keys, targets (class indices, values, or None) and values (class
    probabilities, predictions, or per-subject scores), sorted by sample."""
    rule = SCORE_BY_TASK.get(result["task"], SCORE_BY_TYPE[result["type"]])
    records = sorted(result["eval_samples"], key=lambda r: (r["participant_id"], r["session_id"]))
    keys = [(record["participant_id"], record["session_id"]) for record in records]
    if rule == "dice":
        return rule, keys, None, np.array([record["dice"] for record in records], dtype=float)
    if rule == "ap_log_lift":
        average_precision = np.array([record["average_precision"] for record in records])
        prevalence = np.array([record["prevalence"] for record in records])
        return rule, keys, None, np.log(average_precision / prevalence) / np.log(1 / prevalence)

    (target_key,) = set(records[0]) - NON_TARGET_KEYS
    if rule == "r2":
        targets = np.array([record[target_key] for record in records], dtype=float)
        predictions = np.array([record["prediction"] for record in records], dtype=float)
        return rule, keys, targets, predictions
    labels = np.array([result["classes"].index(record[target_key]) for record in records])
    # probabilities are stored in the order of the classes
    probabilities = np.array([list(record["probabilities"].values()) for record in records])
    return rule, keys, labels, probabilities


def task_score(
    rule: str, targets: np.ndarray | None, values: np.ndarray, index: np.ndarray
) -> float:
    if rule == "auroc":
        return 2 * auroc(targets[index], values[index]) - 1
    if rule == "r2":
        sample_targets = targets[index]
        residual = ((values[index] - sample_targets) ** 2).sum()
        total = ((sample_targets - sample_targets.mean()) ** 2).sum()
        return float(1 - residual / total)
    return float(np.nanmean(values[index]))


def bootstrap_scores(
    results: dict[str, dict[str, dict]], names: list[str], tasks: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Task scores (names x tasks), and the same over paired subject resamples (bootstrap x
    names x tasks): one resample for all models, drawn within each class for AUROC."""
    rng = np.random.default_rng(SEED)
    point = pd.DataFrame(index=names, columns=tasks, dtype=float)
    boot = np.zeros((N_BOOTSTRAP, len(names), len(tasks)))
    for task_index, task in enumerate(tasks):
        samples = {name: sample_values(results[name][task]) for name in names}
        rule, keys, targets, _ = samples[names[0]]
        for name in names:
            assert samples[name][1] == keys, f"{task}: {name} has other samples than {names[0]}"
        everyone = np.arange(len(keys))
        if rule == "auroc":
            strata = [np.flatnonzero(targets == label) for label in np.unique(targets)]
        else:
            strata = [everyone]
        for name in names:
            _, _, targets, values = samples[name]
            point.loc[name, task] = task_score(rule, targets, values, everyone)
        for resample in range(N_BOOTSTRAP):
            index = np.concatenate([rng.choice(stratum, len(stratum)) for stratum in strata])
            for name_index, name in enumerate(names):
                _, _, targets, values = samples[name]
                boot[resample, name_index, task_index] = task_score(rule, targets, values, index)
    return point, boot


def summary_table(point: pd.DataFrame, boot: np.ndarray) -> pd.DataFrame:
    """Mean score over tasks with its 95% CI, best first."""
    low, high = np.percentile(boot.mean(axis=2), [2.5, 97.5], axis=0)
    table = pd.DataFrame({"score": point.mean(axis=1), "low": low, "high": high})
    return table.sort_values("score", ascending=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--tasks", nargs="+", default=["core"], choices=list_tasks() + TASK_GROUPS, metavar="task"
    )
    args = parser.parse_args()

    tasks = expand_tasks(args.tasks)
    results = load_results(args.results)
    subsets = {result["subset"] for by_task in results.values() for result in by_task.values()}
    assert len(subsets) == 1, f"results from more than one subset: {subsets}"
    names = [name for name in results if all(task in results[name] for task in tasks)]
    incomplete = sorted(set(results) - set(names))
    if incomplete:
        print(f"Skipped, missing tasks: {', '.join(incomplete)}")
    if not names:
        return

    point, boot = bootstrap_scores(results, names, tasks)
    table = summary_table(point, boot)
    names = list(table.index)
    means = pd.DataFrame(boot.mean(axis=2), columns=point.index)
    # P(row > column) over the bootstrap
    p_better = pd.DataFrame(
        [[(means[a] > means[b]).mean() for b in names] for a in names], index=names, columns=names
    )

    print(
        f"\n{len(tasks)} tasks, {subsets.pop()} subset, {N_BOOTSTRAP} paired bootstrap resamples\n"
    )
    summary = pd.DataFrame(
        {
            "score": table.score.map("{:.3f}".format),
            "95% CI": [f"[{low:.3f}, {high:.3f}]" for low, high in zip(table.low, table.high)],
        }
    )
    print(summary.to_markdown())
    print("\nPer-task scores\n")
    print(point.loc[names].T.round(3).to_markdown())
    print("\nP(row > column)\n")
    print(p_better.round(2).to_markdown())

    aggregate = {
        "tasks": tasks,
        "scores": {
            name: {
                "score": table.loc[name, "score"],
                "ci": [table.loc[name, "low"], table.loc[name, "high"]],
                "tasks": point.loc[name].to_dict(),
            }
            for name in names
        },
        "p_better": p_better.to_dict(orient="index"),
    }
    path = args.results / "aggregate.json"
    path.write_text(json.dumps(aggregate, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
