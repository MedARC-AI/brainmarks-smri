"""Benchmark tasks.

A task is a plain function returning a `Task`: a dataset, a target and the train/eval ids. It
handles its own special cases inside. `eval_split` is val by default; the CLI never uses test.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from brainmarks_smri.datasets import BrainDataset, create_dataset


@dataclass
class Task:
    name: str
    type: Literal["classification", "regression", "segmentation"]
    dataset: BrainDataset
    # Column of dataset.samples (classification, regression) or mask key (segmentation).
    target: str
    train_ids: np.ndarray
    eval_ids: np.ndarray
    # Classification: target values in order; the last is the positive class for binary tasks.
    classes: list[str | bool] | None = None
    # Segmentation: the mask label values that make up the target region.
    label_values: list[int] | None = None


TASKS: dict[str, Callable[..., Task]] = {}


def register_task(task_fn: Callable[..., Task]) -> Callable[..., Task]:
    TASKS[task_fn.__name__] = task_fn
    return task_fn


def create_task(name: str, **kwargs) -> Task:
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; available: {list_tasks()}")
    return TASKS[name](**kwargs)


def list_tasks() -> list[str]:
    return sorted(TASKS)


def split_ids(
    samples: pd.DataFrame, keep: pd.Series, eval_split: str, max_per_split: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Train and `eval_split` ids among the `keep` samples.

    `max_per_split`: the lowest-rank samples per split (nested, balanced mini-splits). Taken after
    `keep`, so mini-splits are full.
    """
    assert eval_split in ("val", "test"), eval_split
    ids = []
    for split in ("train", eval_split):
        selected = samples[keep & (samples["split"] == split)]
        if max_per_split is not None:
            selected = selected.nsmallest(max_per_split, "rank")
        ids.append(np.sort(selected.index.to_numpy()))
    return ids[0], ids[1]


@register_task
def abide_diagnosis(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("abide1", modality="T1w")
    keep = dataset.samples["diagnosis"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name="abide_diagnosis",
        type="classification",
        dataset=dataset,
        target="diagnosis",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=["TDC", "ASD"],
    )


@register_task
def cnp_diagnosis(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("cnp", modality="T1w")
    samples = dataset.samples
    # 3-way as in Neuro-JEPA: schizophrenia and bipolar disorder are one class.
    samples["diagnosis_3way"] = samples["diagnosis"].replace(
        {"SCHZ": "SCHZ+BIPOLAR", "BIPOLAR": "SCHZ+BIPOLAR"}
    )
    keep = samples["diagnosis_3way"].notna()
    train_ids, eval_ids = split_ids(samples, keep, eval_split, max_per_split)
    return Task(
        name="cnp_diagnosis",
        type="classification",
        dataset=dataset,
        target="diagnosis_3way",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=["CONTROL", "ADHD", "SCHZ+BIPOLAR"],
    )


@register_task
def adhd_diagnosis(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("adhd200", modality="T1w")
    # Brown's test participants have no released labels.
    keep = dataset.samples["adhd"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name="adhd_diagnosis",
        type="classification",
        dataset=dataset,
        target="adhd",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=[False, True],
    )


@register_task
def ixi_age(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("ixi", modality="T1w")
    keep = dataset.samples["age"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name="ixi_age",
        type="regression",
        dataset=dataset,
        target="age",
        train_ids=train_ids,
        eval_ids=eval_ids,
    )


def brats_region(
    name: str,
    modality: str,
    label_values: list[int],
    eval_split: str,
    max_per_split: int | None,
) -> Task:
    dataset = create_dataset("brats2021", modality=modality)
    # Only the official training cases have masks.
    keep = dataset.samples["mask_tumor_path"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name=name,
        type="segmentation",
        dataset=dataset,
        target="mask_tumor",
        train_ids=train_ids,
        eval_ids=eval_ids,
        label_values=label_values,
    )


# The BraTS challenge regions, from labels 1 (necrotic core), 2 (edema), 4 (enhancing tumor).
# Whole tumor is mostly edema, brightest on FLAIR; core and enhancing are defined on T1c.
@register_task
def brats_whole_tumor(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    return brats_region("brats_whole_tumor", "FLAIR", [1, 2, 4], eval_split, max_per_split)


@register_task
def brats_tumor_core(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    return brats_region("brats_tumor_core", "T1c", [1, 4], eval_split, max_per_split)


@register_task
def brats_enhancing_tumor(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    return brats_region("brats_enhancing_tumor", "T1c", [4], eval_split, max_per_split)


@register_task
def openbhb_age(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("openbhb", modality="T1w")
    keep = dataset.samples["age"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name="openbhb_age",
        type="regression",
        dataset=dataset,
        target="age",
        train_ids=train_ids,
        eval_ids=eval_ids,
    )


@register_task
def ucsf_idh(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("ucsf_pdgm", modality="FLAIR")
    keep = dataset.samples["idh"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name="ucsf_idh",
        type="classification",
        dataset=dataset,
        target="idh",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=["wildtype", "mutant"],
    )


@register_task
def upenn_survival_1y(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("upenn_gbm", modality="FLAIR")
    samples = dataset.samples
    # Death within a year of surgery. Follow-up scans measure survival from the scan, so only
    # baselines; patients censored before a year have no label.
    died_in_1y = (samples["os_event"] == 1) & (samples["os_days"] <= 365)
    known = died_in_1y | (samples["os_days"] > 365)
    samples["death_1y"] = died_in_1y.where(known)
    keep = (samples["session_id"] == "baseline") & known
    train_ids, eval_ids = split_ids(samples, keep, eval_split, max_per_split)
    return Task(
        name="upenn_survival_1y",
        type="classification",
        dataset=dataset,
        target="death_1y",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=[False, True],
    )


@register_task
def soop_mrs_poor(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("soop", modality="FLAIR")
    keep = dataset.samples["mrs_poor"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name="soop_mrs_poor",
        type="classification",
        dataset=dataset,
        target="mrs_poor",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=[False, True],
    )


@register_task
def soop_lesion(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("soop", modality="DWI")
    keep = dataset.samples["mask_lesionAcute_path"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name="soop_lesion",
        type="segmentation",
        dataset=dataset,
        target="mask_lesionAcute",
        train_ids=train_ids,
        eval_ids=eval_ids,
        # Binary masks; a few use 2 or 3 as the lesion value instead of 1.
        label_values=[1, 2, 3],
    )
