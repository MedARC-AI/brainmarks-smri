"""Benchmark tasks.

A task is a plain function returning a `Task`: a dataset, a target and the train/eval ids. It
handles its own special cases inside. `eval_split` ("val" or "test") and `max_per_split`
(mini-splits) are run options.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import nibabel as nib
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
    classes: list[str] | None = None
    # Segmentation: the mask label values that make up the target region.
    label_values: list[int] | None = None


def split_ids(
    samples: pd.DataFrame, keep: pd.Series, eval_split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Train and `eval_split` ids from splits.tsv, among the `keep` samples."""
    assert eval_split in ("val", "test"), eval_split
    train_ids = np.flatnonzero(keep & (samples["split"] == "train"))
    eval_ids = np.flatnonzero(keep & (samples["split"] == eval_split))
    return train_ids, eval_ids


def abide_diagnosis(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("abide1", modality="T1w", max_per_split=max_per_split)
    keep = dataset.samples["diagnosis"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split)
    return Task(
        name="abide_diagnosis",
        type="classification",
        dataset=dataset,
        target="diagnosis",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=["TDC", "ASD"],
    )


def cnp_diagnosis(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("cnp", modality="T1w", max_per_split=max_per_split)
    keep = dataset.samples["diagnosis"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split)
    return Task(
        name="cnp_diagnosis",
        type="classification",
        dataset=dataset,
        target="diagnosis",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=["CONTROL", "SCHZ", "BIPOLAR", "ADHD"],
    )


def ixi_age(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("ixi", modality="T1w", max_per_split=max_per_split)
    keep = dataset.samples["age"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split)
    return Task(
        name="ixi_age",
        type="regression",
        dataset=dataset,
        target="age",
        train_ids=train_ids,
        eval_ids=eval_ids,
    )


def brats_region(
    name: str, label_values: list[int], eval_split: str, max_per_split: int | None
) -> Task:
    dataset = create_dataset("brats2021", modality="FLAIR", max_per_split=max_per_split)
    # Only the official training cases have masks.
    keep = dataset.samples["mask_tumor_path"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split)
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
def brats_whole_tumor(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    return brats_region("brats_whole_tumor", [1, 2, 4], eval_split, max_per_split)


def brats_tumor_core(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    return brats_region("brats_tumor_core", [1, 4], eval_split, max_per_split)


def brats_enhancing_tumor(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    return brats_region("brats_enhancing_tumor", [4], eval_split, max_per_split)


def soop_lesion(eval_split: str = "val", max_per_split: int | None = None) -> Task:
    dataset = create_dataset("soop", modality="DWI", max_per_split=max_per_split)
    samples = dataset.samples
    keep = samples["mask_lesionAcute_path"].notna()
    # Drop empty masks (none in the current release).
    for index in np.flatnonzero(keep):
        mask = nib.load(dataset.root / samples["mask_lesionAcute_path"].iloc[index])
        if not np.asanyarray(mask.dataobj).any():
            keep.iloc[index] = False
    train_ids, eval_ids = split_ids(samples, keep, eval_split)
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


TASKS: dict[str, Callable[..., Task]] = {
    task.__name__: task
    for task in [
        abide_diagnosis,
        cnp_diagnosis,
        ixi_age,
        brats_whole_tumor,
        brats_tumor_core,
        brats_enhancing_tumor,
        soop_lesion,
    ]
}
