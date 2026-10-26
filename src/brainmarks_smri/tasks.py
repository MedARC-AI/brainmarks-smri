"""Benchmark tasks.

A task is a plain function returning a `Task`: a dataset, a target and the train/eval ids. It
handles its own special cases inside. `eval_split` is val by default; the CLI never uses test.
Small datasets are all train and use k-fold types: the task assigns the folds.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from brainmarks_smri.datasets import BrainDataset, create_dataset
from brainmarks_smri.models.base import ImageInput

N_FOLDS = 5
FOLD_SEED = 0
# One size for all splits, or a size per split name, e.g. {train: 300, val: 50}.
MaxPerSplit = int | Mapping[str, int] | None


@dataclass
class Task:
    name: str
    type: Literal[
        "classification",
        "regression",
        "segmentation",
        "classification_kfold",
        "segmentation_kfold",
    ]
    dataset: BrainDataset
    # Column of dataset.samples (classification, regression) or mask key (segmentation).
    target: str
    train_ids: np.ndarray
    # Split types only.
    eval_ids: np.ndarray | None = None
    # K-fold types: the fold of each train id.
    folds: np.ndarray | None = None
    # Applied to the image input before the model's transform.
    transform: Callable[[ImageInput], ImageInput] | None = None
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
    samples: pd.DataFrame, keep: pd.Series, eval_split: str, max_per_split: MaxPerSplit
) -> tuple[np.ndarray, np.ndarray]:
    """Train and `eval_split` ids among the `keep` samples.

    `max_per_split`: the lowest-rank samples per split (nested, balanced mini-splits). Taken after
    `keep`, so mini-splits are full. One size for all splits, or a size per split name.
    """
    assert eval_split in ("val", "test"), eval_split
    ids = []
    for split in ("train", eval_split):
        selected = samples[keep & (samples["split"] == split)]
        size = max_per_split.get(split) if isinstance(max_per_split, Mapping) else max_per_split
        if size is not None:
            selected = selected.nsmallest(size, "rank")
        ids.append(np.sort(selected.index.to_numpy()))
    return ids[0], ids[1]


def kfold_ids(
    samples: pd.DataFrame, keep: pd.Series, stratify_by: str | None, max_per_split: MaxPerSplit
) -> tuple[np.ndarray, np.ndarray]:
    """The `keep` train samples and their folds, stratified on `stratify_by` if given.

    `max_per_split`: the lowest-rank samples, as in `split_ids`.
    """
    selected = samples[keep & (samples["split"] == "train")]
    size = max_per_split.get("train") if isinstance(max_per_split, Mapping) else max_per_split
    if size is not None:
        selected = selected.nsmallest(size, "rank")
    assert not selected["participant_id"].duplicated().any(), "k-fold needs one sample per person"
    ids = np.sort(selected.index.to_numpy())

    if stratify_by is None:
        splitter = KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
        labels = None
    else:
        splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
        labels = samples.loc[ids, stratify_by].to_numpy()
    folds = np.zeros(len(ids), dtype=int)
    for fold, (_, fold_ids) in enumerate(splitter.split(ids, labels)):
        folds[fold_ids] = fold
    return ids, folds


@register_task
def abide_diagnosis(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
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
def cnp_diagnosis(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
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
def adhd_diagnosis(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
    dataset = create_dataset("adhd200", modality="T1w")
    samples = dataset.samples
    # Site and sex predict the label (WashU is all controls; most ADHD are male), so keep equal
    # numbers of ADHD and TDC per site, sex and split, lowest rank first. Brown's test labels were
    # never released.
    keep = pd.Series(False, index=samples.index)
    labeled = samples[samples["adhd"].notna()]
    for _, stratum in labeled.groupby(["site", "sex", "split"]):
        class_counts = stratum["adhd"].value_counts()
        n_per_class = 0 if len(class_counts) < 2 else class_counts.min()
        for _, rows in stratum.groupby("adhd"):
            keep[rows.nsmallest(n_per_class, "rank").index] = True
    train_ids, eval_ids = split_ids(samples, keep, eval_split, max_per_split)
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
def ixi_age(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
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
    max_per_split: MaxPerSplit,
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
def brats_whole_tumor(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
    return brats_region("brats_whole_tumor", "FLAIR", [1, 2, 4], eval_split, max_per_split)


@register_task
def brats_tumor_core(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
    return brats_region("brats_tumor_core", "T1c", [1, 4], eval_split, max_per_split)


@register_task
def brats_enhancing_tumor(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
    return brats_region("brats_enhancing_tumor", "T1c", [4], eval_split, max_per_split)


@register_task
def openbhb_age(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
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
def ucsf_idh(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
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
def upenn_survival_1y(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
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
def soop_mrs_poor(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
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
def soop_lesion(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
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


@register_task
def atlas_lesion(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
    dataset = create_dataset("atlas", modality="T1w")
    keep = dataset.samples["mask_lesion_path"].notna()
    train_ids, eval_ids = split_ids(dataset.samples, keep, eval_split, max_per_split)
    return Task(
        name="atlas_lesion",
        type="segmentation",
        dataset=dataset,
        target="mask_lesion",
        train_ids=train_ids,
        eval_ids=eval_ids,
        label_values=[1],
    )


@register_task
def oasis1_dementia(eval_split: str = "val", max_per_split: MaxPerSplit = None) -> Task:
    dataset = create_dataset("oasis1", modality="T1w")
    samples = dataset.samples
    # CDR is only given for first sessions (MR1), mostly 60+.
    samples["dementia"] = (samples["cdr"] > 0).where(samples["cdr"].notna())
    keep = (samples["session_id"] == "MR1") & samples["cdr"].notna()
    train_ids, eval_ids = split_ids(samples, keep, eval_split, max_per_split)
    return Task(
        name="oasis1_dementia",
        type="classification",
        dataset=dataset,
        target="dementia",
        train_ids=train_ids,
        eval_ids=eval_ids,
        classes=[False, True],
    )


@register_task
def oasis2_dementia(max_per_split: MaxPerSplit = None) -> Task:
    dataset = create_dataset("oasis2", modality="T1w")
    samples = dataset.samples
    # CDR > 0 at the first visit only, one scan per participant (converters count as CDR 0).
    samples["dementia"] = (samples["cdr"] > 0).where(samples["cdr"].notna())
    keep = (samples["visit"] == 1) & samples["cdr"].notna()
    ids, folds = kfold_ids(samples, keep, "dementia", max_per_split)
    return Task(
        name="oasis2_dementia",
        type="classification_kfold",
        dataset=dataset,
        target="dementia",
        train_ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def oasis2_conversion(max_per_split: MaxPerSplit = None) -> Task:
    dataset = create_dataset("oasis2", modality="T1w")
    samples = dataset.samples
    # Nondemented at the first visit: later converted to dementia vs stayed nondemented.
    samples["converted"] = samples["group"] == "converted"
    keep = (samples["visit"] == 1) & samples["group"].isin(["nondemented", "converted"])
    ids, folds = kfold_ids(samples, keep, "converted", max_per_split)
    return Task(
        name="oasis2_conversion",
        type="classification_kfold",
        dataset=dataset,
        target="converted",
        train_ids=ids,
        folds=folds,
        classes=[False, True],
    )


AP_EXTENT_MM = 133


def strip_and_crop_ap(sample: ImageInput) -> ImageInput:
    """Zero the image outside the brain mask and outside an `AP_EXTENT_MM` window along the
    second (anterior) axis, centered on the brain."""
    image = sample["image"]
    brain_mask = sample["brain_mask"]
    assert nib.aff2axcodes(image.affine) == ("R", "A", "S"), "expected RAS images"
    keep = np.asanyarray(brain_mask.dataobj) > 0
    width = round(AP_EXTENT_MM / image.header.get_zooms()[1])
    brain_ap = np.flatnonzero(keep.any(axis=(0, 2)))
    start = max(round((brain_ap[0] + brain_ap[-1] - width) / 2), 0)
    keep[:, :start] = False
    keep[:, start + width :] = False

    data = np.where(keep, np.asanyarray(image.dataobj), 0)
    return {
        "image": nib.Nifti1Image(data, image.affine, image.header),
        "brain_mask": nib.Nifti1Image(keep.astype(np.uint8), brain_mask.affine, brain_mask.header),
        "mni_affine": sample["mni_affine"],
    }


@register_task
def fomo26_task5_pmg(max_per_split: MaxPerSplit = None) -> Task:
    dataset = create_dataset("fomo26_task5", modality="T1w")
    keep = dataset.samples["pmg"].notna()
    ids, folds = kfold_ids(dataset.samples, keep, "pmg", max_per_split)
    return Task(
        name="fomo26_task5_pmg",
        type="classification_kfold",
        dataset=dataset,
        target="pmg",
        train_ids=ids,
        folds=folds,
        classes=["control", "polymicrogyria"],
        transform=strip_and_crop_ap,
    )


def fomo26_task4_region(name: str, label_values: list[int], max_per_split: MaxPerSplit) -> Task:
    dataset = create_dataset("fomo26_task4", modality="T2w")
    keep = dataset.samples["mask_trigeminal_path"].notna()
    ids, folds = kfold_ids(dataset.samples, keep, None, max_per_split)
    return Task(
        name=name,
        type="segmentation_kfold",
        dataset=dataset,
        target="mask_trigeminal",
        train_ids=ids,
        folds=folds,
        label_values=label_values,
    )


@register_task
def fomo26_task4_nerve(max_per_split: MaxPerSplit = None) -> Task:
    return fomo26_task4_region("fomo26_task4_nerve", [1], max_per_split)


@register_task
def fomo26_task4_vessel(max_per_split: MaxPerSplit = None) -> Task:
    return fomo26_task4_region("fomo26_task4_vessel", [2], max_per_split)
