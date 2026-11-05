"""Benchmark tasks.

A task is a plain function returning a `Task`: a dataset, a target, and the samples with their
cross-validation folds. It handles its own special cases inside.
"""

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd

from brainmarks_smri.datasets import BrainDataset, create_dataset
from brainmarks_smri.models.base import ImageInput

Subset = Literal["dev", "test"]
N_FOLDS = 5
SEED = 0
# Samples per task and subset; bounds the cost of the largest datasets.
MAX_SAMPLES = 500
# Continuous targets are stratified by quantile bins.
N_STRATA_BINS = 5
# sha256 of the maintainers' key for the test subset
TEST_KEY_SHA256 = "638de7459ed1b2d936c41ee87b8e7a554ffad5f4fed81e3f4df9a48199c788f4"


@dataclass
class Task:
    name: str
    type: Literal["classification", "regression", "segmentation"]
    subset: Subset
    dataset: BrainDataset
    # Column of dataset.samples (classification, regression) or mask key (segmentation).
    target: str
    # Rows of dataset.samples, and the fold of each. The probe fit on the other folds predicts a
    # fold; fold -1 is train only (e.g. for a fixed train/eval split).
    ids: np.ndarray
    folds: np.ndarray
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


def fold_ids(
    samples: pd.DataFrame,
    keep: pd.Series,
    subset: Subset,
    stratify_by: str | None,
    max_samples: int | None = MAX_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """The `keep` samples of `subset` and their folds, balanced on `stratify_by` if given.

    The samples are ordered so that the strata interleave in proportion to their sizes: within
    each stratum, sample k of n sits at (k + u) / n, with a random offset u. Any contiguous block
    of the order is then balanced. The first `max_samples` are kept, so smaller caps are nested in
    larger ones, and the folds are `N_FOLDS` contiguous blocks.
    """
    assert subset in ("dev", "test"), subset
    if subset == "test":
        key = os.environ.get("BRAINMARKS_SMRI_TEST_KEY", "")
        if hashlib.sha256(key.encode()).hexdigest() != TEST_KEY_SHA256:
            raise RuntimeError(
                "The test subset is held out for rare checks by the maintainers; don't develop, "
                "select or compare models on it. Use the dev subset."
            )
    selected = samples[keep & (samples["split"] == subset)]
    assert not selected["participant_id"].duplicated().any(), "one sample per participant"

    if stratify_by is None:
        strata = pd.Series("all", index=selected.index)
    else:
        values = selected[stratify_by]
        continuous = pd.api.types.is_float_dtype(values) and values.nunique() > N_STRATA_BINS
        if continuous:
            strata = pd.qcut(values, N_STRATA_BINS, labels=False, duplicates="drop")
        else:
            strata = values
        strata = strata.astype(str)

    rng = np.random.default_rng(SEED)
    position = pd.Series(np.nan, index=selected.index)
    for _, group in selected.groupby(strata):  # groupby sorts the keys
        group_ids = rng.permutation(np.sort(group.index.to_numpy()))
        position[group_ids] = (np.arange(len(group_ids)) + rng.uniform()) / len(group_ids)
    ids = position.sort_values(kind="stable").index.to_numpy()
    if max_samples is not None:
        ids = ids[:max_samples]
    folds = np.arange(len(ids)) * N_FOLDS // len(ids)
    return ids, folds


@register_task
def abide_diagnosis(subset: Subset = "dev") -> Task:
    dataset = create_dataset("abide1", modality="T1w")
    keep = dataset.samples["diagnosis"].notna()
    ids, folds = fold_ids(dataset.samples, keep, subset, "diagnosis")
    return Task(
        name="abide_diagnosis",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="diagnosis",
        ids=ids,
        folds=folds,
        classes=["TDC", "ASD"],
    )


@register_task
def cnp_diagnosis(subset: Subset = "dev") -> Task:
    dataset = create_dataset("cnp", modality="T1w")
    samples = dataset.samples
    # 3-way as in Neuro-JEPA: schizophrenia and bipolar disorder are one class.
    samples["diagnosis_3way"] = samples["diagnosis"].replace(
        {"SCHZ": "SCHZ+BIPOLAR", "BIPOLAR": "SCHZ+BIPOLAR"}
    )
    keep = samples["diagnosis_3way"].notna()
    ids, folds = fold_ids(samples, keep, subset, "diagnosis_3way")
    return Task(
        name="cnp_diagnosis",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="diagnosis_3way",
        ids=ids,
        folds=folds,
        classes=["CONTROL", "ADHD", "SCHZ+BIPOLAR"],
    )


@register_task
def adhd_diagnosis(subset: Subset = "dev") -> Task:
    dataset = create_dataset("adhd200", modality="T1w")
    samples = dataset.samples
    # Site and sex predict the label (WashU is all controls; most ADHD are male), so keep equal
    # numbers of ADHD and TDC per site, sex and subset, chosen at random. Brown's labels were never
    # released.
    keep = pd.Series(False, index=samples.index)
    labeled = samples[samples["adhd"].notna()]
    for _, stratum in labeled.groupby(["site", "sex", "split"]):
        class_counts = stratum["adhd"].value_counts()
        n_per_class = 0 if len(class_counts) < 2 else class_counts.min()
        for _, rows in stratum.groupby("adhd"):
            keep[rows.sample(n_per_class, random_state=SEED).index] = True
    ids, folds = fold_ids(samples, keep, subset, "adhd")
    return Task(
        name="adhd_diagnosis",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="adhd",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def ixi_age(subset: Subset = "dev") -> Task:
    dataset = create_dataset("ixi", modality="T1w")
    keep = dataset.samples["age"].notna()
    ids, folds = fold_ids(dataset.samples, keep, subset, "age")
    return Task(
        name="ixi_age",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="age",
        ids=ids,
        folds=folds,
    )


def brats_region(name: str, modality: str, label_values: list[int], subset: Subset) -> Task:
    dataset = create_dataset(
        "brats2021", modality=modality, mask_target="mask_tumor", label_values=label_values
    )
    samples = dataset.samples
    # Only the official training cases have masks; two patients have a follow-up case too.
    masked = samples[samples["mask_tumor_path"].notna()]
    first_cases = masked.sort_values("session_id").groupby("participant_id").head(1)
    keep = samples.index.isin(first_cases.index)
    ids, folds = fold_ids(samples, keep, subset, None)
    return Task(
        name=name,
        type="segmentation",
        subset=subset,
        dataset=dataset,
        target="mask_tumor",
        ids=ids,
        folds=folds,
        label_values=label_values,
    )


# The BraTS challenge regions, from labels 1 (necrotic core), 2 (edema), 4 (enhancing tumor).
# Whole tumor is mostly edema, brightest on FLAIR; core and enhancing are defined on T1c.
@register_task
def brats_whole_tumor(subset: Subset = "dev") -> Task:
    return brats_region("brats_whole_tumor", "FLAIR", [1, 2, 4], subset)


@register_task
def brats_tumor_core(subset: Subset = "dev") -> Task:
    return brats_region("brats_tumor_core", "T1c", [1, 4], subset)


@register_task
def brats_enhancing_tumor(subset: Subset = "dev") -> Task:
    return brats_region("brats_enhancing_tumor", "T1c", [4], subset)


@register_task
def openbhb_age(subset: Subset = "dev") -> Task:
    dataset = create_dataset("openbhb", modality="T1w")
    keep = dataset.samples["age"].notna()
    ids, folds = fold_ids(dataset.samples, keep, subset, "age")
    return Task(
        name="openbhb_age",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="age",
        ids=ids,
        folds=folds,
    )


@register_task
def ucsf_idh(subset: Subset = "dev") -> Task:
    dataset = create_dataset("ucsf_pdgm", modality="FLAIR")
    samples = dataset.samples
    # one scan per person; a few have follow-ups
    keep = (samples["session_id"] == "baseline") & samples["idh"].notna()
    ids, folds = fold_ids(samples, keep, subset, "idh")
    return Task(
        name="ucsf_idh",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="idh",
        ids=ids,
        folds=folds,
        classes=["wildtype", "mutant"],
    )


@register_task
def upenn_survival_1y(subset: Subset = "dev") -> Task:
    dataset = create_dataset("upenn_gbm", modality="FLAIR")
    samples = dataset.samples
    # Death within a year of surgery. Follow-up scans measure survival from the scan, so only
    # baselines; patients censored before a year have no label.
    died_in_1y = (samples["os_event"] == 1) & (samples["os_days"] <= 365)
    known = died_in_1y | (samples["os_days"] > 365)
    samples["death_1y"] = died_in_1y.where(known)
    keep = (samples["session_id"] == "baseline") & known
    ids, folds = fold_ids(samples, keep, subset, "death_1y")
    return Task(
        name="upenn_survival_1y",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="death_1y",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def soop_mrs_poor(subset: Subset = "dev") -> Task:
    dataset = create_dataset("soop", modality="FLAIR")
    keep = dataset.samples["mrs_poor"].notna()
    ids, folds = fold_ids(dataset.samples, keep, subset, "mrs_poor")
    return Task(
        name="soop_mrs_poor",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="mrs_poor",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def soop_lesion(subset: Subset = "dev") -> Task:
    # Binary masks; a few use 2 or 3 as the lesion value instead of 1.
    dataset = create_dataset(
        "soop", modality="DWI", mask_target="mask_lesionAcute", label_values=[1, 2, 3]
    )
    keep = dataset.samples["mask_lesionAcute_path"].notna()
    ids, folds = fold_ids(dataset.samples, keep, subset, None)
    return Task(
        name="soop_lesion",
        type="segmentation",
        subset=subset,
        dataset=dataset,
        target="mask_lesionAcute",
        ids=ids,
        folds=folds,
        label_values=[1, 2, 3],
    )


@register_task
def atlas_lesion(subset: Subset = "dev") -> Task:
    dataset = create_dataset("atlas", modality="T1w", mask_target="mask_lesion", label_values=[1])
    keep = dataset.samples["mask_lesion_path"].notna()
    ids, folds = fold_ids(dataset.samples, keep, subset, None)
    return Task(
        name="atlas_lesion",
        type="segmentation",
        subset=subset,
        dataset=dataset,
        target="mask_lesion",
        ids=ids,
        folds=folds,
        label_values=[1],
    )


@register_task
def oasis1_dementia(subset: Subset = "dev") -> Task:
    dataset = create_dataset("oasis1", modality="T1w")
    samples = dataset.samples
    # CDR is only given for first sessions (MR1), mostly 60+.
    samples["dementia"] = (samples["cdr"] > 0).where(samples["cdr"].notna())
    keep = (samples["session_id"] == "MR1") & samples["cdr"].notna()
    ids, folds = fold_ids(samples, keep, subset, "dementia")
    return Task(
        name="oasis1_dementia",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="dementia",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def oasis2_dementia(subset: Subset = "dev") -> Task:
    dataset = create_dataset("oasis2", modality="T1w")
    samples = dataset.samples
    # CDR > 0 at the first visit only, one scan per participant (converters count as CDR 0).
    samples["dementia"] = (samples["cdr"] > 0).where(samples["cdr"].notna())
    keep = (samples["visit"] == 1) & samples["cdr"].notna()
    ids, folds = fold_ids(samples, keep, subset, "dementia")
    return Task(
        name="oasis2_dementia",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="dementia",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def oasis2_conversion(subset: Subset = "dev") -> Task:
    dataset = create_dataset("oasis2", modality="T1w")
    samples = dataset.samples
    # Nondemented at the first visit: later converted to dementia vs stayed nondemented.
    samples["converted"] = samples["group"] == "converted"
    keep = (samples["visit"] == 1) & samples["group"].isin(["nondemented", "converted"])
    ids, folds = fold_ids(samples, keep, subset, "converted")
    return Task(
        name="oasis2_conversion",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="converted",
        ids=ids,
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
def fomo26_task5_pmg(subset: Subset = "dev") -> Task:
    dataset = create_dataset("fomo26_task5", modality="T1w")
    keep = dataset.samples["pmg"].notna()
    ids, folds = fold_ids(dataset.samples, keep, subset, "pmg")
    return Task(
        name="fomo26_task5_pmg",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="pmg",
        ids=ids,
        folds=folds,
        classes=["control", "polymicrogyria"],
        transform=strip_and_crop_ap,
    )


def fomo26_task4_region(name: str, label_values: list[int], subset: Subset) -> Task:
    dataset = create_dataset(
        "fomo26_task4", modality="T2w", mask_target="mask_trigeminal", label_values=label_values
    )
    keep = dataset.samples["mask_trigeminal_path"].notna()
    ids, folds = fold_ids(dataset.samples, keep, subset, None)
    return Task(
        name=name,
        type="segmentation",
        subset=subset,
        dataset=dataset,
        target="mask_trigeminal",
        ids=ids,
        folds=folds,
        label_values=label_values,
    )


@register_task
def fomo26_task4_nerve(subset: Subset = "dev") -> Task:
    return fomo26_task4_region("fomo26_task4_nerve", [1], subset)


@register_task
def fomo26_task4_vessel(subset: Subset = "dev") -> Task:
    return fomo26_task4_region("fomo26_task4_vessel", [2], subset)


def adni_primary(samples: pd.DataFrame) -> pd.Series:
    """The main ADNI cohort: y0 scans of participants enrolled in ADNIGO onward, all at 3 T.
    ADNI1 (1.5 T, different protocol) is left for dedicated tasks."""
    return (
        (samples["session_id"] == "y0")
        & (samples["protocol"] != "ADNI1")
        & (samples["field_strength"] == 3.0)
    )


@register_task
def adni_ad(subset: Subset = "dev") -> Task:
    dataset = create_dataset("adni", modality="T1w")
    samples = dataset.samples
    # Dementia vs stable CN: CN with no later MCI or dementia diagnosis over at least 3 years.
    dementia = samples["dx"] == "Dementia"
    stable_cn = (
        (samples["dx"] == "CN")
        & samples["years_to_impairment"].isna()
        & (samples["dx_followup_years"] >= 3)
    )
    samples["dementia"] = dementia.where(dementia | stable_cn)
    keep = adni_primary(samples) & samples["dementia"].notna()
    ids, folds = fold_ids(samples, keep, subset, "dementia")
    return Task(
        name="adni_ad",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="dementia",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def adni_pmci(subset: Subset = "dev") -> Task:
    dataset = create_dataset("adni", modality="T1w")
    samples = dataset.samples
    # MCI at y0: dementia within 3 years (progressive) vs none over at least 3 years (stable).
    progressive = samples["years_to_dementia"] <= 3
    stable = ~progressive & (samples["dx_followup_years"] >= 3)
    samples["progressive"] = progressive.where(progressive | stable)
    keep = adni_primary(samples) & (samples["dx"] == "MCI") & samples["progressive"].notna()
    ids, folds = fold_ids(samples, keep, subset, "progressive")
    return Task(
        name="adni_pmci",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="progressive",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def adni_amyloid(subset: Subset = "dev") -> Task:
    dataset = create_dataset("adni", modality="T1w")
    samples = dataset.samples
    # Amyloid PET Centiloid (UC Berkeley), all diagnoses.
    keep = adni_primary(samples) & samples["centiloid"].notna()
    ids, folds = fold_ids(samples, keep, subset, "centiloid")
    return Task(
        name="adni_amyloid",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="centiloid",
        ids=ids,
        folds=folds,
    )


# Flortaucipir temporal meta-ROI SUVR (inferior cerebellar reference) from which a scan is tau
# positive: the Youden cutoff of Guo et al. 2020 (Alzheimers Res Ther 12:97) on ADNI, UC Berkeley
# processing. Their conservative alternative (mean + 2 SD of amyloid-negative CU) is 1.34.
TAU_POSITIVE_SUVR = 1.25


@register_task
def adni_tau(subset: Subset = "dev") -> Task:
    dataset = create_dataset("adni", modality="T1w")
    samples = dataset.samples
    # Tau PET came with ADNI3, so many participants have it only at y2 or y4: take each one's
    # first session with flortaucipir PET (other tracers have other SUVR scales).
    suvr = samples["tau_meta_temporal_suvr"]
    samples["tau_positive"] = (suvr >= TAU_POSITIVE_SUVR).where(suvr.notna())
    eligible = (
        (samples["protocol"] != "ADNI1")
        & (samples["field_strength"] == 3.0)
        & (samples["tau_tracer"] == "FTP")
        & suvr.notna()
    )
    first = samples[eligible].sort_values("session_id").groupby("participant_id").head(1)
    keep = samples.index.isin(first.index)
    ids, folds = fold_ids(samples, keep, subset, "tau_positive")
    return Task(
        name="adni_tau",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="tau_positive",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def adni_wmh(subset: Subset = "dev") -> Task:
    dataset = create_dataset("adni", modality="FLAIR")
    samples = dataset.samples
    # Log white matter hyperintensity volume (UC Davis, from FLAIR); the raw volume is very skewed.
    samples["log_wmh_volume"] = np.log(samples["wmh_volume"])
    keep = adni_primary(samples) & samples["wmh_volume"].notna()
    ids, folds = fold_ids(samples, keep, subset, "log_wmh_volume")
    return Task(
        name="adni_wmh",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="log_wmh_volume",
        ids=ids,
        folds=folds,
    )


@register_task
def adni_ptau217(subset: Subset = "dev") -> Task:
    dataset = create_dataset("adni", modality="T1w")
    samples = dataset.samples
    # Log plasma pTau217 (Fujirebio), all diagnoses; the raw values are very skewed.
    samples["log_plasma_ptau217"] = np.log(samples["plasma_ptau217"])
    keep = adni_primary(samples) & samples["plasma_ptau217"].notna()
    ids, folds = fold_ids(samples, keep, subset, "log_plasma_ptau217")
    return Task(
        name="adni_ptau217",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="log_plasma_ptau217",
        ids=ids,
        folds=folds,
    )


@register_task
def adni_hippocampus(subset: Subset = "dev") -> Task:
    dataset = create_dataset("adni", modality="T1w")
    samples = dataset.samples
    # FreeSurfer 7 left + right hippocampal volume per 1000 of intracranial volume.
    hippocampus = samples["fs_vol_left_hippocampus"] + samples["fs_vol_right_hippocampus"]
    samples["hippocampus_icv"] = 1000 * hippocampus / samples["fs_ctxvol_icv"]
    keep = adni_primary(samples) & samples["hippocampus_icv"].notna()
    ids, folds = fold_ids(samples, keep, subset, "hippocampus_icv")
    return Task(
        name="adni_hippocampus",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="hippocampus_icv",
        ids=ids,
        folds=folds,
    )


@register_task
def a4learn_amyloid(subset: Subset = "dev") -> Task:
    dataset = create_dataset("a4learn", modality="T1w")
    samples = dataset.samples
    # Florbetapir Centiloid at screening, from the baseline scan of A4 and LEARN.
    keep = (
        (samples["session_id"] == "bl")
        & samples["group"].isin(["A4", "LEARN"])
        & samples["screening_centiloid"].notna()
    )
    ids, folds = fold_ids(samples, keep, subset, "screening_centiloid")
    return Task(
        name="a4learn_amyloid",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="screening_centiloid",
        ids=ids,
        folds=folds,
    )


@register_task
def a4learn_cohort(subset: Subset = "dev") -> Task:
    dataset = create_dataset("a4learn", modality="T1w")
    samples = dataset.samples
    # A4 enrolled cognitively normal people with elevated amyloid, LEARN those without.
    samples["a4"] = samples["group"] == "A4"
    keep = (samples["session_id"] == "bl") & samples["group"].isin(["A4", "LEARN"])
    ids, folds = fold_ids(samples, keep, subset, "a4")
    return Task(
        name="a4learn_cohort",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="a4",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


@register_task
def a4learn_ptau217(subset: Subset = "dev") -> Task:
    dataset = create_dataset("a4learn", modality="T1w")
    samples = dataset.samples
    # Log plasma pTau217 (Lilly) at baseline, A4 and LEARN; the raw values are very skewed.
    samples["log_plasma_ptau217"] = np.log(samples["plasma_ptau217"])
    keep = (
        (samples["session_id"] == "bl")
        & samples["group"].isin(["A4", "LEARN"])
        & samples["plasma_ptau217"].notna()
    )
    ids, folds = fold_ids(samples, keep, subset, "log_plasma_ptau217")
    return Task(
        name="a4learn_ptau217",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="log_plasma_ptau217",
        ids=ids,
        folds=folds,
    )


@register_task
def a4learn_hippocampus(subset: Subset = "dev") -> Task:
    dataset = create_dataset("a4learn", modality="T1w")
    samples = dataset.samples
    # Left + right hippocampal volume per 1000 of intracranial volume at baseline (A4 only;
    # LEARN has no volumes).
    hippocampus = samples["vol_left_hippocampus"] + samples["vol_right_hippocampus"]
    samples["hippocampus_icv"] = 1000 * hippocampus / samples["vol_intra_cranial_volume"]
    keep = (
        (samples["session_id"] == "bl")
        & (samples["group"] == "A4")
        & samples["hippocampus_icv"].notna()
    )
    ids, folds = fold_ids(samples, keep, subset, "hippocampus_icv")
    return Task(
        name="a4learn_hippocampus",
        type="regression",
        subset=subset,
        dataset=dataset,
        target="hippocampus_icv",
        ids=ids,
        folds=folds,
    )


@register_task
def a4learn_pacc_decline(subset: Subset = "dev") -> Task:
    dataset = create_dataset("a4learn", modality="T1w")
    samples = dataset.samples
    # From the baseline scan of A4 (amyloid-elevated) participants: a PACC change by week 240
    # below the 5th percentile of LEARN (amyloid not elevated), i.e. decline beyond normal aging.
    baseline = samples["session_id"] == "bl"
    change = samples["pacc_change_wk240"]
    learn = baseline & (samples["group"] == "LEARN")
    cutoff = change[learn].quantile(0.05)
    samples["pacc_decline"] = (change < cutoff).where(change.notna())
    # Older participants decline more often (age alone: AUROC 0.65), so keep the subset's decliner
    # rate in every age fifth, dropping the excess class at random.
    candidates = samples[baseline & (samples["group"] == "A4") & change.notna()]
    age_fifth = pd.qcut(candidates["age"], 5, labels=False)
    keep = pd.Series(False, index=samples.index)
    for _, subset_rows in candidates.groupby("split"):
        rate = subset_rows["pacc_decline"].astype(bool).mean()
        for _, stratum in subset_rows.groupby(age_fifth):
            declined = stratum["pacc_decline"].astype(bool)
            n_declined = min(declined.sum(), round((~declined).sum() * rate / (1 - rate)))
            n_stable = min((~declined).sum(), round(declined.sum() * (1 - rate) / rate))
            keep[stratum[declined].sample(n_declined, random_state=SEED).index] = True
            keep[stratum[~declined].sample(n_stable, random_state=SEED).index] = True
    ids, folds = fold_ids(samples, keep, subset, "pacc_decline")
    return Task(
        name="a4learn_pacc_decline",
        type="classification",
        subset=subset,
        dataset=dataset,
        target="pacc_decline",
        ids=ids,
        folds=folds,
        classes=[False, True],
    )


# Groups of tasks, usable in place of task names; after the tasks, so that `all` has them.
TASK_GROUPS: dict[str, list[str]] = {
    # The benchmark score is their mean: the primary target of each cohort, plus a few more tasks
    # with distinct, reliable signal.
    "core": [
        "abide_diagnosis",
        "adhd_diagnosis",
        "cnp_diagnosis",
        "ucsf_idh",
        "upenn_survival_1y",
        "oasis1_dementia",
        "adni_ad",
        "a4learn_amyloid",
        "ixi_age",
        "openbhb_age",
        "fomo26_task5_pmg",
        "brats_enhancing_tumor",
        "soop_lesion",
        "atlas_lesion",
        "fomo26_task4_vessel",
        "adni_pmci",
        "soop_mrs_poor",
        "brats_whole_tumor",
    ],
    # The core tasks on hosted datasets: no data access application needed.
    "open": [
        "abide_diagnosis",
        "adhd_diagnosis",
        "cnp_diagnosis",
        "ucsf_idh",
        "upenn_survival_1y",
        "ixi_age",
        "openbhb_age",
        "brats_enhancing_tumor",
        "soop_lesion",
        "soop_mrs_poor",
        "brats_whole_tumor",
    ],
    "all": list_tasks(),
}


def expand_tasks(names: list[str]) -> list[str]:
    """Task names with the groups expanded, in order, without duplicates."""
    expanded = []
    for name in names:
        expanded += TASK_GROUPS.get(name, [name])
    return list(dict.fromkeys(expanded))
