"""Benchmark datasets."""

import gzip
import io
import logging
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import platformdirs
import torch.utils.data

from brainmarks_smri.models.base import ImageInput

DATA_ROOT = os.getenv("BRAINMARKS_SMRI_DATA", str(Path(__file__).resolve().parents[2] / "datasets"))
CACHE_DIR = Path(os.getenv("BRAINMARKS_SMRI_CACHE", platformdirs.user_cache_dir("brainmarks_smri")))
# Derivative files next to each image's mirrored path, see scripts/preprocess.py.
MASK_SUFFIX = "_mask.nii.gz"
AFFINE_SUFFIX = "_mni.txt"

logger = logging.getLogger(__name__)


class BrainDataset(torch.utils.data.Dataset):
    """Torch map-style dataset for the Brainmarks dataset format.

    Samples are of the form (image_input, targets).

    Args:
        derivatives_modality: use the session's image of this modality for the derivatives.
        desc_pattern: regex for the input images' desc; by default, images without a desc.
        mask_target: a mask key, e.g. "mask_tumor", added to the targets as a bool tensor with the
            brain mask and the image affine.
        label_values: the mask labels that count as the target region; by default, any nonzero.
    """

    def __init__(
        self,
        name: str,
        modality: str,
        root: str | Path | None = None,
        derivatives_modality: str | None = None,
        desc_pattern: str | None = None,
        transform: Callable[[ImageInput], Any] | None = None,
        mask_target: str | None = None,
        label_values: list[int] | None = None,
    ):
        self.name = name
        if root is None:
            root = DATA_ROOT
        if str(root).startswith("hf://"):
            from huggingface_hub import snapshot_download

            # hf://datasets/<org>/<repo>: download this dataset's folder once, into the cache
            repo_path = str(root).removeprefix("hf://")
            assert repo_path.startswith("datasets/"), f"expected hf://datasets/<org>/<repo>: {root}"
            repo_id = repo_path.removeprefix("datasets/")
            root = CACHE_DIR / "datasets"
            snapshot_download(
                repo_id, repo_type="dataset", allow_patterns=f"{self.name}/**", local_dir=root
            )
        self.root = Path(root) / self.name
        self.transform = transform
        self.mask_target = mask_target
        self.label_values = label_values
        self.samples = self.load_samples(modality, derivatives_modality, desc_pattern)
        shards = pd.read_csv(self.root / "tables" / "shards.tsv", sep="\t")
        self.shard_of_path = dict(zip(shards["path"], shards["shard"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.samples.iloc[index]
        shard_path = self.root / self.shard_of_path[row["image_path"]]
        with zipfile.ZipFile(shard_path) as shard:
            image = _load_nifti(shard.read(row["image_path"]))
            # Some DWI/ADC are stored as 4D with a single volume.
            if image.ndim == 4 and image.shape[3] == 1:
                image = nib.funcs.squeeze_image(image)
            brain_mask = _load_nifti(shard.read(row["brain_mask_path"]))
            image_input: ImageInput = {
                "image": image,
                "brain_mask": brain_mask,
                "mni_affine": np.loadtxt(io.BytesIO(shard.read(row["mni_affine_path"]))),
            }
            targets = row.to_dict()
            if self.mask_target is not None:
                label_image = _load_nifti(shard.read(row[f"{self.mask_target}_path"]))
                assert label_image.shape == image.shape, f"{self.name}: mask not on image grid"
                assert brain_mask.shape == image.shape, f"{self.name}: brain mask not on image grid"
                labels = np.asanyarray(label_image.dataobj)
                if self.label_values is None:
                    label_mask = labels > 0
                else:
                    label_mask = np.isin(labels, self.label_values)
                targets[self.mask_target] = torch.from_numpy(label_mask)
                targets["brain_mask"] = torch.from_numpy(np.asanyarray(brain_mask.dataobj) > 0)
                targets["affine"] = image.affine

        if self.transform is not None:
            image_input = self.transform(image_input)
        return image_input, targets

    def load_samples(
        self, modality: str, derivatives_modality: str | None, desc_pattern: str | None
    ) -> pd.DataFrame:
        """The samples with an image of `modality`, from the dataset tables."""
        ids = ["participant_id", "session_id"]
        read_kwargs = dict(sep="\t", dtype={"participant_id": str, "session_id": str})
        images = pd.read_csv(self.root / "tables" / "images.tsv", **read_kwargs)
        samples = pd.read_csv(self.root / "tables" / "samples.tsv", **read_kwargs)
        splits = pd.read_csv(self.root / "tables" / "splits.tsv", **read_kwargs)
        derivatives = pd.read_csv(self.root / "tables" / "derivatives.tsv", sep="\t")
        # copy consolidates the frame: read_csv gives one block per column (pandas 3)
        samples = samples.merge(splits, on="participant_id").copy()

        derivatives_modality = derivatives_modality or modality
        if desc_pattern is None:
            desc_matches = images["desc"].isna()
        else:
            desc_matches = images["desc"].fillna("").str.fullmatch(desc_pattern)
        for column, image_modality in [
            ("image_path", modality),
            ("derivatives_of", derivatives_modality),
        ]:
            selected = images[(images["modality"] == image_modality) & desc_matches]
            assert len(selected), f"{self.name}: no {image_modality} images"
            assert not selected.duplicated(ids).any(), f"{self.name}: several {image_modality}"
            selected = selected[ids + ["path"]].rename(columns={"path": column})
            samples = samples.merge(selected, on=ids)
        for desc, rows in images[images["modality"] == "mask"].groupby("desc"):
            rows = rows[ids + ["path"]].rename(columns={"path": f"mask_{desc}_path"})
            samples = samples.merge(rows, on=ids, how="left")

        has_derivatives = samples["derivatives_of"].isin(derivatives["path"])
        if not has_derivatives.all():
            dropped = (~has_derivatives).sum()
            logger.warning(f"{self.name}: dropping {dropped} samples without derivatives")
            samples = samples[has_derivatives]

        # source/<dir>/<name>.nii.gz -> derivatives/<dir>/<name>
        stems = "derivatives/" + samples.pop("derivatives_of").str.removeprefix("source/")
        stems = stems.str.removesuffix(".gz").str.removesuffix(".nii")
        samples["brain_mask_path"] = stems + MASK_SUFFIX
        samples["mni_affine_path"] = stems + AFFINE_SUFFIX

        return samples.sort_values(ids).reset_index(drop=True)


def _load_nifti(buf: bytes) -> nib.Nifti1Image:
    return nib.Nifti1Image.from_bytes(gzip.decompress(buf))


DATASETS: dict[str, Callable[..., BrainDataset]] = {}


def register_dataset(dataset_fn: Callable[..., BrainDataset]) -> Callable[..., BrainDataset]:
    DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn


def create_dataset(name: str, **kwargs) -> BrainDataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; available: {list_datasets()}")
    return DATASETS[name](**kwargs)


def list_datasets() -> list[str]:
    return sorted(DATASETS)


@register_dataset
def a4learn(modality: str = "T1w", **kwargs) -> BrainDataset:
    """A4/LEARN: preclinical AD (amyloid, cognitive decline, treatment arm), T1w at bl, wk84 and
    wk240. Built on the user side."""
    return BrainDataset("a4learn", modality, **kwargs)


@register_dataset
def abide1(modality: str = "T1w", **kwargs) -> BrainDataset:
    """ABIDE I: autism diagnosis, T1w."""
    return BrainDataset("abide1", modality, **kwargs)


@register_dataset
def adhd200(modality: str = "T1w", **kwargs) -> BrainDataset:
    """ADHD-200: ADHD diagnosis, T1w."""
    return BrainDataset("adhd200", modality, **kwargs)


@register_dataset
def adni(modality: str = "T1w", **kwargs) -> BrainDataset:
    """ADNI: AD diagnosis, progression and biomarkers, T1w and FLAIR at y0, y2 and y4. Built on
    the user side."""
    return BrainDataset("adni", modality, **kwargs)


@register_dataset
def atlas(modality: str = "T1w", **kwargs) -> BrainDataset:
    """ATLAS R2.1: chronic stroke lesion segmentation, skull-stripped T1w. Built on the user side."""
    return BrainDataset("atlas", modality, **kwargs)


@register_dataset
def brats2021(modality: str = "FLAIR", **kwargs) -> BrainDataset:
    """BraTS 2021: tumor segmentation (training cases only), MGMT. Skull-stripped, co-registered
    T1w, T1c, T2w, FLAIR, so every sequence uses the session's T1w derivatives."""
    return BrainDataset("brats2021", modality, derivatives_modality="T1w", **kwargs)


@register_dataset
def cnp(modality: str = "T1w", **kwargs) -> BrainDataset:
    """CNP: psychiatric diagnosis, T1w."""
    return BrainDataset("cnp", modality, **kwargs)


@register_dataset
def fomo26_task4(modality: str = "T2w", **kwargs) -> BrainDataset:
    """FOMO26 Task 4: trigeminal nerve and vessel segmentation, T2w. Built on the user side."""
    return BrainDataset("fomo26_task4", modality, **kwargs)


@register_dataset
def fomo26_task5(modality: str = "T1w", **kwargs) -> BrainDataset:
    """FOMO26 Task 5: polymicrogyria vs control, T1w. Built on the user side."""
    return BrainDataset("fomo26_task5", modality, **kwargs)


@register_dataset
def ixi(modality: str = "T1w", **kwargs) -> BrainDataset:
    """IXI: age, T1w, T2w, PD."""
    return BrainDataset("ixi", modality, **kwargs)


@register_dataset
def oasis1(modality: str = "T1w", **kwargs) -> BrainDataset:
    """OASIS-1: dementia (CDR), T1w (the averaged native-space image). Built on the user side."""
    return BrainDataset("oasis1", modality, **kwargs)


@register_dataset
def oasis2(modality: str = "T1w", **kwargs) -> BrainDataset:
    """OASIS-2: dementia and conversion, longitudinal T1w (mpr-1). Built on the user side."""
    return BrainDataset("oasis2", modality, **kwargs)


@register_dataset
def openbhb(modality: str = "T1w", **kwargs) -> BrainDataset:
    """OpenBHB: brain age, T1w (the original rawdata images)."""
    return BrainDataset("openbhb", modality, **kwargs)


@register_dataset
def soop(modality: str = "DWI", **kwargs) -> BrainDataset:
    """SOOP: stroke lesion segmentation (masks on the DWI/ADC grid), discharge mRS, NIHSS. FLAIR,
    DWI, ADC are not co-registered (head motion), so each image has its own derivatives."""
    return BrainDataset("soop", modality, **kwargs)


@register_dataset
def ucsf_pdgm(modality: str = "FLAIR", **kwargs) -> BrainDataset:
    """UCSF-PDGM: IDH and other glioma targets, tumor segmentation. Skull-stripped, co-registered
    T1w, T1c, T2w, FLAIR, DWI, ADC (not the `bias` copies), so every sequence uses the session's
    T1w derivatives."""
    return BrainDataset("ucsf_pdgm", modality, derivatives_modality="T1w", **kwargs)


@register_dataset
def upenn_gbm(modality: str = "FLAIR", **kwargs) -> BrainDataset:
    """UPENN-GBM: survival, IDH1, MGMT, tumor segmentation. Skull-stripped, co-registered T1w,
    T1c, T2w, FLAIR (not the `unstripped` or `old` copies), so every sequence uses the session's
    T1w derivatives."""
    return BrainDataset("upenn_gbm", modality, derivatives_modality="T1w", **kwargs)
