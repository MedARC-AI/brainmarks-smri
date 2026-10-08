"""Benchmark datasets.

One map-style class over the common on-disk layout (`<root>/<name>/{source,derivatives,tables}`),
and a registry of one small function per dataset. A dataset holds the sessions that have one image
modality.

`samples` has one row per sample: the samples.tsv and splits.tsv columns, plus file paths relative
to the dataset directory: `image_path`, `brain_mask_path`, `mni_affine_path` and, for each
segmentation mask of the session, `mask_<desc>_path`.

`__getitem__` returns `(image_input, targets)`: the `ImageInput` (after `transform`), and the row's
values plus the mask images as `mask_<desc>` (after `target_transform`). Images are loaded lazily
by nibabel; nothing is collated.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import platformdirs
import torch.utils.data

from brainmarks_smri.models.base import ImageInput

DATA_ROOT = Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parents[2] / "datasets"))
CACHE_DIR = Path(os.getenv("BRAINMARKS_SMRI_CACHE", platformdirs.user_cache_dir("brainmarks_smri")))
# Derivative files next to each image's mirrored path, see scripts/preprocess.py.
MASK_SUFFIX = "_mask.nii.gz"
AFFINE_SUFFIX = "_mni.txt"

logger = logging.getLogger(__name__)


class BrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str,
        modality: str,
        root: str | Path | None = None,
        derivatives_modality: str | None = None,
        desc_pattern: str | None = None,
        transform: Callable[[ImageInput], Any] | None = None,
        target_transform: Callable[[dict[str, Any]], Any] | None = None,
    ):
        """`derivatives_modality`: use the session's image of this modality for the derivatives.
        `desc_pattern`: regex for the images' desc; by default, images without a desc.
        """
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
        self.target_transform = target_transform
        self.samples = self.load_samples(modality, derivatives_modality, desc_pattern)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.samples.iloc[index]
        image = nib.load(self.root / row["image_path"])
        # Some DWI/ADC are stored as 4D with a single volume.
        if image.ndim == 4 and image.shape[3] == 1:
            image = nib.funcs.squeeze_image(image)
        image_input: ImageInput = {
            "image": image,
            "brain_mask": nib.load(self.root / row["brain_mask_path"]),
            "mni_affine": np.loadtxt(self.root / row["mni_affine_path"]),
        }
        targets = row.to_dict()
        for column in self.samples.columns:
            if column.startswith("mask_") and pd.notna(row[column]):
                targets[column.removesuffix("_path")] = nib.load(self.root / row[column])
        if self.transform is not None:
            image_input = self.transform(image_input)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
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
        samples = samples.merge(splits, on="participant_id")

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

        # source/<dir>/<name>.nii.gz -> derivatives/<dir>/<name>
        stems = "derivatives/" + samples.pop("derivatives_of").str.removeprefix("source/")
        stems = stems.str.removesuffix(".gz").str.removesuffix(".nii")
        samples["brain_mask_path"] = stems + MASK_SUFFIX
        samples["mni_affine_path"] = stems + AFFINE_SUFFIX
        exists = samples["mni_affine_path"].map(lambda path: (self.root / path).exists())
        if not exists.all():
            logger.warning(f"{self.name}: dropping {(~exists).sum()} samples without derivatives")
            samples = samples[exists]

        return samples.sort_values(ids).reset_index(drop=True)


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
def abide1(modality: str = "T1w", **kwargs) -> BrainDataset:
    """ABIDE I: autism diagnosis, T1w."""
    return BrainDataset("abide1", modality, **kwargs)


@register_dataset
def adhd200(modality: str = "T1w", **kwargs) -> BrainDataset:
    """ADHD-200: ADHD diagnosis, T1w. One T1w per participant; Peking_3's carry an `acq-` label."""
    return BrainDataset("adhd200", modality, desc_pattern=r"(acq-\d_)?run-1", **kwargs)


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
def ixi(modality: str = "T1w", **kwargs) -> BrainDataset:
    """IXI: age, T1w, T2w, PD."""
    return BrainDataset("ixi", modality, **kwargs)


@register_dataset
def openbhb(modality: str = "T1w", **kwargs) -> BrainDataset:
    """OpenBHB: brain age, T1w (the original rawdata images)."""
    return BrainDataset("openbhb", modality, **kwargs)


@register_dataset
def soop(modality: str = "DWI", **kwargs) -> BrainDataset:
    """SOOP: stroke lesion segmentation (masks on the DWI/ADC grid), discharge mRS, NIHSS. T1w,
    FLAIR, DWI, ADC are not co-registered (head motion), so each image has its own derivatives."""
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
