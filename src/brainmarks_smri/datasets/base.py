"""Dataset contract.

One map-style dataset class per benchmark dataset, over the common on-disk layout
(`<root>/<name>/{source,derivatives,tables}`). A dataset holds the sessions that have one image
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

DATA_ROOT = Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parents[3] / "datasets"))
CACHE_DIR = Path(os.getenv("BRAINMARKS_SMRI_CACHE", platformdirs.user_cache_dir("brainmarks_smri")))
# Derivative files next to each image's mirrored path, see scripts/preprocess.py.
MASK_SUFFIX = "_mask.nii.gz"
AFFINE_SUFFIX = "_mni.txt"

logger = logging.getLogger(__name__)


class BrainDataset(torch.utils.data.Dataset):
    name: str
    modalities: tuple[str, ...]

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable[[ImageInput], Any] | None = None,
        target_transform: Callable[[dict[str, Any]], Any] | None = None,
    ):
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
        self.samples = pd.DataFrame()

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
        self,
        modality: str,
        max_per_split: int | None = None,
        derivatives_modality: str | None = None,
    ) -> None:
        """Set `self.samples` from the dataset tables.

        `derivatives_modality`: use the session's image of this modality for the derivatives.
        `max_per_split`: keep the lowest-rank samples per split (nested, balanced mini-splits).
        """
        assert modality in self.modalities, f"{self.name}: {modality!r} not in {self.modalities}"
        ids = ["participant_id", "session_id"]
        read_kwargs = dict(sep="\t", dtype={"participant_id": str, "session_id": str})
        images = pd.read_csv(self.root / "tables" / "images.tsv", **read_kwargs)
        samples = pd.read_csv(self.root / "tables" / "samples.tsv", **read_kwargs)
        splits = pd.read_csv(self.root / "tables" / "splits.tsv", **read_kwargs)
        samples = samples.merge(splits, on="participant_id")

        derivatives_modality = derivatives_modality or modality
        for column, image_modality in [
            ("image_path", modality),
            ("derivatives_of", derivatives_modality),
        ]:
            selected = images[(images["modality"] == image_modality) & images["desc"].isna()]
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

        if max_per_split is not None:
            samples = samples.sort_values("rank").groupby("split").head(max_per_split)
        self.samples = samples.sort_values(ids).reset_index(drop=True)


DATASETS: dict[str, type[BrainDataset]] = {}


def register_dataset(dataset_class: type[BrainDataset]) -> type[BrainDataset]:
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def create_dataset(name: str, **kwargs) -> BrainDataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; available: {list_datasets()}")
    return DATASETS[name](**kwargs)


def list_datasets() -> list[str]:
    return sorted(DATASETS)
