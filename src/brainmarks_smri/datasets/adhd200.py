"""ADHD-200: ADHD diagnosis, T1w. One T1w per participant; Peking_3's carry an `acq-` label."""

from pathlib import Path

from brainmarks_smri.datasets.base import BrainDataset, register_dataset


@register_dataset
class ADHD200(BrainDataset):
    name = "adhd200"
    modalities = ("T1w",)

    def __init__(
        self,
        root: str | Path | None = None,
        modality: str = "T1w",
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform, target_transform)
        self.load_samples(modality, desc_pattern=r"(acq-\d_)?run-1")
