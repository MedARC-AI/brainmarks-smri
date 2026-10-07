"""BraTS 2021: tumor segmentation (training cases only) and MGMT. Skull-stripped, co-registered
sequences, so every sequence uses the session's T1w derivatives."""

from pathlib import Path

from brainmarks_smri.datasets.base import BrainDataset, register_dataset


@register_dataset
class BraTS2021(BrainDataset):
    name = "brats2021"
    modalities = ("T1w", "T1c", "T2w", "FLAIR")

    def __init__(
        self,
        root: str | Path | None = None,
        modality: str = "FLAIR",
        max_per_split: int | None = None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform, target_transform)
        self.load_samples(modality, max_per_split, derivatives_modality="T1w")
