"""ABIDE I: autism diagnosis, T1w."""

from pathlib import Path

from brainmarks_smri.datasets.base import BrainDataset, register_dataset


@register_dataset
class ABIDE1(BrainDataset):
    name = "abide1"
    modalities = ("T1w",)

    def __init__(
        self,
        root: str | Path | None = None,
        modality: str = "T1w",
        max_per_split: int | None = None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform, target_transform)
        self.load_samples(modality, max_per_split)
