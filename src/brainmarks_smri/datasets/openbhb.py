"""OpenBHB: brain age, T1w (the original rawdata images)."""

from pathlib import Path

from brainmarks_smri.datasets.base import BrainDataset, register_dataset


@register_dataset
class OpenBHB(BrainDataset):
    name = "openbhb"
    modalities = ("T1w",)

    def __init__(
        self,
        root: str | Path | None = None,
        modality: str = "T1w",
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform, target_transform)
        self.load_samples(modality)
