"""SOOP: stroke lesion segmentation (masks on the DWI/ADC grid), discharge mRS, NIHSS. Sequences
are not co-registered (head motion), so each image has its own derivatives."""

from pathlib import Path

from brainmarks_smri.datasets.base import BrainDataset, register_dataset


@register_dataset
class SOOP(BrainDataset):
    name = "soop"
    modalities = ("T1w", "FLAIR", "DWI", "ADC")

    def __init__(
        self,
        root: str | Path | None = None,
        modality: str = "DWI",
        max_per_split: int | None = None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform, target_transform)
        self.load_samples(modality, max_per_split)
