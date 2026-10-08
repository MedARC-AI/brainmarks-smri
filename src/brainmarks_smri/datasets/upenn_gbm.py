"""UPENN-GBM: survival, IDH1, MGMT, tumor segmentation. Skull-stripped, co-registered sequences
(not the `unstripped` or `old` copies), so every sequence uses the session's T1w derivatives."""

from pathlib import Path

from brainmarks_smri.datasets.base import BrainDataset, register_dataset


@register_dataset
class UPENNGBM(BrainDataset):
    name = "upenn_gbm"
    modalities = ("T1w", "T1c", "T2w", "FLAIR")

    def __init__(
        self,
        root: str | Path | None = None,
        modality: str = "FLAIR",
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform, target_transform)
        self.load_samples(modality, derivatives_modality="T1w")
