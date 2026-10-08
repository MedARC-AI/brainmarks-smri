"""UCSF-PDGM: IDH and other glioma targets, tumor segmentation. Skull-stripped, co-registered
sequences (not the `bias` copies), so every sequence uses the session's T1w derivatives."""

from pathlib import Path

from brainmarks_smri.datasets.base import BrainDataset, register_dataset


@register_dataset
class UCSFPDGM(BrainDataset):
    name = "ucsf_pdgm"
    modalities = ("T1w", "T1c", "T2w", "FLAIR", "DWI", "ADC")

    def __init__(
        self,
        root: str | Path | None = None,
        modality: str = "FLAIR",
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform, target_transform)
        self.load_samples(modality, derivatives_modality="T1w")
