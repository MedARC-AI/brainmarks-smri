"""Neuro-JEPA (NYUMedML/Neuro-JEPA): a JEPA-pretrained 3D ViT-B with a sparse MoE.

Pretraining used 1 mm scans affine-registered to MNI152 and skull-stripped. Given `mni_affine` and
`brain_mask`, we resample onto a 1 mm MNI grid and mask instead of running FLIRT + SynthStrip.
N4 bias correction is skipped. The 96x108x96 input gives an 8x9x8 grid of 12-voxel patches,
mean-pooled for the global embedding. The input shape must not change: the RoPE position split
uses the wrong axis sizes in a way the model was trained with.

The spline resamples dominate the cost (7 s on CPU, 0.2 s on GPU), so `transform` only
canonicalizes the arrays and the MONAI pipeline runs on the model's device.
"""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai import transforms
from monai.data import MetaTensor
from neurojepa.utils.init_utils import load_backbone_from_hf
from torch import Tensor

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

REPO_ID = "NYUMedML/Neuro-JEPA"
REVISION = "f55b154091aa7ac5ca23632918e216cd92d9f2cb"
IMG_SIZE = (96, 108, 96)
PATCH_SIZE = (12, 12, 12)

# FSL's MNI152 1 mm grid (182x218x182), in RAS voxel order
MNI_SHAPE = (182, 218, 182)
MNI_AFFINE = np.array(
    [
        [1.0, 0.0, 0.0, -90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class NeuroJEPA(nn.Module):
    name = "neurojepa"
    embed_dim = 768
    patch_size = PATCH_SIZE

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.out_layers = None  # the last block only, whatever the checkpoint config says
        self.preprocess = upstream_transform()
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def transform(self, sample: ImageInput) -> dict[str, Any]:
        image: nib.Nifti1Image = sample["image"]
        data = np.nan_to_num(image.get_fdata(dtype=np.float32))
        mask = sample.get("brain_mask")
        if mask is not None:
            mask = np.asarray(mask.dataobj) > 0
        mni_affine = sample.get("mni_affine")
        return {
            "image": torch.from_numpy(data)[None],
            "affine": torch.from_numpy(image.affine),
            "mask": None if mask is None else torch.from_numpy(mask)[None],
            "mni_affine": None if mni_affine is None else torch.from_numpy(mni_affine),
        }

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        volumes = []
        affines = []
        for sample in samples:
            volume, affine = self.prepare_volume(sample)
            volumes.append(volume)
            affines.append(affine)
        batch = torch.stack(volumes)  # (B, 1, 96, 108, 96)

        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            tokens, _moe_scores = self.backbone(batch)
        tokens = tokens.float()  # (B, 576, 768)

        grid = tuple(size // patch for size, patch in zip(IMG_SIZE, PATCH_SIZE))
        outputs = []
        for sample_tokens, affine in zip(tokens, affines):
            output = EmbeddingOutput(
                global_embedding=sample_tokens.mean(0),
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                # PatchEmbed3D flattens the conv output in C order, so tokens are (x y z) ordered
                output["dense_embedding"] = sample_tokens.reshape(*grid, self.embed_dim)
                output["dense_affine"] = affine
            outputs.append(output)
        return outputs

    def prepare_volume(self, sample: dict[str, Any]) -> tuple[Tensor, np.ndarray]:
        """The (1, 96, 108, 96) model input and its voxel -> image world mm affine."""
        device = self.device
        affine = sample["affine"].double()
        mni_affine = sample["mni_affine"]

        # With an MNI affine, the MetaTensor's world becomes MNI space
        world_affine = affine if mni_affine is None else mni_affine.double() @ affine
        volume = MetaTensor(sample["image"].to(device), affine=world_affine.to(device))
        mask = sample["mask"]
        if mask is not None:
            mask = MetaTensor(mask.to(device).float(), affine=world_affine.to(device))

        if mni_affine is not None:
            resample = transforms.SpatialResample()
            mni_affine_grid = torch.from_numpy(MNI_AFFINE).to(device)
            volume = resample(volume, dst_affine=mni_affine_grid, spatial_size=MNI_SHAPE, mode=3)
            if mask is not None:
                mask = resample(
                    mask, dst_affine=mni_affine_grid, spatial_size=MNI_SHAPE, mode="nearest"
                )

        if mask is not None:
            volume = volume * (mask > 0.5)

        volume = self.preprocess(volume)
        input_affine = np.asarray(volume.affine.cpu(), dtype=np.float64)
        if mni_affine is not None:
            input_affine = np.linalg.inv(mni_affine.double().numpy()) @ input_affine
        return volume.as_tensor(), input_affine


@register_model
def neurojepa(checkpoint: str | Path = REPO_ID, revision: str = REVISION) -> NeuroJEPA:
    """Load with upstream `load_backbone_from_hf`, from an HF repo (gated; `HF_TOKEN`) or a local
    dir with `config.json`."""
    return NeuroJEPA(load_backbone_from_hf(str(checkpoint), device="cpu", revision=revision))


def upstream_transform() -> transforms.Compose:
    """`loading_transforms` + test-mode `vit3d_transforms` from neurojepa.data.transforms.

    Upstream are dict transforms over a file path. These are the array versions, applied after
    loading and NaN removal. `test_neurojepa.py` checks them against upstream.
    """
    return transforms.Compose(
        [
            transforms.Orientation(axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
            transforms.ScaleIntensityRangePercentiles(
                lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True
            ),
            transforms.ResizeWithPadOrCrop(spatial_size=(180, 216, 180), mode="edge"),
            transforms.Spacing(pixdim=(1.0, 1.0, 1.0), mode=5),
            transforms.CropForeground(select_fn=lambda x: x > 0.0, margin=4, allow_smaller=True),
            transforms.Resize(spatial_size=(100, 120, 100)),
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=IMG_SIZE, mode="constant", value=0),
            transforms.CenterSpatialCrop(roi_size=IMG_SIZE),
        ]
    )
