"""BrainDINO: a DINOv3 ViT-B/16 (4 register tokens) pretrained on 2D brain MRI slices.

Weights are not released yet, so this follows the upstream downstream code
(`networks/SliceStudent.py`, dataset loaders) and has only been run with random weights.

Each volume is z-scored over nonzero voxels (after a 1-99 percentile clip), resized to 128
slices along array axis 2, and each slice is resized to 224x224 and repeated to 3 channels.
Upstream never reorients, so the slice axis is whatever the file stores as axis 2. We reorient
to RAS first so the slices are axial. Upstream z-scores over nonzero voxels, which assumes
skull-stripped input, so `brain_mask` is applied when given. `mni_affine` is unused.

Global embedding: per-slice CLS tokens averaged over the 128 slices, as upstream. Dense: the
14x14 patch tokens of every slice, a 14x14x128 grid of 16x16x1 patches on the 224x224x128 input.
"""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov3.models.vision_transformer import vit_base

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

N_SLICES = 128
SLICE_SIZE = 224
IMG_SIZE = (SLICE_SIZE, SLICE_SIZE, N_SLICES)
PATCH_SIZE = (16, 16, 1)


class BrainDINO(nn.Module):
    name = "braindino"
    embed_dim = 768
    patch_size = PATCH_SIZE

    def __init__(self):
        super().__init__()
        # as upstream SliceStudent
        self.backbone = vit_base(
            layerscale_init=1e-5, n_storage_tokens=4, qkv_bias=False, mask_k_bias=True
        )
        self.backbone.init_weights()
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def transform(self, sample: ImageInput) -> dict[str, Any]:
        image: nib.Nifti1Image = sample["image"]
        data = np.nan_to_num(image.get_fdata(dtype=np.float32))
        if sample.get("brain_mask") is not None:
            data = data * (np.asarray(sample["brain_mask"].dataobj) > 0)
        image = nib.as_closest_canonical(nib.Nifti1Image(data, image.affine))
        data = np.asarray(image.dataobj, dtype=np.float32)  # (X, Y, Z), RAS

        volume = torch.from_numpy(zscore_normalize(data))
        volume = volume.permute(2, 0, 1)[None, None]  # (1, 1, Z, X, Y), upstream's (D, H, W)
        x_size, y_size = data.shape[:2]
        volume = F.interpolate(
            volume, size=(N_SLICES, x_size, y_size), mode="trilinear", align_corners=False
        )
        volume = F.interpolate(
            volume[0], size=(SLICE_SIZE, SLICE_SIZE), mode="bilinear", align_corners=False
        )  # (1, 128, 224, 224)

        # input voxel (x, y, z) on the 224x224x128 grid -> image voxel, then -> world mm
        scale = np.array(data.shape) / np.array(IMG_SIZE)
        resize_affine = np.eye(4)
        resize_affine[:3, :3] = np.diag(scale)
        resize_affine[:3, 3] = 0.5 * scale - 0.5
        return {"volume": volume, "input_affine": image.affine @ resize_affine}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        volumes = torch.stack([sample["volume"] for sample in samples]).to(self.device)
        batch_size = len(samples)
        slices = volumes.reshape(batch_size * N_SLICES, 1, SLICE_SIZE, SLICE_SIZE)
        slices = slices.repeat(1, 3, 1, 1)  # grayscale to 3 channels, no ImageNet normalization

        features = self.backbone.forward_features(slices)
        cls_tokens = features["x_norm_clstoken"].reshape(batch_size, N_SLICES, self.embed_dim)
        grid = SLICE_SIZE // PATCH_SIZE[0]
        patch_tokens = features["x_norm_patchtokens"].reshape(
            batch_size, N_SLICES, grid, grid, self.embed_dim
        )

        outputs = []
        for sample, sample_cls, sample_patches in zip(samples, cls_tokens, patch_tokens):
            output = EmbeddingOutput(
                global_embedding=sample_cls.float().mean(0),
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                # (slice, row, col) = (z, x, y) -> (x, y, z)
                output["dense_embedding"] = sample_patches.float().permute(1, 2, 0, 3)
                output["dense_affine"] = sample["input_affine"]
            outputs.append(output)
        return outputs


# cl: can we also add a model that uses the official dinov3 weights? ideally vit-b size.


@register_model
def braindino(checkpoint: str | Path) -> BrainDINO:
    """Load a DINOv3-style checkpoint (`teacher`, `model` or a bare state dict), as upstream
    SliceStudent does, but strictly."""
    model = BrainDINO()
    dino_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = dino_checkpoint.get("teacher", dino_checkpoint.get("model", dino_checkpoint))
    state_dict = {
        key.replace("backbone.", ""): value
        for key, value in state_dict.items()
        if "ibot" not in key and "dino_head" not in key and not key.startswith("head.")
    }
    model.backbone.load_state_dict(state_dict, strict=True)
    return model


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Upstream `zscore_normalize`: clip to the 1-99 percentiles of nonzero voxels, then z-score
    with the mean and std of the nonzero voxels. Background becomes negative."""
    nonzero = volume[volume > 0]
    if nonzero.size < 10:
        nonzero = volume.reshape(-1)
    volume = np.clip(volume, np.percentile(nonzero, 1), np.percentile(nonzero, 99))
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        nonzero = volume
    std = max(nonzero.std(), 1e-6)
    return ((volume - nonzero.mean()) / std).astype(np.float32)
