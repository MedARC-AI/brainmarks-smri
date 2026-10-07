"""BrainDINO: a DINOv3 ViT-B/16 (4 register tokens) pretrained on 2D brain MRI slices.

Weights are not released yet, so this follows the upstream downstream code
(`networks/SliceStudent.py`, dataset loaders) and has only been run with random weights.

Each volume is z-scored over nonzero voxels (after a 1-99 percentile clip), resized to 128
slices along array axis 2, and each slice is resized to 224x224 and repeated to 3 channels.
Upstream never reorients, so the slice axis is whatever the file stores as axis 2. We reorient
to RAS first so the slices are axial. Upstream z-scores over nonzero voxels, which assumes
skull-stripped input, so `brain_mask` is applied when given. `mni_affine` is unused.

Global embedding: per-slice CLS tokens averaged over the 128 slices, as upstream. Dense: the 14x14
patch tokens of each slice, averaged over blocks of 16 slices, so a 14x14x8 grid of 16x16x16
patches on the 224x224x128 input. The pooling (ours) keeps the dense grid comparable in size to
the other models'; the per-slice grid would be 25k tokens per volume.

`dinov3_vitb16` is the same slice pipeline with Meta's natural-image DINOv3 ViT-B/16, the
DINOv3 baseline in the BrainDINO paper.
"""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov3.hub.backbones import dinov3_vitb16 as official_dinov3_vitb16
from dinov3.models.vision_transformer import DinoVisionTransformer, vit_base
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

N_SLICES = 128
SLICE_SIZE = 224
IMG_SIZE = (SLICE_SIZE, SLICE_SIZE, N_SLICES)
PATCH_SIZE = (16, 16, 16)  # in-plane ViT patches, 16 slices pooled

# timm's copy of Meta's DINOv3 ViT-B/16 (LVD-1689M), ungated. Meta's own HF repo is gated.
DINOV3_REPO_ID = "timm/vit_base_patch16_dinov3.lvd1689m"
DINOV3_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"


class BrainDINO(nn.Module):
    embed_dim = 768
    patch_size = PATCH_SIZE

    def __init__(self, backbone: DinoVisionTransformer, name: str = "braindino"):
        super().__init__()
        self.name = name
        self.backbone = backbone
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
        n_slice_blocks = N_SLICES // PATCH_SIZE[2]
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
                # mean over each block of PATCH_SIZE[2] slices
                blocks = sample_patches.float().reshape(
                    n_slice_blocks, PATCH_SIZE[2], grid, grid, self.embed_dim
                )
                pooled = blocks.mean(dim=1)
                # (slice block, row, col) = (z, x, y) -> (x, y, z)
                output["dense_embedding"] = pooled.permute(1, 2, 0, 3)
                output["dense_affine"] = sample["input_affine"]
            outputs.append(output)
        return outputs


@register_model
def braindino(checkpoint: str | Path) -> BrainDINO:
    """Load a DINOv3-style checkpoint (`teacher`, `model` or a bare state dict), as upstream
    SliceStudent does, but strictly."""
    # as upstream SliceStudent
    backbone = vit_base(layerscale_init=1e-5, n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    dino_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = dino_checkpoint.get("teacher", dino_checkpoint.get("model", dino_checkpoint))
    state_dict = {
        key.replace("backbone.", ""): value
        for key, value in state_dict.items()
        if "ibot" not in key and "dino_head" not in key and not key.startswith("head.")
    }
    backbone.load_state_dict(state_dict, strict=True)
    return BrainDINO(backbone)


@register_model
def dinov3_vitb16(revision: str = DINOV3_REVISION) -> BrainDINO:
    """Meta's DINOv3 ViT-B/16 in the official architecture, with weights from timm's copy.

    timm renames a few keys and drops what it doesn't use: the qkv biases (all zero in the
    distilled ViT-B), the RoPE periods, the pretraining mask token, and the k-bias mask. Those
    come from the model's own init. `test_braindino.py` checks the output against timm's model.
    """
    backbone = official_dinov3_vitb16(pretrained=False)
    timm_state_dict = load_file(
        hf_hub_download(DINOV3_REPO_ID, "model.safetensors", revision=revision)
    )
    state_dict = {}
    for key, value in timm_state_dict.items():
        key = key.replace("reg_token", "storage_tokens")
        key = key.replace("gamma_1", "ls1.gamma").replace("gamma_2", "ls2.gamma")
        state_dict[key] = value
    for key, value in backbone.state_dict().items():
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(value) if key.endswith("qkv.bias") else value
    backbone.load_state_dict(state_dict, strict=True)
    return BrainDINO(backbone, name="dinov3_vitb16")


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
