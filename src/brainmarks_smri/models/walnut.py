"""Walnut (MedARC-AI/walnut-fomo26): an MAE-pretrained 3D ViT-L over 8^3 patches.

Pretraining used 1 mm scans rigidly registered to MNI152NLin2009cAsym (193x229x193), centred in
a 208x240x208 grid, masked with SynthSeg, and z-scored over the mask with zeros outside. Patches
without brain voxels are not tokens. Given `mni_affine` and `brain_mask`, we resample onto that
grid with the rigid part of `mni_affine` (our affines target MNI152NLin6Asym, which is within
~2 mm) and mask. Without them, we follow upstream's eval transform (`fomo_tune.backbone`): RAS,
1 mm, centred, masked with `data > data.mean()`. Upstream resamples with B-splines; we use
trilinear. The global embedding is the mean of the live patch tokens, as in the FOMO26
submission, not the CLS token.

`transform` only computes the sampling grid; the resample runs on the model's device. The
encoder packs the live tokens into nested-tensor attention, which has no CPU path.
"""

from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from huggingface_hub import hf_hub_download
from smri_mae.model_mae import MaskedViT
from torch import Tensor

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

REPO_ID = "medarc/walnut"
REVISION = "c74b76b39c398df6a4c18977b8d7dba2dd630f6c"
CHECKPOINT = "checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"
IMG_SIZE = (208, 240, 208)
PATCH_SIZE = (8, 8, 8)
GRID_SIZE = (26, 30, 26)

# Model input voxel -> MNI mm: the 1 mm MNI152NLin2009cAsym grid (origin -96, -132, -78)
# centred in IMG_SIZE, i.e. shifted by (7, 5, 7) voxels.
MNI_INPUT_AFFINE = np.array(
    [
        [1.0, 0.0, 0.0, -103.0],
        [0.0, 1.0, 0.0, -137.0],
        [0.0, 0.0, 1.0, -85.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class Walnut(nn.Module):
    name = "walnut"
    embed_dim = 1024
    patch_size = PATCH_SIZE

    def __init__(self, encoder: MaskedViT, dense_pool: int = 1):
        super().__init__()
        self.encoder = encoder
        # average dense tokens over dense_pool^3 blocks, for comparing at a coarser patch size
        self.dense_pool = dense_pool
        self.patch_size = tuple(size * dense_pool for size in PATCH_SIZE)
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def transform(self, sample: ImageInput) -> dict[str, Any]:
        image: nib.Nifti1Image = sample["image"]
        image = nib.funcs.squeeze_image(nib.Nifti1Image(image.dataobj, image.affine, image.header))
        data = np.nan_to_num(image.get_fdata(dtype=np.float32))
        mask = sample.get("brain_mask")
        if mask is not None:
            mask = np.asarray(mask.dataobj).reshape(data.shape) > 0
        mni_affine = sample.get("mni_affine")
        if mni_affine is None:
            input_affine = upstream_input_affine(image)
        else:
            input_affine = rigid_mni_to_image(mni_affine) @ MNI_INPUT_AFFINE
        return {
            "image": torch.from_numpy(data),
            "mask": None if mask is None else torch.from_numpy(mask),
            # model input voxel -> image voxel
            "sampling_affine": torch.from_numpy(np.linalg.inv(image.affine) @ input_affine),
            "input_affine": input_affine,
        }

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        volumes = []
        masks = []
        for sample in samples:
            volume, mask = self.prepare_volume(sample)
            volumes.append(volume)
            masks.append(mask)
        volumes = torch.stack(volumes)[:, None]  # (B, 1, 208, 240, 208)
        masks = torch.stack(masks)[:, None]

        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            # patch_ids are the live patches in raster order, padded to the longest sample
            _, _, tokens, _, patch_ids, token_mask = self.encoder(volumes, mask=masks)
        tokens = tokens.float()  # (B, L, 1024)

        num_patches = int(np.prod(GRID_SIZE))
        outputs = []
        for sample, sample_tokens, sample_ids, sample_valid in zip(
            samples, tokens, patch_ids, token_mask
        ):
            live_tokens = sample_tokens[sample_valid]
            output = EmbeddingOutput(
                global_embedding=live_tokens.mean(0),
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                live_ids = sample_ids[sample_valid]
                dense = live_tokens.new_zeros(num_patches, self.embed_dim)
                dense[live_ids] = live_tokens
                dense_mask = torch.zeros(num_patches, dtype=torch.bool, device=self.device)
                dense_mask[live_ids] = True
                # Patchify3D flattens patches in C order, so ids are (x y z) ordered
                dense = dense.reshape(*GRID_SIZE, self.embed_dim)
                dense_mask = dense_mask.reshape(GRID_SIZE)
                if self.dense_pool > 1:
                    # mean over the live tokens of each block
                    pattern = "(x i) (y j) (z k) ... -> x y z ..."
                    k = self.dense_pool
                    dense = reduce(dense, pattern, "sum", i=k, j=k, k=k)
                    counts = reduce(dense_mask.int(), pattern, "sum", i=k, j=k, k=k)
                    dense = dense / counts.clamp_min(1)[..., None]
                    dense_mask = counts > 0
                output["dense_embedding"] = dense
                output["dense_mask"] = dense_mask
                output["dense_affine"] = sample["input_affine"]
            outputs.append(output)
        return outputs

    def prepare_volume(self, sample: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """The (208, 240, 208) z-scored model input and its brain mask."""
        device = self.device
        image = sample["image"].to(device)
        sampling_affine = sample["sampling_affine"].to(device)
        volume = resample(image, sampling_affine, mode="bilinear")
        if sample["mask"] is None:
            mask = volume > volume.mean()
        else:
            mask = resample(sample["mask"].to(device).float(), sampling_affine, mode="nearest")
            mask = mask > 0.5

        brain = volume[mask]
        mean = brain.mean()
        std = brain.std(correction=0).clamp_min(1e-6)
        volume = torch.where(mask, (volume - mean) / std, 0.0)
        return volume, mask


@register_model
def walnut(checkpoint: str = CHECKPOINT, revision: str = REVISION, dense_pool: int = 1) -> Walnut:
    """Load the encoder from the MAE checkpoint on the HF hub (public)."""
    path = hf_hub_download(REPO_ID, checkpoint, revision=revision)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"]
    state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    encoder = vit_large()
    encoder.load_state_dict(state)
    return Walnut(encoder, dense_pool=dense_pool)


def vit_large(depth: int = 24) -> MaskedViT:
    """The pretrained encoder architecture (upstream `mae_vit_large`, patch 8, CLS token)."""
    return MaskedViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        depth=depth,
        embed_dim=1024,
        num_heads=16,
        class_token=True,
    )


def resample(image: Tensor, sampling_affine: Tensor, mode: str) -> Tensor:
    """Sample `image` on the model input grid; `sampling_affine` maps input voxels to its voxels."""
    axes = [torch.arange(size, device=image.device, dtype=torch.float32) for size in IMG_SIZE]
    input_voxels = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    sampling_affine = sampling_affine.float()
    image_voxels = input_voxels @ sampling_affine[:3, :3].T + sampling_affine[:3, 3]

    # grid_sample wants (x, y, z) = (last, middle, first) axis, normalized to [-1, 1]
    sizes = torch.tensor(image.shape, device=image.device, dtype=torch.float32)
    grid = 2 * image_voxels / (sizes - 1) - 1
    grid = grid.flip(-1)[None]
    resampled = F.grid_sample(
        image[None, None], grid, mode=mode, padding_mode="zeros", align_corners=True
    )
    return resampled[0, 0]


def upstream_input_affine(image: nib.Nifti1Image) -> np.ndarray:
    """Model input voxel -> image world mm, as in upstream `SmriMaeTransform.resize`.

    Upstream reorients to RAS, rescales to 1 mm if any spacing is off by more than 0.05 mm
    (`F.interpolate`, align_corners=False), then centre crops or pads.
    """
    canonical = nib.as_closest_canonical(image)
    affine = canonical.affine
    shape = np.array(canonical.shape)
    spacing = np.array(canonical.header.get_zooms()[:3])
    if np.abs(spacing - 1.0).max() > 0.05:
        shape = np.floor(shape * spacing).astype(int)
        step = np.diag([*(1 / spacing), 1.0])
        step[:3, 3] = 0.5 / spacing - 0.5
        affine = affine @ step
    pads = np.array(IMG_SIZE) - shape
    fit = np.eye(4)
    fit[:3, 3] = -(pads // 2)
    return affine @ fit


def rigid_mni_to_image(image_to_mni: np.ndarray) -> np.ndarray:
    """The rigid part of an image world -> MNI affine, inverted.

    Pretraining registered rigidly, so head size is kept. We keep the rotation from the polar
    decomposition and fix the image point that the full affine sends to the input grid centre.
    """
    u, _, vt = np.linalg.svd(image_to_mni[:3, :3])
    rotation = u @ vt
    mni_centre = MNI_INPUT_AFFINE @ np.array([*((np.array(IMG_SIZE) - 1) / 2), 1.0])
    image_centre = np.linalg.inv(image_to_mni) @ mni_centre
    mni_to_image = np.eye(4)
    mni_to_image[:3, :3] = rotation.T
    mni_to_image[:3, 3] = image_centre[:3] - rotation.T @ mni_centre[:3]
    return mni_to_image
