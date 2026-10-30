"""nnFoundationViT (MIC-DKFZ/nnFoundationViT): an MAE-pretrained Primus/Eva ViT over 8^3 patches.

Pretrained on 2.1M CT, MRI and PET volumes with nnssl: crop to the nonzero bounding box, z-score
over it (no mask), random 192^3 patches. The median pretraining spacing is 1 mm, and nnU-Net's
"like_pretrained" downstream mode resamples to it. We do the same, then sample a 192^3 grid
(zeros outside the image) centred on the brain mask centroid, or on the `data > data.mean()`
centroid without a mask. Deviations: nnU-Net reads arrays with SimpleITK as stored; we orient the
grid to RAS. It resamples with cubic splines; we use trilinear. The grid snaps to image voxels, so
1 mm images are not interpolated. The global embedding is the mean of all 24^3 tokens after the
final norm; the dense grid is the tokens themselves.

The encoder is the first half of nnssl `EvaMAE`, built from `dynamic_network_architectures`.
"""

from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchEmbed
from einops import rearrange
from huggingface_hub import hf_hub_download
from nibabel.affines import apply_affine
from scipy import ndimage
from timm.layers import RotaryEmbeddingCat

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

REPO_ID = "MIC-DKFZ/nnFoundationViT"
REVISION = "c691f2115140458ec64017712eb8ebbc8dde6ae4"
CHECKPOINT = "checkpoint_final.pth"
IMG_SIZE = (192, 192, 192)
PATCH_SIZE = (8, 8, 8)
EMBED_DIM = 1056
DEPTH = 40
NUM_HEADS = 16


class RotaryEmbeddingCat3D(RotaryEmbeddingCat):
    """timm>=1.0.23 sizes the cached table for 2D grids; rebuild it at its real size."""

    def _init_buffers(self) -> None:
        self.pos_embed = self._get_pos_embed_values(self.feat_shape)


class NNFoundationViT(nn.Module):
    name = "nnfoundation_vit"
    embed_dim = EMBED_DIM
    patch_size = PATCH_SIZE

    def __init__(
        self,
        depth: int = DEPTH,
        img_size: tuple[int, int, int] = IMG_SIZE,
        max_batch_size: int = 8,
    ):
        super().__init__()
        self.img_size = img_size
        self.grid_size = tuple(size // patch for size, patch in zip(img_size, PATCH_SIZE))
        self.max_batch_size = max_batch_size
        # names match the checkpoint's `network_weights`
        self.down_projection = PatchEmbed(PATCH_SIZE, 1, EMBED_DIM)
        self.eva = Eva(
            embed_dim=EMBED_DIM,
            depth=depth,
            num_heads=NUM_HEADS,
            ref_feat_shape=self.grid_size,
            num_reg_tokens=0,
            use_rot_pos_emb=True,
            use_abs_pos_emb=True,
            mlp_ratio=4 * 2 / 3,
            init_values=0.1,
            scale_attn_inner=True,
            rope_impl=RotaryEmbeddingCat3D,
        )
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
        if mask is None:
            mask = data > data.mean()
        else:
            mask = np.asarray(mask.dataobj).reshape(data.shape) > 0
        centre = apply_affine(image.affine, ndimage.center_of_mass(mask))
        input_affine = self.input_affine(image, centre)

        # as nnssl: crop to the nonzero bounding box, z-score over it
        nonzero = data != 0
        lower = []
        upper = []
        for axis in range(3):
            other_axes = tuple(other for other in range(3) if other != axis)
            present = np.flatnonzero(nonzero.any(other_axes))
            lower.append(present[0])
            upper.append(present[-1] + 1)
        lower = np.array(lower)
        data = data[tuple(slice(start, stop) for start, stop in zip(lower, upper))]
        data = (data - data.mean()) / max(data.std(), 1e-8)

        # model input voxel -> cropped image voxel
        sampling_affine = np.linalg.inv(image.affine) @ input_affine
        sampling_affine[:3, 3] -= lower
        volume = ndimage.affine_transform(
            data,
            sampling_affine[:3, :3],
            sampling_affine[:3, 3],
            output_shape=self.img_size,
            order=1,
            cval=0.0,
        )
        return {"volume": torch.from_numpy(volume.astype(np.float32)), "input_affine": input_affine}

    def input_affine(self, image: nib.Nifti1Image, centre: np.ndarray) -> np.ndarray:
        """Model input voxel -> image world mm: a 1 mm RAS grid centred on `centre`, snapped to the
        image voxels."""
        affine = image.affine
        zooms = np.array(image.header.get_zooms()[:3])
        directions = np.zeros((3, 3))
        for image_axis, (ras_axis, flip) in enumerate(nib.io_orientation(affine)):
            directions[:, int(ras_axis)] = flip * affine[:3, image_axis] / zooms[image_axis]
        start = centre - directions @ ((np.array(self.img_size) - 1) / 2)
        start_voxel = np.round(apply_affine(np.linalg.inv(affine), start))
        input_affine = np.eye(4)
        input_affine[:3, :3] = directions
        input_affine[:3, 3] = apply_affine(affine, start_voxel)
        return input_affine

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        outputs = []
        for chunk_start in range(0, len(samples), self.max_batch_size):
            chunk = samples[chunk_start : chunk_start + self.max_batch_size]
            volumes = torch.stack([sample["volume"] for sample in chunk])
            volumes = volumes.to(self.device)[:, None]  # (B, 1, 192, 192, 192)
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                patches = self.down_projection(volumes)
                tokens = rearrange(patches, "b c x y z -> b (x y z) c")
                tokens, _ = self.eva(tokens)
            tokens = tokens.float()  # (B, 13824, 1056)

            for sample, sample_tokens in zip(chunk, tokens):
                output = EmbeddingOutput(
                    global_embedding=sample_tokens.mean(0),
                    dense_embedding=None,
                    dense_affine=None,
                    dense_mask=None,
                )
                if return_dense:
                    output["dense_embedding"] = sample_tokens.reshape(*self.grid_size, EMBED_DIM)
                    output["dense_affine"] = sample["input_affine"]
                outputs.append(output)
        return outputs


@register_model
def nnfoundation_vit(revision: str = REVISION, max_batch_size: int = 8) -> NNFoundationViT:
    """Load the MAE encoder from the HF hub (public)."""
    path = hf_hub_download(REPO_ID, CHECKPOINT, revision=revision)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = {
        key: value
        for key, value in checkpoint["network_weights"].items()
        if key.startswith(("down_projection.", "eva."))
    }
    model = NNFoundationViT(max_batch_size=max_batch_size)
    model.load_state_dict(state)
    return model
