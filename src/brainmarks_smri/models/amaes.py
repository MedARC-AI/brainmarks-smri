"""AMAES (FOMO-MRI/AMAES_resenc_b_fomo300k): a ResEnc-B U-Net encoder pretrained with MAE on FOMO300K.

Pretraining used raw FOMO300K scans (mostly not skull-stripped) in RAS at native spacing,
z-scored over the whole volume, as random 160^3 crops. Upstream's linear probe pools the last
encoder stage over a centre 160^3 crop. We resample to 1 mm (trilinear) and centre the crop on
the brain-mask centroid (else on `data > data.mean()`), zero padded as upstream. `mni_affine` is
ignored. A 160 mm crop keeps ~99% of the brain (97.5% worst over 26 masks; it clips the frontal
pole and cerebellum of long heads).

Global: mean of the last stage (320 ch, stride 32). Dense: stage `dense_stage` (default 3: 256 ch,
stride 8, a 20^3 grid).

Upstream builds the encoder with gardening_tools; it is the same network as the ResEnc-L of
dynamic_network_architectures up to key names, with bit-identical outputs, so we use the latter.
"""

from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from huggingface_hub import hf_hub_download

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

REPO_ID = "FOMO-MRI/AMAES_resenc_b_fomo300k"
REVISION = "75cb10b89a9855377aa29e1da53b66dade7f9229"
CHECKPOINT = "resenc_unet_b.ckpt"
IMG_SIZE = (160, 160, 160)
FEATURES_PER_STAGE = (32, 64, 128, 256, 320, 320)


class AMAES(nn.Module):
    name = "amaes"
    embed_dim = FEATURES_PER_STAGE[-1]

    def __init__(self, encoder: ResidualEncoder, dense_stage: int = 3):
        super().__init__()
        self.encoder = encoder
        self.dense_stage = dense_stage
        stride = 2**dense_stage
        self.patch_size = (stride, stride, stride)
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def transform(self, sample: ImageInput) -> dict[str, Any]:
        image: nib.Nifti1Image = sample["image"]
        image = nib.funcs.squeeze_image(nib.Nifti1Image(image.dataobj, image.affine, image.header))
        canonical = nib.as_closest_canonical(image)
        data = np.nan_to_num(canonical.get_fdata(dtype=np.float32))
        spacing = np.array(canonical.header.get_zooms()[:3], dtype=np.float64)

        # 1 mm in RAS; with align_corners=False, output voxel i sits at input voxel (i + 0.5) / s - 0.5
        volume = F.interpolate(
            torch.from_numpy(data)[None, None],
            scale_factor=tuple(spacing),
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )[0, 0]
        step = np.diag([*(1 / spacing), 1.0])
        step[:3, 3] = 0.5 / spacing - 0.5
        resampled_affine = canonical.affine @ step

        mask = sample.get("brain_mask")
        if mask is not None:
            mask_voxels = np.argwhere(np.asarray(mask.dataobj).reshape(image.shape) > 0)
            world_centre = image.affine[:3, :3] @ mask_voxels.mean(0) + image.affine[:3, 3]
            inverse = np.linalg.inv(resampled_affine)
            centre = inverse[:3, :3] @ world_centre + inverse[:3, 3]
        else:
            centre = torch.argwhere(volume > volume.mean()).double().mean(0).numpy()

        mean = volume.mean()
        std = volume.std(correction=0).clamp_min(1e-6)
        volume = (volume - mean) / std

        # crop/pad to IMG_SIZE around the centre, zeros outside
        start = np.round(centre - (np.array(IMG_SIZE) - 1) / 2).astype(int)
        crop = torch.zeros(IMG_SIZE)
        source = []
        target = []
        for axis_start, crop_size, volume_size in zip(start, IMG_SIZE, volume.shape):
            low = max(axis_start, 0)
            high = min(axis_start + crop_size, volume_size)
            source.append(slice(low, max(high, low)))
            target.append(slice(low - axis_start, max(high, low) - axis_start))
        crop[tuple(target)] = volume[tuple(source)]

        shift = np.eye(4)
        shift[:3, 3] = start
        return {"volume": crop, "input_affine": resampled_affine @ shift}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        volumes = torch.stack([sample["volume"] for sample in samples])[:, None]
        volumes = volumes.to(self.device)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            stages = self.encoder(volumes)
        global_embeddings = stages[-1].float().mean((2, 3, 4))  # (B, 320)

        outputs = []
        for index, sample in enumerate(samples):
            output = EmbeddingOutput(
                global_embedding=global_embeddings[index],
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                dense = stages[self.dense_stage][index].float()  # (C, X, Y, Z)
                output["dense_embedding"] = dense.permute(1, 2, 3, 0).contiguous()
                output["dense_affine"] = sample["input_affine"]
            outputs.append(output)
        return outputs


@register_model
def amaes(checkpoint: str = CHECKPOINT, revision: str = REVISION, dense_stage: int = 3) -> AMAES:
    """Load the encoder from the Lightning checkpoint on the HF hub (public)."""
    path = hf_hub_download(REPO_ID, checkpoint, revision=revision)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    # gardening_tools names -> dynamic_network_architectures names
    state = {
        key.removeprefix("model.encoder.")
        .replace("stem.conv1.", "stem.convs.0.")
        .replace(".norm_op.", ".norm."): value
        for key, value in state.items()
        if key.startswith("model.encoder.")
    }
    encoder = resenc_b_encoder()
    encoder.load_state_dict(state)
    return AMAES(encoder, dense_stage=dense_stage)


def resenc_b_encoder(n_blocks_per_stage: tuple[int, ...] = (1, 3, 4, 6, 6, 6)) -> ResidualEncoder:
    """The encoder of asparagus `resenc_unet_b`, returning all 6 stage outputs."""
    n_stages = len(FEATURES_PER_STAGE)
    return ResidualEncoder(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=list(FEATURES_PER_STAGE),
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n_stages,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1),
        n_blocks_per_stage=list(n_blocks_per_stage),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        return_skips=True,
    )
