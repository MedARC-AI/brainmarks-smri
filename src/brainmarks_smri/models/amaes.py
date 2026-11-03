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
from nibabel.affines import apply_affine
from scipy import ndimage

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

REPO_ID = "FOMO-MRI/AMAES_resenc_b_fomo300k"
REVISION = "75cb10b89a9855377aa29e1da53b66dade7f9229"
CHECKPOINT = "resenc_unet_b.ckpt"
IMG_SIZE = (160, 160, 160)
SPACING = 1.0
STAGE_CHANNELS = (32, 64, 128, 256, 320, 320)
STAGE_STRIDES = (1, 2, 4, 8, 16, 32)


def resenc_b_encoder(n_blocks_per_stage: tuple[int, ...] = (1, 3, 4, 6, 6, 6)) -> ResidualEncoder:
    """The encoder of asparagus `resenc_unet_b`, returning all 6 stage outputs."""
    n_stages = len(STAGE_CHANNELS)
    return ResidualEncoder(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=list(STAGE_CHANNELS),
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


class AMAES(nn.Module):
    name = "amaes"
    embed_dim = STAGE_CHANNELS[-1]

    def __init__(self, encoder: ResidualEncoder, dense_stage: int = 3):
        super().__init__()
        self.encoder = encoder
        self.dense_stage = dense_stage
        self.patch_size = (STAGE_STRIDES[dense_stage],) * 3
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def transform(self, sample: ImageInput) -> dict[str, Any]:
        image: nib.Nifti1Image = sample["image"]
        image = nib.funcs.squeeze_image(nib.Nifti1Image(image.dataobj, image.affine, image.header))
        data = np.nan_to_num(image.get_fdata(dtype=np.float32))

        # brain centre in world mm
        mask = sample.get("brain_mask")
        if mask is None:
            mask = data > data.mean()
        else:
            mask = np.asarray(mask.dataobj).reshape(data.shape) > 0
        centre = apply_affine(image.affine, ndimage.center_of_mass(mask))

        # RAS at SPACING mm; output voxel i samples input voxel (i + 0.5) / scale - 0.5
        canonical = nib.as_closest_canonical(nib.Nifti1Image(data, image.affine))
        scale = np.array(canonical.header.get_zooms()[:3], dtype=np.float64) / SPACING
        volume = torch.from_numpy(np.ascontiguousarray(canonical.dataobj, dtype=np.float32))
        volume = F.interpolate(
            volume[None, None],
            scale_factor=tuple(scale),
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )[0, 0]
        resampled_to_canonical = np.diag([*(1 / scale), 1.0])
        resampled_to_canonical[:3, 3] = 0.5 / scale - 0.5
        resampled_affine = canonical.affine @ resampled_to_canonical

        # as asparagus: z-score over the whole volume
        volume = (volume - volume.mean()) / volume.std(correction=0).clamp_min(1e-6)

        # crop / zero pad to IMG_SIZE around the brain centre
        centre_voxel = apply_affine(np.linalg.inv(resampled_affine), centre)
        start = np.round(centre_voxel - (np.array(IMG_SIZE) - 1) / 2).astype(int)
        crop = torch.zeros(IMG_SIZE)
        source = []
        target = []
        for axis in range(3):
            low = max(start[axis], 0)
            high = max(min(start[axis] + IMG_SIZE[axis], volume.shape[axis]), low)
            source.append(slice(low, high))
            target.append(slice(low - start[axis], high - start[axis]))
        crop[tuple(target)] = volume[tuple(source)]
        crop_to_resampled = np.eye(4)
        crop_to_resampled[:3, 3] = start
        return {"volume": crop, "input_affine": resampled_affine @ crop_to_resampled}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        volumes = torch.stack([sample["volume"] for sample in samples])[:, None].to(self.device)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            stages = self.encoder(volumes)  # (B, C, X, Y, Z) per stage
        global_embeddings = stages[-1].float().mean((2, 3, 4))

        outputs = []
        for index, sample in enumerate(samples):
            output = EmbeddingOutput(
                global_embedding=global_embeddings[index],
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                dense = stages[self.dense_stage][index].float()
                output["dense_embedding"] = dense.permute(1, 2, 3, 0)
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
