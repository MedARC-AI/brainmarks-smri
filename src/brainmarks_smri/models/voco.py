"""VoCo-B (Luffy503/VoCo): a SwinUNETR-B encoder self-supervised on 160k CT volumes.

Pretraining used RAS, 1.5 mm, the CT window [-175, 250] HU scaled to [0, 1] and 64^3 crops. For
MRI we keep RAS and 1.5 mm and scale the intensity to [0, 1] between the image's 0.5th and 99.5th
percentiles, clipped, as Neuro-JEPA's frozen VoCo baseline did; this keeps the pretraining input
range. (VoCo's own MRI fine-tuning z-scores nonzero voxels instead.) The 128^3 input (192 mm) is
centred on the brain mask centroid, else on the `data > data.mean()` centroid. Resampling is
trilinear, in one step from the stored image. `mni_affine` is ignored.

The global embedding is upstream's VoCo head input: the mean over space of encoder1/2/3/4/10,
concatenated (1152-d). Dense features come from one of these five maps (`dense_stage`), by
default encoder4 (192 ch, stride 8, 12 mm patches).

Upstream pins monai==1.3.0. MONAI 1.5 changed the 3D slice order of the legacy `PatchMerging`
used by SwinUNETR, so we restore the old order to match the pretrained weights.
"""

from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.nets.swin_unetr import PatchMergingV2, SwinTransformer
from scipy import ndimage
from torch import Tensor

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

REPO_ID = "Luffy503/VoCo"
REVISION = "68df573b67d32d869adcc3dc45307d93e9440628"
CHECKPOINT = "VoCo_B_SSL_head.pt"
IMG_SIZE = (128, 128, 128)
SPACING = 1.5
FEATURE_SIZE = 48
# encoder1, encoder2, encoder3, encoder4, encoder10
STAGE_CHANNELS = (48, 48, 96, 192, 768)
STAGE_STRIDES = (1, 2, 4, 8, 32)
INTENSITY_PERCENTILES = (0.5, 99.5)


class LegacyPatchMerging(PatchMergingV2):
    """MONAI 1.3's `PatchMerging`: it repeats two of the eight 2x2x2 slices and skips two."""

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        if d % 2 or h % 2 or w % 2:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class VoCoEncoder(nn.Module):
    """Upstream `voco_head.Swin`: the SwinUNETR-B encoder without the decoder."""

    def __init__(self):
        super().__init__()
        self.swinViT = SwinTransformer(
            in_chans=1,
            embed_dim=FEATURE_SIZE,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            spatial_dims=3,
            downsample=LegacyPatchMerging,
            use_v2=True,
        )
        self.encoder1 = basic_block(1, FEATURE_SIZE)
        self.encoder2 = basic_block(FEATURE_SIZE, FEATURE_SIZE)
        self.encoder3 = basic_block(2 * FEATURE_SIZE, 2 * FEATURE_SIZE)
        self.encoder4 = basic_block(4 * FEATURE_SIZE, 4 * FEATURE_SIZE)
        self.encoder10 = basic_block(16 * FEATURE_SIZE, 16 * FEATURE_SIZE)

    def forward(self, x: Tensor) -> list[Tensor]:
        hidden_states = self.swinViT(x, normalize=True)
        return [
            self.encoder1(x),
            self.encoder2(hidden_states[0]),
            self.encoder3(hidden_states[1]),
            self.encoder4(hidden_states[2]),
            self.encoder10(hidden_states[4]),
        ]


def basic_block(in_channels: int, out_channels: int) -> UnetrBasicBlock:
    return UnetrBasicBlock(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        res_block=True,
    )


class VoCo(nn.Module):
    name = "voco"
    embed_dim = sum(STAGE_CHANNELS)

    def __init__(self, encoder: VoCoEncoder, dense_stage: int = 3):
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

        mask = sample.get("brain_mask")
        if mask is None:
            mask = data > data.mean()
        else:
            mask = np.asarray(mask.dataobj).reshape(data.shape) > 0
        centre_voxel = np.argwhere(mask).mean(0)
        centre_world = image.affine[:3, :3] @ centre_voxel + image.affine[:3, 3]

        # RAS axes as in MONAI's Orientation (keeps any obliquity), at 1.5 mm, centred on the brain
        canonical_affine = nib.as_closest_canonical(image).affine
        directions = canonical_affine[:3, :3] / np.linalg.norm(canonical_affine[:3, :3], axis=0)
        input_affine = np.eye(4)
        input_affine[:3, :3] = directions * SPACING
        grid_centre = (np.array(IMG_SIZE) - 1) / 2
        input_affine[:3, 3] = centre_world - input_affine[:3, :3] @ grid_centre

        input_to_image = np.linalg.inv(image.affine) @ input_affine
        volume = ndimage.affine_transform(
            data, input_to_image, output_shape=IMG_SIZE, order=1, mode="constant", cval=0.0
        )
        lower, upper = np.percentile(data, INTENSITY_PERCENTILES)
        volume = (np.clip(volume, lower, upper) - lower) / max(upper - lower, 1e-6)
        return {
            "volume": torch.from_numpy(volume.astype(np.float32)),
            "input_affine": input_affine,
        }

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        volumes = torch.stack([sample["volume"] for sample in samples])[:, None]
        volumes = volumes.to(self.device)  # (B, 1, 128, 128, 128)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            stages = self.encoder(volumes)
        stages = [stage.float() for stage in stages]
        global_embeddings = torch.cat([stage.mean((2, 3, 4)) for stage in stages], dim=1)

        outputs = []
        for index, sample in enumerate(samples):
            output = EmbeddingOutput(
                global_embedding=global_embeddings[index],
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                output["dense_embedding"] = stages[self.dense_stage][index].permute(1, 2, 3, 0)
                output["dense_affine"] = sample["input_affine"]
            outputs.append(output)
        return outputs


@register_model
def voco(checkpoint: str = CHECKPOINT, revision: str = REVISION, dense_stage: int = 3) -> VoCo:
    """Load the encoder from the VoCo SSL checkpoint on the HF hub (public, Apache-2.0)."""
    path = hf_hub_download(REPO_ID, checkpoint, revision=revision)
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder = VoCoEncoder()
    encoder.load_state_dict(state)
    return VoCo(encoder, dense_stage=dense_stage)
