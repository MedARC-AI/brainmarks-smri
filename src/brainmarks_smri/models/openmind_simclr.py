"""OpenMind SimCLR (MIC-DKFZ/ResEncL-OpenMind-SimCLR): a ResEnc-L CNN encoder, SimCLR-pretrained
on brain MRI from OpenNeuro (nnssl).

Pretraining plan (`onemmiso`): 1 mm isotropic, z-score (no mask) over the nonzero bounding box
before resampling, downstream patch 160^3. We do the same, except: upstream reads images with
SimpleITK without reorienting; we reorient to RAS. We resample with trilinear interpolation
(nnU-Net uses third-order splines).
The model input is a 160^3 crop centred on the brain (mask centroid, else `data > data.mean()`),
zero padded. The global embedding concatenates the means of all 6 stage outputs, as the SimCLR
projection head does (1120 dims). The dense embedding is one stage's output, by default stage 3
(stride 8, 256 channels).
"""

from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from huggingface_hub import hf_hub_download
from scipy import ndimage

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

REPO_ID = "MIC-DKFZ/ResEncL-OpenMind-SimCLR"
REVISION = "f82b7420d77d558ae9a9d44ed8aa63da3d8c69ff"
CHECKPOINT = "checkpoint_final.pth"
IMG_SIZE = (160, 160, 160)
STAGE_CHANNELS = (32, 64, 128, 256, 320, 320)
STAGE_STRIDES = (1, 2, 4, 8, 16, 32)


class OpenMindSimCLR(nn.Module):
    name = "openmind_simclr"
    embed_dim = sum(STAGE_CHANNELS)

    def __init__(self, encoder: ResidualEncoder, dense_stage: int = 3):
        super().__init__()
        self.encoder = encoder
        self.dense_stage = dense_stage
        self.dense_dim = STAGE_CHANNELS[dense_stage]
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
        centre_world = nib.affines.apply_affine(image.affine, ndimage.center_of_mass(mask))

        # as nnssl: z-score over the nonzero bounding box, before resampling; zeros outside
        nonzero_box = tuple(
            slice(indices.min(), indices.max() + 1) for indices in np.nonzero(data != 0)
        )
        box = data[nonzero_box]
        normalized = np.zeros_like(data)
        normalized[nonzero_box] = (box - box.mean()) / max(box.std(), 1e-8)
        data = normalized

        # RAS, then 1 mm; interpolate maps output voxel i to input (i + 0.5) / scale - 0.5
        canonical = nib.as_closest_canonical(nib.Nifti1Image(data, image.affine))
        spacing = np.array(canonical.header.get_zooms()[:3], dtype=np.float64)
        volume = torch.from_numpy(np.ascontiguousarray(canonical.dataobj, dtype=np.float32))
        volume = F.interpolate(
            volume[None, None],
            scale_factor=tuple(spacing),
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )[0, 0]
        step = np.diag([*(1 / spacing), 1.0])
        step[:3, 3] = 0.5 / spacing - 0.5
        resampled_affine = canonical.affine @ step

        # crop / zero pad to IMG_SIZE around the brain centre
        centre_voxel = nib.affines.apply_affine(np.linalg.inv(resampled_affine), centre_world)
        start = np.round(centre_voxel - (np.array(IMG_SIZE) - 1) / 2).astype(int)
        crop = torch.zeros(IMG_SIZE)
        source = []
        target = []
        for axis in range(3):
            low = max(start[axis], 0)
            high = min(start[axis] + IMG_SIZE[axis], volume.shape[axis])
            source.append(slice(low, high))
            target.append(slice(low - start[axis], high - start[axis]))
        crop[tuple(target)] = volume[tuple(source)]

        shift = np.eye(4)
        shift[:3, 3] = start
        return {"volume": crop, "input_affine": resampled_affine @ shift}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        volumes = torch.stack([sample["volume"] for sample in samples])[:, None].to(self.device)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            stage_outputs = self.encoder(volumes)  # list of (B, C, X, Y, Z)
        global_embeddings = torch.cat([stage.float().mean((2, 3, 4)) for stage in stage_outputs], 1)

        outputs = []
        for index, sample in enumerate(samples):
            output = EmbeddingOutput(
                global_embedding=global_embeddings[index],
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                dense = stage_outputs[self.dense_stage][index].float()
                output["dense_embedding"] = dense.permute(1, 2, 3, 0)
                output["dense_affine"] = sample["input_affine"]
            outputs.append(output)
        return outputs


@register_model
def openmind_simclr(
    checkpoint: str = CHECKPOINT, revision: str = REVISION, dense_stage: int = 3
) -> OpenMindSimCLR:
    """Load the encoder from the nnssl checkpoint on the HF hub (public); drop the SimCLR head."""
    path = hf_hub_download(REPO_ID, checkpoint, revision=revision)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["network_weights"]
    state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    encoder = res_enc_l()
    encoder.load_state_dict(state)
    return OpenMindSimCLR(encoder, dense_stage=dense_stage)


def res_enc_l(n_blocks_per_stage: tuple[int, ...] = (1, 3, 4, 6, 6, 6)) -> ResidualEncoder:
    """The encoder of nnssl's `get_res_enc_l`, returning all 6 stage outputs."""
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
