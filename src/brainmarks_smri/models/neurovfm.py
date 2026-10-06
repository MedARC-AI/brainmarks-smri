"""NeuroVFM (mlinslab/neurovfm-encoder): a Vol-JEPA-pretrained ViT-B over 4x16x16 patches.

Built for raw clinical scans, so `brain_mask` and `mni_affine` are ignored. Each volume is
reoriented, resampled to 1x1 mm in-plane and 4 mm through-plane, and the slice axis is moved
first. Background tokens are dropped, so the token count varies per scan: the dense grid is
masked and the global embedding is the mean of the foreground tokens. A token covers a 16 mm
cube.
"""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from neurovfm.data.preprocess import prepare_for_inference, tokenize_volume
from neurovfm.data.utils import preprocess_image
from neurovfm.pipelines.encoder import load_encoder
from neurovfm.systems.utils import NormalizationModule

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

REPO_ID = "mlinslab/neurovfm-encoder"
REVISION = "d5194fc70a162185f8ef062e362bd522a35312a9"
PATCH_SIZE = (4, 16, 16)

# `transpose_to_dhw` moves the 4 mm axis to the front of SimpleITK's (z, y, x) array. These
# are the SimpleITK (x, y, z) index axes behind the resulting (D, H, W) axes, per `view`.
SITK_AXES = {0: [2, 1, 0], 1: [1, 2, 0], 2: [0, 1, 2]}

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


class NeuroVFM(nn.Module):
    name = "neurovfm"
    embed_dim = 768
    patch_size = PATCH_SIZE

    def __init__(self, backbone: nn.Module, norm_module: NormalizationModule):
        super().__init__()
        self.backbone = backbone
        self.norm_module = norm_module
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def transform(self, sample: ImageInput) -> dict[str, Any]:
        """Upstream `StudyPreprocessor.load_study` for one in-memory MRI volume.

        Upstream reads a file with SimpleITK. `test_neurovfm.py` checks this against it.
        """
        image_sitk = preprocess_image(nifti_to_sitk(sample["image"]))
        arrays, background_mask, view = prepare_for_inference(image_sitk, mode="mri")
        volume = arrays[0]  # (D, H, W), values in [0, 1]
        # Upstream drops every patch with any background voxel (at or below the 10th
        # percentile), as in pretraining. On skull-stripped scans this loses the outer ~16-32 mm
        # of brain.
        tokens, coords, _ = tokenize_volume(
            volume, background_mask, patch_size=PATCH_SIZE, remove_background=True
        )
        # Model input voxel (d, h, w) -> SimpleITK index (x, y, z) -> RAS world mm
        permutation = np.zeros((4, 4))
        permutation[SITK_AXES[view], [0, 1, 2]] = 1
        permutation[3, 3] = 1
        input_affine = LPS_TO_RAS @ sitk_affine(image_sitk) @ permutation
        grid_shape = tuple(size // patch for size, patch in zip(volume.shape, PATCH_SIZE))
        return {
            "tokens": torch.from_numpy(tokens).float(),
            "coords": torch.from_numpy(coords).long(),
            "grid_shape": grid_shape,
            "input_affine": input_affine,
        }

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        lengths = [len(sample["tokens"]) for sample in samples]
        cu_seqlens = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32, device=self.device)
        tokens = torch.cat([sample["tokens"] for sample in samples]).to(self.device)
        coords = torch.cat([sample["coords"] for sample in samples]).to(self.device)
        modes = ["mri"] * len(samples)
        paths = [""] * len(samples)  # upstream only reads these to pick a CT window
        tokens = self.norm_module.normalize(tokens, modes, paths, cu_seqlens=cu_seqlens)

        # Upstream builds the attention projections in bf16 and always runs under bf16 autocast
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            embeddings = self.backbone(
                tokens, coords, cu_seqlens=cu_seqlens, max_seqlen=max(lengths), use_flash_attn=False
            )
        embeddings = embeddings.float()  # (N_total, 768), sequences packed back to back

        outputs = []
        for sample, sample_embeddings in zip(samples, embeddings.split(lengths)):
            output = EmbeddingOutput(
                global_embedding=sample_embeddings.mean(0),
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                grid_shape = sample["grid_shape"]
                depth_index, height_index, width_index = sample["coords"].to(self.device).unbind(1)
                dense = sample_embeddings.new_zeros(*grid_shape, self.embed_dim)
                dense[depth_index, height_index, width_index] = sample_embeddings
                mask = torch.zeros(grid_shape, dtype=torch.bool, device=self.device)
                mask[depth_index, height_index, width_index] = True
                output["dense_embedding"] = dense
                output["dense_affine"] = sample["input_affine"]
                output["dense_mask"] = mask
            outputs.append(output)
        return outputs


@register_model
def neurovfm(checkpoint: str | Path = REPO_ID, revision: str = REVISION) -> NeuroVFM:
    """Load with upstream `load_encoder`, from a local dir or an HF repo (gated; `HF_TOKEN`)."""
    if not Path(checkpoint).is_dir():
        checkpoint = snapshot_download(str(checkpoint), revision=revision)
    pipeline, _ = load_encoder(str(checkpoint), device="cpu")
    return NeuroVFM(pipeline.model, pipeline.norm_module)


def nifti_to_sitk(image: nib.Nifti1Image) -> sitk.Image:
    """A nifti as a SimpleITK image, with the RAS affine converted to LPS origin/direction."""
    lps = LPS_TO_RAS @ image.affine
    # read spacing from header to match the ITK reader exactly. affine column norms can
    # be off by float noise, which breaks the "all spacings equal -> axial" check.
    spacing = np.array(image.header.get_zooms()[:3], dtype=np.float64)
    # keep the original dtype rather than cast to float32.
    # ITK resamples int16 images in int16, which results in rounding.
    data = np.asanyarray(image.dataobj)
    if data.dtype == bool:
        data = data.astype(np.uint8)
    if data.dtype == np.float64 and image.get_data_dtype() != np.float64:
        data = data.astype(np.float32)  # ITK reads scaled integer data as float32
    out = sitk.GetImageFromArray(np.ascontiguousarray(data.transpose(2, 1, 0)))
    out.SetSpacing(spacing.tolist())
    # unit axis directions: normalize the affine columns by their own length
    direction = lps[:3, :3] / np.linalg.norm(lps[:3, :3], axis=0)
    out.SetDirection(direction.ravel().tolist())
    out.SetOrigin(lps[:3, 3].tolist())
    return out


def sitk_affine(image: sitk.Image) -> np.ndarray:
    """A SimpleITK image's (x, y, z) index -> LPS world mm affine."""
    affine = np.eye(4)
    direction = np.array(image.GetDirection()).reshape(3, 3)
    affine[:3, :3] = direction * np.array(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    return affine
