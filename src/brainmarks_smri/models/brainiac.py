"""BrainIAC (Tak et al. 2026): a SimCLR-pretrained MONAI ViT-B over 96^3 volumes.

Upstream preprocessing registers each scan rigidly to its own template (`temp_head.nii.gz`),
skull-strips it with HD-BET, resizes the whole array to 96^3 and z-scores the nonzero voxels.
Here the registration is the rigid part of `mni_affine` composed with a fixed MNI -> template
affine, and `brain_mask` stands in for HD-BET. N4 bias correction is skipped. Without
`mni_affine` the image is only reoriented to the template's LAS axes. Upstream never reorients,
so a scan stored in another orientation is resized along the wrong axes.

The 16^3 patches give a 6x6x6 token grid. Upstream's "CLS token" is `features[0][:, 0]`, but
MONAI's `ViT(classification=False)` has no CLS token, so that is the corner patch. We use the
mean of all 216 tokens instead.
"""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai import transforms
from monai.networks.nets import ViT
from scipy import ndimage

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

IMG_SIZE = (96, 96, 96)
PATCH_SIZE = (16, 16, 16)

# Header of upstream src/preprocessing/atlases/temp_head.nii.gz (1 mm, LAS)
TEMPLATE_SHAPE = (170, 206, 162)
TEMPLATE_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 84.0],
        [0.0, 1.0, 0.0, -119.0],
        [0.0, 0.0, 1.0, -69.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# MNI152NLin6Asym world mm -> template world mm. 12-DOF SimpleITK registration of the
# TemplateFlow 1 mm T1w to temp_head (Mattes MI). Head Dice 0.945. The template is smaller than
# MNI (scales ~0.91, 0.92, 0.85), but only the rigid part of the composite is used.
MNI_TO_TEMPLATE = np.array(
    [
        [0.912604, -0.001763, 0.004258, -0.792807],
        [-0.004213, 0.92376, 0.024093, -1.965898],
        [0.002228, -0.034612, 0.845302, 1.588296],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class BrainIAC(nn.Module):
    name = "brainiac"
    embed_dim = 768
    patch_size = PATCH_SIZE

    def __init__(self):
        super().__init__()
        # cl: I don't see why break the pattern of neurovfm, neurojepa to construct the encoder in
        # the model fn?
        self.backbone = ViT(
            in_channels=1,
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
        )
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

        mni_affine = sample.get("mni_affine")
        if mni_affine is None:
            image = nib.Nifti1Image(data, image.affine)
            las = nib.orientations.axcodes2ornt(("L", "A", "S"))
            image = image.as_reoriented(
                nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine), las)
            )
            data = np.asarray(image.dataobj, dtype=np.float32)
            grid_affine = image.affine
        else:
            template_to_image = rigid_template_to_image(MNI_TO_TEMPLATE @ mni_affine)
            grid_affine = template_to_image @ TEMPLATE_AFFINE
            # template voxel -> image voxel, linear like upstream's SimpleITK resample
            voxel_map = np.linalg.inv(image.affine) @ grid_affine
            data = ndimage.affine_transform(
                data, voxel_map, output_shape=TEMPLATE_SHAPE, order=1, mode="constant", cval=0.0
            )

        volume = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))[None]
        volume = upstream_transform()(volume)

        # Resize (align_corners=False) maps output voxel j to input coordinate (j + 0.5) * s - 0.5
        scale = np.array(data.shape) / np.array(IMG_SIZE)
        resize_affine = np.eye(4)
        resize_affine[:3, :3] = np.diag(scale)
        resize_affine[:3, 3] = 0.5 * scale - 0.5
        return {"volume": torch.as_tensor(volume), "input_affine": grid_affine @ resize_affine}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        batch = torch.stack([sample["volume"] for sample in samples]).to(self.device)
        tokens, _hidden_states = self.backbone(batch)  # (B, 216, 768), after the final norm

        grid = tuple(size // patch for size, patch in zip(IMG_SIZE, PATCH_SIZE))
        outputs = []
        for sample, sample_tokens in zip(samples, tokens):
            output = EmbeddingOutput(
                global_embedding=sample_tokens.mean(0),
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                # the conv patch embedding flattens in C order, so tokens are (x y z) ordered
                output["dense_embedding"] = sample_tokens.reshape(*grid, self.embed_dim)
                output["dense_affine"] = sample["input_affine"]
            outputs.append(output)
        return outputs


@register_model
def brainiac(checkpoint: str | Path) -> BrainIAC:
    """Load the upstream `BrainIAC.ckpt` (Lightning checkpoint, from their Dropbox folder)."""
    # cl: any way we can download and cache the checkpoint from dropbox in this code here?
    # feel free to set up a BRAINMARKS_SMRI_CACHE, which we will use for checkpoints and datasets.
    model = BrainIAC()
    lightning_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = lightning_checkpoint.get("state_dict", lightning_checkpoint)
    prefix = "backbone."
    state_dict = {
        key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
    }
    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    # MONAI >= 1.4 adds cross-attention norms that are unused without cross-attention
    missing = [key for key in missing if ".norm_cross_attn." not in key]
    if missing or unexpected:
        raise RuntimeError(f"missing keys {missing}, unexpected keys {unexpected}")
    return model


# cl: why are we doing this 12dof affine to rigid conversion every time rather than just once
# offline?
def rigid_template_to_image(image_to_template: np.ndarray) -> np.ndarray:
    """The rigid part of an image world -> template world affine, inverted.

    Upstream registers rigidly, so head size is kept. We keep the rotation from the polar
    decomposition and fix the image point that the full affine sends to the template centre.
    """
    u, _, vt = np.linalg.svd(image_to_template[:3, :3])
    rotation = u @ vt
    template_centre = TEMPLATE_AFFINE @ np.array([*((np.array(TEMPLATE_SHAPE) - 1) / 2), 1.0])
    image_centre = np.linalg.inv(image_to_template) @ template_centre
    template_to_image = np.eye(4)
    template_to_image[:3, :3] = rotation.T
    template_to_image[:3, 3] = image_centre[:3] - rotation.T @ template_centre[:3]
    return template_to_image


def upstream_transform() -> transforms.Compose:
    """Upstream `get_validation_transform` after loading: resize to 96^3, z-score nonzero voxels."""
    return transforms.Compose(
        [
            transforms.Resize(spatial_size=IMG_SIZE, mode="trilinear"),
            transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
        ]
    )
