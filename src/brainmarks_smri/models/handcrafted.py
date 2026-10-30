"""Hand-crafted features: a shallow, untrained baseline.

The recipe follows what coding agents built for our Harbor tasks: align to a template, normalize
intensity, split the brain into three intensity classes, then measure intensity and tissue amounts
in blocks at several scales.

`transform` (CPU) resamples the image onto a 2 mm grid in MNI space with the full 12-DOF
`mni_affine` and masks it with `brain_mask`. Without `mni_affine` the grid is RAS, 2 mm, centred
on the brain; without `brain_mask` the brain is `data > data.mean()`. Intensity is divided by the
brain median. A 3-class Gaussian mixture on brain intensities gives soft class maps (CSF, GM, WM on
a T1w). These are scaled by the affine's volume change, so they measure tissue amount in the
subject ("modulated", as in VBM).

`forward_embeddings` computes the features on the model's device:
- global: brain volume, mixture parameters, intensity histogram, and block means of intensity and
  the class maps at 8, 16 and 32 mm. Only blocks at least half inside the MNI152 brain mask are
  kept: blocks at the edge of the brain are rare and blow up when the probe standardizes them;
- dense, per 8 mm patch: means of the four maps, intensity SD, the maps smoothed at 4, 8 and
  16 mm, left-right differences, brain fraction at three scales, and the patch's position.

The smoothing kernels are frozen parameters, so the model moves to a device like the others.
"""

import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import platformdirs
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from torch import Tensor

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

CACHE_DIR = Path(os.getenv("BRAINMARKS_SMRI_CACHE", platformdirs.user_cache_dir("brainmarks_smri")))
MNI_MASK_URL = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz"

VOXEL_SIZE = 2.0
GRID_SIZE = (96, 112, 96)
PATCH_SIZE = (4, 4, 4)
GLOBAL_BLOCK_SIZES = (4, 8, 16)  # voxels: 8, 16 and 32 mm
SMOOTHING_SIGMAS = (2.0, 4.0, 8.0)  # voxels: 4, 8 and 16 mm
HISTOGRAM_BINS = 32
HISTOGRAM_MAX = 3.0  # in units of the brain median
NUM_CLASSES = 3
NUM_MAPS = 1 + NUM_CLASSES  # intensity and the class maps

# Model input voxel -> MNI mm: 2 mm grid centred on (0, -18, 18), which covers the MNI152 brain.
MNI_INPUT_AFFINE = np.array(
    [
        [VOXEL_SIZE, 0.0, 0.0, -95.0],
        [0.0, VOXEL_SIZE, 0.0, -129.0],
        [0.0, 0.0, VOXEL_SIZE, -77.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

NUM_SCALARS = 2 + 3 * NUM_CLASSES + NUM_CLASSES + HISTOGRAM_BINS
DENSE_DIM = NUM_MAPS + 1 + NUM_MAPS * len(SMOOTHING_SIGMAS) + 2 * NUM_MAPS + 3 + 3


class Handcrafted(nn.Module):
    name = "handcrafted"
    dense_dim = DENSE_DIM
    patch_size = PATCH_SIZE

    def __init__(self, support: np.ndarray):
        """`support`: brain fraction of each voxel of the input grid, e.g. a template brain mask."""
        super().__init__()
        support = torch.as_tensor(support, dtype=torch.float32)[None, None]
        self.embed_dim = NUM_SCALARS
        for block in GLOBAL_BLOCK_SIZES:
            kept = F.avg_pool3d(support, block)[0, 0] >= 0.5
            self.register_buffer(f"kept_blocks_{block}", kept)
            self.embed_dim += NUM_MAPS * int(kept.sum())
        kernels = []
        for sigma in SMOOTHING_SIGMAS:
            radius = int(3 * sigma)
            offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
            kernels.append(nn.Parameter(kernel / kernel.sum(), requires_grad=False))
        self.kernels = nn.ParameterList(kernels)
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

        mni_affine = sample.get("mni_affine")
        if mni_affine is None:
            # RAS, centred on the brain
            centre = nib.affines.apply_affine(image.affine, np.argwhere(mask).mean(0))
            input_affine = np.diag([VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1.0])
            input_affine[:3, 3] = centre - VOXEL_SIZE * (np.array(GRID_SIZE) - 1) / 2
            volume_scale = 1.0
        else:
            input_affine = np.linalg.inv(mni_affine) @ MNI_INPUT_AFFINE
            # subject volume per MNI volume
            volume_scale = 1.0 / abs(np.linalg.det(mni_affine[:3, :3]))

        sampling_affine = np.linalg.inv(image.affine) @ input_affine
        matrix, offset = sampling_affine[:3, :3], sampling_affine[:3, 3]
        volume = ndimage.affine_transform(data, matrix, offset, output_shape=GRID_SIZE, order=1)
        brain = ndimage.affine_transform(
            mask.astype(np.float32), matrix, offset, output_shape=GRID_SIZE, order=1
        )
        brain = brain > 0.5

        volume = np.where(brain, volume / np.median(volume[brain]), 0.0)
        values = volume[brain]

        initial_means = np.percentile(values, [10, 50, 90])[:, None]
        mixture = GaussianMixture(NUM_CLASSES, means_init=initial_means, random_state=0)
        probabilities = mixture.fit(values[:, None]).predict_proba(values[:, None])
        order = np.argsort(mixture.means_[:, 0])
        class_maps = np.zeros((NUM_CLASSES, *GRID_SIZE), dtype=np.float32)
        class_maps[:, brain] = probabilities[:, order].T * volume_scale

        voxel_volume_ml = abs(np.linalg.det(image.affine[:3, :3])) / 1000
        brain_volume_ml = mask.sum() * voxel_volume_ml
        histogram, _ = np.histogram(
            values, bins=HISTOGRAM_BINS, range=(0, HISTOGRAM_MAX), density=True
        )
        scalars = np.concatenate(
            [
                [brain_volume_ml / 1000, volume_scale],
                mixture.means_[order, 0],
                np.sqrt(mixture.covariances_[order].ravel()),
                mixture.weights_[order],
                mixture.weights_[order] * brain_volume_ml / 1000,
                histogram,
            ]
        )
        # float16 halves the copy out of the loader workers
        maps = np.concatenate([volume[None], class_maps]).astype(np.float16)
        return {
            "maps": torch.from_numpy(maps),
            "brain": torch.from_numpy(brain),
            "scalars": torch.from_numpy(scalars.astype(np.float32)),
            "input_affine": input_affine,
        }

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        device = self.device
        maps = torch.stack([sample["maps"] for sample in samples]).to(device).float()  # (B, 4, ...)
        brain = torch.stack([sample["brain"] for sample in samples]).to(device)[:, None].float()

        global_features = [torch.stack([sample["scalars"] for sample in samples]).to(device)]
        for block in GLOBAL_BLOCK_SIZES:
            kept = getattr(self, f"kept_blocks_{block}")
            global_features.append(F.avg_pool3d(maps, block)[:, :, kept].flatten(1))
        global_features = torch.cat(global_features, dim=1)

        outputs = []
        for index, sample in enumerate(samples):
            output = EmbeddingOutput(
                global_embedding=global_features[index],
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            outputs.append(output)
        if not return_dense:
            return outputs

        dense_features, brain_fraction = self.dense_features(maps, brain)
        for output, sample, dense, fraction in zip(
            outputs, samples, dense_features, brain_fraction
        ):
            dense_mask = fraction[0] > 0
            dense = dense.permute(1, 2, 3, 0)  # (X, Y, Z, D)
            output["dense_embedding"] = torch.where(dense_mask[..., None], dense, 0.0)
            output["dense_mask"] = dense_mask
            output["dense_affine"] = sample["input_affine"]
        return outputs

    def dense_features(self, maps: Tensor, brain: Tensor) -> tuple[Tensor, Tensor]:
        """(B, DENSE_DIM, X, Y, Z) features per patch, and the patch brain fraction (B, 1, X, Y, Z)."""
        patch = PATCH_SIZE[0]
        mean = F.avg_pool3d(maps, patch)
        intensity_mean = mean[:, :1]
        intensity_square = F.avg_pool3d(maps[:, :1] ** 2, patch)
        intensity_std = (intensity_square - intensity_mean**2).clamp_min(0).sqrt()

        smoothed = [F.avg_pool3d(self.smooth(maps, kernel), patch) for kernel in self.kernels]
        # left-right: the grid's first axis is MNI x, mirrored about the grid centre
        asymmetry = [mean - mean.flip(2), smoothed[1] - smoothed[1].flip(2)]

        brain_fraction = F.avg_pool3d(brain, patch)
        # depth proxies: brain fraction around the patch
        brain_smoothed = [
            F.avg_pool3d(self.smooth(brain, kernel), patch) for kernel in self.kernels[1:]
        ]

        # patch centre in [-1, 1] along each grid axis
        axes = [torch.linspace(-1, 1, size // patch, device=maps.device) for size in GRID_SIZE]
        position = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        position = position.expand(maps.shape[0], -1, -1, -1, -1)

        features = torch.cat(
            [mean, intensity_std, *smoothed, *asymmetry, brain_fraction, *brain_smoothed, position],
            dim=1,
        )
        return features, brain_fraction

    def smooth(self, volumes: Tensor, kernel: Tensor) -> Tensor:
        """Gaussian smoothing of (B, C, X, Y, Z) volumes, one separable pass per axis."""
        channels = volumes.shape[1]
        radius = kernel.shape[0] // 2
        for axis in range(3):
            shape = [1, 1, 1]
            shape[axis] = kernel.shape[0]
            weight = kernel.reshape(1, 1, *shape).expand(channels, 1, *shape)
            padding = [0, 0, 0]
            padding[axis] = radius
            volumes = F.conv3d(volumes, weight, padding=padding, groups=channels)
        return volumes


@register_model
def handcrafted() -> Handcrafted:
    """Support from the MNI152NLin6Asym brain mask (TemplateFlow), downloaded into `CACHE_DIR`."""
    path = CACHE_DIR / "templates" / MNI_MASK_URL.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(MNI_MASK_URL, str(path), progress=False)
    template = nib.load(path)
    sampling_affine = np.linalg.inv(template.affine) @ MNI_INPUT_AFFINE
    support = ndimage.affine_transform(
        np.asarray(template.dataobj, dtype=np.float32),
        sampling_affine[:3, :3],
        sampling_affine[:3, 3],
        output_shape=GRID_SIZE,
        order=1,
    )
    return Handcrafted(support)
