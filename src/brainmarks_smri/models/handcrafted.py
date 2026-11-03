"""Hand-crafted features: a shallow, untrained baseline.

Images are affinely aligned to MNI space at 2 mm and normalized by the brain median intensity. A
3-class Gaussian mixture on brain intensities gives soft tissue maps (CSF, GM, WM on a T1w), scaled
by the affine's volume change so they measure tissue amount in the subject, as in VBM. Features are
block means of intensity and tissue amount at a few scales, plus a few whole-brain statistics
(global), and local statistics per 8 mm patch (dense).
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
from nibabel.affines import apply_affine
from scipy import ndimage
from scipy.special import softmax
from torch import Tensor

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model

CACHE_DIR = Path(os.getenv("BRAINMARKS_SMRI_CACHE", platformdirs.user_cache_dir("brainmarks_smri")))
MNI_MASK_URL = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz"

VOXEL_SIZE = 2.0
GRID_SIZE = (96, 112, 96)
PATCH_SIZE = (4, 4, 4)
GLOBAL_BLOCK_SIZES = (8, 16)  # voxels: 16 and 32 mm
SMOOTHING_SIGMAS = (2.0, 4.0, 8.0)  # voxels: 4, 8 and 16 mm
HISTOGRAM_BINS = 32
HISTOGRAM_MAX = 3.0  # in units of the brain median
GMM_BINS = 300
NUM_CLASSES = 3
INITIAL_QUANTILES = (0.1, 0.5, 0.9)
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
    patch_size = PATCH_SIZE

    def __init__(self, support: np.ndarray):
        """`support`: brain fraction of each voxel of the input grid, e.g. a template brain mask."""
        super().__init__()
        support = torch.as_tensor(support, dtype=torch.float32)[None, None]
        # masks of valid blocks that contain brain data
        kept_blocks = []
        for block in GLOBAL_BLOCK_SIZES:
            kept = F.avg_pool3d(support, block)[0, 0] >= 0.5
            kept_blocks.append(nn.Parameter(kept, requires_grad=False))
        self.kept_blocks = nn.ParameterList(kept_blocks)
        self.embed_dim = NUM_SCALARS + NUM_MAPS * sum(int(kept.sum()) for kept in kept_blocks)
        self.smoothing = nn.ModuleList(GaussianSmoothing(sigma) for sigma in SMOOTHING_SIGMAS)
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
            centre = apply_affine(image.affine, ndimage.center_of_mass(mask))
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

        # intensity relative to the brain median, comparable across scanners
        volume = np.where(brain, volume / np.median(volume[brain]), 0.0)
        values = volume[brain]

        # tissue amount in the subject per grid voxel
        probabilities, means, stds, weights = intensity_classes(
            values, NUM_CLASSES, INITIAL_QUANTILES, GMM_BINS, HISTOGRAM_MAX
        )
        class_maps = np.zeros((NUM_CLASSES, *GRID_SIZE), dtype=np.float32)
        class_maps[:, brain] = probabilities.T * volume_scale

        voxel_volume_ml = abs(np.linalg.det(image.affine[:3, :3])) / 1000
        brain_volume_ml = mask.sum() * voxel_volume_ml
        histogram, _ = np.histogram(
            values, bins=HISTOGRAM_BINS, range=(0, HISTOGRAM_MAX), density=True
        )
        scalars = np.concatenate(
            [
                [brain_volume_ml / 1000, volume_scale],
                means,
                stds,
                weights,
                weights * brain_volume_ml / 1000,
                histogram,
            ]
        )
        maps = np.concatenate([volume[None], class_maps]).astype(np.float32)
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
        maps = torch.stack([sample["maps"] for sample in samples]).to(device)  # (B, 4, X, Y, Z)
        brain = torch.stack([sample["brain"] for sample in samples]).to(device)[:, None].float()

        global_features = [torch.stack([sample["scalars"] for sample in samples]).to(device)]
        for block, kept in zip(GLOBAL_BLOCK_SIZES, self.kept_blocks):
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
        # mean intensity and tissue composition per patch
        mean = F.avg_pool3d(maps, patch)
        intensity_mean = mean[:, :1]
        intensity_square = F.avg_pool3d(maps[:, :1] ** 2, patch)
        intensity_std = (intensity_square - intensity_mean**2).clamp_min(0).sqrt()

        # context around the patch at 4, 8 and 16 mm
        smoothed = [F.avg_pool3d(smoothing(maps), patch) for smoothing in self.smoothing]
        # difference from the mirror patch, e.g. for lesions
        # the grid's first axis is MNI x, mirrored about the grid centre
        asymmetry = [mean - mean.flip(2), smoothed[1] - smoothed[1].flip(2)]

        brain_fraction = F.avg_pool3d(brain, patch)
        # smooth the brain mask as a way to estimate "depth", i.e. distance from boundary
        brain_smoothed = [F.avg_pool3d(smoothing(brain), patch) for smoothing in self.smoothing[1:]]

        # position of each patch in [-1, 1] along each grid axis
        axes = [torch.linspace(-1, 1, size // patch, device=maps.device) for size in GRID_SIZE]
        position = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        position = position.expand(maps.shape[0], -1, -1, -1, -1)

        features = torch.cat(
            [mean, intensity_std, *smoothed, *asymmetry, brain_fraction, *brain_smoothed, position],
            dim=1,
        )
        return features, brain_fraction


class GaussianSmoothing(nn.Module):
    """Gaussian smoothing of (B, C, X, Y, Z) volumes, one separable pass per axis, zero padded."""

    def __init__(self, sigma: float):
        super().__init__()
        radius = int(3 * sigma)
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        self.kernel = nn.Parameter(kernel / kernel.sum(), requires_grad=False)

    def forward(self, volumes: Tensor) -> Tensor:
        channels = volumes.shape[1]
        size = self.kernel.shape[0]
        for axis in range(3):
            shape = [1, 1, 1]
            shape[axis] = size
            weight = self.kernel.reshape(1, 1, *shape).expand(channels, 1, *shape)
            padding = [0, 0, 0]
            padding[axis] = size // 2
            volumes = F.conv3d(volumes, weight, padding=padding, groups=channels)
        return volumes


def intensity_classes(
    values: np.ndarray,
    num_classes: int = 3,
    initial_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    num_bins: int = 300,
    max_value: float = 3.0,
    max_iterations: int = 2000,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a Gaussian mixture to 1D intensities, e.g. the brain voxels of a T1w (CSF, GM, WM).

    EM runs on a histogram of the values over [0, max_value], weighted by the bin counts; values
    outside are clipped into the end bins. Returns the class probabilities of each value (N, K) and the class means, SDs
    and weights (K,), with classes ordered dark to bright.
    """
    clipped = np.clip(values, 0, max_value)
    counts, edges = np.histogram(clipped, num_bins, (0, max_value))
    centres = (edges[:-1] + edges[1:]) / 2

    # start from the partition of the bins by the nearest initial mean
    means = np.quantile(clipped, initial_quantiles)
    labels = np.abs(centres[:, None] - means).argmin(axis=1)
    weights = np.zeros(num_classes)
    variances = np.zeros(num_classes)
    for class_id in range(num_classes):
        class_counts = counts[labels == class_id]
        class_centres = centres[labels == class_id]
        weights[class_id] = class_counts.sum() / counts.sum()
        variances[class_id] = np.average(
            (class_centres - means[class_id]) ** 2, weights=class_counts
        )

    for _ in range(max_iterations):
        log_density = (
            -0.5 * (centres[:, None] - means) ** 2 / variances
            - 0.5 * np.log(variances)
            + np.log(weights)
        )
        responsibilities = softmax(log_density, axis=1) * counts[:, None]
        totals = responsibilities.sum(axis=0)
        new_means = (responsibilities * centres[:, None]).sum(axis=0) / totals
        variances = (responsibilities * (centres[:, None] - new_means) ** 2).sum(axis=0) / totals
        variances = variances + 1e-6
        weights = totals / counts.sum()
        converged = np.abs(new_means - means).max() < tolerance
        means = new_means
        if converged:
            break

    log_density = (
        -0.5 * (values[:, None] - means) ** 2 / variances
        - 0.5 * np.log(variances)
        + np.log(weights)
    )
    probabilities = softmax(log_density, axis=1)
    order = np.argsort(means)
    return probabilities[:, order], means[order], np.sqrt(variances[order]), weights[order]


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
