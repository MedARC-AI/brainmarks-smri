"""Brain masks and affine registration to MNI152NLin6Asym."""

import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import platformdirs
from scipy import ndimage

CACHE_DIR = Path(
    os.getenv("BRAINMARKS_SMRI_CACHE", platformdirs.user_cache_dir("brainmarks_smri")),
    "preprocessing",
)

FREESURFER_COMMIT = "cf4bccf24875a47245b4df9fc9372d0f1c3d784f"
SYNTHSTRIP_SCRIPT_URL = f"https://raw.githubusercontent.com/freesurfer/freesurfer/{FREESURFER_COMMIT}/mri_synthstrip/mri_synthstrip"
SYNTHSTRIP_WEIGHTS_URL = (
    "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt"
)
MNI_TEMPLATE_URL = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz"
MNI_TEMPLATE_MASK_URL = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz"


def cached_download(url: str, cache_dir: Path) -> Path:
    path = cache_dir / url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".tmp-{os.getpid()}-{path.name}")
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)
    return path


def mask_image_like(mask: np.ndarray, image: nib.Nifti1Image) -> nib.Nifti1Image:
    """uint8 mask image on the grid and header of `image`."""
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(mask.astype(np.uint8), image.affine, header)


def synthstrip(image: nib.Nifti1Image) -> nib.Nifti1Image:
    """SynthStrip brain mask."""
    script = cached_download(SYNTHSTRIP_SCRIPT_URL, CACHE_DIR)
    weights = cached_download(SYNTHSTRIP_WEIGHTS_URL, CACHE_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        nib.save(image, f"{tmp}/image.nii")
        command = [
            sys.executable, script,
            "-i", f"{tmp}/image.nii",
            "-m", f"{tmp}/mask.nii",
            "--model", weights,
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SynthStrip exit {result.returncode}: {result.stderr[-2000:]}")
        mask = np.asanyarray(nib.load(f"{tmp}/mask.nii").dataobj) > 0
    return mask_image_like(mask, image)


def threshold_mask(
    image: nib.Nifti1Image,
    relative_threshold: float = 1e-3,
    percentile: float = 99,
    largest_component: bool = True,
    fill_holes: bool = True,
) -> nib.Nifti1Image:
    """Foreground mask by a threshold relative to the image intensity."""
    data = np.asanyarray(image.dataobj)
    mask = data > relative_threshold * np.percentile(data[data != 0], percentile)
    if largest_component:
        labels, _ = ndimage.label(mask)
        mask = labels == np.bincount(labels.ravel())[1:].argmax() + 1
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    return mask_image_like(mask, image)


def nifti_to_ants(image: nib.Nifti1Image) -> ants.ANTsImage:
    """nibabel image (RAS) -> ANTs image (LPS), so both agree on world coordinates."""
    lps_affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ image.affine
    spacing = np.linalg.norm(lps_affine[:3, :3], axis=0)
    return ants.from_numpy(
        image.get_fdata(dtype=np.float32),
        origin=lps_affine[:3, 3].tolist(),
        spacing=spacing.tolist(),
        direction=lps_affine[:3, :3] / spacing,
    )


def ants_to_ras_affine(parameters: np.ndarray, center: np.ndarray) -> np.ndarray:
    """ANTs registration transform to nibabel-style RAS affine."""
    parameters = np.asarray(parameters, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    matrix = parameters[:9].reshape(3, 3)
    translation = parameters[9:]

    template_to_image_lps = np.eye(4)
    template_to_image_lps[:3, :3] = matrix
    template_to_image_lps[:3, 3] = translation + center - matrix @ center
    lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    template_to_image_ras = lps @ template_to_image_lps @ lps
    return np.linalg.inv(template_to_image_ras)


def mni_affine_registration(
    image: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    seed: int = 1,
) -> np.ndarray:
    """Image RAS mm -> MNI RAS mm affine: ANTs 12-DOF, Mattes MI, masked image to template
    brain. ANTs treats seed 0 as unseeded."""
    template = nib.load(cached_download(MNI_TEMPLATE_URL, CACHE_DIR))
    masked_data = image.get_fdata(dtype=np.float32) * (np.asanyarray(mask.dataobj) > 0)
    masked_image = nib.Nifti1Image(masked_data, image.affine)
    ants.config._random_seed = seed
    with tempfile.TemporaryDirectory() as tmp:
        registration = ants.registration(
            fixed=nifti_to_ants(template),
            moving=nifti_to_ants(masked_image),
            type_of_transform="Affine",
            aff_metric="mattes",
            outprefix=f"{tmp}/",
        )
        transform = ants.read_transform(registration["fwdtransforms"][0])
    return ants_to_ras_affine(transform.parameters, transform.fixed_parameters)


def mni_affine_qc(mask: nib.Nifti1Image, mni_affine: np.ndarray) -> dict:
    """Brain volume (L), linear scale to MNI, Dice with the template brain mask."""
    mask_data = np.asanyarray(mask.dataobj) > 0
    template_mask = nib.load(cached_download(MNI_TEMPLATE_MASK_URL, CACHE_DIR))
    template_to_mask_voxels = (
        np.linalg.inv(mask.affine) @ np.linalg.inv(mni_affine) @ template_mask.affine
    )
    mask_in_template = ndimage.affine_transform(
        mask_data.astype(np.float32),
        template_to_mask_voxels,
        order=1,
        output_shape=template_mask.shape,
    )
    mask_in_template = mask_in_template > 0.5
    template_brain = np.asanyarray(template_mask.dataobj) > 0
    overlap = (mask_in_template & template_brain).sum()
    dice = 2 * overlap / (mask_in_template.sum() + template_brain.sum())
    scale = np.linalg.det(mni_affine[:3, :3]) ** (1 / 3)
    volume = mask_data.sum() * np.prod(mask.header.get_zooms()[:3]) / 1e6
    return {
        "brain_volume": round(volume, 4),
        "mni_scale": round(scale, 4),
        "template_dice": round(dice, 4),
    }


def rigid_from_affine(affine: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Rigid part of `affine`: the rotation from its polar decomposition, mapping `anchor` where
    `affine` does."""
    u, _, vt = np.linalg.svd(affine[:3, :3])
    rotation = u @ vt
    rigid = np.eye(4)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = nib.affines.apply_affine(affine, anchor) - rotation @ anchor
    return rigid


def resample_to_template(
    image: nib.Nifti1Image,
    template: nib.Nifti1Image,
    affine: np.ndarray,
    order: int = 1,
    anti_aliasing: bool = True,
) -> nib.Nifti1Image:
    """Image resampled onto the template grid; `affine` maps image world mm to template world mm.

    With `anti_aliasing`, image axes finer than the template spacing are Gaussian smoothed first.
    """
    data = image.get_fdata(dtype=np.float32)
    if anti_aliasing:
        factors = min(template.header.get_zooms()[:3]) / np.array(image.header.get_zooms()[:3])
        sigmas = np.maximum(0, (factors - 1) / 2)
        if sigmas.any():
            data = ndimage.gaussian_filter(data, sigmas)
    template_to_image_voxels = np.linalg.inv(image.affine) @ np.linalg.inv(affine) @ template.affine
    data = ndimage.affine_transform(
        data, template_to_image_voxels, order=order, output_shape=template.shape
    )
    return nib.Nifti1Image(data, template.affine)
