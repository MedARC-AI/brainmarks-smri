"""Brain mask and affine to MNI152NLin6Asym for one 3D image."""

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import platformdirs
from scipy import ndimage

CACHE_DIR = Path(os.getenv("BRAINMARKS_SMRI_CACHE", platformdirs.user_cache_dir("brainmarks_smri")))

FREESURFER_COMMIT = "cf4bccf24875a47245b4df9fc9372d0f1c3d784f"
SYNTHSTRIP_SCRIPT = (
    f"https://raw.githubusercontent.com/freesurfer/freesurfer/{FREESURFER_COMMIT}/mri_synthstrip/mri_synthstrip",
    "291c253ab2f7c0cbcb84afa769ae10909549537e727db5b92d268754b7cc7871",
)
SYNTHSTRIP_WEIGHTS = (
    "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
    "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33",
)
TEMPLATE = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz",
    "72f8edc99d7e696da7717db20b73897d8da5909d7f30b1c9a7e2a79e9f64ff2f",
)
TEMPLATE_MASK = (
    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz",
    "6d540075dac093fa5070a929dc5ea52795d54cc9ab57fe1272086dc70314b428",
)
# ANTs treats seed 0 as "no seed".
ANTS_SEED = 1

# QC warning ranges (template brain mask Dice, linear scale to MNI, brain volume in litres)
MIN_DICE = 0.9
SCALE_RANGE = (0.9, 1.4)
VOLUME_RANGE = (0.8, 2.2)

MASK_SUFFIX = "_mask.nii.gz"
# 4x4 text matrix (np.loadtxt): image world mm (RAS) -> MNI152NLin6Asym mm (RAS).
AFFINE_SUFFIX = "_mni.txt"


@dataclass(frozen=True)
class Resources:
    synthstrip_script: Path
    synthstrip_weights: Path
    template: Path
    template_mask: Path


def fetch(url: str, sha256: str) -> Path:
    path = CACHE_DIR / "preprocess" / url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".tmp-{os.getpid()}-{path.name}")
        urllib.request.urlretrieve(url, tmp_path)
        digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        if digest != sha256:
            raise RuntimeError(f"{url}: sha256 {digest} != {sha256}")
        os.replace(tmp_path, path)
    return path


def fetch_resources() -> Resources:
    return Resources(
        synthstrip_script=fetch(*SYNTHSTRIP_SCRIPT),
        synthstrip_weights=fetch(*SYNTHSTRIP_WEIGHTS),
        template=fetch(*TEMPLATE),
        template_mask=fetch(*TEMPLATE_MASK),
    )


def load_image(path: Path) -> nib.Nifti1Image:
    image = nib.load(path)
    # Some DWI/ADC are stored as 4D with a single volume.
    if image.ndim == 4 and image.shape[3] == 1:
        image = image.slicer[..., 0]
    assert image.ndim == 3, image.shape
    return image


def synthstrip(image: nib.Nifti1Image, resources: Resources) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        nib.save(image, f"{tmp}/image.nii")
        command = [
            sys.executable, resources.synthstrip_script,
            "-i", f"{tmp}/image.nii",
            "-m", f"{tmp}/mask.nii",
            "--model", resources.synthstrip_weights,
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SynthStrip exit {result.returncode}: {result.stderr[-2000:]}")
        return np.asanyarray(nib.load(f"{tmp}/mask.nii").dataobj) > 0


def brain_mask(
    image: nib.Nifti1Image,
    method: str,
    resources: Resources,
    provided_mask_path: Path | None = None,
) -> np.ndarray:
    """Brain mask on the image grid.

    Methods: `synthstrip`; `nonzero` for skull-stripped images; `threshold` for skull-stripped
    images with a noisy background; `provided` loads `provided_mask_path`.
    """
    if method == "synthstrip":
        return synthstrip(image, resources)
    if method == "nonzero":
        return np.asanyarray(image.dataobj) != 0
    if method == "threshold":
        data = np.asanyarray(image.dataobj)
        foreground = data > 1e-3 * np.percentile(data[data != 0], 99)
        labels, _ = ndimage.label(foreground)
        largest = labels == np.bincount(labels.ravel())[1:].argmax() + 1
        return ndimage.binary_fill_holes(largest)
    if method == "provided":
        provided = nib.load(provided_mask_path)
        assert np.allclose(provided.affine, image.affine) and provided.shape == image.shape
        return np.asanyarray(provided.dataobj) > 0
    raise ValueError(f"unknown mask method {method!r}")


def to_ants(data: np.ndarray, affine: np.ndarray) -> ants.ANTsImage:
    """nibabel array + RAS affine -> ANTs image (LPS), so both agree on world coordinates."""
    lps_affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
    spacing = np.linalg.norm(lps_affine[:3, :3], axis=0)
    return ants.from_numpy(
        data.astype(np.float32),
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


def register_to_mni(image: nib.Nifti1Image, mask: np.ndarray, resources: Resources) -> np.ndarray:
    """Image RAS mm -> MNI RAS mm affine: ANTs 12-DOF, Mattes MI, masked image to template brain."""
    template = nib.load(resources.template)
    masked_data = image.get_fdata(dtype=np.float32) * mask
    ants.config._random_seed = ANTS_SEED
    with tempfile.TemporaryDirectory() as tmp:
        registration = ants.registration(
            fixed=to_ants(template.get_fdata(dtype=np.float32), template.affine),
            moving=to_ants(masked_data, image.affine),
            type_of_transform="Affine",
            aff_metric="mattes",
            outprefix=f"{tmp}/",
        )
        transform = ants.read_transform(registration["fwdtransforms"][0])
    return ants_to_ras_affine(transform.parameters, transform.fixed_parameters)


def quality_check(
    mask_image: nib.Nifti1Image, mni_affine: np.ndarray, resources: Resources
) -> dict:
    """Brain volume (L), linear scale to MNI, Dice with the template brain mask, and warnings."""
    mask = np.asanyarray(mask_image.dataobj) > 0
    template_mask = nib.load(resources.template_mask)
    template_to_mask_voxels = (
        np.linalg.inv(mask_image.affine) @ np.linalg.inv(mni_affine) @ template_mask.affine
    )
    mask_in_template = ndimage.affine_transform(
        mask.astype(np.float32), template_to_mask_voxels, order=1, output_shape=template_mask.shape
    )
    mask_in_template = mask_in_template > 0.5
    template_brain = np.asanyarray(template_mask.dataobj) > 0
    overlap = (mask_in_template & template_brain).sum()
    dice = 2 * overlap / (mask_in_template.sum() + template_brain.sum())
    scale = np.linalg.det(mni_affine[:3, :3]) ** (1 / 3)
    volume = mask.sum() * np.prod(mask_image.header.get_zooms()[:3]) / 1e6

    warnings = []
    if dice < MIN_DICE:
        warnings.append("dice")
    if not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
        warnings.append("scale")
    if not VOLUME_RANGE[0] <= volume <= VOLUME_RANGE[1]:
        warnings.append("volume")
    return {
        "brain_volume": round(volume, 4),
        "mni_scale": round(scale, 4),
        "template_dice": round(dice, 4),
        "warnings": ",".join(warnings) or "n/a",
    }


def preprocess_image(
    image_path: Path,
    stem: Path,
    mask_method: str,
    resources: Resources,
    provided_mask_path: Path | None = None,
) -> dict:
    """Write `<stem>_mask.nii.gz` and `<stem>_mni.txt` for one image and return its QC.

    Finished images (the affine exists) are not recomputed, only checked.
    """
    mask_path = Path(f"{stem}{MASK_SUFFIX}")
    affine_path = Path(f"{stem}{AFFINE_SUFFIX}")

    if not affine_path.exists():
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = load_image(image_path)
        mask = brain_mask(image, mask_method, resources, provided_mask_path)
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        tmp_mask_path = mask_path.with_name(f".tmp-{mask_path.name}")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine, header), tmp_mask_path)
        os.replace(tmp_mask_path, mask_path)

        mni_affine = register_to_mni(image, mask, resources)
        tmp_affine_path = affine_path.with_name(f".tmp-{affine_path.name}")
        np.savetxt(tmp_affine_path, mni_affine, fmt="%.10f")
        os.replace(tmp_affine_path, affine_path)

    return quality_check(nib.load(mask_path), np.loadtxt(affine_path), resources)
