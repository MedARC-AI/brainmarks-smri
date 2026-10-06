"""Brain mask and affine to MNI152NLin6Asym for every 3D image of a dataset.

    uv run --extra preprocess scripts/preprocess.py datasets/<name> --jobs 32

Outputs in `derivatives/` inside each dataset folder.
"""

import os

# Before importing ants/torch: one thread per job, needed for reproducible registration.
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import hashlib
import logging
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import ants
import nibabel as nib
import numpy as np
import pandas as pd
import platformdirs
from scipy import ndimage

CACHE_DIR = Path(os.getenv("BRAINMARKS_SMRI_CACHE", platformdirs.user_cache_dir("brainmarks_smri")))
REPO = Path(__file__).resolve().parents[1]

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

# Included image modalities. 4D DWI and DTI are skipped.
MODALITIES = {"T1w", "T1c", "T2w", "FLAIR", "PD", "DWI", "ADC"}
# Sequences co-registered by the source on one grid per session: only the T1w, whose
# derivatives serve the whole session.
DATASET_MODALITIES = {"brats2021": {"T1w"}, "ucsf_pdgm": {"T1w"}, "upenn_gbm": {"T1w"}}
# Ignore extra image variants that won't go into benchmarks.
SKIP_DESC = {
    "ucsf_pdgm": {"bias"},
    "upenn_gbm": {"unstripped", "old", "old_unstripped"},
}
# Skip skull-stripping for datasets that are already stripped.
SKULL_STRIPPED = {"brats2021", "upenn_gbm"}

# QC warning ranges (template brain mask Dice, linear scale to MNI, brain volume in litres)
MIN_DICE = 0.9
SCALE_RANGE = (0.9, 1.4)
VOLUME_RANGE = (0.8, 2.2)

MASK_SUFFIX = "_mask.nii.gz"
# 4x4 text matrix (np.loadtxt): image world mm (RAS) -> MNI152NLin6Asym mm (RAS).
AFFINE_SUFFIX = "_mni.txt"

logger = logging.getLogger("preprocess")


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


def load_image(path: Path) -> nib.Nifti1Image:
    image = nib.load(path)
    # Some DWI/ADC are stored as 4D with a single volume.
    if image.ndim == 4 and image.shape[3] == 1:
        image = image.slicer[..., 0]
    assert image.ndim == 3, image.shape
    return image


def synthstrip(image: nib.Nifti1Image, files: dict[str, Path]) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        nib.save(image, f"{tmp}/image.nii")
        command = [
            sys.executable, files["synthstrip_script"],
            "-i", f"{tmp}/image.nii",
            "-m", f"{tmp}/mask.nii",
            "--model", files["synthstrip_weights"],
            "-t", "1",
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SynthStrip exit {result.returncode}: {result.stderr[-2000:]}")
        return np.asanyarray(nib.load(f"{tmp}/mask.nii").dataobj) > 0


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


def derivative_stem(path: str) -> str:
    """Image `path` from images.tsv -> derivative stem, relative to the dataset directory."""
    source_path = PurePosixPath(path)
    assert source_path.parts[0] == "source", path
    name = source_path.name.removesuffix(".gz").removesuffix(".nii")
    return str(PurePosixPath("derivatives", *source_path.parent.parts[1:], name))


def preprocess_image(image_row: dict, dataset_dir: Path, files: dict[str, Path]) -> dict:
    stem = dataset_dir / derivative_stem(image_row["path"])
    mask_path = Path(f"{stem}{MASK_SUFFIX}")
    aff_path = Path(f"{stem}{AFFINE_SUFFIX}")

    # The transform is written last, so its presence marks a finished image.
    if not aff_path.exists():
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = load_image(dataset_dir / image_row["path"])

        # Brain mask. SynthStrip if no mask provided / not already stripped.
        if image_row["mask_method"] == "provided":
            brain_mask = nib.load(dataset_dir / image_row["provided_mask"])
            assert np.allclose(brain_mask.affine, image.affine) and brain_mask.shape == image.shape
            mask = np.asanyarray(brain_mask.dataobj) > 0
        elif image_row["mask_method"] == "nonzero":
            mask = np.asanyarray(image.dataobj) != 0
        else:
            mask = synthstrip(image, files)
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        tmp_mask_path = mask_path.with_name(f".tmp-{mask_path.name}")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine, header), tmp_mask_path)
        os.replace(tmp_mask_path, mask_path)

        # ANTs 12-DOF registration to MNI template with Mattes MI.
        template = nib.load(files["template"])
        masked_data = image.get_fdata(dtype=np.float32) * mask
        ants.config.set_ants_deterministic(True, seed_value=ANTS_SEED)
        with tempfile.TemporaryDirectory() as tmp:
            registration = ants.registration(
                fixed=to_ants(template.get_fdata(dtype=np.float32), template.affine),
                moving=to_ants(masked_data, image.affine),
                type_of_transform="Affine",
                aff_metric="mattes",
                outprefix=f"{tmp}/",
            )
            transform = ants.read_transform(registration["fwdtransforms"][0])
        mni_affine = ants_to_ras_affine(transform.parameters, transform.fixed_parameters)
        tmp_aff_path = aff_path.with_name(f".tmp-{aff_path.name}")
        np.savetxt(tmp_aff_path, mni_affine, fmt="%.10f")
        os.replace(tmp_aff_path, aff_path)

    # QC from the outputs, so re-runs report finished images too.
    mask_image = nib.load(mask_path)
    mask = np.asanyarray(mask_image.dataobj) > 0
    mni_affine = np.loadtxt(aff_path)
    template_mask = nib.load(files["template_mask"])
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
        "path": image_row["path"],
        "modality": image_row["modality"],
        "mask_method": image_row["mask_method"],
        "brain_volume": round(volume, 4),
        "mni_scale": round(scale, 4),
        "template_dice": round(dice, 4),
        "warnings": ",".join(warnings) or "n/a",
    }


def run_safely(image_row: dict, dataset_dir: Path, files: dict[str, Path]) -> dict:
    start = time.time()
    try:
        result = preprocess_image(image_row, dataset_dir, files)
    except Exception:
        return {"path": image_row["path"], "error": traceback.format_exc()}
    result["seconds"] = time.time() - start
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="e.g. datasets/ixi")
    # SynthStrip needs up to ~10 GB of memory, so choose --jobs by memory as well as cores.
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--limit", type=int, help="only the first N images (for trial runs)")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    dataset_dir = Path(args.dataset_dir).resolve()
    name = dataset_dir.name
    images = pd.read_csv(
        dataset_dir / "tables" / "images.tsv", sep="\t", dtype=str, keep_default_na=False
    )

    provided_masks = images[(images["modality"] == "mask") & (images["desc"] == "brain")]
    provided_masks = provided_masks.set_index(["participant_id", "session_id"])["path"].to_dict()

    modalities = DATASET_MODALITIES.get(name, MODALITIES)
    skip_desc = SKIP_DESC.get(name, set())
    selected = images["modality"].isin(modalities) & ~images["desc"].isin(skip_desc)
    image_rows = images[selected].to_dict("records")
    for image_row in image_rows:
        provided_mask = provided_masks.get((image_row["participant_id"], image_row["session_id"]))
        if provided_mask:
            image_row["mask_method"] = "provided"
            image_row["provided_mask"] = provided_mask
        elif name in SKULL_STRIPPED:
            image_row["mask_method"] = "nonzero"
        else:
            image_row["mask_method"] = "synthstrip"
    image_rows = image_rows[: args.limit]

    files = {
        "synthstrip_script": fetch(*SYNTHSTRIP_SCRIPT),
        "synthstrip_weights": fetch(*SYNTHSTRIP_WEIGHTS),
        "template": fetch(*TEMPLATE),
        "template_mask": fetch(*TEMPLATE_MASK),
    }
    logger.info("%s: %d images, %d jobs", name, len(image_rows), args.jobs)

    results = []
    failures = []
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_safely, row, dataset_dir, files) for row in image_rows]
        for count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if "error" in result:
                failures.append(result)
                logger.error(
                    "[%d/%d] %s failed:\n%s", count, len(futures), result["path"], result["error"]
                )
                continue
            results.append(result)
            message = "[%d/%d] %s: %s mask %.2f L, scale %.3f, dice %.3f (%.0f s)"
            values = (
                count, len(futures), result["path"], result["mask_method"], result["brain_volume"],
                result["mni_scale"], result["template_dice"], result.pop("seconds"),
            )  # fmt: skip
            if result["warnings"] == "n/a":
                logger.info(message, *values)
            else:
                logger.warning(message + " check: %s", *values, result["warnings"])

    if not results:
        sys.exit(1)
    derivatives_dir = dataset_dir / "derivatives"
    qc = pd.DataFrame(results).sort_values("path")
    qc.to_csv(derivatives_dir / "qc.tsv", sep="\t", index=False, lineterminator="\n")
    flagged = (qc["warnings"] != "n/a").sum()
    logger.info(
        "done: %d ok (%d with QC warnings), %d failed", len(results), flagged, len(failures)
    )
    if failures or args.limit:
        sys.exit(1 if failures else 0)

    # Leftovers from killed runs.
    for tmp_path in derivatives_dir.rglob(".tmp-*"):
        tmp_path.unlink()
    # Same format and order as `manifest.sha256` (sha256sum, LC_ALL=C sort).
    paths = [path.relative_to(dataset_dir).as_posix() for path in derivatives_dir.rglob("*")]
    manifest = []
    for path in sorted(paths, key=str.encode):
        if (dataset_dir / path).is_file():
            digest = hashlib.sha256((dataset_dir / path).read_bytes()).hexdigest()
            manifest.append(f"{digest}  {path}\n")
    (dataset_dir / "derivatives.sha256").write_text("".join(manifest))
    tracked_dir = REPO / "scripts" / name
    if tracked_dir.is_dir():
        (tracked_dir / "derivatives.sha256").write_text("".join(manifest))
    logger.info("wrote derivatives.sha256 (%d files)", len(manifest))


if __name__ == "__main__":
    main()
