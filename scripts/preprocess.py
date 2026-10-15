"""Brain mask and affine to MNI152NLin6Asym for every 3D image of a dataset.

    uv run --extra preprocess scripts/preprocess.py datasets/<name> --jobs 32

Writes the masks and affines to `derivatives/` inside the dataset folder, and
`tables/derivatives.tsv` (QC per image) and `tables/derivatives.sha256`.
"""

import argparse
import hashlib
import logging
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
import pandas as pd

from brainmarks_smri.utils import preprocessing

REPO = Path(__file__).resolve().parents[1]

# Included image modalities. 4D DWI and DTI are skipped.
MODALITIES = {"T1w", "T1c", "T2w", "FLAIR", "PD", "DWI", "ADC"}
# Sequences co-registered by the source on one grid per session: only the T1w, whose
# derivatives serve the whole session.
DATASET_MODALITIES = {"brats2021": {"T1w"}, "ucsf_pdgm": {"T1w"}, "upenn_gbm": {"T1w"}}
# Ignore extra image variants that won't go into benchmarks.
SKIP_DESC = {
    "ucsf_pdgm": {"bias"},
    "upenn_gbm": {"unstripped", "old", "old_unstripped"},
    "oasis1": {"mpr-1", "mpr-2", "mpr-3", "mpr-4", "mpr-5", "mpr-6", "t88", "t88_masked"},
    "oasis2": {"mpr-2", "mpr-3", "mpr-4", "3906-3"},
}
# Skip skull-stripping for datasets that are already stripped.
SKULL_STRIPPED = {"atlas", "brats2021", "upenn_gbm"}
# Skull-stripped images whose background is noise and speckles, not zeros.
NOISY_BACKGROUND = {"atlas": "source/Training_Raw/R029/*"}

# ANTs treats seed 0 as "no seed".
ANTS_SEED = 1

# QC warning ranges (template brain mask Dice, linear scale to MNI, brain volume in litres)
MIN_DICE = 0.9
SCALE_RANGE = (0.9, 1.4)
VOLUME_RANGE = (0.8, 2.2)

MASK_SUFFIX = "_mask.nii.gz"
# 4x4 text matrix (np.loadtxt): image world mm (RAS) -> MNI152NLin6Asym mm (RAS).
AFFINE_SUFFIX = "_mni.txt"

logger = logging.getLogger("preprocess")


def derivative_stem(path: str) -> str:
    """Image `path` from images.tsv -> derivative stem, relative to the dataset directory."""
    source_path = PurePosixPath(path)
    assert source_path.parts[0] == "source", path
    name = source_path.name.removesuffix(".gz").removesuffix(".nii")
    return str(PurePosixPath("derivatives", *source_path.parent.parts[1:], name))


def preprocess_image(image_row: dict, dataset_dir: Path) -> dict:
    """Write the mask and MNI affine of one image (unless done) and return its QC."""
    stem = dataset_dir / derivative_stem(image_row["path"])
    mask_path = Path(f"{stem}{MASK_SUFFIX}")
    affine_path = Path(f"{stem}{AFFINE_SUFFIX}")

    # Both outputs are saved after both succeed, the affine last: it marks a finished image.
    if not affine_path.exists():
        image = nib.squeeze_image(nib.load(dataset_dir / image_row["path"]))
        assert image.ndim == 3, image.shape

        if image_row["mask_method"] == "provided":
            provided = nib.load(dataset_dir / image_row["provided_mask"])
            assert np.allclose(provided.affine, image.affine) and provided.shape == image.shape
            mask = preprocessing.mask_image_like(np.asanyarray(provided.dataobj) > 0, image)
        elif image_row["mask_method"] == "nonzero":
            mask = preprocessing.mask_image_like(np.asanyarray(image.dataobj) != 0, image)
        elif image_row["mask_method"] == "threshold":
            mask = preprocessing.threshold_mask(image)
        else:
            mask = preprocessing.synthstrip(image)
        mni_affine = preprocessing.mni_affine_registration(image, mask, seed=ANTS_SEED)

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_mask_path = mask_path.with_name(f".tmp-{mask_path.name}")
        nib.save(mask, tmp_mask_path)
        os.replace(tmp_mask_path, mask_path)
        tmp_affine_path = affine_path.with_name(f".tmp-{affine_path.name}")
        np.savetxt(tmp_affine_path, mni_affine, fmt="%.10f")
        os.replace(tmp_affine_path, affine_path)

    # QC from the outputs, so re-runs report finished images too.
    qc = preprocessing.mni_affine_qc(nib.load(mask_path), np.loadtxt(affine_path))
    warnings = []
    if qc["template_dice"] < MIN_DICE:
        warnings.append("dice")
    if not SCALE_RANGE[0] <= qc["mni_scale"] <= SCALE_RANGE[1]:
        warnings.append("scale")
    if not VOLUME_RANGE[0] <= qc["brain_volume"] <= VOLUME_RANGE[1]:
        warnings.append("volume")
    return {**qc, "warnings": ",".join(warnings) or "n/a"}


def run_safely(image_row: dict, dataset_dir: Path) -> dict:
    start = time.time()
    try:
        qc = preprocess_image(image_row, dataset_dir)
    except Exception:
        return {"path": image_row["path"], "error": traceback.format_exc()}
    return {
        "path": image_row["path"],
        "modality": image_row["modality"],
        "mask_method": image_row["mask_method"],
        **qc,
        "seconds": time.time() - start,
    }


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

    manifest_path = dataset_dir / "tables" / "derivatives.sha256"
    if manifest_path.exists():
        logger.info("manifest %s exists for dataset %s; exiting", manifest_path, name)
        sys.exit(0)

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
        elif fnmatch(image_row["path"], NOISY_BACKGROUND.get(name, "")):
            image_row["mask_method"] = "threshold"
        elif name in SKULL_STRIPPED:
            image_row["mask_method"] = "nonzero"
        else:
            image_row["mask_method"] = "synthstrip"
    image_rows = image_rows[: args.limit]

    logger.info("%s: %d images, %d jobs", name, len(image_rows), args.jobs)

    results = []
    failures = []
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_safely, row, dataset_dir) for row in image_rows]
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

    derivatives_dir = dataset_dir / "derivatives"
    derivatives_table = pd.DataFrame(results).sort_values("path")
    derivatives_table.to_csv(
        dataset_dir / "tables" / "derivatives.tsv", sep="\t", index=False, lineterminator="\n"
    )
    flagged = (derivatives_table["warnings"] != "n/a").sum()
    logger.info(
        "done: %d ok (%d with QC warnings), %d failed", len(results), flagged, len(failures)
    )
    if args.limit:
        sys.exit(0)

    # Leftovers from killed runs.
    for tmp_path in derivatives_dir.rglob(".tmp-*"):
        tmp_path.unlink()
    # Same format and order as `source.sha256` (sha256sum, LC_ALL=C sort).
    paths = [path.relative_to(dataset_dir).as_posix() for path in derivatives_dir.rglob("*")]
    manifest = []
    for path in sorted(paths, key=str.encode):
        if (dataset_dir / path).is_file():
            digest = hashlib.sha256((dataset_dir / path).read_bytes()).hexdigest()
            manifest.append(f"{digest}  {path}\n")
    manifest_path.write_text("".join(manifest))
    tracked_dir = REPO / "scripts" / name / "tables"
    if tracked_dir.is_dir():
        (tracked_dir / "derivatives.sha256").write_text("".join(manifest))
        shutil.copy(dataset_dir / "tables" / "derivatives.tsv", tracked_dir / "derivatives.tsv")
    logger.info("wrote derivatives.tsv and derivatives.sha256 (%d files)", len(manifest))


if __name__ == "__main__":
    main()
