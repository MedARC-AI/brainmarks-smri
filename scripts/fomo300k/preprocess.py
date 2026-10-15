"""Preprocess one batch of the filtered FOMO300K sessions.

uv run --extra preprocess scripts/fomo300k/preprocess.py <out_dir> --batch 0 --num-batches 1000
"""

import argparse
import gzip
import json
import logging
import os
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from brainmarks_smri.utils import preprocessing

REPO_ID = "FOMO-MRI/FOMO300K"
REVISION = "0ea0b15cfcc08a96b7b9fd43d952cec3fdb7cfc7"
SESSIONS = Path(__file__).parent / "filtered_sessions.txt"
SHUFFLE_SEED = 0
ANTS_SEED = 1

# QC warning ranges (template brain mask Dice, linear scale to MNI, brain volume in litres)
MIN_DICE = 0.9
SCALE_RANGE = (0.9, 1.4)
VOLUME_RANGE = (0.8, 2.2)

QUANTILES = (0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999)

VIEWS = ("sag", "cor", "ax")
THUMBNAIL_SIZE = 224
JPEG_QUALITY = 85

logger = logging.getLogger("preprocess_fomo300k")


def preprocess_image(image: nib.Nifti1Image, stem: Path) -> dict:
    start = time.time()
    image = nib.squeeze_image(image)
    assert image.ndim == 3, image.shape
    mask = preprocessing.synthstrip(image)
    mni_affine = preprocessing.mni_affine_registration(image, mask, seed=ANTS_SEED)
    qc = preprocessing.mni_affine_qc(mask, mni_affine)

    mask_data = np.asanyarray(mask.dataobj) > 0
    mask_voxels = np.argwhere(mask_data)
    brain_centre = nib.affines.apply_affine(image.affine, mask_voxels.mean(axis=0))
    rigid = preprocessing.rigid_from_affine(mni_affine, brain_centre)
    data = image.get_fdata(dtype=np.float32)
    template = nib.load(
        preprocessing.cached_download(preprocessing.MNI_TEMPLATE_URL, preprocessing.CACHE_DIR)
    )
    preproc = preprocessing.resample_to_template(image, template, rigid)
    preproc_mask = preprocessing.resample_to_template(
        mask, template, rigid, order=0, anti_aliasing=False
    )
    preproc_data = preproc.get_fdata(dtype=np.float32) * (preproc_mask.get_fdata() > 0)
    preproc = nib.Nifti1Image(preproc_data, template.affine)
    preproc.set_data_dtype(np.int16)

    brain_values = data[mask_data]
    quantiles = np.quantile(brain_values, QUANTILES)
    thumbnails = make_thumbnails(preproc, quantiles[1], quantiles[-2])

    warnings = []
    if qc["template_dice"] < MIN_DICE:
        warnings.append("dice")
    if not SCALE_RANGE[0] <= qc["mni_scale"] <= SCALE_RANGE[1]:
        warnings.append("scale")
    if not VOLUME_RANGE[0] <= qc["brain_volume"] <= VOLUME_RANGE[1]:
        warnings.append("volume")

    spacing = np.array(image.header.get_zooms()[:3], dtype=float)
    mask_extent = (mask_voxels.max(axis=0) - mask_voxels.min(axis=0) + 1) * spacing
    info = {
        **qc,
        "warnings": warnings,
        "dtype": str(image.get_data_dtype()),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "shape": list(image.shape),
        "spacing": spacing.round(4).tolist(),
        "mask_extent": mask_extent.round(1).tolist(),
        "min": float(brain_values.min()),
        "max": float(brain_values.max()),
        "mean": float(brain_values.mean()),
        "std": float(brain_values.std()),
        "quantiles": dict(zip(map(str, QUANTILES), quantiles.tolist())),
        "seconds": round(time.time() - start, 1),
    }

    stem.parent.mkdir(parents=True, exist_ok=True)
    nib.save(mask, f"{stem}_mask.nii.gz")
    np.savetxt(f"{stem}_mni.txt", mni_affine, fmt="%.10f")
    nib.save(preproc, f"{stem}_preproc.nii.gz")
    for view, thumbnail in thumbnails.items():
        thumbnail.save(f"{stem}_{view}.jpg", quality=JPEG_QUALITY)
    info_path = Path(f"{stem}_info.json")
    tmp_info_path = info_path.with_name(f".tmp-{info_path.name}")
    tmp_info_path.write_text(json.dumps(info))
    os.replace(tmp_info_path, info_path)
    return info


def preprocess_session(session: str, out_dir: Path) -> dict:
    """Preprocess one session zip, e.g. PT001_ClevelandCCF/sub-01/ses-01.zip; returns counts."""
    session_dir = out_dir / session.removesuffix(".zip")
    session_json = session_dir / "session.json"
    if session_json.exists():
        return {"images": 0, "failed": 0, "warned": 0, "skipped": 1}

    counts = {"images": 0, "failed": 0, "warned": 0, "skipped": 0}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            zip_path = hf_hub_download(
                REPO_ID, session, repo_type="dataset", revision=REVISION, local_dir=tmp
            )
        except Exception:
            logger.error("%s: download failed:\n%s", session, traceback.format_exc())
            counts["failed"] += 1
            return counts

        with zipfile.ZipFile(zip_path) as archive:
            # Zip members are <session>/<modality>/<name>.nii.gz.
            names = sorted(name for name in archive.namelist() if name.endswith(".nii.gz"))
            for name in names:
                stem = session_dir.parent / name.removesuffix(".nii.gz")
                if Path(f"{stem}_info.json").exists():
                    continue
                counts["images"] += 1
                image_name = f"{Path(session).parent}/{name}"
                try:
                    image = nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(name)))
                    info = preprocess_image(image, stem)
                except Exception:
                    counts["failed"] += 1
                    logger.error("%s failed:\n%s", image_name, traceback.format_exc())
                    continue
                if info["warnings"]:
                    counts["warned"] += 1
                    logger.warning(
                        "%s: check %s (mask %.2f L, scale %.3f, dice %.3f)",
                        image_name, ",".join(info["warnings"]), info["brain_volume"],
                        info["mni_scale"], info["template_dice"],
                    )  # fmt: skip

    if counts["failed"] == 0:
        session_dir.mkdir(parents=True, exist_ok=True)
        session_json.write_text(json.dumps({"images": len(names)}))
    return counts


def make_thumbnails(image: nib.Nifti1Image, vmin: float, vmax: float) -> dict[str, Image.Image]:
    data = image.get_fdata(dtype=np.float32)
    mask = data != 0
    brain_voxels = np.argwhere(mask)
    if len(brain_voxels) > 0:
        centre = brain_voxels.mean(axis=0).astype(int)
    else:
        centre = np.array(data.shape) // 2

    data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    data = np.where(mask, data, 0.5)
    thumbnails = {}
    for axis, view in enumerate(VIEWS):
        plane = np.flipud(np.take(data, centre[axis], axis=axis).T)
        plane = Image.fromarray((plane * 255).astype(np.uint8))
        thumbnail = Image.new("L", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), 127)
        offset = ((THUMBNAIL_SIZE - plane.width) // 2, (THUMBNAIL_SIZE - plane.height) // 2)
        thumbnail.paste(plane, offset)
        thumbnails[view] = thumbnail
    return thumbnails


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--batch", type=int, default=int(os.getenv("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only the first N sessions of the batch")
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    sessions = SESSIONS.read_text().split()
    rng = np.random.default_rng(SHUFFLE_SEED)
    sessions = [sessions[index] for index in rng.permutation(len(sessions))]
    offsets = np.linspace(0, len(sessions), args.num_batches + 1, dtype=int)
    batch = sessions[offsets[args.batch] : offsets[args.batch + 1]][: args.limit]
    logger.info("batch %d of %d: %d sessions", args.batch, args.num_batches, len(batch))

    start = time.time()
    totals = {"images": 0, "failed": 0, "warned": 0, "skipped": 0}
    for count, session in enumerate(batch, start=1):
        counts = preprocess_session(session, args.out_dir)
        for key in totals:
            totals[key] += counts[key]
        if count % 10 == 0 or count == len(batch):
            hours = (time.time() - start) / 3600
            logger.info(
                "[%d/%d sessions] %d images, %d failed, %d with QC warnings, %d sessions "
                "already done; %.2f h, %.0f images/h",
                count, len(batch), totals["images"], totals["failed"], totals["warned"],
                totals["skipped"], hours, totals["images"] / max(hours, 1e-6),
            )  # fmt: skip


if __name__ == "__main__":
    main()
