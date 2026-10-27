"""Convert the selected ADNI DICOM series to NIfTI with dcm2niix, straight from the IDA zips.

    uv run python scripts/adni/convert.py <dir of IDA zips> <selected_images.tsv> <out dir>

Writes `<out>/sub-<id>/ses-<y0|y2|y4>/anat/sub-<id>_ses-<..>[_acq-3T]_<T1w|FLAIR>.nii.gz` and the
dcm2niix JSON sidecar, and `<out>/image_dates.tsv` (acquisition date per image ID, from the zips). Each series is extracted to a temporary folder, converted and deleted, so
the zips are never unpacked whole. Existing outputs are skipped.

dcm2niix writes a second, resampled copy (`_Eq_1`) when the slice spacing is uneven; we keep
that one, since the plain output cannot represent the spacing. Any other extra output, a missing
sidecar or a 4D image (stacked acquisitions) is an error.
"""

import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import pandas as pd

SERIES_DIR = re.compile(r"ADNI/[^/]+/[^/]+/(\d{4}-\d{2}-\d{2})_[^/]*/(I\d+)/")
CHUNK = 50


def target_stem(row: pd.Series) -> Path:
    name = f"{row.participant_id}_ses-{row.session_id}"
    if isinstance(row.desc, str):
        name += f"_{row.desc}"
    return Path(row.participant_id) / f"ses-{row.session_id}" / "anat" / f"{name}_{row.modality}"


def convert_series(archive: zipfile.ZipFile, members: list[str], out_stem: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        dicom_dir = Path(tmp) / "dicom"
        nifti_dir = Path(tmp) / "nifti"
        dicom_dir.mkdir()
        nifti_dir.mkdir()
        for member in members:
            (dicom_dir / Path(member).name).write_bytes(archive.read(member))
        result = subprocess.run(
            [
                "dcm2niix",
                "-z",
                "y",
                "-b",
                "y",
                "-ba",
                "y",
                "-f",
                "image",
                "-o",
                nifti_dir,
                dicom_dir,
            ],
            capture_output=True,
            text=True,
        )
        outputs = sorted(path.name for path in nifti_dir.iterdir())
        if "image.nii.gz" in outputs and len(nib.load(nifti_dir / "image.nii.gz").shape) != 3:
            return f"failed: 4D output {nib.load(nifti_dir / 'image.nii.gz').shape}"
        if outputs == ["image.json", "image.nii.gz"]:
            stem = "image"
        elif outputs == ["image.json", "image.nii.gz", "image_Eq_1.nii.gz"]:
            stem = "image_Eq_1"
        else:
            tail = " | ".join(result.stdout.strip().splitlines()[-2:])[:300]
            return f"failed: outputs {outputs}; {tail}"
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        # Copy under a temporary name, then rename, so an interrupted run leaves no partial file.
        for source, suffix in [
            (nifti_dir / "image.json", ".json"),
            (nifti_dir / f"{stem}.nii.gz", ".nii.gz"),
        ]:
            partial = out_stem.with_suffix(suffix + ".part")
            shutil.copyfile(source, partial)
            partial.replace(out_stem.with_suffix(suffix))
        return "ok" if stem == "image" else "ok (Eq_1)"


def convert_chunk(
    zip_path: Path, tasks: list[tuple[str, list[str], Path]]
) -> list[tuple[str, str]]:
    results = []
    with zipfile.ZipFile(zip_path) as archive:
        for image_id, members, out_stem in tasks:
            results.append((image_id, convert_series(archive, members, out_stem)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_dir", type=Path)
    parser.add_argument("selected", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    selected = pd.read_csv(args.selected, sep="\t", na_values="n/a")
    todo = {}
    for row in selected.itertuples():
        out_stem = args.out_dir / target_stem(row)
        if not out_stem.with_suffix(".nii.gz").exists():
            todo[row.image_id] = out_stem
    print(f"{len(selected)} selected, {len(todo)} to convert")

    members: dict[str, list[str]] = {}
    zip_of: dict[str, Path] = {}
    dates: dict[str, str] = {}
    wanted = set(selected.image_id)
    for zip_path in sorted(args.zip_dir.glob("*.zip")):
        if not zipfile.is_zipfile(zip_path):
            continue
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                match = SERIES_DIR.match(name)
                if not match or match[2] not in wanted or not name.endswith(".dcm"):
                    continue
                date, image_id = match.groups()
                dates[image_id] = date
                if image_id in todo:
                    members.setdefault(image_id, []).append(name)
                    zip_of[image_id] = zip_path
    missing = sorted(wanted - set(dates))
    assert not missing, f"{len(missing)} selected series not in the zips, e.g. {missing[:5]}"
    # Scan dates are DUA data, so they come from the user's zips rather than the tracked list.
    date_table = pd.DataFrame(sorted(dates.items()), columns=["image_id", "acq_date"])
    date_table.to_csv(args.out_dir / "image_dates.tsv", sep="\t", index=False, lineterminator="\n")

    chunks = []
    for zip_path in sorted(set(zip_of.values())):
        image_ids = sorted(image_id for image_id in todo if zip_of[image_id] == zip_path)
        for start in range(0, len(image_ids), CHUNK):
            chunk = image_ids[start : start + CHUNK]
            chunks.append((zip_path, [(i, sorted(members[i]), todo[i]) for i in chunk]))

    failed = []
    done = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(convert_chunk, zip_path, tasks) for zip_path, tasks in chunks]
        for future in futures:
            for image_id, status in future.result():
                done += 1
                if not status.startswith("ok"):
                    failed.append(image_id)
                    print(f"{image_id}: {status}", flush=True)
                elif status != "ok":
                    print(f"{image_id}: {status}", flush=True)
            print(f"{done}/{len(todo)} done", flush=True)
    assert not failed, (
        f"{len(failed)} series failed (add them to EXCLUDE in select_images.py): {failed}"
    )


if __name__ == "__main__":
    main()
