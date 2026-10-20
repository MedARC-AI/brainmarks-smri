"""Image table of the preprocessed FOMO300K: one row per image with an `_info.json`.

    uv run scripts/fomo300k/build_tables.py datasets/fomo300k

Joins FOMO's own participants.tsv, mri_info.tsv and mapping.tsv (pinned revision, gated: needs
HF_TOKEN with access). Writes `tables/images.parquet` inside the dataset folder.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "FOMO-MRI/FOMO300K"
REVISION = "0ea0b15cfcc08a96b7b9fd43d952cec3fdb7cfc7"
INFO_SUFFIX = "_info.json"

PARTICIPANT_COLUMNS = ["age", "sex", "handedness", "group"]
MRI_INFO_COLUMNS = {
    "MagneticFieldStrength": "field_strength",
    "Manufacturer": "manufacturer",
    "ManufacturersModelName": "model_name",
    "SoftwareVersions": "software_versions",
    "MRAcquisitionType": "acquisition_type",
    "SeriesDescription": "series_description",
    "ProtocolName": "protocol_name",
    "ScanningSequence": "scanning_sequence",
    "SequenceVariant": "sequence_variant",
    "ScanOptions": "scan_options",
    "SequenceName": "sequence_name",
    "EchoTime": "echo_time",
    "RepetitionTime": "repetition_time",
    "InversionTime": "inversion_time",
    "FlipAngle": "flip_angle",
    "SliceThickness": "slice_thickness",
}
NUMERIC_COLUMNS = [
    "echo_time",
    "repetition_time",
    "inversion_time",
    "flip_angle",
    "slice_thickness",
]


def image_row(derivatives_dir: Path, info_path: str) -> dict:
    """Table row of one image from its `_info.json` path."""
    stem = os.path.relpath(info_path, derivatives_dir).removesuffix(INFO_SUFFIX)
    parts = stem.split("/")
    num_collection_parts = 2 if parts[0] == "PT030_OpenNeuro" else 1
    collection = "/".join(parts[:num_collection_parts])
    subject, session, modality, name = parts[num_collection_parts:]
    suffix = "_".join(part for part in name.split("_") if "-" not in part)

    info = json.loads(Path(info_path).read_text())
    quantiles = info.pop("quantiles")
    del info["seconds"]
    return {
        "path": f"source/{stem}.nii.gz",
        "collection": collection,
        "subject": subject,
        "session": session,
        "modality": modality,
        "suffix": suffix,
        **info,
        **{f"q{level}": value for level, value in quantiles.items()},
    }


def read_fomo_table(file_name: str) -> pd.DataFrame:
    path = hf_hub_download(REPO_ID, file_name, repo_type="dataset", revision=REVISION)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def parse_age(value: str) -> float:
    """Years; ranges ('26-30') as their midpoint, '89+' as 89, anything else NaN."""
    value = value.strip()
    if re.fullmatch(r"[0-9.]+\+?", value):
        return float(value.removesuffix("+"))
    match = re.fullmatch(r"([0-9.]+)-([0-9.]+)", value)
    if match:
        return (float(match[1]) + float(match[2])) / 2
    return np.nan


def parse_field_strength(value: str) -> float:
    """Tesla; values in gauss (15000) are converted."""
    match = re.match(r"[0-9.]+", value)
    if not match:
        return np.nan
    tesla = float(match[0])
    return tesla / 10000 if tesla > 100 else tesla


def dwi_type(old_file_name: str) -> str:
    """FOMO splits each 4D DWI run into a b0 and, per shell, one 3D image: the mean of three
    axis-aligned volumes or a trace ADC (old name ends in `_trace`). The new name doesn't say which."""
    if "sbref" in old_file_name.lower():
        return "sbref"
    match = re.search(r"bval_(\d+)", old_file_name)
    if not match:
        return "other"
    if int(match[1]) <= 5:
        return "b0"
    if old_file_name.endswith("_trace.nii.gz"):
        return "adc"
    return "mean"


def source_metadata() -> pd.DataFrame:
    """FOMO's metadata per image, indexed by `source/<collection>/<path>`."""
    participants = read_fomo_table("participants.tsv")
    # Some OpenNeuro collections have sex in the age column.
    sex_in_age = participants["age"].str.strip().isin(["M", "F"]) & (participants["sex"] == "")
    participants.loc[sex_in_age, "sex"] = participants.loc[sex_in_age, "age"].str.strip()
    participants["age"] = participants["age"].map(parse_age)
    participants = participants.rename(columns={"dataset": "collection"})

    mapping = read_fomo_table("mapping.tsv")
    mapping = mapping.rename(
        columns={"dataset": "collection", "participant_id": "subject", "session_id": "session"}
    )
    mapping["source_path"] = mapping["old_path"]
    mapping["dwi_type"] = mapping["old_filename"].map(dwi_type).where(mapping["modality"] == "dwi")

    mri_info = read_fomo_table("mri_info.tsv")
    mri_info = mri_info.rename(columns={"dataset": "collection", "filename": "new_path"})
    mri_info = mri_info[["collection", "new_path", *MRI_INFO_COLUMNS]].rename(
        columns=MRI_INFO_COLUMNS
    )
    mri_info["field_strength"] = mri_info["field_strength"].map(parse_field_strength)
    mri_info["acquisition_type"] = mri_info["acquisition_type"].str.upper()
    for column in NUMERIC_COLUMNS:
        mri_info[column] = pd.to_numeric(mri_info[column], errors="coerce")

    metadata = mapping[["collection", "subject", "session", "new_path", "source_path", "dwi_type"]]
    metadata = metadata.merge(
        participants[["collection", "participant_id", "session_id", *PARTICIPANT_COLUMNS]],
        left_on=["collection", "subject", "session"],
        right_on=["collection", "participant_id", "session_id"],
        how="left",
        validate="many_to_one",
    )
    metadata = metadata.merge(
        mri_info, on=["collection", "new_path"], how="left", validate="one_to_one"
    )
    metadata.index = "source/" + metadata["collection"] + "/" + metadata["new_path"]
    columns = ["dwi_type", "source_path", *PARTICIPANT_COLUMNS, *MRI_INFO_COLUMNS.values()]
    return metadata[columns].replace("", pd.NA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="e.g. datasets/fomo300k")
    args = parser.parse_args()

    derivatives_dir = args.dataset_dir / "derivatives"
    info_paths = []
    for root, _, files in os.walk(derivatives_dir):
        info_paths += [os.path.join(root, name) for name in files if name.endswith(INFO_SUFFIX)]
    print(f"{len(info_paths)} images")

    with ThreadPoolExecutor(32) as pool:
        rows = list(pool.map(lambda path: image_row(derivatives_dir, path), info_paths))
    images = pd.DataFrame(rows).sort_values("path", ignore_index=True)

    metadata = source_metadata()
    assert images["path"].isin(metadata.index).all(), "images missing from mapping.tsv"
    identity = ["path", "collection", "subject", "session", "modality", "suffix"]
    images = pd.concat(
        [
            images[identity],
            metadata.loc[images["path"]].reset_index(drop=True),
            images.drop(columns=identity),
        ],
        axis=1,
    )

    tables_dir = args.dataset_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    images.to_parquet(tables_dir / "images.parquet", index=False)
    print(f"wrote {tables_dir / 'images.parquet'}")


if __name__ == "__main__":
    main()
