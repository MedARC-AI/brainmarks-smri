"""Image table of the preprocessed FOMO300K: one row per image with an `_info.json`.

    uv run scripts/fomo300k/build_tables.py datasets/fomo300k

Writes `tables/images.parquet` inside the dataset folder.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

INFO_SUFFIX = "_info.json"


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
    tables_dir = args.dataset_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    images.to_parquet(tables_dir / "images.parquet", index=False)
    print(f"wrote {tables_dir / 'images.parquet'}")


if __name__ == "__main__":
    main()
