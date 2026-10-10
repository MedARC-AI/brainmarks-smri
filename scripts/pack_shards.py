"""Pack a dataset's images, their sidecars and derivatives into zip shards.

    uv run scripts/pack_shards.py datasets/<name> [--shard-size-gb 1]

Writes `shards/<name>-NNN.zip` and `tables/shards.tsv` (path, shard) inside the dataset folder, and
copies shards.tsv to scripts/<name>/tables/. The input files are not changed. Each shard holds whole
participants, in participant_id order: the images of images.tsv, their .json sidecars, and their
masks and affines from derivatives.tsv. Members are stored uncompressed under their paths relative
to the dataset folder, so unzipping the shards there restores the original files. Every member is
checked against tables/source.sha256 and tables/derivatives.sha256.
"""

import argparse
import hashlib
import logging
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

logger = logging.getLogger("pack_shards")


def sidecar_path(path: str) -> str:
    return path.removesuffix(".gz").removesuffix(".nii") + ".json"


def list_packed_files(dataset_dir: Path) -> pd.DataFrame:
    """One row per file to pack: participant_id, path, size."""
    tables_dir = dataset_dir / "tables"
    images = pd.read_csv(tables_dir / "images.tsv", sep="\t", dtype=str)
    images = images[["participant_id", "path"]]
    sidecars = images.assign(path=images["path"].map(sidecar_path))
    derivatives = pd.read_csv(tables_dir / "derivatives.tsv", sep="\t", dtype=str)
    derivatives = derivatives.merge(images, on="path")
    masks = derivatives[["participant_id", "mask_path"]].rename(columns={"mask_path": "path"})
    affines = derivatives[["participant_id", "mni_affine_path"]].rename(
        columns={"mni_affine_path": "path"}
    )

    files = pd.concat([images, sidecars, masks, affines], ignore_index=True)
    files = files[files["path"].map(lambda path: (dataset_dir / path).exists())]
    files = files.drop_duplicates("path")
    files["size"] = files["path"].map(lambda path: (dataset_dir / path).stat().st_size)
    return files.sort_values(["participant_id", "path"], ignore_index=True)


def assign_shards(files: pd.DataFrame, name: str, shard_size: int) -> pd.Series:
    """Shard file name per row: whole participants, cut into equal parts of about `shard_size`."""
    participant_sizes = files.groupby("participant_id", sort=True)["size"].sum()
    num_shards = max(1, round(participant_sizes.sum() / shard_size))
    bytes_before = participant_sizes.cumsum() - participant_sizes
    shard_index = (bytes_before * num_shards // participant_sizes.sum()).astype(int)
    shard_names = shard_index.map(lambda index: f"shards/{name}-{index:03d}.zip")
    return files["participant_id"].map(shard_names)


def write_shard(shard_path: Path, dataset_dir: Path, member_paths: list[str]) -> None:
    with zipfile.ZipFile(shard_path, "w", compression=zipfile.ZIP_STORED) as shard:
        for path in sorted(member_paths):
            shard.write(dataset_dir / path, arcname=path)


def read_manifest(path: Path) -> dict[str, str]:
    """`sha256sum` output -> {path: sha256}."""
    manifest = {}
    for line in path.read_text().splitlines():
        digest, file_path = line.split("  ", 1)
        manifest[file_path] = digest
    return manifest


def check_shard(shard_path: Path, expected: dict[str, str]) -> None:
    with zipfile.ZipFile(shard_path) as shard:
        for info in shard.infolist():
            digest = hashlib.sha256(shard.read(info)).hexdigest()
            if digest != expected[info.filename]:
                raise RuntimeError(
                    f"{shard_path.name}: {info.filename} does not match the manifest"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="e.g. datasets/ixi")
    parser.add_argument("--shard-size-gb", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    dataset_dir = Path(args.dataset_dir).resolve()
    name = dataset_dir.name
    expected = read_manifest(dataset_dir / "tables" / "source.sha256")
    expected |= read_manifest(dataset_dir / "tables" / "derivatives.sha256")

    files = list_packed_files(dataset_dir)
    missing = set(files["path"]) - set(expected)
    assert not missing, f"{len(missing)} files not in the manifests, e.g. {sorted(missing)[:3]}"
    files["shard"] = assign_shards(files, name, int(args.shard_size_gb * 1e9))
    logger.info(
        "%s: %d files, %.1f GB, %d participants -> %d shards",
        name,
        len(files),
        files["size"].sum() / 1e9,
        files["participant_id"].nunique(),
        files["shard"].nunique(),
    )

    tmp_dir = dataset_dir / ".tmp-shards"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir()
    for shard, shard_files in files.groupby("shard", sort=True):
        shard_path = tmp_dir / Path(shard).name
        write_shard(shard_path, dataset_dir, shard_files["path"].tolist())
        check_shard(shard_path, expected)
        logger.info(
            "%s: %d files, %.2f GB", shard, len(shard_files), shard_path.stat().st_size / 1e9
        )
    shutil.rmtree(dataset_dir / "shards", ignore_errors=True)
    os.replace(tmp_dir, dataset_dir / "shards")
    index_path = dataset_dir / "tables" / "shards.tsv"
    files[["path", "shard"]].to_csv(index_path, sep="\t", index=False, lineterminator="\n")
    shutil.copy(index_path, REPO / "scripts" / name / "tables" / "shards.tsv")

    packed = set(files["path"])
    loose = [path for path in expected if path not in packed]
    logger.info("done: %d files packed, %d left loose", len(packed), len(loose))


if __name__ == "__main__":
    main()
