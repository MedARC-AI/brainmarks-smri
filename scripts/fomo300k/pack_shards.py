"""Pack the preprocessed FOMO300K into zip shards, one shard per call.

    uv run scripts/fomo300k/pack_shards.py datasets/fomo300k --count
    uv run scripts/fomo300k/pack_shards.py datasets/fomo300k --shard 0
    uv run scripts/fomo300k/pack_shards.py datasets/fomo300k --index

Shards hold whole participants, cut into equal parts by estimated size. Per image: `<stem>.npz`
(the source image from the hub, masked with the saved mask, cropped to the brain, encoded with
`encode_sparse_npz` plus an `mni_affine` field) and every derivative file, plus each session's
`session.json`. Members are stored uncompressed under their paths relative to the dataset folder,
so unzipping the shards there restores derivatives/. `--shard` writes `shards/fomo300k-NNN.zip`
(skipped if it exists) and its member index; `--index` merges the member indexes into
`tables/shards.parquet` (path, shard, offset, size, sha256), where `offset` is the start of the
member's data.
"""

import argparse
import gzip
import hashlib
import logging
import os
import struct
import tempfile
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from brainmarks_smri.utils import preprocessing
from brainmarks_smri.utils.encoding import encode_sparse_npz

REPO_ID = "FOMO-MRI/FOMO300K"
REVISION = "0ea0b15cfcc08a96b7b9fd43d952cec3fdb7cfc7"
NAME = "fomo300k"
DERIVATIVE_SUFFIXES = ("_mask.nii.gz", "_mni.txt", "_info.json", "_sag.jpg", "_cor.jpg", "_ax.jpg")
INDEX_PARTS_DIR = ".shard-index"

logger = logging.getLogger("pack_shards")


def assign_shards(images: pd.DataFrame, shard_size: int) -> pd.Series:
    """Shard index per image: whole participants, cut into equal parts of about `shard_size`."""
    spacing = np.array(images["spacing"].tolist(), dtype=float)
    encoded_bytes = images["brain_volume"] * 1e6 / spacing.prod(axis=1) * 2
    participant = images["collection"] + "/" + images["subject"]
    participant_bytes = encoded_bytes.groupby(participant, sort=True).sum()
    num_shards = max(1, round(participant_bytes.sum() / shard_size))
    bytes_before = participant_bytes.cumsum() - participant_bytes
    shard_index = bytes_before * num_shards // participant_bytes.sum()
    shard_index = shard_index.rank(method="dense").astype(int) - 1
    return participant.map(shard_index)


def derivative_stem(path: str) -> str:
    """Image path `source/<...>.nii.gz` -> derivative stem `derivatives/<...>`."""
    return "derivatives/" + path.removeprefix("source/").removesuffix(".nii.gz")


def member_data_offset(shard_file, header_offset: int) -> int:
    """Start of a member's data, from its local file header."""
    shard_file.seek(header_offset)
    header = shard_file.read(30)
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    return header_offset + 30 + name_length + extra_length


def add_member(shard_zip: zipfile.ZipFile, members: list, path: str, data: bytes) -> None:
    info = zipfile.ZipInfo(path, date_time=time.localtime()[:6])
    info.external_attr = 0o644 << 16
    shard_zip.writestr(info, data)
    members.append({"path": path, "sha256": hashlib.sha256(data).hexdigest()})


def add_file(shard_zip: zipfile.ZipFile, members: list, dataset_dir: Path, path: str) -> None:
    shard_zip.write(dataset_dir / path, arcname=path)
    sha256 = hashlib.sha256((dataset_dir / path).read_bytes()).hexdigest()
    members.append({"path": path, "sha256": sha256})


def pack_shard(dataset_dir: Path, images: pd.DataFrame, shard: int) -> None:
    shard_name = f"shards/{NAME}-{shard:03d}.zip"
    shard_path = dataset_dir / shard_name
    if shard_path.exists():
        logger.info("%s exists, skipping", shard_name)
        return
    shard_path.parent.mkdir(exist_ok=True)
    tmp_path = shard_path.with_name(f".tmp-{shard_path.name}")

    images = images.sort_values("path")
    session_zips = images["path"].map(
        lambda path: "/".join(path.removeprefix("source/").split("/")[:-2]) + ".zip"
    )
    members = []
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as shard_zip:
        for session_zip, session_images in images.groupby(session_zips, sort=True):
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = hf_hub_download(
                    REPO_ID, session_zip, repo_type="dataset", revision=REVISION, local_dir=tmp
                )
                with zipfile.ZipFile(zip_path) as source_zip:
                    for path in session_images["path"]:
                        stem = derivative_stem(path)
                        member_name = "/".join(path.split("/")[-3:])
                        image_bytes = gzip.decompress(source_zip.read(member_name))
                        image = nib.squeeze_image(nib.Nifti1Image.from_bytes(image_bytes))
                        mask = nib.load(dataset_dir / f"{stem}_mask.nii.gz")
                        mni_affine = np.loadtxt(dataset_dir / f"{stem}_mni.txt")
                        mask_data = np.asanyarray(mask.dataobj) > 0
                        masked_data = image.get_fdata(dtype=np.float32) * mask_data
                        masked = nib.Nifti1Image(masked_data, image.affine)
                        cropped, _ = preprocessing.crop_image(masked)
                        encoded = encode_sparse_npz(cropped, mni_affine=mni_affine)
                        add_member(shard_zip, members, f"{stem}.npz", encoded)
                        for suffix in DERIVATIVE_SUFFIXES:
                            add_file(shard_zip, members, dataset_dir, f"{stem}{suffix}")
            session_json = "derivatives/" + session_zip.removesuffix(".zip") + "/session.json"
            if (dataset_dir / session_json).exists():
                add_file(shard_zip, members, dataset_dir, session_json)
            logger.info("%s: %s, %d images", shard_name, session_zip, len(session_images))

    index = pd.DataFrame(members)
    index["shard"] = shard_name
    with zipfile.ZipFile(tmp_path) as shard_zip, open(tmp_path, "rb") as shard_file:
        infos = {info.filename: info for info in shard_zip.infolist()}
        index["offset"] = [
            member_data_offset(shard_file, infos[path].header_offset) for path in index["path"]
        ]
        index["size"] = [infos[path].file_size for path in index["path"]]
        for path, offset, size, sha256 in index[["path", "offset", "size", "sha256"]].itertuples(
            index=False
        ):
            shard_file.seek(offset)
            assert hashlib.sha256(shard_file.read(size)).hexdigest() == sha256, path

    index_part = dataset_dir / INDEX_PARTS_DIR / f"{NAME}-{shard:03d}.parquet"
    index_part.parent.mkdir(exist_ok=True)
    index[["path", "shard", "offset", "size", "sha256"]].to_parquet(index_part, index=False)
    os.replace(tmp_path, shard_path)
    logger.info(
        "%s: %d images, %d members, %.2f GB",
        shard_name,
        len(images),
        len(index),
        shard_path.stat().st_size / 1e9,
    )


def merge_index(dataset_dir: Path, num_shards: int) -> None:
    parts = []
    for shard in range(num_shards):
        shard_path = dataset_dir / "shards" / f"{NAME}-{shard:03d}.zip"
        index_part = dataset_dir / INDEX_PARTS_DIR / f"{NAME}-{shard:03d}.parquet"
        assert shard_path.exists() and index_part.exists(), f"shard {shard} is not packed"
        parts.append(pd.read_parquet(index_part))
    index = pd.concat(parts, ignore_index=True)
    index.to_parquet(dataset_dir / "tables" / "shards.parquet", index=False)
    logger.info("tables/shards.parquet: %d members in %d shards", len(index), num_shards)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="e.g. datasets/fomo300k")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--count", action="store_true", help="print the number of shards")
    action.add_argument("--shard", type=int, default=os.getenv("SLURM_ARRAY_TASK_ID"))
    action.add_argument("--index", action="store_true", help="merge the shard member indexes")
    parser.add_argument("--shard-size-gb", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    images = pd.read_parquet(args.dataset_dir / "tables" / "images.parquet")
    images["shard"] = assign_shards(images, int(args.shard_size_gb * 1e9))
    num_shards = images["shard"].max() + 1
    if args.count:
        print(num_shards)
    elif args.index:
        merge_index(args.dataset_dir, num_shards)
    else:
        assert args.shard is not None, "--shard or SLURM_ARRAY_TASK_ID"
        pack_shard(args.dataset_dir, images[images["shard"] == int(args.shard)], int(args.shard))


if __name__ == "__main__":
    main()
