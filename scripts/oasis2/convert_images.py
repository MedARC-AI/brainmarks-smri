"""Rewrite every NIfTI pair (<name>.nifti.hdr/.img) under a folder as <name>.nii.gz, in place.

The trailing size-1 axis is dropped. Header (incl. qform) and voxel values are otherwise unchanged.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def convert(hdr_path: Path) -> None:
    pair = nib.load(hdr_path)
    assert isinstance(pair, nib.Nifti1Pair) and not isinstance(pair, nib.Nifti1Image), hdr_path
    assert pair.ndim == 4 and pair.shape[3] == 1, f"{hdr_path}: shape {pair.shape}"
    image = nib.funcs.squeeze_image(nib.Nifti1Image.from_image(pair))
    stem = hdr_path.name.removesuffix(".nifti.hdr")
    nifti_path = hdr_path.with_name(f"{stem}.nii.gz")
    tmp_path = hdr_path.with_name(f"{stem}.tmp.nii.gz")
    nib.save(image, tmp_path)
    check = nib.load(tmp_path)
    assert np.array_equal(check.affine, pair.affine), hdr_path
    assert np.array_equal(np.asanyarray(check.dataobj), np.asanyarray(pair.dataobj)[..., 0]), (
        hdr_path
    )
    tmp_path.rename(nifti_path)
    hdr_path.with_suffix(".img").unlink()
    hdr_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    for hdr_path in sorted(args.folder.rglob("*.nifti.hdr")):
        convert(hdr_path)


if __name__ == "__main__":
    main()
