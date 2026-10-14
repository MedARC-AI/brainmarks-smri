"""Convert every Analyze 7.5 image (.hdr/.img) under a folder to .nii.gz, in place.

Analyze headers carry no affine, only an `orient` code. OASIS uses 0 (transverse) and
2 (sagittal). We map them to LAS and ASL (as FreeSurfer reads them), centered on the volume.
Images written by the 4dfp tools (`*.4dfp.hdr`) have all three axes flipped: PIR.
Voxel values and dtype are unchanged.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

# Columns: world direction (RAS) of voxel axes 0, 1, 2.
AXES_BY_ORIENT = {
    0: np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # LAS
    2: np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]]),  # ASL
}
AXES_4DFP_SAGITTAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # PIR


def convert(hdr_path: Path) -> None:
    image = nib.load(hdr_path)
    header = image.header
    orient = int(header.binaryblock[252])
    assert orient in AXES_BY_ORIENT, f"{hdr_path}: orient {orient}"
    slope, intercept = header.get_slope_inter()
    assert slope in (None, 1) and intercept in (None, 0), (
        f"{hdr_path}: scaling {slope}, {intercept}"
    )

    data = np.asanyarray(image.dataobj)
    assert data.ndim == 4 and data.shape[3] == 1, f"{hdr_path}: shape {data.shape}"
    data = data[..., 0]
    data = data.astype(data.dtype.newbyteorder("="))

    zooms = np.array(header.get_zooms()[:3], dtype=float)
    axes = AXES_BY_ORIENT[orient]
    if hdr_path.name.endswith(".4dfp.hdr"):
        assert orient == 2, f"{hdr_path}: orient {orient}"
        axes = AXES_4DFP_SAGITTAL
    rotation = axes * zooms
    center = (np.array(data.shape) - 1) / 2
    affine = np.eye(4)
    affine[:3, :3] = rotation
    affine[:3, 3] = -rotation @ center

    nifti = nib.Nifti1Image(data, affine)
    nifti.header.set_xyzt_units("mm")
    nifti.set_qform(affine, code=1)
    nifti.set_sform(affine, code=1)
    nifti_path = hdr_path.with_suffix(".nii.gz")
    tmp_path = hdr_path.with_suffix(".tmp.nii.gz")
    nib.save(nifti, tmp_path)
    tmp_path.rename(nifti_path)
    hdr_path.with_suffix(".img").unlink()
    hdr_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    for hdr_path in sorted(args.folder.rglob("*.hdr")):
        convert(hdr_path)


if __name__ == "__main__":
    main()
