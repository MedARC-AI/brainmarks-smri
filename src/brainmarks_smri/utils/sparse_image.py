"""Brain-masked 3D images as uncompressed npz: in-mask int16 values plus a run-length mask.

Fields: `values` (int16, in-mask voxels in C order), `mask_runs` (uint32 run lengths of the
mask, alternating outside/inside, starting outside), `shape`, `affine` (of the cropped grid),
`scale` (image = values * scale) and `mni_affine` (image world mm -> MNI world mm).
"""

import io

import nibabel as nib
import numpy as np

INT16_MAX = np.iinfo(np.int16).max


def encode_image(image: nib.Nifti1Image, mask: nib.Nifti1Image, mni_affine: np.ndarray) -> bytes:
    """Image masked and cropped to the mask's bounding box, as npz bytes."""
    data = image.get_fdata(dtype=np.float32)
    mask_data = np.asanyarray(mask.dataobj) > 0
    assert data.shape == mask_data.shape, (data.shape, mask_data.shape)

    brain_voxels = np.argwhere(mask_data)
    start = brain_voxels.min(axis=0)
    stop = brain_voxels.max(axis=0) + 1
    crop = tuple(slice(a, b) for a, b in zip(start, stop))
    mask_data = mask_data[crop]
    values = data[crop][mask_data]
    affine = image.affine @ nib.affines.from_matvec(np.eye(3), start)

    peak = float(np.abs(values).max()) if values.size else 0.0
    if peak == 0 or (np.all(values == np.round(values)) and peak <= INT16_MAX):
        scale = 1.0
    else:
        scale = peak / INT16_MAX
    values = np.round(values / scale).astype(np.int16)

    flat_mask = mask_data.ravel()
    run_starts = np.flatnonzero(flat_mask[1:] != flat_mask[:-1]) + 1
    mask_runs = np.diff(np.concatenate([[0], run_starts, [flat_mask.size]]))
    if flat_mask[0]:
        mask_runs = np.concatenate([[0], mask_runs])

    buffer = io.BytesIO()
    np.savez(
        buffer,
        values=values,
        mask_runs=mask_runs.astype(np.uint32),
        shape=np.array(mask_data.shape, dtype=np.int64),
        affine=affine,
        scale=np.float64(scale),
        mni_affine=mni_affine,
    )
    return buffer.getvalue()


def decode_image(
    data: bytes,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray]:
    """Image (float32, zero outside the mask), mask and MNI affine."""
    fields = np.load(io.BytesIO(data))
    shape = tuple(int(size) for size in fields["shape"])
    mask_runs = fields["mask_runs"]
    run_is_inside = np.arange(len(mask_runs)) % 2 == 1
    mask_data = np.repeat(run_is_inside, mask_runs).reshape(shape)

    image_data = np.zeros(shape, dtype=np.float32)
    image_data[mask_data] = fields["values"] * np.float32(fields["scale"])
    affine = fields["affine"]
    image = nib.Nifti1Image(image_data, affine)
    mask = nib.Nifti1Image(mask_data.astype(np.uint8), affine)
    return image, mask, fields["mni_affine"]
