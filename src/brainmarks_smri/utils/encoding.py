import io

import nibabel as nib
import numpy as np

from brainmarks_smri.utils.rle import brle_to_dense, dense_to_brle

INT16_MAX = np.iinfo(np.int16).max


def encode_sparse_npz(image: nib.Nifti1Image, **extra_fields: np.ndarray) -> bytes:
    """Image as uncompressed npz: nonzero values as int16 times a scale, plus a run-length mask."""
    data = image.get_fdata(dtype=np.float32)
    mask = data != 0
    values = data[mask]

    peak = float(np.abs(values).max()) if values.size else 0.0
    if peak == 0 or (np.all(values == np.round(values)) and peak <= INT16_MAX):
        scale = 1.0
    else:
        scale = peak / INT16_MAX
    values = np.round(values / scale).astype(np.int16)

    buffer = io.BytesIO()
    np.savez(
        buffer,
        values=values,
        scale=np.float64(scale),
        mask_rle=dense_to_brle(mask.ravel(), dtype=np.uint8),
        shape=np.array(mask.shape, dtype=np.int64),
        affine=image.affine,
        **extra_fields,
    )
    return buffer.getvalue()


def decode_sparse_npz(data: bytes) -> nib.Nifti1Image:
    """Reference decoder: float32 image. Extra fields are read with `np.load`."""
    fields = np.load(io.BytesIO(data))
    shape = tuple(int(size) for size in fields["shape"])
    mask = brle_to_dense(fields["mask_rle"]).reshape(shape)
    image_data = np.zeros(shape, dtype=np.float32)
    image_data[mask] = fields["values"] * np.float32(fields["scale"])
    return nib.Nifti1Image(image_data, fields["affine"])
