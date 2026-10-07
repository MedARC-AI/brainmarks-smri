import numpy as np
import torch
from scipy import ndimage

from brainmarks_smri.probes import patchify, resample, unpatchify


def test_patchify_round_trip():
    volume = torch.randn(8, 12, 4, 3)
    patches = patchify(volume, (4, 6, 2))
    assert patches.shape == (2, 2, 2, 48, 3)
    # the first patch holds the first block of voxels
    assert torch.equal(patches[0, 0, 0].reshape(4, 6, 2, 3), volume[:4, :6, :2])
    assert torch.equal(unpatchify(patches, (4, 6, 2)), volume)


def test_resample_matches_scipy():
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((20, 24, 16)).astype(np.float32)
    # oblique, scaled, flipped, shifted output -> input voxel map
    angle = np.deg2rad(10)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag([0.8, 1.1, -0.9])
    matrix[:3, 3] = [3.0, -1.5, 14.0]
    output_shape = (18, 22, 15)

    expected = ndimage.affine_transform(
        volume, matrix, output_shape=output_shape, order=1, cval=0.0
    )
    actual = resample(torch.from_numpy(volume)[None], matrix, output_shape, "bilinear")[0].numpy()
    # grid_sample also blends with zero padding at the border; compare the interior
    inside = (
        ndimage.affine_transform(
            np.ones_like(volume), matrix, output_shape=output_shape, order=0, cval=0
        )
        > 0
    )
    inside = ndimage.binary_erosion(inside, iterations=2)
    assert inside.sum() > 1000
    np.testing.assert_allclose(actual[inside], expected[inside], atol=1e-4)
