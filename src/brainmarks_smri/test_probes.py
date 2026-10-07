import numpy as np
import torch
from scipy import ndimage

from brainmarks_smri.logistic import LogisticRegressionVal
from brainmarks_smri.metrics import dice
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


def test_logistic_regression_val_learns_linear_rule():
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(600, 8, generator=generator)
    # 2 outputs x 2 channels, each a linear rule on the features
    weights = torch.randn(8, 4, generator=generator)
    targets = (features @ weights > 0.5).reshape(600, 2, 2)
    classifier = LogisticRegressionVal(alphas=(1e-1, 1e1, 1e5), max_negative_ratio=None)
    classifier.fit(features[:400], targets[:400], features[400:500], targets[400:500])
    predicted = classifier.predict(features[500:])
    assert predicted.shape == (100, 2, 2)
    assert classifier.alpha_ != 1e5  # heavy penalty can't fit the rule
    channel_dice = dice(predicted.permute(2, 0, 1), targets[500:].permute(2, 0, 1))
    assert (channel_dice > 0.9).all()
