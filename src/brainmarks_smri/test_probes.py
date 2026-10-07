import numpy as np
import torch
from scipy import ndimage

from brainmarks_smri.logistic import TorchLogisticRegressionCV
from brainmarks_smri.metrics import average_precision, dice, voxel_auroc
from brainmarks_smri.probes import resample


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


def test_logistic_regression_cv_learns_linear_rule():
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(600, 8, generator=generator)
    # 4 outputs, each a linear rule on the features
    weights = torch.randn(8, 4, generator=generator)
    targets = features @ weights > 0.5
    groups = np.arange(600) // 10
    classifier = TorchLogisticRegressionCV(alphas=(1e-1, 1e1, 1e5), max_negative_ratio=None)
    classifier.fit(features[:500], targets[:500], groups[:500])
    assert classifier.alpha_ != 1e5  # heavy penalty can't fit the rule
    assert dice(classifier.predict(features[500:]), targets[500:]) > 0.9


def test_voxel_ranking_metrics():
    target = torch.tensor([False, False, True, True])
    perfect = torch.tensor([0.1, 0.2, 0.8, 0.9])
    assert voxel_auroc(perfect, target) == 1.0
    assert average_precision(perfect, target) == 1.0
    swapped = torch.tensor([0.1, 0.9, 0.8, 0.2])
    # ranked: 0.9 (neg), 0.8 (pos), 0.2 (pos), 0.1 (neg)
    assert voxel_auroc(swapped, target) == 0.5
    assert abs(average_precision(swapped, target) - (1 / 2 + 2 / 3) / 2) < 1e-6
