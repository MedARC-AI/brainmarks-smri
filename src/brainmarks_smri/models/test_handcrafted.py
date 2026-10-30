import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.affines import apply_affine
from scipy import ndimage

from brainmarks_smri.models import create_model
from brainmarks_smri.models.handcrafted import (
    DENSE_DIM,
    GRID_SIZE,
    NUM_SCALARS,
    PATCH_SIZE,
    SMOOTHING_SIGMAS,
    Handcrafted,
)

DENSE_GRID = tuple(size // patch for size, patch in zip(GRID_SIZE, PATCH_SIZE))


@pytest.fixture(scope="module")
def model() -> Handcrafted:
    # every block kept
    return Handcrafted(np.ones(GRID_SIZE))


def test_template_support():
    model = create_model("handcrafted")
    kept = [int(kept.sum()) for kept in model.kept_blocks]
    # the MNI152 brain is ~1.8 l: 432 blocks of 16 mm and 56 of 32 mm
    assert kept == [432, 56]
    assert model.embed_dim == NUM_SCALARS + 4 * sum(kept)


def make_head(
    zooms: tuple[float, float, float] = (1.2, 1.0, 1.5),
    blob_world: tuple[float, float, float] | None = None,
) -> nib.Nifti1Image:
    """An ellipsoid 'head' stored LPI, centred on world 0, with three nested intensity shells.

    With `blob_world`, a dark 6 mm ball at that point.
    """
    zooms = np.array(zooms)
    shape = tuple(np.round(np.array([180.0, 220.0, 165.0]) / zooms).astype(int))
    affine = np.diag([-zooms[0], -zooms[1], -zooms[2], 1.0])
    affine[:3, 3] = zooms * (np.array(shape) - 1) / 2
    world = apply_affine(affine, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    radius = np.sqrt(((world / [70.0, 90.0, 70.0]) ** 2).sum(-1))
    data = np.select([radius < 0.5, radius < 0.8, radius < 1.0], [100.0, 60.0, 30.0], 0.0)
    data += np.random.default_rng(0).normal(0, 2, shape) * (radius < 1.0)
    if blob_world is not None:
        near_blob = np.linalg.norm(world - np.array(blob_world), axis=-1) < 6
        data[near_blob] = 0.0
    return nib.Nifti1Image(data.astype(np.float32), affine)


def scaled_rotation(scale: float) -> np.ndarray:
    angle = np.deg2rad(10)
    mni_affine = np.eye(4)
    mni_affine[:3, :3] = scale * np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    mni_affine[:3, 3] = [3.0, -5.0, 8.0]
    return mni_affine


@pytest.mark.parametrize("use_mni", [False, True])
def test_input_affine_locates_blob(model: Handcrafted, use_mni: bool):
    blob = (20.0, -30.0, 15.0)
    sample = {"image": make_head(blob_world=blob)}
    if use_mni:
        sample["mni_affine"] = scaled_rotation(1.1)

    transformed = model.transform(sample)
    intensity = transformed["maps"][0].numpy()
    # the blob is the only dark component that doesn't touch the volume border
    dark, _ = ndimage.label(intensity < 0.1)
    edges = [dark[[0, -1]], dark[:, [0, -1]], dark[:, :, [0, -1]]]
    border = set(np.concatenate([edge.ravel() for edge in edges])) | {0}
    interior = [label for label in np.unique(dark) if label not in border]
    assert len(interior) == 1
    found = apply_affine(transformed["input_affine"], np.argwhere(dark == interior[0]).mean(0))
    assert np.linalg.norm(found - blob) < 1.5, found


@pytest.mark.parametrize("scale", [1.0, 1.25])
def test_class_maps(model: Handcrafted, scale: float):
    image = make_head()
    transformed = model.transform({"image": image, "mni_affine": scaled_rotation(scale)})
    maps = transformed["maps"].numpy()
    brain = transformed["brain"].numpy()
    intensity, class_maps = maps[0], maps[1:]

    # brain median is 1; the class maps are modulated by the volume change
    assert np.median(intensity[brain]) == pytest.approx(1.0)
    np.testing.assert_allclose(class_maps.sum(0)[brain], 1 / scale**3, rtol=1e-4)
    assert (class_maps[:, ~brain] == 0).all()
    # classes are ordered dark to bright and follow the shells
    labels = class_maps.argmax(0)[brain]
    for label, level in enumerate([0.5, 1.0, 1.67]):
        shell = np.abs(intensity[brain] - level) < 0.1
        assert (labels[shell] == label).mean() > 0.99


def test_forward_embeddings(model: Handcrafted):
    samples = [
        model.transform({"image": make_head(zooms)}) for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    outputs = model.forward_embeddings(samples, return_dense=True)

    assert len(outputs) == 2
    for sample, output in zip(samples, outputs):
        assert output["global_embedding"].shape == (model.embed_dim,)
        assert output["dense_embedding"].shape == (*DENSE_GRID, DENSE_DIM)
        dense_mask = output["dense_mask"]
        patches = sample["brain"].reshape(DENSE_GRID[0], 4, DENSE_GRID[1], 4, DENSE_GRID[2], 4)
        assert torch.equal(dense_mask, patches.any(5).any(3).any(1))
        assert output["dense_embedding"][~dense_mask].eq(0).all()
        assert torch.isfinite(output["dense_embedding"]).all()

        # the first block features are 16 mm block means of the intensity map
        blocks = sample["maps"][0].reshape(12, 8, 14, 8, 12, 8).mean((1, 3, 5))
        torch.testing.assert_close(
            output["global_embedding"][NUM_SCALARS : NUM_SCALARS + blocks.numel()],
            blocks.flatten(),
        )

        # batching doesn't mix samples
        alone = model.forward_embeddings([sample], return_dense=True)[0]
        torch.testing.assert_close(output["global_embedding"], alone["global_embedding"])
        torch.testing.assert_close(output["dense_embedding"], alone["dense_embedding"])

    outputs = model.forward_embeddings(samples)
    assert outputs[0]["dense_embedding"] is None


def test_smooth_matches_scipy(model: Handcrafted):
    volume = np.random.default_rng(0).normal(size=(40, 44, 36)).astype(np.float32)
    for smoothing, sigma in zip(model.smoothing, SMOOTHING_SIGMAS):
        smoothed = smoothing(torch.from_numpy(volume)[None, None])[0, 0].numpy()
        expected = ndimage.gaussian_filter(volume, sigma, mode="constant", truncate=3.0)
        np.testing.assert_allclose(smoothed, expected, atol=1e-3)
