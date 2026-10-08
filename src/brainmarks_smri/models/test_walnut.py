import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.affines import apply_affine
from scipy import ndimage

from brainmarks_smri.models import create_model
from brainmarks_smri.models.walnut import GRID_SIZE, Walnut, resample, vit_large

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="nested-tensor attention has no CPU path"
)


@pytest.fixture(scope="module")
def model() -> Walnut:
    # depth 0: no attention blocks, so the forward pass runs on CPU
    return Walnut(vit_large(depth=0))


def make_head(
    zooms: tuple[float, float, float] = (1.2, 1.0, 1.5),
    blob_world: tuple[float, float, float] | None = None,
) -> nib.Nifti1Image:
    """An ellipsoid 'head' stored LPI, centred on world 0.

    Noisy by default. With `blob_world`, noise-free with a dark 5 mm ball at that point.
    """
    zooms = np.array(zooms)
    shape = tuple(np.round(np.array([180.0, 220.0, 165.0]) / zooms).astype(int))
    affine = np.diag([-zooms[0], -zooms[1], -zooms[2], 1.0])
    affine[:3, 3] = zooms * (np.array(shape) - 1) / 2 * [1, 1, 1]
    world = apply_affine(affine, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    inside = ((world / [70.0, 90.0, 70.0]) ** 2).sum(-1) < 1
    data = np.where(inside, 100.0, 0.0)
    if blob_world is None:
        data += np.random.default_rng(0).normal(0, 5, shape)
    else:
        near_blob = np.linalg.norm(world - np.array(blob_world), axis=-1) < 5
        data[near_blob] = 0.0
    return nib.Nifti1Image(data.astype(np.float32), affine)


@pytest.mark.parametrize("zooms", [(1.0, 1.0, 1.0), (1.2, 1.0, 1.5)])
def test_transform_matches_upstream(model: Walnut, zooms):
    from fomo_tune.backbone import SmriMaeTransform

    image = make_head(zooms)
    expected = SmriMaeTransform()(image)

    sample = model.transform({"image": image})
    volume, mask = model.prepare_volume(sample)

    np.testing.assert_allclose(sample["input_affine"], expected["affine"].numpy(), atol=1e-4)
    if zooms == (1.0, 1.0, 1.0):
        torch.testing.assert_close(volume, expected["image"][0], atol=1e-3, rtol=1e-3)
        assert torch.equal(mask, expected["mask"][0])
    else:
        # upstream's interpolate clamps at the volume edge, grid_sample pads with zeros
        assert (mask != expected["mask"][0]).float().mean() < 1e-3
        assert (volume - expected["image"][0]).abs().mean() < 1e-2


@pytest.mark.parametrize("use_mni", [False, True])
def test_dense_affine_locates_blob(model: Walnut, use_mni: bool):
    blob = (20.0, -30.0, 15.0)
    sample = {"image": make_head(blob_world=blob)}
    if use_mni:
        angle = np.deg2rad(10)
        mni_affine = np.eye(4)
        mni_affine[:3, :3] = 1.1 * np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        mni_affine[:3, 3] = [3.0, -5.0, 8.0]
        sample["mni_affine"] = mni_affine

    sample = model.transform(sample)
    input_affine = sample["input_affine"]
    volume = resample(sample["image"], sample["sampling_affine"], mode="bilinear")
    # the blob is the only dark component that doesn't touch the volume border
    dark, _ = ndimage.label(volume.numpy() < 50)
    edges = [dark[[0, -1]], dark[:, [0, -1]], dark[:, :, [0, -1]]]
    border = set(np.concatenate([edge.ravel() for edge in edges])) | {0}
    interior = [label for label in np.unique(dark) if label not in border]
    assert len(interior) == 1
    found = apply_affine(input_affine, np.argwhere(dark == interior[0]).mean(0))
    assert np.linalg.norm(found - blob) < 1.0, found
    # rigid, so head size is kept
    np.testing.assert_allclose(np.linalg.det(input_affine[:3, :3]), 1.0, atol=1e-6)


def test_brain_mask_selects_tokens(model: Walnut):
    image = make_head()
    brain = (np.asarray(image.dataobj) > 50).astype(np.uint8)
    sample = model.transform({"image": image, "brain_mask": nib.Nifti1Image(brain, image.affine)})
    volume, mask = model.prepare_volume(sample)

    assert volume[~mask].eq(0).all()
    assert abs(volume[mask].mean()) < 1e-4
    assert abs(volume[mask].std(correction=0) - 1) < 1e-4

    output = model.forward_embeddings([sample], return_dense=True)[0]
    patches = mask.reshape(GRID_SIZE[0], 8, GRID_SIZE[1], 8, GRID_SIZE[2], 8).any((1, 3, 5))
    assert torch.equal(output["dense_mask"], patches)


def test_forward_embeddings(model: Walnut):
    samples = [
        model.transform({"image": make_head(zooms)}) for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    outputs = model.forward_embeddings(samples, return_dense=True)

    assert len(outputs) == 2
    for sample, output in zip(samples, outputs):
        dense = output["dense_embedding"]
        dense_mask = output["dense_mask"]
        assert output["global_embedding"].shape == (1024,)
        assert dense.shape == (*GRID_SIZE, 1024)
        assert dense_mask.shape == GRID_SIZE
        assert output["dense_affine"].shape == (4, 4)
        assert dense[~dense_mask].eq(0).all()
        torch.testing.assert_close(output["global_embedding"], dense[dense_mask].mean(0))

        # without attention, a token is a function of its own patch, so this checks placement
        volume, _ = model.prepare_volume(sample)
        encoder = model.encoder
        patches = encoder.patchify(volume[None, None])
        expected = encoder.norm(encoder.pos_embed(encoder.patch_embed(patches)))[0]
        expected = expected.reshape(*GRID_SIZE, 1024)
        torch.testing.assert_close(dense[dense_mask], expected[dense_mask], atol=1e-4, rtol=1e-4)

    outputs = model.forward_embeddings(samples)
    assert outputs[0]["dense_embedding"] is None


@requires_cuda
def test_pretrained():
    model = create_model("walnut").cuda()
    samples = [
        model.transform({"image": make_head(zooms)}) for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    batched = model.forward_embeddings(samples, return_dense=True)
    for sample, output in zip(samples, batched):
        assert torch.isfinite(output["dense_embedding"]).all()
        # packing several volumes doesn't leak between them (bf16 noise)
        alone = model.forward_embeddings([sample])[0]
        torch.testing.assert_close(
            output["global_embedding"], alone["global_embedding"], atol=0.05, rtol=0.05
        )
