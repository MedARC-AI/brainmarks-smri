import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.affines import apply_affine
from scipy import ndimage

from brainmarks_smri.models.amaes import IMG_SIZE, AMAES, amaes, resenc_b_encoder

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="full size is slow")


@pytest.fixture(scope="module")
def model() -> AMAES:
    torch.manual_seed(0)
    # one block per stage keeps the CPU forward cheap
    return AMAES(resenc_b_encoder(n_blocks_per_stage=(1, 1, 1, 1, 1, 1)))


def make_head(
    zooms: tuple[float, float, float] = (1.2, 1.0, 1.5),
    blob_world: tuple[float, float, float] | None = None,
    size_mm: tuple[float, float, float] = (180.0, 220.0, 165.0),
) -> nib.Nifti1Image:
    """An ellipsoid 'head' stored LPI, centred on world 0.

    Noisy by default. With `blob_world`, noise-free with a dark 5 mm ball at that point.
    """
    zooms = np.array(zooms)
    shape = tuple(np.round(np.array(size_mm) / zooms).astype(int))
    affine = np.diag([-zooms[0], -zooms[1], -zooms[2], 1.0])
    affine[:3, 3] = zooms * (np.array(shape) - 1) / 2
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
def test_dense_affine_locates_blob(model: AMAES, zooms):
    blob = (20.0, -30.0, 15.0)
    sample = model.transform({"image": make_head(zooms, blob_world=blob)})
    volume = sample["volume"].float().numpy()
    input_affine = sample["input_affine"]

    # 1 mm, axes along RAS
    np.testing.assert_allclose(input_affine[:3, :3], np.eye(3), atol=1e-6)
    # the blob is the only dark component that doesn't touch the volume border
    dark, _ = ndimage.label(volume < (volume.max() + volume.min()) / 2)
    edges = [dark[[0, -1]], dark[:, [0, -1]], dark[:, :, [0, -1]]]
    border = set(np.concatenate([edge.ravel() for edge in edges])) | {0}
    interior = [label for label in np.unique(dark) if label not in border]
    assert len(interior) == 1
    found = apply_affine(input_affine, np.argwhere(dark == interior[0]).mean(0))
    assert np.linalg.norm(found - blob) < 1.0, found


def test_crop_centred_on_brain_mask(model: AMAES):
    image = make_head()
    world = apply_affine(image.affine, np.indices(image.shape).reshape(3, -1).T)
    brain_centre = np.array([10.0, 25.0, -5.0])
    brain = np.linalg.norm(world - brain_centre, axis=-1) < 40
    brain_mask = nib.Nifti1Image(brain.reshape(image.shape).astype(np.uint8), image.affine)
    sample = model.transform({"image": image, "brain_mask": brain_mask})

    crop_centre = apply_affine(sample["input_affine"], (np.array(IMG_SIZE) - 1) / 2)
    assert np.abs(crop_centre - brain_centre).max() <= 1.0, crop_centre


def test_zscore_and_padding(model: AMAES):
    image = make_head(zooms=(1.0, 1.0, 1.0), size_mm=(120.0, 200.0, 100.0))
    sample = model.transform({"image": image})
    volume = sample["volume"].float()

    data = np.asarray(image.dataobj)
    first = (data[0, 20, 0] - data.mean()) / data.std()
    # the image fits in the crop, so it is whole and zero padded around
    inside = volume != 0
    assert inside.sum() == np.prod([120, 160, 100])
    nonzero = torch.argwhere(inside)
    lower = nonzero.min(0).values.tolist()
    assert lower == [20, 0, 30]
    # stored LPI: voxel (0, 20, 0) is RAS voxel (119, 179, 99)
    assert abs(volume[139, 159, 129] - first) < 1e-2


def test_forward_embeddings(model: AMAES):
    samples = [
        model.transform({"image": make_head(zooms)}) for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    outputs = model.forward_embeddings(samples, return_dense=True)

    assert model.patch_size == (8, 8, 8)
    assert len(outputs) == 2
    for sample, output in zip(samples, outputs):
        assert output["global_embedding"].shape == (320,)
        assert output["dense_embedding"].shape == (20, 20, 20, 256)
        assert output["dense_mask"] is None
        assert output["dense_affine"] is sample["input_affine"]

        stages = model.encoder(sample["volume"].float()[None, None])
        torch.testing.assert_close(output["global_embedding"], stages[-1][0].mean((1, 2, 3)))
        torch.testing.assert_close(output["dense_embedding"], stages[3][0].permute(1, 2, 3, 0))

        alone = model.forward_embeddings([sample])[0]
        assert alone["dense_embedding"] is None
        torch.testing.assert_close(alone["global_embedding"], output["global_embedding"])


def test_pretrained_keys():
    model = amaes()  # strict load of every encoder key
    assert sum(param.numel() for param in model.parameters()) == pytest.approx(90.3e6, rel=0.01)


@requires_cuda
def test_pretrained():
    model = amaes().cuda()
    samples = [
        model.transform({"image": make_head(zooms)}) for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    batched = model.forward_embeddings(samples, return_dense=True)
    for sample, output in zip(samples, batched):
        assert torch.isfinite(output["global_embedding"]).all()
        assert torch.isfinite(output["dense_embedding"]).all()
        alone = model.forward_embeddings([sample])[0]
        torch.testing.assert_close(
            output["global_embedding"], alone["global_embedding"], atol=0.05, rtol=0.05
        )
