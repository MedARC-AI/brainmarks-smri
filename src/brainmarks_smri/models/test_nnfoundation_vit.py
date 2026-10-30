import nibabel as nib
import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from nibabel.affines import apply_affine
from scipy import ndimage

from brainmarks_smri.models.nnfoundation_vit import (
    CHECKPOINT,
    DEPTH,
    EMBED_DIM,
    IMG_SIZE,
    NUM_HEADS,
    PATCH_SIZE,
    REPO_ID,
    REVISION,
    NNFoundationViT,
    nnfoundation_vit,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="full size needs a GPU")


@pytest.fixture(scope="module")
def shallow_model() -> NNFoundationViT:
    # depth 0: no attention blocks, so a token is a function of its own patch
    return NNFoundationViT(depth=0)


@pytest.fixture(scope="module")
def small_model() -> NNFoundationViT:
    torch.manual_seed(0)
    return NNFoundationViT(depth=2, img_size=(32, 48, 24))


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
def test_dense_affine_locates_blob(shallow_model: NNFoundationViT, zooms):
    blob = (20.0, -30.0, 15.0)
    image = make_head(zooms, blob_world=blob)
    sample = shallow_model.transform({"image": image})
    input_affine = sample["input_affine"]
    volume = sample["volume"].float().numpy()
    assert volume.shape == IMG_SIZE

    # noise-free, so the nonzero box is the head's and the z-score threshold follows from it
    data = image.get_fdata()
    box = data[ndimage.find_objects((data != 0).astype(int))[0]]
    threshold = (50 - box.mean()) / box.std()
    dark, num_labels = ndimage.label(volume < threshold)
    # the blob (~520 voxels) is the only small dark component
    sizes = ndimage.sum_labels(np.ones_like(dark), dark, range(1, num_labels + 1))
    small = np.flatnonzero(sizes < 2000) + 1
    assert len(small) == 1
    found = apply_affine(input_affine, np.argwhere(dark == small[0]).mean(0))
    assert np.linalg.norm(found - blob) < 1.0, found
    # 1 mm, RAS
    np.testing.assert_allclose(input_affine[:3, :3], np.eye(3), atol=1e-6)


def test_one_mm_is_not_interpolated(shallow_model: NNFoundationViT):
    image = make_head((1.0, 1.0, 1.0))
    sample = shallow_model.transform({"image": image})
    data = image.get_fdata(dtype=np.float32)
    zscored = (data - data.mean()) / data.std()

    input_voxels = np.indices(IMG_SIZE).reshape(3, -1).T[::997]
    image_voxels = apply_affine(np.linalg.inv(image.affine) @ sample["input_affine"], input_voxels)
    np.testing.assert_allclose(image_voxels, np.round(image_voxels), atol=1e-6)
    image_voxels = np.round(image_voxels).astype(int)
    inside = ((image_voxels >= 0) & (image_voxels < data.shape)).all(1)
    assert inside.mean() > 0.5
    expected = zscored[tuple(image_voxels[inside].T)]
    actual = sample["volume"].float().numpy()[tuple(input_voxels[inside].T)]
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
    assert (sample["volume"][tuple(input_voxels[~inside].T)] == 0).all()


def test_centred_on_brain_mask(shallow_model: NNFoundationViT):
    image = make_head()
    world = apply_affine(image.affine, np.indices(image.shape).reshape(3, -1).T)
    brain_centre = np.array([10.0, 20.0, -15.0])
    brain = np.linalg.norm(world - brain_centre, axis=-1).reshape(image.shape) < 30
    brain_mask = nib.Nifti1Image(brain.astype(np.uint8), image.affine)

    sample = shallow_model.transform({"image": image, "brain_mask": brain_mask})
    grid_centre = apply_affine(sample["input_affine"], (np.array(IMG_SIZE) - 1) / 2)
    # snapped to image voxels of up to 1.5 mm
    assert np.abs(grid_centre - brain_centre).max() <= 0.75 + 1e-6, grid_centre


def test_dense_placement(shallow_model: NNFoundationViT):
    sample = shallow_model.transform({"image": make_head()})
    output = shallow_model.forward_embeddings([sample], return_dense=True)[0]
    dense = output["dense_embedding"]
    grid = tuple(size // patch for size, patch in zip(IMG_SIZE, PATCH_SIZE))
    assert dense.shape == (*grid, EMBED_DIM)
    assert output["global_embedding"].shape == (EMBED_DIM,)
    assert output["dense_mask"] is None
    np.testing.assert_array_equal(output["dense_affine"], sample["input_affine"])
    torch.testing.assert_close(output["global_embedding"], dense.reshape(-1, EMBED_DIM).mean(0))

    volume = sample["volume"].float()
    eva = shallow_model.eva
    for x, y, z in [(0, 0, 0), (12, 3, 20), (23, 17, 5)]:
        patch = volume[8 * x : 8 * x + 8, 8 * y : 8 * y + 8, 8 * z : 8 * z + 8]
        token = shallow_model.down_projection(patch[None, None]).reshape(EMBED_DIM)
        token = eva.norm(token + eva.pos_embed[0, (x * grid[1] + y) * grid[2] + z])
        torch.testing.assert_close(dense[x, y, z], token, atol=1e-4, rtol=1e-4)

    assert shallow_model.forward_embeddings([sample])[0]["dense_embedding"] is None


def test_batched_matches_alone(small_model: NNFoundationViT):
    samples = [
        small_model.transform({"image": make_head(zooms)})
        for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    assert samples[0]["volume"].shape == (32, 48, 24)
    batched = small_model.forward_embeddings(samples, return_dense=True)
    small_model.max_batch_size = 1
    try:
        alone = small_model.forward_embeddings(samples, return_dense=True)
    finally:
        small_model.max_batch_size = 8
    for batched_output, alone_output in zip(batched, alone):
        assert batched_output["dense_embedding"].shape == (4, 6, 3, EMBED_DIM)
        torch.testing.assert_close(
            batched_output["dense_embedding"], alone_output["dense_embedding"]
        )
    assert not torch.allclose(batched[0]["global_embedding"], batched[1]["global_embedding"])


def test_rope_table(small_model: NNFoundationViT):
    rope = small_model.eva.rope
    # one row per token; sin and cos over the full head dim
    assert rope.pos_embed.shape == (4 * 6 * 3, 2 * EMBED_DIM // NUM_HEADS)
    assert "eva.rope.pos_embed" not in small_model.state_dict()
    # a registered buffer, so it follows the module
    moved = NNFoundationViT(depth=0, img_size=(16, 16, 16)).to(torch.float64)
    assert moved.eva.rope.pos_embed.dtype == torch.float64


def test_pretrained_keys():
    path = hf_hub_download(REPO_ID, CHECKPOINT, revision=REVISION)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    arch = checkpoint["nnssl_adaptation_plan"]["architecture_plans"]["arch_kwargs"]
    assert arch["embed_dim"] == EMBED_DIM
    assert arch["encoder_eva_depth"] == DEPTH
    assert arch["encoder_eva_numheads"] == NUM_HEADS
    assert tuple(arch["patch_embed_size"]) == PATCH_SIZE
    assert tuple(arch["input_shape"]) == IMG_SIZE
    assert arch["init_values"] == 0.1 and arch["scale_attn_inner"]
    # strict load of down_projection.* and eva.*
    nnfoundation_vit()


@requires_cuda
def test_pretrained():
    model = nnfoundation_vit().cuda()
    samples = [
        model.transform({"image": make_head(zooms)}) for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    batched = model.forward_embeddings(samples, return_dense=True)
    for sample, output in zip(samples, batched):
        assert output["dense_embedding"].shape == (24, 24, 24, EMBED_DIM)
        assert torch.isfinite(output["dense_embedding"]).all()
        alone = model.forward_embeddings([sample])[0]
        torch.testing.assert_close(
            output["global_embedding"], alone["global_embedding"], atol=0.05, rtol=0.05
        )
