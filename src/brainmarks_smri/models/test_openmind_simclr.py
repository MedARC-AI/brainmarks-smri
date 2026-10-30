import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.affines import apply_affine
from scipy import ndimage

from brainmarks_smri.models.openmind_simclr import (
    CHECKPOINT,
    IMG_SIZE,
    REPO_ID,
    REVISION,
    OpenMindSimCLR,
    openmind_simclr,
    res_enc_l,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="full size is slow on CPU")


@pytest.fixture(scope="module")
def model() -> OpenMindSimCLR:
    # one block per stage keeps the CPU forward cheap
    torch.manual_seed(0)
    return OpenMindSimCLR(res_enc_l(n_blocks_per_stage=(1,) * 6))


def make_head(
    zooms: tuple[float, float, float] = (1.2, 1.0, 1.5),
    blob_world: tuple[float, float, float] | None = None,
    fov: tuple[float, float, float] = (180.0, 220.0, 165.0),
    flip: bool = True,
) -> nib.Nifti1Image:
    """An ellipsoid 'head' centred on world 0, stored LPI if `flip` else RAS.

    Noisy by default. With `blob_world`, noise-free with a dark 5 mm ball at that point.
    """
    zooms = np.array(zooms)
    shape = tuple(np.round(np.array(fov) / zooms).astype(int))
    sign = -1 if flip else 1
    affine = np.diag([sign * zooms[0], sign * zooms[1], sign * zooms[2], 1.0])
    affine[:3, 3] = -sign * zooms * (np.array(shape) - 1) / 2
    world = apply_affine(affine, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    inside = ((world / [70.0, 90.0, 70.0]) ** 2).sum(-1) < 1
    data = np.where(inside, 100.0, 0.0)
    if blob_world is None:
        data += np.random.default_rng(0).normal(0, 5, shape)
    else:
        near_blob = np.linalg.norm(world - np.array(blob_world), axis=-1) < 5
        data[near_blob] = 0.0
    return nib.Nifti1Image(data.astype(np.float32), affine)


@pytest.mark.parametrize("use_mask", [False, True])
def test_dense_affine_locates_blob(model: OpenMindSimCLR, use_mask: bool):
    blob = (20.0, -30.0, 15.0)
    image = make_head(blob_world=blob)
    sample = {"image": image}
    if use_mask:
        # an off-centre mask moves the crop
        world = apply_affine(image.affine, np.indices(image.shape).reshape(3, -1).T)
        brain = (np.linalg.norm(world - [0.0, -20.0, 10.0], axis=-1) < 50).reshape(image.shape)
        sample["brain_mask"] = nib.Nifti1Image(brain.astype(np.uint8), image.affine)

    sample = model.transform(sample)
    volume = sample["volume"].numpy()
    input_affine = sample["input_affine"]
    assert volume.shape == IMG_SIZE
    np.testing.assert_allclose(np.abs(np.linalg.det(input_affine[:3, :3])), 1.0, atol=1e-6)
    # input axes are RAS
    assert (np.diag(input_affine[:3, :3]) > 0).all()

    centre = apply_affine(input_affine, (np.array(IMG_SIZE) - 1) / 2)
    expected_centre = [0.0, -20.0, 10.0] if use_mask else [0.0, 0.0, 0.0]
    assert np.linalg.norm(centre - expected_centre) < 1.5, centre

    # the blob is the only dark component that doesn't touch the volume border
    dark, _ = ndimage.label(volume < (volume.max() + volume.min()) / 2)
    edges = [dark[[0, -1]], dark[:, [0, -1]], dark[:, :, [0, -1]]]
    border = set(np.concatenate([edge.ravel() for edge in edges])) | {0}
    interior = [label for label in np.unique(dark) if label not in border]
    assert len(interior) == 1
    found = apply_affine(input_affine, np.argwhere(dark == interior[0]).mean(0))
    assert np.linalg.norm(found - blob) < 1.0, found


def test_transform_ras_1mm_is_crop_of_zscore(model: OpenMindSimCLR):
    image = make_head(zooms=(1.0, 1.0, 1.0), flip=False)
    data = image.get_fdata(dtype=np.float32)
    sample = model.transform({"image": image})

    start = np.round(sample["input_affine"][:3, 3] - image.affine[:3, 3]).astype(int)
    np.testing.assert_allclose(sample["input_affine"][:3, :3], np.eye(3), atol=1e-6)
    # z-scored over the nonzero bounding box, zeros outside it
    box = tuple(slice(indices.min(), indices.max() + 1) for indices in np.nonzero(data))
    expected = np.zeros_like(data)
    expected[box] = (data[box] - data[box].mean()) / data[box].std()
    expected = expected[tuple(slice(s, s + n) for s, n in zip(start, IMG_SIZE))]
    np.testing.assert_allclose(sample["volume"].numpy(), expected, atol=1e-4)


def test_transform_pads_small_image(model: OpenMindSimCLR):
    image = make_head(zooms=(1.0, 1.0, 1.0), fov=(150.0, 190.0, 150.0))
    sample = model.transform({"image": image})
    volume = sample["volume"]
    # 150 mm centred in 160: 5 zero slices each side
    assert volume[:5].eq(0).all() and volume[-5:].eq(0).all()
    assert volume[5].ne(0).any() and volume[-6].ne(0).any()


def test_forward_embeddings(model: OpenMindSimCLR):
    samples = [
        model.transform({"image": make_head(zooms)}) for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    outputs = model.forward_embeddings(samples, return_dense=True)

    assert len(outputs) == 2
    for sample, output in zip(samples, outputs):
        dense = output["dense_embedding"]
        assert output["global_embedding"].shape == (1120,)
        assert dense.shape == (20, 20, 20, 256)
        assert output["dense_mask"] is None
        assert output["dense_affine"] is sample["input_affine"]

        stage_outputs = model.encoder(sample["volume"][None, None])
        expected_global = torch.cat([stage[0].mean((1, 2, 3)) for stage in stage_outputs])
        # batched == alone: instance norm is per sample
        torch.testing.assert_close(output["global_embedding"], expected_global)
        torch.testing.assert_close(dense, stage_outputs[3][0].permute(1, 2, 3, 0))

    assert model.patch_size == (8, 8, 8)
    assert model.forward_embeddings(samples)[0]["dense_embedding"] is None


def test_dense_stage(model: OpenMindSimCLR):
    stage_model = OpenMindSimCLR(model.encoder, dense_stage=2)
    assert stage_model.patch_size == (4, 4, 4)
    sample = model.transform({"image": make_head()})
    output = stage_model.forward_embeddings([sample], return_dense=True)[0]
    assert output["dense_embedding"].shape == (40, 40, 40, 128)


def test_checkpoint_keys():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, CHECKPOINT, revision=REVISION)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["network_weights"]
    encoder_keys = {key.removeprefix("encoder.") for key in state if key.startswith("encoder.")}
    other_keys = {key for key in state if not key.startswith("encoder.")}
    assert encoder_keys == set(res_enc_l().state_dict())
    assert {key.split(".")[0] for key in other_keys} == {"projector"}
    assert state["projector.layer1.0.weight"].shape == (2048, 1120)


@requires_cuda
def test_pretrained():
    model = openmind_simclr().cuda()
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
