import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.affines import apply_affine
from scipy import ndimage

from brainmarks_smri.models.voco import (
    CHECKPOINT,
    IMG_SIZE,
    REPO_ID,
    REVISION,
    LegacyPatchMerging,
    VoCo,
    VoCoEncoder,
    voco,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="full-size forward")


@pytest.fixture(scope="module")
def model() -> VoCo:
    torch.manual_seed(0)
    return VoCo(VoCoEncoder())


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
def test_dense_affine_locates_blob(model: VoCo, zooms):
    blob = (20.0, -30.0, 15.0)
    sample = model.transform({"image": make_head(zooms, blob_world=blob)})
    input_affine = sample["input_affine"]
    volume = sample["volume"].float().numpy()
    assert volume.shape == IMG_SIZE
    np.testing.assert_allclose(input_affine[:3, :3], np.diag([1.5, 1.5, 1.5]), atol=1e-9)

    # the blob is the only dark component that doesn't touch the volume border
    dark, _ = ndimage.label(volume < 0.5)
    edges = [dark[[0, -1]], dark[:, [0, -1]], dark[:, :, [0, -1]]]
    border = set(np.concatenate([edge.ravel() for edge in edges])) | {0}
    interior = [label for label in np.unique(dark) if label not in border]
    assert len(interior) == 1
    found = apply_affine(input_affine, np.argwhere(dark == interior[0]).mean(0))
    assert np.linalg.norm(found - blob) < 1.0, found


def test_brain_mask_centres_input(model: VoCo):
    image = make_head()
    world = apply_affine(image.affine, np.indices(image.shape).reshape(3, -1).T)
    brain = np.linalg.norm(world - [10.0, 20.0, -5.0], axis=-1).reshape(image.shape) < 30
    mask = nib.Nifti1Image(brain.astype(np.uint8), image.affine)
    sample = model.transform({"image": image, "brain_mask": mask})
    centre = apply_affine(sample["input_affine"], (np.array(IMG_SIZE) - 1) / 2)
    np.testing.assert_allclose(centre, [10.0, 20.0, -5.0], atol=0.5)


def test_intensity_scaling(model: VoCo):
    image = make_head()
    sample = model.transform({"image": image})
    volume = sample["volume"].float()
    assert volume.min() >= 0 and volume.max() <= 1
    # [0.5th, 99.5th] percentiles -> [0, 1]; the noisy background puts the lower one below 0
    lower, upper = np.percentile(image.get_fdata(), (0.5, 99.5))
    outside = (0 - lower) / (upper - lower)
    assert volume[0, 0, 0].item() == pytest.approx(outside, abs=1e-4)
    # head 100 +- 5
    centre = volume[54:74, 54:74, 54:74]
    assert centre.mean().item() == pytest.approx((100 - lower) / (upper - lower), abs=0.02)


def test_forward_embeddings(model: VoCo):
    samples = [
        model.transform({"image": make_head(zooms)}) for zooms in [(1.0,) * 3, (1.2, 1.0, 1.5)]
    ]
    outputs = model.forward_embeddings(samples, return_dense=True)
    volumes = torch.stack([sample["volume"].float() for sample in samples])[:, None]
    with torch.inference_mode():
        stages = model.encoder(volumes)

    assert len(outputs) == 2
    for index, (sample, output) in enumerate(zip(samples, outputs)):
        assert output["global_embedding"].shape == (1152,)
        assert output["dense_embedding"].shape == (16, 16, 16, 192)
        assert output["dense_mask"] is None
        np.testing.assert_array_equal(output["dense_affine"], sample["input_affine"])
        expected = torch.cat([stage[index].mean((1, 2, 3)) for stage in stages])
        torch.testing.assert_close(output["global_embedding"], expected)
        torch.testing.assert_close(output["dense_embedding"], stages[3][index].permute(1, 2, 3, 0))

        alone = model.forward_embeddings([sample])[0]
        assert alone["dense_embedding"] is None
        torch.testing.assert_close(
            output["global_embedding"], alone["global_embedding"], atol=1e-4, rtol=1e-4
        )


def test_dense_stage(model: VoCo):
    stage_model = VoCo(model.encoder, dense_stage=1)
    assert stage_model.patch_size == (2, 2, 2)
    # encoder2 is a conv block on the patch embedding, so a bright cube shows up where it is
    volume = torch.zeros(IMG_SIZE)
    volume[40:48, 80:88, 20:28] = 1
    sample = {"volume": volume, "input_affine": np.eye(4)}
    dense = stage_model.forward_embeddings([sample], return_dense=True)[0]["dense_embedding"]
    assert dense.shape == (64, 64, 64, 48)
    # instance norm shifts the constant background, and zero padding changes the border
    interior = dense[4:-4, 4:-4, 4:-4]
    changed = (interior - interior[-1, -1, -1]).abs().amax(-1) > 1e-4
    centroid = torch.argwhere(changed).float().mean(0) + 4
    torch.testing.assert_close(centroid, torch.tensor([21.5, 41.5, 11.5]), atol=0.5, rtol=0)


def test_legacy_patch_merging(model: VoCo):
    for stage in [model.encoder.swinViT.layers1, model.encoder.swinViT.layers4]:
        assert isinstance(stage[0].downsample, LegacyPatchMerging)


def test_pretrained_keys():
    from huggingface_hub import hf_hub_download
    from monai.networks.nets import SwinUNETR

    path = hf_hub_download(REPO_ID, CHECKPOINT, revision=REVISION)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state) == set(VoCoEncoder().state_dict())

    network = SwinUNETR(in_channels=1, out_channels=1, feature_size=48, use_v2=True)
    missing, unexpected = network.load_state_dict(state, strict=False)
    assert not unexpected
    assert {key.split(".")[0] for key in missing} == {
        "decoder1",
        "decoder2",
        "decoder3",
        "decoder4",
        "decoder5",
        "out",
    }


@requires_cuda
def test_pretrained():
    model = voco().cuda()
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
