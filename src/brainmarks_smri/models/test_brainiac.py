import os

import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.affines import apply_affine

from brainmarks_smri.models import create_model
from brainmarks_smri.models.brainiac import IMG_SIZE, BrainIAC, rigid_template_to_image

BLOB = np.array([20.0, -30.0, 15.0])


@pytest.fixture(scope="module")
def model() -> BrainIAC:
    return BrainIAC()


def make_head(axcodes: tuple[str, str, str] = ("L", "A", "S")) -> nib.Nifti1Image:
    """A noisy ellipsoid 'head' at 1.2 x 1.0 x 1.5 mm with a bright 5 mm ball at BLOB."""
    shape = (150, 200, 110)
    zooms = np.array([1.2, 1.0, 1.5])
    affine = np.diag([*zooms, 1.0])
    affine[:3, 3] = -zooms * (np.array(shape) - 1) / 2
    world = apply_affine(affine, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    inside = ((world / [70.0, 90.0, 70.0]) ** 2).sum(-1) < 1
    rng = np.random.default_rng(0)
    data = np.where(inside, 100.0 + rng.normal(0, 5, shape), 0.0)
    data[np.linalg.norm(world - BLOB, axis=-1) < 5] = 1000.0
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    target = nib.orientations.axcodes2ornt(axcodes)
    return image.as_reoriented(
        nib.orientations.ornt_transform(nib.orientations.io_orientation(affine), target)
    )


def blob_world(sample: dict) -> np.ndarray:
    volume = sample["volume"][0].numpy()
    bright = np.argwhere(volume > volume.max() / 2)
    return apply_affine(sample["input_affine"], bright.mean(0))


def test_transform_matches_upstream(model: BrainIAC, tmp_path):
    """Upstream `get_validation_transform` on a file already in template (LAS) orientation."""
    from monai import transforms

    upstream = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Resized(keys=["image"], spatial_size=IMG_SIZE, mode="trilinear"),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
    )
    image = make_head(("L", "A", "S"))
    path = tmp_path / "head.nii.gz"
    nib.save(image, path)
    expected = upstream({"image": str(path)})["image"]

    sample = model.transform({"image": image})

    torch.testing.assert_close(sample["volume"], expected.as_tensor())


def test_transform_ignores_storage_orientation(model: BrainIAC):
    las = model.transform({"image": make_head(("L", "A", "S"))})
    rpi = model.transform({"image": make_head(("R", "P", "I"))})
    torch.testing.assert_close(las["volume"], rpi["volume"])
    np.testing.assert_allclose(las["input_affine"], rpi["input_affine"])


@pytest.mark.parametrize("use_mni", [False, True])
def test_dense_affine_locates_blob(model: BrainIAC, use_mni: bool):
    sample = {"image": make_head(("R", "A", "S"))}
    if use_mni:
        angle = np.deg2rad(10)
        mni_affine = np.diag([1.1, 1.1, 1.1, 1.0])  # the rigid part drops the scaling
        mni_affine[:3, :3] = mni_affine[:3, :3] @ [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        mni_affine[:3, 3] = [3.0, -5.0, 8.0]
        sample["mni_affine"] = mni_affine

    found = blob_world(model.transform(sample))
    assert np.linalg.norm(found - BLOB) < 1.5, found


def test_rigid_part_is_rigid():
    affine = np.diag([1.2, 0.8, 1.1, 1.0])
    affine[:3, 3] = [5.0, -3.0, 2.0]
    rigid = rigid_template_to_image(affine)
    np.testing.assert_allclose(rigid[:3, :3] @ rigid[:3, :3].T, np.eye(3), atol=1e-10)


def test_forward_embeddings(model: BrainIAC):
    samples = [model.transform({"image": make_head()}) for _ in range(2)]
    outputs = model.forward_embeddings(samples, return_dense=True)

    assert len(outputs) == 2
    for output in outputs:
        assert output["global_embedding"].shape == (768,)
        assert output["dense_embedding"].shape == (6, 6, 6, 768)
        assert output["dense_affine"].shape == (4, 4)
        torch.testing.assert_close(
            output["global_embedding"], output["dense_embedding"].mean((0, 1, 2))
        )


@pytest.mark.skipif("BRAINIAC_CKPT" not in os.environ, reason="set BRAINIAC_CKPT to BrainIAC.ckpt")
def test_pretrained():
    model = create_model("brainiac", checkpoint=os.environ["BRAINIAC_CKPT"])
    outputs = model.forward_embeddings([model.transform({"image": make_head()})])
    assert torch.isfinite(outputs[0]["global_embedding"]).all()
