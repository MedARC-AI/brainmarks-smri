import os

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from nibabel.affines import apply_affine

from brainmarks_smri.models import create_model
from brainmarks_smri.models.braindino import BrainDINO, zscore_normalize

CENTRE = np.array([10.0, -20.0, 5.0])
RADII = np.array([60.0, 85.0, 45.0])


@pytest.fixture(scope="module")
def model() -> BrainDINO:
    from dinov3.models.vision_transformer import vit_base

    # as upstream SliceStudent, random weights
    backbone = vit_base(layerscale_init=1e-5, n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    backbone.init_weights()
    return BrainDINO(backbone)


def make_head(axcodes: tuple[str, str, str] = ("R", "A", "S")) -> nib.Nifti1Image:
    """A noisy ellipsoid 'head' at CENTRE with RADII, 1.2 x 1.0 x 1.5 mm, stored in `axcodes`."""
    shape = (180, 220, 140)
    zooms = np.array([1.2, 1.0, 1.5])
    affine = np.diag([*zooms, 1.0])
    affine[:3, 3] = -zooms * (np.array(shape) - 1) / 2
    world = apply_affine(affine, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    inside = (((world - CENTRE) / RADII) ** 2).sum(-1) < 1
    rng = np.random.default_rng(0)
    data = np.where(inside, 100.0 + rng.normal(0, 5, shape), 0.0)
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    target = nib.orientations.axcodes2ornt(axcodes)
    return image.as_reoriented(
        nib.orientations.ornt_transform(nib.orientations.io_orientation(affine), target)
    )


def test_transform_matches_upstream(model: BrainDINO):
    """Upstream ABIDE loader (`load_nii` + `__getitem__`, no augmentation) on a RAS file."""
    image = make_head(("R", "A", "S"))
    volume = np.transpose(image.get_fdata().astype(np.float32), (2, 0, 1))
    volume = torch.from_numpy(zscore_normalize(volume))[None, None]
    volume = F.interpolate(
        volume, size=(128, volume.shape[3], volume.shape[4]), mode="trilinear", align_corners=False
    )[0, 0]
    expected = F.interpolate(volume[None], size=(224, 224), mode="bilinear", align_corners=False)

    sample = model.transform({"image": image})

    torch.testing.assert_close(sample["volume"], expected)


def test_transform_ignores_storage_orientation(model: BrainDINO):
    ras = model.transform({"image": make_head(("R", "A", "S"))})
    lpi = model.transform({"image": make_head(("L", "P", "I"))})
    torch.testing.assert_close(ras["volume"], lpi["volume"])
    np.testing.assert_allclose(ras["input_affine"], lpi["input_affine"])


def test_input_affine_places_head(model: BrainDINO):
    sample = model.transform({"image": make_head(("L", "P", "I"))})
    volume = sample["volume"][0].permute(1, 2, 0).numpy()  # (x, y, z) input grid
    inside = np.argwhere(volume > volume.min() / 2)
    world = apply_affine(sample["input_affine"], inside)

    np.testing.assert_allclose(world.mean(0), CENTRE, atol=1.0)
    np.testing.assert_allclose(world.max(0) - world.min(0), 2 * RADII, atol=4.0)


def test_forward_embeddings(model: BrainDINO):
    outputs = model.forward_embeddings([model.transform({"image": make_head()})], return_dense=True)

    output = outputs[0]
    assert output["global_embedding"].shape == (768,)
    assert output["dense_embedding"].shape == (14, 14, 128, 768)
    assert output["dense_affine"].shape == (4, 4)
    assert output["dense_mask"] is None


@pytest.mark.skipif("BRAINDINO_CKPT" not in os.environ, reason="BrainDINO weights are unreleased")
def test_pretrained():
    model = create_model("braindino", checkpoint=os.environ["BRAINDINO_CKPT"])
    outputs = model.forward_embeddings([model.transform({"image": make_head()})])
    assert torch.isfinite(outputs[0]["global_embedding"]).all()


def test_dinov3_matches_timm():
    """The official architecture with converted weights reproduces timm's model exactly."""
    import timm

    model = create_model("dinov3_vitb16")
    reference = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True).eval()
    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ours = model.backbone.forward_features(images)
        expected = reference.forward_features(images)  # (B, 1 + 4 + 196, 768), after the norm
    torch.testing.assert_close(ours["x_norm_clstoken"], expected[:, 0])
    torch.testing.assert_close(ours["x_norm_patchtokens"], expected[:, 5:])

    outputs = model.forward_embeddings([model.transform({"image": make_head()})])
    assert outputs[0]["global_embedding"].shape == (768,)
