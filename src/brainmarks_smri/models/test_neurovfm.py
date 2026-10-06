import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from nibabel.affines import apply_affine

from brainmarks_smri.models import create_model
from brainmarks_smri.models.neurovfm import PATCH_SIZE, REPO_ID, REVISION, NeuroVFM

CENTRE = np.array([10.0, -20.0, 5.0])
RADII = np.array([60.0, 85.0, 45.0])


# `vision_backbone_cf` in upstream examples/config/mil.yaml, for a random-weight model.
# The pretrained model takes its config from the checkpoint.
BACKBONE_CONFIG = {
    "which": "vit_base",
    "params": {
        "token_dim": 1024,
        "embed_layer_cf": {
            "which": "voxel",
            "params": {"patch_hw_size": 16, "patch_d_size": 4, "in_chans": 1, "embed_dim": 738},
        },
        "pos_emb_cf": {
            "which": "pe3d",
            "params": {"d": 30, "d_size": 128, "hw_size": 192, "concat": True},
        },
    },
}


@pytest.fixture(scope="module")
def model() -> NeuroVFM:
    from neurovfm.models import get_vit_backbone
    from neurovfm.systems.utils import NormalizationModule

    return NeuroVFM(get_vit_backbone(**BACKBONE_CONFIG), NormalizationModule())


def make_head(
    zooms: tuple[float, float, float],
    flips: tuple[int, int, int],
    degrees: float = 0.0,
    fov: tuple[float, float, float] = (220.0, 220.0, 220.0),
    dtype: type = np.float32,
) -> nib.Nifti1Image:
    """A noisy ellipsoid 'head' at CENTRE with RADII, sampled at `zooms` mm with axis `flips`,
    the grid rotated by `degrees` about z, covering `fov` mm."""
    zooms = np.array(zooms)
    shape = np.ceil(np.array(fov) / zooms).astype(int)
    angle = np.deg2rad(degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag(zooms * flips)
    affine[:3, 3] = -affine[:3, :3] @ (shape - 1) / 2
    affine = affine.astype(np.float32).astype(np.float64)  # nifti stores the sform in float32
    world = apply_affine(affine, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    inside = (((world - CENTRE) / RADII) ** 2).sum(-1) < 1
    rng = np.random.default_rng(0)
    data = np.where(inside, 100.0, 0.0) + np.abs(rng.normal(0, 2, shape))
    image = nib.Nifti1Image(data.astype(dtype), affine)
    image.header.set_data_dtype(dtype)
    return image


HEADS = {
    "iso_ras": ((1.0, 1.0, 1.0), (1, 1, 1)),
    "thick_axial_lpi": ((0.9, 0.9, 5.0), (-1, -1, -1)),
    "thick_sagittal": ((5.0, 1.0, 1.1), (1, 1, 1)),
    "thick_coronal_las": ((1.0, 4.0, 1.2), (-1, 1, 1)),
    # column norms of a rotated affine are off by float noise; pixdims are exact
    "oblique_iso_sagittal_fov": ((1.0, 1.0, 1.0), (1, 1, 1), 7.0, (176.0, 220.0, 220.0)),
    # upstream resamples in the on-disk type, so int16 rounding decides the background
    "int16_axial": ((1.0, 1.0, 1.2), (-1, 1, 1), 0.0, (220.0, 220.0, 220.0), np.int16),
}


def test_transform_matches_upstream(model: NeuroVFM, tmp_path):
    from neurovfm.pipelines.preprocessor import StudyPreprocessor

    for name, params in HEADS.items():
        image = make_head(*params)
        path = tmp_path / f"{name}.nii.gz"
        nib.save(image, path)
        expected = StudyPreprocessor().load_study(path, modality="mri")

        sample = model.transform({"image": image})

        torch.testing.assert_close(sample["tokens"], expected["img"], msg=name)
        torch.testing.assert_close(sample["coords"], expected["coords"], msg=name)


@pytest.mark.parametrize("name", HEADS)
def test_dense_affine_places_head(model: NeuroVFM, name: str):
    """Foreground tokens sit inside the head, and points well inside it fall in foreground tokens."""
    sample = model.transform({"image": make_head(*HEADS[name])})
    patch = np.array(PATCH_SIZE)
    coords = sample["coords"].numpy()

    world = apply_affine(sample["input_affine"], coords * patch + (patch - 1) / 2)
    assert ((((world - CENTRE) / RADII) ** 2).sum(-1) < 1).all()

    probes = CENTRE + np.concatenate([np.zeros((1, 3)), np.diag(RADII / 2), -np.diag(RADII / 2)])
    voxels = apply_affine(np.linalg.inv(sample["input_affine"]), probes)
    probe_coords = np.floor((voxels + 0.5) / patch).astype(int)
    foreground = {tuple(coord) for coord in coords}
    assert all(tuple(coord) in foreground for coord in probe_coords)


def test_forward_embeddings(model: NeuroVFM):
    samples = [model.transform({"image": make_head(*HEADS[name])}) for name in HEADS]
    outputs = model.forward_embeddings(samples, return_dense=True)

    assert len(outputs) == len(samples)
    for sample, output in zip(samples, outputs):
        mask = output["dense_mask"]
        assert output["global_embedding"].shape == (768,)
        assert output["dense_embedding"].shape == (*sample["grid_shape"], 768)
        assert mask.shape == sample["grid_shape"]
        assert mask.sum() == len(sample["tokens"])
        torch.testing.assert_close(
            output["global_embedding"], output["dense_embedding"][mask].mean(0)
        )

    # packing must not mix sequences: the same sample alone gives the same embedding, up to bf16
    # noise (in fp32 the two agree to 5e-7)
    alone = model.forward_embeddings(samples[:1])
    torch.testing.assert_close(
        alone[0]["global_embedding"], outputs[0]["global_embedding"], atol=5e-3, rtol=0
    )


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="gated weights need HF_TOKEN")
def test_pretrained():
    from huggingface_hub import snapshot_download

    model = create_model("neurovfm")
    # upstream loads with strict=False; check nothing was silently skipped
    weights = torch.load(Path(snapshot_download(REPO_ID, revision=REVISION)) / "pytorch_model.bin")
    assert set(weights.get("state_dict", weights)) == set(model.backbone.state_dict())

    outputs = model.forward_embeddings([model.transform({"image": make_head(*HEADS["iso_ras"])})])
    assert torch.isfinite(outputs[0]["global_embedding"]).all()
