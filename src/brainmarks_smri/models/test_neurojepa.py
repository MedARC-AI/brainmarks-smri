import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy import ndimage
from nibabel.affines import apply_affine

from brainmarks_smri.models import create_model
from brainmarks_smri.models.neurojepa import IMG_SIZE, PATCH_SIZE, REPO_ID, REVISION, NeuroJEPA

# Architecture from the upstream finetune config, for a random-weight model. The pretrained
# model takes its config from the checkpoint.
BACKBONE_KWARGS = dict(
    model_name="vit_base",
    img_size=list(IMG_SIZE),
    patch_size=list(PATCH_SIZE),
    in_chans=1,
    uniform_power=True,
    use_sdpa=True,
    use_rope=True,
    use_silu=False,
    wide_silu=True,
    use_moe=True,
    moe_params=SimpleNamespace(
        moe_type="bias",
        dim=768,
        n_shared_experts=2,
        n_routed_experts=16,
        n_activated_experts=6,
        moe_inter_dim=384,
        score_func="softmax",
        route_scale=4.0,
        moe_layer_indices=[1, 3, 5, 7, 9, 11],
    ),
)


@pytest.fixture(scope="module")
def model() -> NeuroJEPA:
    from neurojepa.utils.init_utils import init_backbone

    return NeuroJEPA(init_backbone(device="cpu", **BACKBONE_KWARGS))


def make_head(blob_world: tuple[float, float, float] | None = None) -> nib.Nifti1Image:
    """An ellipsoid 'head' stored LPI at 1.2 x 1.0 x 1.5 mm, centred on world 0.

    Noisy by default. With `blob_world`, noise-free with a dark 5 mm ball at that point, which
    survives percentile scaling (a bright one saturates with the head).
    """
    shape = (150, 200, 110)
    zooms = np.array([1.2, 1.0, 1.5])
    affine = np.diag([-zooms[0], -zooms[1], -zooms[2], 1.0])
    affine[:3, 3] = zooms * (np.array(shape) - 1) / 2 * [1, 1, 1]
    rng = np.random.default_rng(0)
    world = apply_affine(affine, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    inside = ((world / [70.0, 90.0, 70.0]) ** 2).sum(-1) < 1
    data = np.where(inside, 100.0, 0.0)
    if blob_world is None:
        data += rng.normal(0, 5, shape)
    else:
        near_blob = np.linalg.norm(world - np.array(blob_world), axis=-1) < 5
        data[near_blob] = 0.0
    return nib.Nifti1Image(data.astype(np.float32), affine)


def test_transform_matches_upstream(model: NeuroJEPA, tmp_path):
    from neurojepa.data.transforms import loading_transforms, vit3d_transforms

    image = make_head()
    path = tmp_path / "head.nii.gz"
    nib.save(image, path)
    config = SimpleNamespace(data=SimpleNamespace(img_size=list(IMG_SIZE)))
    expected = loading_transforms(roi=IMG_SIZE)({"image": str(path)})
    expected = vit3d_transforms(config, mode="test")(expected)["image"]

    volume, affine = model.prepare_volume(model.transform({"image": image}))

    assert volume.shape == (1, *IMG_SIZE)
    torch.testing.assert_close(volume, expected.as_tensor(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(affine, expected.affine.numpy(), atol=1e-4)


@pytest.mark.parametrize("use_mni", [False, True])
def test_dense_affine_locates_blob(model: NeuroJEPA, use_mni: bool):
    blob = (20.0, -30.0, 15.0)
    image = make_head(blob_world=blob)
    sample = {"image": image}
    if use_mni:
        angle = np.deg2rad(10)
        mni_affine = np.eye(4)
        mni_affine[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        mni_affine[:3, 3] = [3.0, -5.0, 8.0]
        sample["mni_affine"] = mni_affine
        mask = (np.asarray(make_head().dataobj) > 50).astype(np.uint8)  # includes the blob
        sample["brain_mask"] = nib.Nifti1Image(mask, image.affine)

    volume, affine = model.prepare_volume(model.transform(sample))
    # the blob is the only dark component that doesn't touch the volume border
    dark, _ = ndimage.label(volume[0].numpy() < 0.5)
    edges = [dark[[0, -1]], dark[:, [0, -1]], dark[:, :, [0, -1]]]
    border = set(np.concatenate([edge.ravel() for edge in edges])) | {0}
    interior = [label for label in np.unique(dark) if label not in border]
    assert len(interior) == 1
    found = apply_affine(affine, np.argwhere(dark == interior[0]).mean(0))
    assert np.linalg.norm(found - blob) < 1.0, found


def test_forward_embeddings(model: NeuroJEPA):
    samples = [model.transform({"image": make_head()}) for _ in range(2)]
    outputs = model.forward_embeddings(samples, return_dense=True)

    assert len(outputs) == 2
    for output in outputs:
        assert output["global_embedding"].shape == (768,)
        assert output["dense_embedding"].shape == (8, 9, 8, 768)
        assert output["dense_affine"].shape == (4, 4)
        assert output["dense_mask"] is None
        torch.testing.assert_close(
            output["global_embedding"], output["dense_embedding"].mean((0, 1, 2))
        )

    outputs = model.forward_embeddings(samples)
    assert outputs[0]["dense_embedding"] is None


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="gated weights need HF_TOKEN")
def test_pretrained():
    from huggingface_hub import snapshot_download
    from neurojepa.utils.init_utils import (
        _clean_backbone_state_dict,
        _extract_backbone_state_dict,
        _load_hf_or_local_checkpoint,
    )

    model = create_model("neurojepa")
    # upstream loads with strict=False; check nothing was silently skipped
    path = Path(snapshot_download(REPO_ID, revision=REVISION)) / "model.safetensors"
    weights = _clean_backbone_state_dict(
        _extract_backbone_state_dict(_load_hf_or_local_checkpoint(path))
    )
    assert set(weights) == set(model.backbone.state_dict())

    outputs = model.forward_embeddings([model.transform({"image": make_head()})])
    assert torch.isfinite(outputs[0]["global_embedding"]).all()
