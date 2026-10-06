"""Model contract."""

from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

import numpy as np
import torch
import torch.nn as nn
from nibabel import Nifti1Image


class ImageInput(TypedDict):
    image: Nifti1Image  # one 3D volume
    brain_mask: NotRequired[Nifti1Image | None]  # on the image grid
    mni_affine: NotRequired[np.ndarray | None]  # (4, 4) image world mm -> MNI152 world mm


class EmbeddingOutput(TypedDict):
    global_embedding: torch.Tensor  # (D,)
    dense_embedding: torch.Tensor | None  # (X, Y, Z, D) patch features
    dense_affine: np.ndarray | None  # (4, 4) model input voxel index -> image world mm
    dense_mask: torch.Tensor | None  # (X, Y, Z) bool, False where the model dropped a token


class Model(nn.Module):
    """A frozen encoder.

    The dense grid is the model's input voxel grid cut into `patch_size` blocks, so the input
    grid has shape `(X, Y, Z) * patch_size` and `dense_affine` maps its voxels to world mm.
    """

    name: str
    embed_dim: int
    patch_size: tuple[int, int, int]  # in model input voxels

    def transform(self, sample: ImageInput) -> dict[str, Any]:
        """Preprocess one sample on CPU.

        Outputs are not collated. Implementation can be ommitted if there is no CPU preprocessing.
        """
        raise NotImplementedError

    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        """Embed a list of transformed samples.

        Returns float32 tensors on the model's device. The model handles batching and picks its own
        amp and dtype.
        """
        raise NotImplementedError


MODELS: dict[str, Callable[..., Model]] = {}


def register_model(model_fn: Callable[..., Model]) -> Callable[..., Model]:
    MODELS[model_fn.__name__] = model_fn
    return model_fn


def create_model(name: str, **kwargs) -> Model:
    if name not in MODELS:
        raise ValueError(
            f"Unknown model {name!r}; available: {list_models()}. "
            f"If it is a built-in baseline, install its extra. E.g. `uv sync --all-extras`."
        )
    return MODELS[name](**kwargs)


def list_models() -> list[str]:
    return sorted(MODELS)
