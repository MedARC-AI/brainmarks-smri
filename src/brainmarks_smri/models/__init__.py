import importlib

from brainmarks_smri.models.base import (
    EmbeddingOutput,
    ImageInput,
    Model,
    create_model,
    list_models,
    register_model,
)

# Each model needs its own extra. Importing its module registers it; skip the ones not installed.
for _module in ("brainiac", "braindino", "neurojepa", "neurovfm", "walnut"):
    try:
        importlib.import_module(f"{__name__}.{_module}")
    except ModuleNotFoundError:
        pass

__all__ = [
    "EmbeddingOutput",
    "ImageInput",
    "Model",
    "create_model",
    "list_models",
    "register_model",
]
