# Brainmarks-sMRI

An open benchmark for structural brain MRI foundation models. We train linear probes on frozen model embeddings to predict clinical and demographic targets and to segment lesions. The benchmark is a work in progress, so tasks and APIs may change.

## Install

```sh
git clone https://github.com/MedARC-AI/brainmarks-smri && cd brainmarks-smri
uv sync --all-extras
```

Each baseline model is an optional extra: `brainiac`, `braindino`, `neurojepa`, `neurovfm`, `walnut`. Some weights are gated on Hugging Face. Accept their terms first.

## Data

The benchmark uses ten public datasets. We add standard metadata tables and fixed train/val/test splits to each one. Every image also has a brain mask and an affine to MNI space. The dataset card, [`README_hf.md`](README_hf.md), lists the datasets and their licenses.

We plan to host the datasets at [medarc/brainmarks-smri](https://huggingface.co/datasets/medarc/brainmarks-smri) and download them automatically. This is still under construction. For now, the benchmark reads from a local `datasets/` folder (or `$DATA_ROOT`). The scripts in `scripts/` show how we downloaded and built each dataset.

## Run

```sh
uv run --all-extras python -m brainmarks_smri neurojepa ixi_age
```

Each run evaluates one model on one task. The results go to `output/<model>/<task>.json`. This file has the metrics with bootstrap confidence intervals, and the prediction for each sample. Default settings are in [`config/default.yaml`](src/brainmarks_smri/config/default.yaml). You can change them on the command line, e.g. `--overrides max_per_split=50`.

| Type | Tasks | Probe | Metric |
|---|---|---|---|
| classification | ABIDE autism, ADHD-200, CNP diagnosis, UCSF IDH, UPENN 1-year survival, SOOP mRS | logistic regression | AUROC |
| regression | IXI age, OpenBHB age | ridge | R² |
| segmentation | BraTS tumor regions (3), SOOP stroke lesion | patch-level logistic regression | Dice |

Probe hyperparameters are tuned by cross-validation on the train split. Scores are reported on the val split. By default, each split is capped at 300 samples.

## Add a model

A model is an `nn.Module` with two methods, `transform` and `forward_embeddings`. You also register a function that builds the model and loads its weights. The full contract is in [`models/base.py`](src/brainmarks_smri/models/base.py).

```python
@register_model
def my_model() -> MyModel:  # name, embed_dim, patch_size
    ...

# transform(sample) -> anything: preprocess one image on CPU.
#   sample = {"image": Nifti1Image, "brain_mask": Nifti1Image, "mni_affine": 4x4 array}
# forward_embeddings(samples, return_dense=False) -> one dict per sample:
#   global_embedding (D,), and for segmentation dense_embedding (X, Y, Z, D) + dense_affine
```

## License

The code is MIT licensed. Each dataset keeps its original license. Three of them are non-commercial. Please cite the sources listed in each dataset's README.
