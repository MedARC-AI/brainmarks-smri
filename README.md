# Brainmarks-sMRI

An open benchmark for structural brain MRI foundation models. We train linear probes on frozen model embeddings to predict clinical and demographic targets and to segment lesions. The benchmark is a work in progress, so tasks and APIs may change.

## Install

```sh
git clone https://github.com/MedARC-AI/brainmarks-smri && cd brainmarks-smri
uv sync --all-extras
```

Each baseline model is an optional extra: `brainiac`, `braindino`, `neurojepa`, `neurovfm`, `walnut`. Some weights are gated on Hugging Face. Accept their terms first.

## Data

The benchmark uses public datasets. We add standard metadata tables and fixed train/val/test splits to each one. Every image also has a brain mask and an affine to MNI space. The dataset card, [`README_hf.md`](README_hf.md), lists the datasets and their licenses.

- **Hosted**: nine datasets are hosted at [medarc/brainmarks-smri](https://huggingface.co/datasets/medarc/brainmarks-smri). The repo is gated, so accept its terms and log in with `hf auth login` first. Then set `BRAINMARKS_SMRI_DATA=hf://datasets/medarc/brainmarks-smri`, and each dataset is downloaded to a local cache when a run first needs it. You can also download the repo yourself and set `BRAINMARKS_SMRI_DATA` to its folder.
- **Manual**: you build ADNI, A4/LEARN, ATLAS, OASIS-1, OASIS-2 and FOMO26 Tasks 4 and 5 yourself in your local `BRAINMARKS_SMRI_DATA` folder (default `datasets/`), because their licenses don't allow redistribution. Each one's README explains how to get access.

```sh
bash scripts/oasis1/download.sh
uv run python scripts/oasis1/build_tables.py
uv run --all-extras scripts/preprocess.py datasets/oasis1 --jobs 32
uv run scripts/pack_shards.py datasets/oasis1
```

Our tables are tracked in `scripts/<name>/tables/` (without `samples.tsv`), so `git diff` shows whether your build matches ours.

The hub also has [FOMO300K](scripts/fomo300k/README.md) for pretraining, without the collections that share subjects with the evaluation datasets. The scripts in `scripts/` show how we downloaded and built each dataset.

## Run

```sh
uv run --all-extras python -m brainmarks_smri neurojepa ixi_age abide_diagnosis
```

Each run evaluates one model on the given tasks. The results go to `output/<model>/<task>.json`, one file per task. Tasks with a results file are skipped. This file has the metrics with bootstrap confidence intervals, and the prediction for each sample. Default settings are in [`config/default.yaml`](src/brainmarks_smri/config/default.yaml). You can change them on the command line, e.g. `--overrides max_per_split=50`.

| Type | Tasks | Probe | Metric |
|---|---|---|---|
| classification | ABIDE autism, ADHD-200, CNP diagnosis, UCSF IDH, UPENN 1-year survival, SOOP mRS, OASIS-1 dementia | logistic regression | AUROC |
| regression | IXI age, OpenBHB age | ridge | R² |
| segmentation | BraTS tumor regions (3), SOOP stroke lesion, ATLAS stroke lesion | patch-level logistic regression | Dice |
| classification, k-fold | OASIS-2 dementia, OASIS-2 conversion, FOMO26 polymicrogyria | logistic regression | AUROC |
| segmentation, k-fold | FOMO26 trigeminal nerve, FOMO26 vessels | patch-level logistic regression | AP |

Probe hyperparameters are tuned by cross-validation on the train split. Scores are reported on the val split. By default, each split is capped at 300 samples.

The small datasets have no val split. Their tasks use 5-fold cross-validation, and the pooled out-of-fold predictions are scored once.

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

A model outside this package registers itself the same way. Pass its module in `imports`, e.g. `python -m brainmarks_smri my_model ixi_age --overrides 'imports=[my_package.model]' model_kwargs.checkpoint=ckpt.pth`.

## License

The code is MIT licensed. Each dataset keeps its original license. Several are for non-commercial use only. Please cite the sources listed in each dataset's README.
