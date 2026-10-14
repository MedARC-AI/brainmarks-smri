---
pretty_name: Brainmarks-sMRI
license: other
license_name: per-dataset
tags:
  - medical
  - neuroimaging
  - mri
  - brain
  - nifti
task_categories:
  - image-classification
  - image-segmentation
size_categories:
  - 10K<n<100K
extra_gated_prompt: >-
  This collection redistributes 10 public datasets, each under its own license:
  CC0 (Pixar, CNP, SOOP); CC BY 4.0 + the TCIA Data Usage Policy (UCSF-PDGM,
  UPENN-GBM, BraTS 2021); CC BY-SA 3.0 (IXI); CC BY-NC-SA 3.0 (ABIDE I, OpenBHB);
  CC BY-NC (ADHD-200). OpenBHB additionally asks users to accept the most
  restrictive data usage agreement of its source cohorts. Each dataset's README
  states its license, terms and required citations.
extra_gated_fields:
  I will use each dataset according to its license and terms, and cite the original sources: checkbox
configs:
  - {config_name: abide1, data_files: abide1/tables/samples.tsv}
  - {config_name: adhd200, data_files: adhd200/tables/samples.tsv}
  - {config_name: brats2021, data_files: brats2021/tables/samples.tsv}
  - {config_name: cnp, data_files: cnp/tables/samples.tsv}
  - {config_name: ixi, data_files: ixi/tables/samples.tsv}
  - {config_name: openbhb, data_files: openbhb/tables/samples.tsv}
  - {config_name: pixar, data_files: pixar/tables/samples.tsv}
  - {config_name: soop, data_files: soop/tables/samples.tsv}
  - {config_name: ucsf_pdgm, data_files: ucsf_pdgm/tables/samples.tsv}
  - {config_name: upenn_gbm, data_files: upenn_gbm/tables/samples.tsv}
---

# Brainmarks-sMRI

Ten public structural brain MRI datasets for evaluating sMRI foundation models, with classification, regression and segmentation targets. The images are the original releases, unmodified, packed into zip shards. Each dataset adds harmonized metadata tables, fixed train/val/test splits, and a brain mask and MNI affine for every image.

## Datasets

| Dataset | Participants | Images | Targets | License | Size |
|---|---|---|---|---|---|
| [ABIDE I](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) | 1,102 | T1w | autism diagnosis, age, sex | CC BY-NC-SA 3.0 | 7.4 GB |
| [ADHD-200](https://fcon_1000.projects.nitrc.org/indi/adhd200/) | 961 | T1w | ADHD diagnosis | CC BY-NC | 9.0 GB |
| [BraTS 2021](https://www.med.upenn.edu/cbica/brats2021/) | 1,468 | T1w, T1c, T2w, FLAIR, tumor mask | tumor segmentation, MGMT | CC BY 4.0 | 15.8 GB |
| [CNP](https://openneuro.org/datasets/ds000030) | 265 | T1w | psychiatric diagnosis (4-way) | CC0 | 3.2 GB |
| [IXI](https://brain-development.org/ixi-dataset/) | 582 | T1w, T2w, PD | age | CC BY-SA 3.0 | 12.8 GB |
| [OpenBHB](https://baobablab.github.io/bhb/) | 3,984 | T1w | age (with site debiasing) | CC BY-NC-SA 3.0 | 32 GB |
| [Pixar](https://openneuro.org/datasets/ds000228) | 155 | T1w | age | CC0 | 1.0 GB |
| [SOOP](https://openneuro.org/datasets/ds004889) | 1,715 | FLAIR, DWI, ADC, lesion mask | stroke lesion segmentation, discharge mRS, NIHSS | CC0 | 27 GB |
| [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) | 495 | T1w, T1c, T2w, FLAIR, DWI, ADC, tumor mask | IDH, MGMT, 1p/19q, grade, survival, tumor segmentation | CC BY 4.0 | 15.9 GB |
| [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) | 630 | T1w, T1c, T2w, FLAIR, tumor mask | survival, IDH1, MGMT, tumor segmentation | CC BY 4.0 | 25.3 GB |

Each dataset folder's `README.md` has its source, version, license, citation, and split details.

## Layout

```
<dataset>/
  README.md            # source, version, license, citation, splits
  shards/              # zip files of source/ (the original release) and derivatives/
  tables/
    images.tsv         # one row per image: participant_id, session_id, modality, desc, path
    samples.tsv        # one row per scan session: participant_id, session_id, age, sex, site, targets...
    samples.json       # column descriptions, levels and units
    splits.tsv         # one row per participant: participant_id, split, official_split, rank, complete
    derivatives.tsv    # one row per processed image: path, mask method, QC metrics
    shards.tsv         # one row per packed file: path, shard
    source.sha256      # checksums of the original release (source/)
    derivatives.sha256 # checksums of the masks and affines (derivatives/)
```

- All paths in the tables are relative to `<dataset>/`.
- Derivatives: a brain mask on the image grid (`<image>_mask.nii.gz`) and a 4×4 nibabel-style affine matrix from the image to MNI152NLin6Asym (`<image>_mni.txt`). `derivatives.tsv` lists the images that have them.
- `<dataset>-NNN.zip` shards hold whole participants: their images, JSON sidecars, brain masks and MNI affines, about 1 GB per shard. `<dataset>-meta.zip` holds the rest of the release: participant tables, phenotypes and docs.
- Files in the shards keep their original paths under `source/` and `derivatives/`. Unzipping all shards in `<dataset>/` restores the full release, which you can check with `sha256sum -c tables/source.sha256 tables/derivatives.sha256`.
- Splits are by participant. Official splits are kept where they exist; otherwise the split is 60/20/20, stratified, with a fixed seed.
- `complete` marks participants with all core images and the primary targets.
- `rank` orders participants within each split so that the lowest-ranked N form a balanced subset; subsets are nested as N grows.

## Usage

```sh
hf download medarc/brainmarks-smri --repo-type dataset --include "pixar/*" --local-dir brainmarks-smri
```

```python
import gzip
import zipfile

import nibabel as nib
import pandas as pd

root = "brainmarks-smri/pixar"
images = pd.read_csv(f"{root}/tables/images.tsv", sep="\t")
samples = pd.read_csv(f"{root}/tables/samples.tsv", sep="\t")
splits = pd.read_csv(f"{root}/tables/splits.tsv", sep="\t")
shards = pd.read_csv(f"{root}/tables/shards.tsv", sep="\t").set_index("path")["shard"]

# A balanced 50-participant training subset.
train = splits[(splits.split == "train") & splits.complete]
train_50 = train.nsmallest(50, "rank").participant_id

# Read one image from its shard.
path = images["path"][0]
with zipfile.ZipFile(f"{root}/{shards[path]}") as shard:
    image = nib.Nifti1Image.from_bytes(gzip.decompress(shard.read(path)))
```

To work with plain files instead, unzip the shards: `cd brainmarks-smri/pixar && unzip -q 'shards/*.zip'`.

## License and citation

Datasets and their derivatives are released under their original licenses. Non-commercial terms apply to ABIDE I, ADHD-200 and OpenBHB. If you use a dataset, use the citation given in the README and follow all dataset-specific acknowledgement conditions.

## Reproducing

Every file here can be re-downloaded from its original source and checked against `tables/source.sha256`. The download scripts and table-building code are at <https://github.com/MedARC-AI/brainmarks-smri>.
