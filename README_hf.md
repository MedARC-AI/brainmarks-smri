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
  - 100K<n<1M
extra_gated_prompt: >-
  This collection redistributes public datasets, each under its own license.
  Evaluation: CC0 (CNP, SOOP); CC BY 4.0 + the TCIA Data Usage Policy (UCSF-PDGM,
  UPENN-GBM, BraTS 2021); CC BY-SA 3.0 (IXI); CC BY-NC-SA 3.0 (ABIDE I, OpenBHB);
  CC BY-NC (ADHD-200). OpenBHB additionally asks users to accept the most
  restrictive data usage agreement of its source cohorts. Pretraining: FOMO300K
  is CC BY-NC-SA 4.0, each of its collections keeps its own license, and its HCP
  WU-Minn collections are under the HCP Open Access Data Use Terms
  (https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
  Each dataset's README states its license, terms and required citations.
extra_gated_fields:
  I will use each dataset according to its license and terms, and cite the original sources: checkbox
  I accept the HCP WU-Minn Open Access Data Use Terms for the HCP data in FOMO300K: checkbox
configs:
  - {config_name: abide1, data_files: abide1/tables/samples.tsv}
  - {config_name: adhd200, data_files: adhd200/tables/samples.tsv}
  - {config_name: brats2021, data_files: brats2021/tables/samples.tsv}
  - {config_name: cnp, data_files: cnp/tables/samples.tsv}
  - {config_name: ixi, data_files: ixi/tables/samples.tsv}
  - {config_name: openbhb, data_files: openbhb/tables/samples.tsv}
  - {config_name: soop, data_files: soop/tables/samples.tsv}
  - {config_name: ucsf_pdgm, data_files: ucsf_pdgm/tables/samples.tsv}
  - {config_name: upenn_gbm, data_files: upenn_gbm/tables/samples.tsv}
  - {config_name: fomo300k, data_files: fomo300k/tables/images.parquet}
---

# Brainmarks-sMRI

Public structural brain MRI datasets for building and evaluating sMRI foundation models: evaluation datasets with classification, regression and segmentation targets, and a large pretraining set. Code and benchmark: <https://github.com/MedARC-AI/brainmarks-smri>.

The datasets come in two kinds:

- **Hosted**: the full data is here.
- **Manual**: you download the data from its source and build the dataset with our scripts. Only the README and the tables (without `samples.tsv`) are here. Their licenses don't allow redistribution.

## Evaluation datasets

Each evaluation dataset has harmonized metadata tables, fixed train/val/test splits, and a brain mask and MNI affine for every image. The images are the original releases, unmodified.

### Hosted

| Dataset | Participants | Images | Targets | License | Size |
|---|---|---|---|---|---|
| [ABIDE I](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) | 1,102 | T1w | autism diagnosis, age, sex | CC BY-NC-SA 3.0 | 7.4 GB |
| [ADHD-200](https://fcon_1000.projects.nitrc.org/indi/adhd200/) | 961 | T1w | ADHD diagnosis | CC BY-NC | 9.0 GB |
| [BraTS 2021](https://www.med.upenn.edu/cbica/brats2021/) | 1,468 | T1w, T1c, T2w, FLAIR, tumor mask | tumor segmentation, MGMT | CC BY 4.0 | 15.8 GB |
| [CNP](https://openneuro.org/datasets/ds000030) | 265 | T1w | psychiatric diagnosis (4-way) | CC0 | 3.2 GB |
| [IXI](https://brain-development.org/ixi-dataset/) | 582 | T1w, T2w, PD | age | CC BY-SA 3.0 | 12.8 GB |
| [OpenBHB](https://baobablab.github.io/bhb/) | 3,984 | T1w | age (with site debiasing) | CC BY-NC-SA 3.0 | 32 GB |
| [SOOP](https://openneuro.org/datasets/ds004889) | 1,715 | FLAIR, DWI, ADC, lesion mask | stroke lesion segmentation, discharge mRS, NIHSS | CC0 | 27 GB |
| [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) | 495 | T1w, T1c, T2w, FLAIR, DWI, ADC, tumor mask | IDH, MGMT, 1p/19q, grade, survival, tumor segmentation | CC BY 4.0 | 15.9 GB |
| [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) | 630 | T1w, T1c, T2w, FLAIR, tumor mask | survival, IDH1, MGMT, tumor segmentation | CC BY 4.0 | 25.3 GB |

### Manual

| Dataset | Participants | Images | Targets | Access | Size |
|---|---|---|---|---|---|
| [OASIS-1](https://sites.wustl.edu/oasisbrains/home/oasis-1/) | 416 | T1w | dementia (CDR) | accept the OASIS terms | 16 GB |
| [OASIS-2](https://sites.wustl.edu/oasisbrains/home/oasis-2/) | 150 | T1w, longitudinal | dementia, conversion | accept the OASIS terms | 20 GB |
| [FOMO26 Task 4](https://fomo26.github.io/) | 40 | T2w, nerve and vessel mask | trigeminal nerve and vessel segmentation | download link from the organizers | 2.2 GB |
| [FOMO26 Task 5](https://fomo26.github.io/) | 48 | T1w | polymicrogyria | download link from the organizers | 1.1 GB |
| [ATLAS R2.1](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) | 955 | T1w, lesion mask | stroke lesion segmentation | apply for a decryption key | 2.9 GB |
| [ADNI](https://adni.loni.usc.edu/) | 3,070 | T1w, FLAIR, longitudinal | diagnosis, conversion, cognition, amyloid and tau PET, CSF, plasma, FreeSurfer volumes | apply for ADNI data access (LONI IDA) | 65 GB |
| [A4/LEARN](https://www.a4studydata.org/) | 1,773 | T1w, longitudinal | amyloid, tau PET, plasma pTau217, cognition (PACC), treatment arm | apply for A4 data access (LONI IDA) | 43 GB |

Each dataset folder's `README.md` has its source, version, license, citation, and split details.

### Layout

```
<dataset>/
  README.md            # source, version, license, citation, splits
  shards/              # zip files of source/ (the original release) and derivatives/ (hosted only)
  tables/
    images.tsv         # one row per image: participant_id, session_id, modality, desc, path
    samples.tsv        # one row per scan session: participant_id, session_id, age, sex, site, targets... (hosted only)
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
- For the manual datasets, the checksums let you check your own build against ours.
- Splits are by participant. Official splits are kept where they exist; otherwise the split is 60/20/20, stratified, with a fixed seed. The smallest datasets have only a train split, for k-fold cross-validation.
- `complete` marks participants with all core images and the primary targets.
- `rank` orders participants within each split so that the lowest-ranked N form a balanced subset; subsets are nested as N grows.

### Usage

```sh
hf download medarc/brainmarks-smri --repo-type dataset --include "cnp/*" --local-dir brainmarks-smri
```

```python
import gzip
import zipfile

import nibabel as nib
import pandas as pd

root = "brainmarks-smri/cnp"
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

To work with plain files instead, unzip the shards: `cd brainmarks-smri/cnp && unzip -q 'shards/*.zip'`.

## Pretraining dataset

| Dataset | Subjects | Images | Sequences | License | Size |
|---|---|---|---|---|---|
| [FOMO300K](fomo300k/README.md) | 47,404 | 266,450 | T1w, T2w, FLAIR, T1c, DWI, ADC, GRE, ASL, ... | CC BY-NC-SA 4.0 + per collection | 692 GB |

FOMO300K without the collections that share subjects with the evaluation datasets. Each image is masked to the brain, cropped and stored as a sparse int16 npz at native resolution, with its brain mask, MNI affine, QC and thumbnails. `tables/images.parquet` has one row per image with all the metadata for filtering. `tables/shards.parquet` gives each file's byte offset in its shard. See [`fomo300k/README.md`](fomo300k/README.md) for the format and a reading example.

## License and citation

Datasets and their derivatives are released under their original licenses. Non-commercial terms apply to ABIDE I, ADHD-200, OpenBHB and FOMO300K. FOMO300K's HCP collections are under the HCP WU-Minn Open Access Data Use Terms. If you use a dataset, use the citation given in its README and follow all dataset-specific acknowledgement conditions.

## Reproducing

Every evaluation dataset can be re-downloaded from its original source and checked against `tables/source.sha256`. The download scripts, table-building and preprocessing code are at <https://github.com/MedARC-AI/brainmarks-smri>.
