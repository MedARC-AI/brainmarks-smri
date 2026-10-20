# ATLAS R2.1

Anatomical Tracings of Lesions After Stroke: T1w MRIs of chronic stroke patients from 33 sites, with manually traced lesion masks. We use it for stroke lesion segmentation.

- **Homepage:** <https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html>
- **Source:** INDI S3 (`s3://fcp-indi/data/Projects/INDI/ATLAS/R2.1/`), downloaded with `curl`. The archive is encrypted with `openssl`. To get the key, accept the terms of use and fill in the Google form on the homepage. Each user must apply for their own key.
- **Version:** R2.1, minimally processed (`atlas21_training_raw.tar.gz`, 2026-04-22). The same 955 scans as R2.0, now with the masks of the 300 test scans.
- **DOI:** [10.1038/s41597-022-01401-7](https://doi.org/10.1038/s41597-022-01401-7) (R2.0 data descriptor)
- **License:** ATLAS terms of use, accepted with the key request. Not redistributable, so our mirror has only this README and the tables without `samples.tsv`. Each user downloads the data and builds the rest.
- **Citation:** Liew, S.-L., Lo, B. P., Donnelly, M. R., et al. (2022). A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms. *Scientific Data*, 9, 320.
- **Code:** [`scripts/atlas/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/atlas) downloads `source/` and builds `tables/`, `derivatives/` and `shards/`.

## Samples

One sample per participant (955), each with a T1w and a lesion mask. The official R2.0 split is kept: Testing (300) is our `test`, and Training (655) is split 80/20 into train and val, stratified by site. Complete = T1w and lesion mask (all 955). The release has no age or sex.

| split | participants | complete | sites | days post stroke |
|---|---|---|---|---|
| train | 526 | 526 | 33 | 981.6 ± 1363.0 |
| val | 129 | 129 | 30 | 847.8 ± 1117.3 |
| test | 300 | 300 | 24 | 873.8 ± 1204.0 |
| total | 955 | 955 | 33 | 928.4 ± 1281.6 |

## Contents

`source/`: the extracted `Training_Raw/` folder (2,865 files, 2.9 GB), one BIDS-like folder per subject, `Training_Raw/<site>/sub-<id>/ses-1/anat/`:

- `*_space-orig_desc-brain_T1w.nii.gz`: skull-stripped T1w in native space.
- `*_space-orig_label-lesion_desc-T1lesion_mask.nii.gz`: manual lesion mask on the T1w grid.
- `*_metadata.csv`: R2.0 split, days post stroke, chronicity and site.

`tables/` (derived from `source/`):

- `images.tsv`: T1w and lesion mask.
- `samples.tsv` + `samples.json`: site, days post stroke and chronicity. Not on our mirror.
- `splits.tsv`: split, official split, rank and complete per participant.

## Excluded

- The fully preprocessed R2.1 release (`atlas21_training_preprocessed.tar.gz`, MNI-normalized).
- 6 `.DS_Store` files in the archive.

## Notes

- The background of one site (R029) is noise and speckles, not zeros, so its brain masks are made by thresholding instead of from the nonzero voxels.
