# ABIDE I

Autism Brain Imaging Data Exchange I: 1112 subjects (539 ASD, 573 typical controls, age 6.5–64 y) from 17 institutions. We use the T1w scans for autism classification (ASD vs TDC), age regression and sex classification.

- **Source:** INDI / FCP public S3 bucket, `s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/` (BIDS form of the raw release) plus `Phenotypic_V1_0b.csv` from `s3://fcp-indi/data/Projects/ABIDE/`. Downloaded with `aws s3 sync --no-sign-request`. Project page: <https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html>. The phenotypic legend `ABIDE_LEGEND_V1.02.pdf` comes from the project page (not on S3).
- **Version:** S3 is the release here and there are no version tags. Pinned by the S3 listing taken 2026-10-01T22:29Z and by `manifest.sha256`. Object timestamps: images, per-site metadata and the phenotypic CSV 2016-12-16; `RawDataBIDS/sidecards/` JSONs 2022-04-13. `dataset_description.json` says `BIDSVersion: 1.4.2`. The original release was August 2012; the composite phenotypic file is version `V1_0b`, the legend V1.02 (last modified 2016-06-24).
- **DOI:** none.
- **License:** CC BY-NC-SA 3.0 (`License` field in each site's `dataset_description.json`, and the project page). Non-commercial research use. The project page asks users to register with NITRC and INDI. The NITRC tarballs are behind a login, but the S3 bucket is open (no sign-in).
- **Citation:** Di Martino, A., Yan, C.-G., Li, Q., et al. (2014). The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*, 19, 659–667. The project page also asks you to name the site datasets used and acknowledge their funding sources (see the site profiles on the ABIDE I page).
- **Contents:** (`source/`, mirrors the S3 tree)
  - `RawDataBIDS/<site>/`: 24 BIDS site folders (`CMU_a`, `CMU_b`, `MaxMun_a`–`_d`, ... = 20 `SITE_ID`s = 17 institutions). Each has `dataset_description.json`, `participants.tsv` (phenotype columns as in the composite CSV), a site-level `T1w.json`, and `sub-<id>/anat/sub-<id>_T1w.nii.gz`. 1102 T1w images.
  - `RawDataBIDS/sidecards/sub-<id>/ses-<site>/anat/sub-<id>_ses-<site>_T1w.json`: per-subject T1w sidecars added by INDI in 2022 (1102). The folder name `sidecards` is INDI's typo. Note these use a `ses-<site>` level that the images don't have.
  - `Phenotypic_V1_0b.csv`: the composite phenotypic file, 1112 rows. Labels: `DX_GROUP` (1 = ASD, 2 = control), `AGE_AT_SCAN`, `SEX` (1 = M, 2 = F), `SITE_ID`, plus DSM-IV, IQ, ADI-R, ADOS, SRS, etc. Missing values are `-9999` or empty.
  - `ABIDE_LEGEND_V1.02.pdf`: the data legend (column definitions and coding).
- **Excluded:**
  - rs-fMRI: `func/` and `task-rest_bold.json` (56.4 GB of the 63.8 GB `RawDataBIDS/`), and the func sidecars in `sidecards/`.
  - `RawData/` (64 GB): the same release in the old FCP layout (`mprage.nii.gz` + `rest.nii.gz`). Its 1102 `mprage.nii.gz` match the BIDS T1w set. Its only extra content is 109 UCLA `hires.nii.gz` (80 MB), which the BIDS form drops. These are 128x128x34 images at 1.5x1.5x4 mm, an EPI-matched registration scan rather than a usable structural.
  - `Derivatives/`, `Outputs/`, `Resources/`, `Phenotypic_V1_0b_preprocessed*.csv`: ABIDE Preprocessed (PCP) outputs and CPAC configs/atlases.
  - `PhenotypicData/phenotypic_<SITE>.csv`: per-site phenotypic CSVs, redundant with the composite file and `participants.tsv`.
- **Notes:**
  - Counts check against `Phenotypic_V1_0b.csv`: 1112 subjects (539 ASD / 573 TDC; 948 M / 164 F), 20 `SITE_ID`s from 17 institutions (CMU, Leuven, UCLA, UM and MaxMun are split into sub-samples). Every phenotypic subject has a BIDS folder and vice versa.
  - **10 subjects have no T1w**, only rs-fMRI (plus the low-resolution UCLA `hires` scan in `RawData/`): UCLA_1 51232, 51233, 51242–51247, 51270 and UCLA_2 51310 (8 ASD, 2 TDC). The usable T1w cohort is **1102 subjects (531 ASD / 571 TDC; 939 M / 163 F)**, 20 sites.
  - Subject IDs are 7-digit zero-padded in BIDS (`sub-0050002`) and plain integers in `SUB_ID` (`50002`).
  - OpenBHB includes ABIDE I subjects (see `datasets.md`).

```sh
bash scripts/abide1/download.sh   # 7.4 GB; resumable
```
