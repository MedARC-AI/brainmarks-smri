# ABIDE I

Autism Brain Imaging Data Exchange I: 1112 subjects (539 ASD, 573 typical controls, age 6.5–64 y) from 17 institutions. We use the T1w scans for autism classification (ASD vs TDC), age regression and sex classification.

- **Homepage:** <https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html>
- **Source:** INDI / FCP public S3 bucket: `s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/` (the BIDS form of the raw release) plus `Phenotypic_V1_0b.csv` from `s3://fcp-indi/data/Projects/ABIDE/`. Downloaded with `aws s3 sync --no-sign-request`. The phenotypic legend `ABIDE_LEGEND_V1.02.pdf` comes from the [project page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) (not on S3).
- **Version:** S3 is the release and has no version tags. Pinned by the S3 listing (2026-10-01T22:29Z) and `manifest.sha256`. Object dates: images, per-site metadata and the phenotypic CSV 2016-12-16; `RawDataBIDS/sidecards/` 2022-04-13. Phenotypic file version `V1_0b`, legend V1.02, `BIDSVersion` 1.4.2. Original release: August 2012.
- **DOI:** none.
- **License:** CC BY-NC-SA 3.0 (`License` in each site's `dataset_description.json`, and the project page). The project page asks users to register with NITRC and INDI. The NITRC tarballs are behind a login, but the S3 bucket is open.
- **Citation:** Di Martino, A., Yan, C.-G., Li, Q., et al. (2014). The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*, 19, 659–667. The project page also asks you to name the site datasets used and acknowledge their funding.
- **Code:** [`scripts/abide1/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/abide1) re-downloads `source/` and rebuilds `tables/`.

## Samples

One sample per participant (1112; 1102 have a T1w). Split 60/20/20, stratified by diagnosis × site; there is no official split. Complete = T1w and diagnosis.

| split | participants | complete | age | female | sites | diagnosis |
|---|---|---|---|---|---|---|
| train | 665 | 660 | 16.9 ± 7.9 | 15% | 20 | ASD 321 / TDC 344 |
| val | 223 | 220 | 17.0 ± 7.7 | 15% | 20 | ASD 109 / TDC 114 |
| test | 224 | 222 | 17.4 ± 8.8 | 13% | 20 | ASD 109 / TDC 115 |
| total | 1112 | 1102 | 17.0 ± 8.0 | 15% | 20 | ASD 539 / TDC 573 |

## Contents

`source/` mirrors the S3 tree (2278 files, 7.4 GB):

- `RawDataBIDS/<site>/`: 24 BIDS site folders (`CMU_a`, `CMU_b`, `MaxMun_a`–`_d`, ...; 20 `SITE_ID`s from 17 institutions). Each has `dataset_description.json`, `participants.tsv`, a site-level `T1w.json`, and `sub-<id>/anat/sub-<id>_T1w.nii.gz`. 1102 T1w images.
- `RawDataBIDS/sidecards/sub-<id>/ses-<site>/anat/sub-<id>_ses-<site>_T1w.json`: per-subject T1w sidecars added by INDI in 2022 (1102). `sidecards` is INDI's spelling. These use a `ses-<site>` level that the images don't have.
- `Phenotypic_V1_0b.csv`: the composite phenotypic file, 1112 rows. Labels: `DX_GROUP` (1 = ASD, 2 = control), `AGE_AT_SCAN`, `SEX` (1 = M, 2 = F), `SITE_ID`, plus DSM-IV, IQ, ADI-R, ADOS, SRS, etc. Missing values are `-9999` or empty.
- `ABIDE_LEGEND_V1.02.pdf`: column definitions and coding.

`tables/` (derived from `source/`):

- `images.tsv`: one T1w per participant, across the 24 BIDS site folders.
- `samples.tsv` + `samples.json`: diagnosis (ASD/TDC, target), age, sex, site (`SITE_ID`, 20 sites) and BIDS site folder, DSM-IV subtype, IQ, ADOS and SRS totals, handedness, medication status. The other phenotypic columns stay in `Phenotypic_V1_0b.csv`. participant_id is the BIDS label (`sub-0050002`).
- `splits.tsv`: split, rank and complete per participant.

## Excluded

- rs-fMRI: `func/` and `task-rest_bold.json` (56.4 GB of the 63.8 GB `RawDataBIDS/`), and the func sidecars in `sidecards/`.
- `RawData/` (64 GB): the same release in the old FCP layout. Its only extra content is 109 UCLA `hires.nii.gz` (80 MB): 128×128×34 images at 1.5×1.5×4 mm, an EPI-matched registration scan rather than a usable structural.
- `Derivatives/`, `Outputs/`, `Resources/`, `Phenotypic_V1_0b_preprocessed*.csv`: ABIDE Preprocessed (PCP) outputs and CPAC configs/atlases.
- `PhenotypicData/phenotypic_<SITE>.csv`: per-site CSVs, redundant with the composite file.

## Notes

- 10 subjects have no T1w, only rs-fMRI (not complete in the tables): UCLA_1 51232, 51233, 51242–51247, 51270 and UCLA_2 51310 (8 ASD, 2 TDC).
- Every phenotypic subject has a BIDS folder and vice versa.
- Subject IDs are 7-digit zero-padded in BIDS (`sub-0050002`) and plain integers in `SUB_ID` (`50002`); the tables use the BIDS form.
- In the source, 20 ASD participants have DSM-IV-TR code 0 (control), and one VIQ (180) is above the legend's range. Use `diagnosis` as the label.
- OpenBHB includes ABIDE I subjects, but its IDs are anonymized (see the [OpenBHB README](../openbhb/README.md)).
