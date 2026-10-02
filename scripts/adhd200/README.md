# ADHD-200

Children and adolescents (7–21 y) with ADHD and typically developing controls, from 8 sites (Peking is split into 3 scanner sub-sites): the cohort of the 2011 ADHD-200 Global Competition. We use only the T1w scans, for ADHD classification (TDC vs ADHD, or TDC / combined / inattentive / hyperactive-impulsive).

- **Source:** INDI / 1000 Functional Connectomes Project public S3 bucket, `s3://fcp-indi/data/Projects/ADHD200/`: the `RawDataBIDS/` tree, plus the phenotypic CSVs from the older `RawData/` tree. Downloaded with `aws s3 sync --no-sign-request`. Docs and the released test-set labels come from the [project page](https://fcon_1000.projects.nitrc.org/indi/adhd200/) over plain HTTPS (no login needed for those files).
- **Version:** S3 is unversioned, so the release is pinned by the S3 listing (2026-10-01T22:29Z) and `manifest.sha256`. Object dates: T1w images 2016-11 to 2020-02; `participants.tsv`/`dataset_description.json` 2020-01/02; phenotypic CSVs 2016-10 (`RawData/`) and 2024-02-23 (`RawDataBIDS/`); NITRC files 2011.
- **DOI:** none for the data. Consortium paper: [10.3389/fnsys.2012.00062](https://doi.org/10.3389/fnsys.2012.00062).
- **License:** CC BY-NC (NITRC category "Attribution Non-Commercial"). The project page says "data usage is unrestricted for non-commercial research purposes", asks users to name the datasets used and acknowledge their funding (per-site `Funding`/`Acknowledgements` are in each `dataset_description.json`), and asks users to register with NITRC and the 1000 Functional Connectomes Project. The S3 copy is open.
- **Citation:** ADHD-200 Consortium (2012). The ADHD-200 Consortium: a model to advance the translational potential of neuroimaging in clinical neuroscience. *Frontiers in Systems Neuroscience*, 6, 62. Plus per-site acknowledgement.

## Usage

```sh
bash scripts/adhd200/download.sh                 # 9.0 GB; resumable
uv run python scripts/adhd200/build_tables.py    # tables/; prints the summary below
```

## Samples

One sample per participant with a T1w (961). The official competition split is kept: the holdout (186 with images) is `test`, and the training release is split 75/25 into train/val, stratified by diagnosis × site (`official_split` holds train/test). Complete = T1w and diagnosis; Brown's 26 test participants never had labels released, so the labeled test set is 160.

| split | participants | complete | age | female | sites | adhd |
|---|---|---|---|---|---|---|
| train | 582 | 582 | 12.1 ± 3.2 | 39% | 9 | False 368 / True 214 |
| val | 193 | 193 | 11.6 ± 3.2 | 34% | 9 | False 122 / True 71 |
| test | 186 | 160 | 12.3 ± 3.8 | 44% | 6 | False 86 / True 74 |
| total | 961 | 935 | 12.1 ± 3.3 | 39% | 10 | False 576 / True 359 |

## Contents

`source/` mirrors the S3 paths (1030 files, 9.0 GB):

- `RawDataBIDS/<site>/sub-<id>/ses-1/anat/*_T1w.nii.gz`: one T1w per subject, **961 subjects** across 10 site folders (Brown 26, KKI 83, NYU 263, NeuroIMAGE 73, OHSU 113, Peking_1 136, Peking_2 67, Peking_3 42, Pittsburgh 98, WashU 60). Peking_3 uses `acq-1..5` T1w variants.
- `RawDataBIDS/<site>/{dataset_description.json, participants.tsv, *T1w.json}`: per-site BIDS metadata. `participants.tsv` has `dx` (diagnosis), age, sex, handedness, IQ, ADHD scores, medication status and QC, **including the released test-set labels**.
- `RawDataBIDS/*_phenotypic.csv` (2024 copies) and `RawData/**/*_phenotypic.csv` (2016 originals in competition format with numeric codes, incl. `*_TestRelease_phenotypic.csv`). Peking_2/Peking_3 have no CSV; their labels are only in `participants.tsv`.
- `nitrc/`:
  - `general/allSubs_testSet_phenotypic_dx.csv`: **the official test-set labels** (all 197 holdout subjects, released Nov 2011).
  - `general/ADHD-200_PhenotypicKey.pdf` (code key for the CSVs), `general/ADHD-200_CompetitionScoring.pdf`, `results.html` (competition results).
  - `json/intro.json`, `json/sites.json`: project-page text (license, usage terms, site descriptions).
  - `fixes/ADHD-200.PhenotypicFix.csv`, `fixes/DeobliqueFixAffectedSubs.txt`: the 2011 fixes. The S3 data postdates them.

Train/test split: subjects in `allSubs_testSet_phenotypic_dx.csv` are the holdout (197); the rest are the training release (776). Diagnosis codes in the CSVs: 0 = TDC, 1 = ADHD-Combined, 2 = ADHD-Hyperactive/Impulsive, 3 = ADHD-Inattentive.

`tables/` (built by `build_tables.py`; layout in `src/brain_datasets/tables.py`):

- `images.tsv`: one T1w per participant. 9 WashU participants have theirs in `ses-2`/`ses-3`/`ses-4` (session_id follows).
- `samples.tsv` + `samples.json`: diagnosis (4 levels) and `adhd` (binary target), age, sex, site (the 10 BIDS site folders), ADHD scores, IQ, medication, handedness and anatomical QC, from the per-site `participants.tsv`. Site-specific phenotypes that are only in the `*_phenotypic.csv` files are not merged. participant_id is the BIDS label (`sub-0010001`).
- `splits.tsv`: split, official split, rank and complete per participant.

## Excluded

- rs-fMRI: `sub-*/ses-*/func/` and `*_bold.json` (1,403 files, 79 GB).
- `RawData/` images (88 GB): the same scans as `RawDataBIDS/` in the older layout, byte-identical.
- `Outputs/` (C-PAC, fMRIPrep, FreeSurfer, MRIQC, ...), `surfaces/`, `Resources/`: preprocessing derivatives and configs. The Neuro Bureau "ADHD-200 Preprocessed" pipelines (Athena/NIAK/Burner) are also not fetched.
- `.DS_Store` files.

## Notes

- **961 vs 973 subjects.** The competition had 973 (776 train / 197 holdout). S3 has 961 with T1w: 775 train (WashU 15019 has a phenotypic row but no images) + 186 test (the **11 KKI test subjects**, IDs 20001–20022, are missing from S3).
- **Brown labels were never released.** All 26 Brown subjects are test-only with DX = "pending", and Brown's `participants.tsv` has no `dx`. They are usable only for unlabeled, age or sex tasks.
- In `source/`, the NYU, Peking_1 and Pittsburgh `participants.tsv` list their test subjects twice (41 + 51 + 9 rows), some sites use numeric codes (KKI IQ measure, WashU QC) and NYU's handedness is a score. The tables deduplicate and decode these.
- Site and diagnosis are strongly confounded: WashU is all controls (60 TDC), and Pittsburgh has 4 ADHD out of 98.
