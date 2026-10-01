# ADHD-200

Children and adolescents (7–21 y) with ADHD and typically developing controls, from 8 sites (Peking is split into 3 scanner sub-sites). The cohort of the 2011 ADHD-200 Global Competition. We use only the T1w scans, for ADHD classification (TDC vs ADHD, or TDC / combined / inattentive / hyperactive-impulsive).

- **Source:** INDI / 1000 Functional Connectomes Project, public S3 bucket [`s3://fcp-indi/data/Projects/ADHD200/`](https://fcp-indi.s3.amazonaws.com/index.html), the `RawDataBIDS/` tree (plus the phenotypic CSVs from the older `RawData/` tree). Downloaded with `aws s3 sync --no-sign-request`. Project page: <https://fcon_1000.projects.nitrc.org/indi/adhd200/>; the docs and the released test-set labels are fetched by plain HTTPS from there (no login needed for those files).
- **Version:** S3 is unversioned, so the release is pinned by the manifest. Listing taken 2026-10-01T22:29Z. Object dates: T1w images 2016-11 to 2020-02, `participants.tsv`/`dataset_description.json` 2020-01/02, phenotypic CSVs 2016-10 (`RawData/`) and 2024-02-23 (`RawDataBIDS/`). NITRC files 2011 (`allSubs_testSet_phenotypic_dx.csv` Last-Modified 2011-11-04).
- **DOI:** none for the data. Consortium paper: [10.3389/fnsys.2012.00062](https://doi.org/10.3389/fnsys.2012.00062).
- **License:** CC BY-NC (NITRC license category "Attribution Non-Commercial"). The project page says "data usage is unrestricted for non-commercial research purposes", asks that the specific datasets used be specified and their funding sources acknowledged (per-site `Funding`/`Acknowledgements` are in each `dataset_description.json`), and "require[s] that user register with the NITRC and 1000 Functional Connectomes Project to gain access". The S3 copy itself is open (no sign-in).
- **Citation:** ADHD-200 Consortium (2012). The ADHD-200 Consortium: a model to advance the translational potential of neuroimaging in clinical neuroscience. *Frontiers in Systems Neuroscience*, 6, 62. Plus per-site acknowledgement.

## Contents

`source/` mirrors the S3 paths:

- `RawDataBIDS/<site>/sub-<id>/ses-1/anat/*_T1w.nii.gz`: one T1w per subject, **961 subjects** across 10 site folders (Brown 26, KKI 83, NYU 263, NeuroIMAGE 73, OHSU 113, Peking_1 136, Peking_2 67, Peking_3 42, Pittsburgh 98, WashU 60). Peking_3 uses `acq-1..5` T1w variants. Images are byte-identical to `RawData/<site>/<id>/session_1/anat_1/mprage.nii.gz`.
- `RawDataBIDS/<site>/{dataset_description.json, participants.tsv, *T1w.json}`: per-site BIDS metadata. `participants.tsv` has `dx` (diagnosis), age, sex, handedness, IQ, ADHD scores, medication status and QC, **including the released test-set labels**. Gotcha: for NYU, Peking_1 and Pittsburgh the test subjects appear twice (identical rows; 41 + 51 + 9 duplicates). Brown has no `dx` column.
- `RawDataBIDS/*_phenotypic.csv` (2024 copies) and `RawData/**/*_phenotypic.csv` (2016, original competition-format CSVs with numeric codes, incl. `*_TestRelease_phenotypic.csv`). No CSV exists for Peking_2/Peking_3; their labels are only in `participants.tsv`.
- `nitrc/`: project-page text (`json/intro.json`, `json/sites.json`: license, usage agreement, site descriptions), `results.html` (competition results), `general/ADHD-200_PhenotypicKey.pdf` (code key for the CSVs), **`general/allSubs_testSet_phenotypic_dx.csv`** (the official post-competition test-set labels, all 197 test subjects), `general/ADHD-200_CompetitionScoring.pdf`, and the 2011 fixes (`fixes/ADHD-200.PhenotypicFix.csv`, `fixes/DeobliqueFixAffectedSubs.txt`). The S3 data postdates these fixes.

**Train/test split**: subjects listed in `allSubs_testSet_phenotypic_dx.csv` are the holdout (197); the rest are the original training release (776). Diagnosis codes (CSV): 0 = TDC, 1 = ADHD-Combined, 2 = ADHD-Hyperactive/Impulsive, 3 = ADHD-Inattentive.

## Excluded

- rs-fMRI: `sub-*/ses-*/func/` and `*_bold.json` (1,403 files, 79 GB).
- `RawData/` images (88 GB; same T1w and rest scans as `RawDataBIDS/`, older layout).
- `Outputs/` (C-PAC, fMRIPrep, FreeSurfer, MRIQC, denoise, mindboggle), `surfaces/`, `Resources/` (C-PAC/QAP configs): preprocessing derivatives. The Neuro Bureau "ADHD-200 Preprocessed" (Athena/NIAK/Burner) pipelines are also not fetched.
- `.DS_Store` files.

## Notes

- **973 vs 961 subjects.** `datasets.md` says 973 (776 train / 197 holdout), which matches the competition. S3 has 961 subjects with T1w: 775 train (WashU 15019 is in `WashU_phenotypic.csv` but has no images) + 186 test (the **11 KKI test subjects**, IDs 20001–20022, are missing from S3).
- **Test labels were released** (Nov 2011, `allSubs_testSet_phenotypic_dx.csv`) for all test sites **except Brown**: all 26 Brown subjects (the whole Brown site, test-only) have DX = "pending", and Brown's `participants.tsv` has no `dx`. So 160 test subjects have both images and labels here.
- Labeled subjects with a T1w: 775 train + 160 test = **935**. Brown's 26 are usable only for unsupervised/age/sex tasks.
- Of the 935, 576 are TDC and 359 ADHD (any subtype). WashU is all controls (60 TDC) and Pittsburgh has 4 ADHD out of 98, so site and diagnosis are strongly confounded.
- License: `datasets.md` says "CC BY-NC (verify)". Confirmed CC BY-NC, with the extra non-commercial and NITRC-registration wording above.

```sh
bash scripts/adhd200/download.sh   # 9.0 GB; resumable
```
