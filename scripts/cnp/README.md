# CNP (LA5c)

UCLA Consortium for Neuropsychiatric Phenomics LA5c study: 272 adults (21–50 y) who are healthy controls or have schizophrenia, bipolar disorder or ADHD. We use the T1w scans for 4-way diagnosis classification.

- **Homepage:** <https://openneuro.org/datasets/ds000030>
- **Source:** OpenNeuro [ds000030](https://openneuro.org/datasets/ds000030/versions/1.0.0). Downloaded with `openneuro-py`.
- **Version:** snapshot 1.0.0 (2020-04-21), the latest.
- **DOI:** [10.18112/openneuro.ds000030.v1.0.0](https://doi.org/10.18112/openneuro.ds000030.v1.0.0)
- **License:** CC0
- **Citation** (per `HowToAcknowledge`): Poldrack, R. A., Congdon, E., Triplett, W., et al. (2016). A phenome-wide examination of neural and cognitive function. *Scientific Data*, 3, 160110. If using derived data, also: Gorgolewski, K. J., Durnez, J., & Poldrack, R. A. (2017). Preprocessed Consortium for Neuropsychiatric Phenomics dataset. *F1000Research*, 6, 1262.
- **Code:** [`scripts/cnp/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/cnp) re-downloads `source/` and rebuilds `tables/`.

## Samples

One sample per participant with a T1w (265). The 7 participants with only fMRI are left out. Split 60/20/20, stratified by diagnosis; there is no official split. Complete = diagnosis (all 265).

| split | participants | complete | age | female | diagnosis |
|---|---|---|---|---|---|
| train | 159 | 159 | 32.9 ± 9.2 | 44% | ADHD 25 / BIPOLAR 29 / CONTROL 75 / SCHZ 30 |
| val | 53 | 53 | 32.7 ± 9.8 | 40% | ADHD 8 / BIPOLAR 10 / CONTROL 25 / SCHZ 10 |
| test | 53 | 53 | 35.0 ± 9.2 | 40% | ADHD 8 / BIPOLAR 10 / CONTROL 25 / SCHZ 10 |
| total | 265 | 265 | 33.3 ± 9.3 | 42% | ADHD 41 / BIPOLAR 49 / CONTROL 125 / SCHZ 50 |

## Contents

`source/`: the structural part of the raw BIDS release (638 files, 3.2 GB):

- `sub-*/anat/`: T1w MPRAGE + JSON for 265 subjects, defaced with `mri_deface`.
- `participants.tsv`: 272 subjects with diagnosis, age, sex, per-scan availability flags, scanner serial number (2 Siemens Trio scanners) and a T1w ghost-artifact flag.
- `phenotype/`: 52 instrument tables + JSON data dictionaries (SCID, BPRS, SANS/SAPS, Hamilton, YMRS, ASRS, WAIS, medication, demographics, ...).
- `dataset_description.json`, `README`, `CHANGES`.

`tables/` (derived from `source/`):

- `images.tsv`: T1w.
- `samples.tsv` + `samples.json`: diagnosis (target), age, sex, scanner serial and `ghost_artifact`. The fMRI availability flags are left out, and the `phenotype/` instruments are not merged.
- `splits.tsv`: split, rank and complete per participant.

## Excluded

- fMRI: `sub-*/func/` and the top-level `task-*_bold.json` (71.7 GB).
- `sub-*/dwi/`: raw 64-direction DTI for 262 subjects (`.nii.gz`, `.bval`, `.bvec`, `.json`). No task uses it.
- `sub-*/beh/`: out-of-scanner stop-signal training logs (6 MB).
- The snapshot has no `derivatives/`.

## Notes

- The paper's protocol also had a high-resolution T2w scan, but this release only has T1w in `anat/`.
- The dataset `README` describes a `derivatives/` folder and links to FreeSurfer/fMRIPrep outputs on a legacy S3 path (`s3://openneuro/ds000030/ds000030_R1.0.5/`). Neither is in snapshot 1.0.0, and we don't fetch them.
- About 20% of T1w scans show a headset aliasing ghost through the temporal lobes (`ghost_artifact`). Consider it as a QC covariate.
- `CHANGES` has duplicated junk test entries from the 2018 republishing ("asdfasdf"); harmless.
