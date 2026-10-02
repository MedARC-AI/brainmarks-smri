# CNP (LA5c)

UCLA Consortium for Neuropsychiatric Phenomics LA5c study: 272 adults (21–50 y) who are healthy controls or have schizophrenia, bipolar disorder or ADHD. We use the T1w (and DWI) scans for 4-way diagnosis classification.

- **Source:** OpenNeuro [ds000030](https://openneuro.org/datasets/ds000030/versions/1.0.0). Downloaded with `openneuro-py`.
- **Version:** snapshot 1.0.0 (2020-04-21), the latest.
- **DOI:** [10.18112/openneuro.ds000030.v1.0.0](https://doi.org/10.18112/openneuro.ds000030.v1.0.0)
- **License:** CC0
- **Citation** (per `HowToAcknowledge`): Poldrack, R. A., Congdon, E., Triplett, W., et al. (2016). A phenome-wide examination of neural and cognitive function. *Scientific Data*, 3, 160110. If using derived data, also: Gorgolewski, K. J., Durnez, J., & Poldrack, R. A. (2017). Preprocessed Consortium for Neuropsychiatric Phenomics dataset. *F1000Research*, 6, 1262.

## Usage

```sh
bash scripts/cnp/download.sh                 # 13.4 GB; resumable
uv run python scripts/cnp/build_tables.py    # tables/; prints the summary below
```

## Samples

One sample per participant. Split 60/20/20, stratified by diagnosis; there is no official split. Complete = T1w and diagnosis: the 7 participants with only fMRI (no T1w in this release) are in the tables but not complete.

| split | participants | complete | age | female | diagnosis |
|---|---|---|---|---|---|
| train | 163 | 159 | 33.1 ± 9.1 | 45% | ADHD 26 / BIPOLAR 29 / CONTROL 78 / SCHZ 30 |
| val | 55 | 54 | 32.1 ± 9.1 | 42% | ADHD 9 / BIPOLAR 10 / CONTROL 26 / SCHZ 10 |
| test | 54 | 52 | 34.8 ± 10.1 | 37% | ADHD 8 / BIPOLAR 10 / CONTROL 26 / SCHZ 10 |
| total | 272 | 265 | 33.2 ± 9.4 | 43% | ADHD 43 / BIPOLAR 49 / CONTROL 130 / SCHZ 50 |

## Contents

`source/`: the structural and diffusion part of the raw BIDS release (1686 files, 13.4 GB):

- `sub-*/anat/`: T1w MPRAGE + JSON for 265 subjects, defaced with `mri_deface`.
- `sub-*/dwi/`: raw 64-direction DWI (`.nii.gz`, `.bval`, `.bvec`, `.json`) for 262 subjects.
- `participants.tsv`: 272 subjects with diagnosis, age, sex, per-scan availability flags, scanner serial number (2 Siemens Trio scanners) and a T1w ghost-artifact flag.
- `phenotype/`: 52 instrument tables + JSON data dictionaries (SCID, BPRS, SANS/SAPS, Hamilton, YMRS, ASRS, WAIS, medication, demographics, ...).
- `dataset_description.json`, `README`, `CHANGES`.

`tables/` (built by `build_tables.py`; layout in `src/brain_datasets/tables.py`):

- `images.tsv`: T1w, and the raw DWI series as modality `DTI` (the `.bval`/`.bvec` sit next to the NIfTI).
- `samples.tsv` + `samples.json`: diagnosis (target), age, sex, scanner serial and `ghost_artifact`. The fMRI availability flags are left out, and the `phenotype/` instruments are not merged.
- `splits.tsv`: split, rank and complete per participant.

## Excluded

- fMRI: `sub-*/func/` and the top-level `task-*_bold.json` (71.7 GB).
- `sub-*/beh/`: out-of-scanner stop-signal training logs (6 MB).
- The snapshot has no `derivatives/`.

## Notes

- The paper's protocol also had a high-resolution T2w scan, but this release only has T1w in `anat/`.
- The dataset `README` describes a `derivatives/` folder and links to FreeSurfer/fMRIPrep outputs on a legacy S3 path (`s3://openneuro/ds000030/ds000030_R1.0.5/`). Neither is in snapshot 1.0.0, and we don't fetch them.
- About 20% of T1w scans show a headset aliasing ghost through the temporal lobes (`ghost_artifact`). Consider it as a QC covariate.
- `CHANGES` has duplicated junk test entries from the 2018 republishing ("asdfasdf"); harmless.
