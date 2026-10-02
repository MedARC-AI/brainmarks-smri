# CNP (LA5c)

UCLA Consortium for Neuropsychiatric Phenomics LA5c study: 272 adults (21–50 y) who are healthy controls or have schizophrenia, bipolar disorder or ADHD. We use the T1w (and DWI) scans for 4-way diagnosis classification.

- **Source:** OpenNeuro [ds000030](https://openneuro.org/datasets/ds000030/versions/1.0.0). Downloaded with `openneuro-py`.
- **Version:** snapshot 1.0.0 (2020-04-21), the latest.
- **DOI:** [10.18112/openneuro.ds000030.v1.0.0](https://doi.org/10.18112/openneuro.ds000030.v1.0.0)
- **License:** CC0
- **Citation** (per `HowToAcknowledge`): Poldrack, R. A., Congdon, E., Triplett, W., et al. (2016). A phenome-wide examination of neural and cognitive function. *Scientific Data*, 3, 160110. If using derived data, also: Gorgolewski, K. J., Durnez, J., & Poldrack, R. A. (2017). Preprocessed Consortium for Neuropsychiatric Phenomics dataset. *F1000Research*, 6, 1262.

## Contents

The structural and diffusion part of the raw BIDS release (1686 files, 13.4 GB):

- `sub-*/anat/`: T1w MPRAGE + JSON for 265 subjects, defaced with `mri_deface`.
- `sub-*/dwi/`: raw 64-direction DWI (`.nii.gz`, `.bval`, `.bvec`, `.json`) for 262 subjects.
- `participants.tsv`: 272 subjects with `diagnosis` (CONTROL 130, SCHZ 50, BIPOLAR 49, ADHD 43), `age`, `gender`, per-scan availability flags, `ScannerSerialNumber` (2 Siemens Trio scanners: 180 / 92 subjects) and `ghost_NoGhost` (55 subjects flagged with a headset aliasing ghost artifact on the T1w).
- `phenotype/`: 52 instrument tables + JSON data dictionaries (SCID, BPRS, SANS/SAPS, Hamilton, YMRS, ASRS, WAIS, medication, demographics, ...).
- `dataset_description.json`, `README`, `CHANGES`.

## Excluded

- fMRI: `sub-*/func/` and the top-level `task-*_bold.json` (71.7 GB).
- `sub-*/beh/`: out-of-scanner stop-signal training logs (6 MB).
- The snapshot has no `derivatives/`.

## Notes

- **Usable T1w cohort: 265** (CONTROL 125, SCHZ 50, BIPOLAR 49, ADHD 41). 7 subjects have only fMRI, so they appear in `participants.tsv` but have no folder here: sub-10299, -10428, -10501, -10971, -11121 (CONTROL) and -70035, -70036 (ADHD).
- The paper's protocol also had a high-resolution T2w scan, but this release only has T1w in `anat/`.
- The dataset `README` describes a `derivatives/` folder and links to FreeSurfer/fMRIPrep outputs on a legacy S3 path (`s3://openneuro/ds000030/ds000030_R1.0.5/`). Neither is in snapshot 1.0.0, and we don't fetch them.
- About 20% of T1w scans show the ghost artifact through the temporal lobes (`ghost_NoGhost`). Consider it as a QC covariate.
- `CHANGES` has duplicated junk test entries from the 2018 republishing ("asdfasdf"); harmless.

## Usage

```sh
bash scripts/cnp/download.sh   # 13.4 GB; resumable
```
