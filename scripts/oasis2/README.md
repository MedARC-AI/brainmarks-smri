# OASIS-2

Open Access Series of Imaging Studies, longitudinal: 150 adults aged 60-96 (Washington University), scanned at 2-5 visits at least a year apart (373 sessions, 3-4 T1w MPRAGE scans each). 72 stayed nondemented, 64 were demented at the first visit, and 14 converted from nondemented to demented. We use it for dementia tasks.

- **Homepage:** <https://sites.wustl.edu/oasisbrains/home/oasis-2/>
- **Source:** two raw image archives from <https://download.nrg.wustl.edu/data/> and the demographics spreadsheet from the homepage, downloaded with `curl`. No login is needed, but users must accept the OASIS data use terms (<https://sites.wustl.edu/oasisbrains/home/access/>). The images are distributed as NIfTI pairs (`.nifti.hdr`/`.img`); we store each as one `.nii.gz` with the same header and voxels.
- **Version:** unversioned; archives dated 2009-03, spreadsheet 2024. Pinned by sha256.
- **DOI:** [10.1162/jocn.2009.21407](https://doi.org/10.1162/jocn.2009.21407) (paper)
- **License:** OASIS data use terms: non-commercial research only, no redistribution without written permission.
- **Citation:** Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI data in nondemented and demented older adults. *Journal of Cognitive Neuroscience*, 22(12), 2677-2684.
- **Code:** [`scripts/oasis2/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/oasis2) downloads `source/` and builds `tables/`, `derivatives/` and `shards/`.

## Samples

One sample per session (373). The main image is the first raw MPRAGE run (`mpr-1`, the one the other runs were registered to). The group (nondemented / demented / converted) is per participant; CDR and MMSE are per visit. Dev/test is 50/50 by participant, stratified by group.

| split | participants | samples | age | female | group | cdr |
|---|---|---|---|---|---|---|
| dev | 75 | 187 | 76.6 ± 8.1 | 53% | converted 18 / demented 75 / nondemented 94 | 0.0 104 / 0.5 58 / 1.0 22 / 2.0 3 |
| test | 75 | 186 | 77.4 ± 7.1 | 61% | converted 19 / demented 71 / nondemented 96 | 0.0 102 / 0.5 65 / 1.0 19 |
| total | 150 | 373 | 77.0 ± 7.6 | 57% | converted 37 / demented 146 / nondemented 190 | 0.0 206 / 0.5 123 / 1.0 41 / 2.0 3 |

The group and CDR counts are per session.

## Contents

`source/`: the two extracted archives (`OAS2_RAW_PART1/`, `OAS2_RAW_PART2/`, 2,008 files, 20 GB), one folder per session (`OAS2_<id>_MR<n>/RAW/`), and `oasis_longitudinal_demographics.xlsx` (group, visit, delay, age, sex, education, SES, MMSE, CDR, eTIV, nWBV, ASF per visit). Each session has its 3-4 raw MPRAGE runs and the 4dfp transforms of each run to `mpr-1` and to the Talairach atlas (`*_to_711-2C_t4`).

`tables/` (derived from `source/`):

- `images.tsv`: the main T1w and the other runs (desc `mpr-<n>`).
- `samples.tsv` + `samples.json`: group, visit, MR delay, age, sex, CDR, MMSE, education, SES, eTIV, nWBV and ASF.
- `splits.tsv`: dev or test per participant.

## Notes

- About 70 of the 150 participants are also in OASIS-1 (see the [OASIS-1 README](../oasis1/README.md)).
- Files kept as distributed: an extra run in OAS2_0012_MR2 (desc `3906-3`, with its logs), a Siemens `.ima.gz` file in OAS2_0081_MR2 (not in the tables), and an `OLD/` folder in OAS2_0036_MR4.
- All participants are right-handed.
