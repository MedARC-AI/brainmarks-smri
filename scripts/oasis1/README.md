# OASIS-1

Open Access Series of Imaging Studies, cross-sectional: 416 adults aged 18-96 (Washington University), each with 3-4 T1w MPRAGE scans from one session. About 100 of those over 60 have very mild to moderate Alzheimer's disease (CDR 0.5-2). A reliability set of 20 nondemented subjects was rescanned within 90 days. We use it for dementia classification.

- **Homepage:** <https://sites.wustl.edu/oasisbrains/home/oasis-1/>
- **Source:** 12 raw image discs from <https://download.nrg.wustl.edu/data/> and the clinical spreadsheets from the homepage, downloaded with `curl`. No login is needed, but users must accept the OASIS data use terms (<https://sites.wustl.edu/oasisbrains/home/access/>). The images are distributed in Analyze 7.5 format (`.hdr`/`.img`), which has no affine. We store each as `.nii.gz` with the same voxels, and set the affine from the header's orientation code as FreeSurfer reads it (sagittal: ASL; transverse: LAS; the two `*.4dfp` images: PIR).
- **Version:** unversioned; discs dated 2007-06, spreadsheets 2024. Pinned by sha256.
- **DOI:** [10.1162/jocn.2007.19.9.1498](https://doi.org/10.1162/jocn.2007.19.9.1498) (paper)
- **License:** OASIS data use terms: non-commercial research only, no redistribution without written permission. Manual dataset: you download and build it yourself, and our mirror has only this README and the tables.
- **Citation:** Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498-1507.
- **Code:** [`scripts/oasis1/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/oasis1) downloads `source/` and builds `tables/`, `derivatives/` and `shards/`.

## Samples

One sample per session (436): one session for each of the 416 participants, plus a rescan within 90 days for the 20 reliability subjects. The main image is the averaged native-space T1w. CDR and MMSE are given only for the first session, and mostly for those over 60. There is no official split: 60/20/20 by participant, stratified by CDR group (0 / 0.5 / ≥ 1 / n/a). Complete = CDR (235).

| split | participants | samples | complete | age | female | cdr |
|---|---|---|---|---|---|---|
| train | 250 | 266 | 141 | 51.2 ± 25.7 | 60% | 0.0 81 / 0.5 42 / 1.0 16 / 2.0 2 |
| val | 83 | 87 | 47 | 52.8 ± 25.7 | 62% | 0.0 27 / 0.5 14 / 1.0 6 |
| test | 83 | 83 | 47 | 50.4 ± 23.7 | 66% | 0.0 27 / 0.5 14 / 1.0 6 |
| total | 416 | 436 | 235 | 51.4 ± 25.3 | 61% | 0.0 135 / 0.5 70 / 1.0 28 / 2.0 2 |

## Contents

`source/`: the 12 extracted raw discs (`disc1/` .. `disc12/`, 12,864 files, 16 GB), one folder per session (`OAS1_<id>_MR<n>/`), and the clinical spreadsheets:

- `RAW/`: the 3-4 raw MPRAGE runs (`mpr-<n>_anon`), plus GIF previews.
- `PROCESSED/MPRAGE/SUBJ_111/`: the runs averaged, native space, 1 mm (our main T1w). `T88_111/`: the average registered to the Talairach atlas, also skull-stripped (`masked_gfc`).
- `FSL_SEG/`: FSL tissue segmentation of the skull-stripped atlas image.
- `.txt`, `.xml`: per-session acquisition and clinical summary.
- `oasis_cross-sectional.xlsx`: age, sex, education, SES, MMSE, CDR, eTIV, nWBV, ASF. `oasis_cross-sectional-reliability.xlsx`: the same for the rescans. `oasis_cross-sectional_facts.pdf`: the fact sheet.

`tables/` (derived from `source/`):

- `images.tsv`: the main T1w, the raw runs (desc `mpr-<n>`), the atlas images (`t88`, `t88_masked`) and the tissue segmentation (mask `fseg`).
- `samples.tsv` + `samples.json`: age, sex, CDR, MMSE, education, SES, eTIV, nWBV, ASF and rescan delay. Not on our mirror.
- `splits.tsv`: split, rank and complete per participant.

## Excluded

- The FreeSurfer discs (`oasis_cs_freesurfer_disc*.tar.gz`, ~105 GB).

## Notes

- About 70 of the 150 OASIS-2 participants are also in OASIS-1 (same site, scanner and protocol). Don't train on one and test on the other.
- All participants are right-handed.
