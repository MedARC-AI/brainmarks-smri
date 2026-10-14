# OASIS

The Open Access Series of Imaging Studies, releases 1 and 2, from the Washington University Knight ADRC. All scans are T1-weighted MPRAGEs from one Siemens 1.5 T Vision scanner, 3–4 repetitions per session.

- **OASIS-1 (cross-sectional):** 416 right-handed adults aged 18–96, one session each. 100 participants over 60 have very mild to moderate Alzheimer's disease (CDR 0.5 / 1 / 2: 70 / 28 / 2). 20 nondemented participants have a second session within 90 days, for test–retest reliability.
- **OASIS-2 (longitudinal):** 150 right-handed adults aged 60–96, 2–5 visits at least a year apart (373 sessions). 72 stayed nondemented, 64 were demented at the first visit and stayed so, and 14 converted from nondemented to demented.

OASIS-2 re-releases 115 OASIS-1 sessions under new IDs (see Notes), so we use the two as one dataset with one image per participant. We use it for dementia classification (CDR).

- **Homepage:** <https://sites.wustl.edu/oasisbrains/home/oasis-1/> and <https://sites.wustl.edu/oasisbrains/home/oasis-2/>
- **Source:** the OASIS website. Plain HTTPS tarballs from `https://download.nrg.wustl.edu/data/` and spreadsheets from `https://sites.wustl.edu/oasisbrains/files/`, fetched with `curl` (no login).
- **Version:** Pinned by the server metadata below (retrieved 2026-10-09), the tarball sha256s in `download.sh` and `manifest.sha256`. The images are the 2007 (OASIS-1) and 2009 (OASIS-2) releases. The spreadsheets were re-uploaded with the 2024 site move.

  | File | Size (bytes) | Last-Modified |
  |---|---|---|
  | `oasis_cross-sectional_disc{1..12}.tar.gz` | 1,247,923,584–1,473,055,117 (16.3 GB total) | 2007-06-27 |
  | `OAS2_RAW_PART1.tar.gz` | 10,874,461,059 | 2009-03-18 |
  | `OAS2_RAW_PART2.tar.gz` | 8,449,289,603 | 2009-03-18 |
  | `oasis_cross-sectional-5708aa0a98d82080.xlsx` | 35,716 | 2025-10-04 |
  | `oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx` | 50,743 | 2025-10-04 |

- **DOI:** none for the data. Papers: [10.1162/jocn.2007.19.9.1498](https://doi.org/10.1162/jocn.2007.19.9.1498) (OASIS-1), [10.1162/jocn.2009.21407](https://doi.org/10.1162/jocn.2009.21407) (OASIS-2).
- **License:** no license file. The [OASIS Data Use Terms](https://sites.wustl.edu/oasisbrains/home/access/) apply.
- **Citation:** Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498–1507. Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI data in nondemented and demented older adults. *Journal of Cognitive Neuroscience*, 22(12), 2677–2684. Acknowledgement: "Data were provided by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; and OASIS-2: Longitudinal: Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382."
- **Code:** [`scripts/oasis/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/oasis) re-downloads `source/` and rebuilds `tables/`.

## Samples

One sample per participant, one T1w each (the session's first MPRAGE): the 416 OASIS-1 participants (session `MR1`) and the 35 OASIS-2 participants who are not in OASIS-1 (first visit). Split 60/20/20, stratified by CDR group (0, ≥ 0.5, unrated) × sex × age bin (< 40, 40–60, 60–75, 75+); there is no official split. Complete = a CDR rating (everyone over 60 and 37 younger participants).

| split | participants | complete | age | female | cdr | release |
|---|---|---|---|---|---|---|
| train | 270 | 161 | 54.3 ± 25.1 | 60% | 0.0 90 / 0.5 50 / 1.0 19 / 2.0 2 | oasis1 248 / oasis2 22 |
| val | 92 | 56 | 53.9 ± 24.2 | 60% | 0.0 32 / 0.5 17 / 1.0 7 | oasis1 84 / oasis2 8 |
| test | 89 | 53 | 54.5 ± 24.6 | 62% | 0.0 29 / 0.5 19 / 1.0 5 | oasis1 84 / oasis2 5 |
| total | 451 | 270 | 54.3 ± 24.8 | 61% | 0.0 151 / 0.5 86 / 1.0 31 / 2.0 2 | oasis1 416 / oasis2 35 |

The CDR (global score of the [CDR scoring table](https://knightadrc.wustl.edu/app/uploads/2021/06/CDR-Table.pdf)) is 0 none, 0.5 very mild / questionable, 1 mild, 2 moderate. OASIS reports everyone with CDR > 0 as probable Alzheimer's disease, so there is no separate MCI or etiology label. The `oasis_dementia` task uses the participants over 60: CDR 0 vs ≥ 0.5 (114 vs 119), with equal numbers of each class per sex and split (200 left: 120 train, 40 val, 40 test).

## Contents

`source/` (19,670 files, 91 GB), one folder per release, each as downloaded:

`source/oasis1/` (48 GB):

- `oasis_cross-sectional.xlsx`: one row per session: ID, M/F, Hand, Age, Educ (1–5), SES (1–5), MMSE, CDR, eTIV, nWBV, ASF, Delay (days to the reliability session).
- `disc1/`…`disc12/` (48 GB): one folder per session, `OAS1_xxxx_MRy/`, each with:
  - `RAW/OAS1_xxxx_MRy_mpr-<n>_anon.{hdr,img}`: the individual MPRAGE scans (3: 59 sessions, 4: 375, 5: 1, 6: 1; 1,688 in total, all marked usable), Analyze 7.5, 256×256×128 at 1×1×1.25 mm, sagittal, int16, defaced. Plus a sagittal `.gif` preview of each.
  - `PROCESSED/MPRAGE/SUBJ_111/…_sbj_111.{hdr,img}`: the motion-corrected average of the scans, native space, 256×256×160 at 1 mm.
  - `PROCESSED/MPRAGE/T88_111/…_t88_gfc.{hdr,img}` and `…_t88_masked_gfc.{hdr,img}`: the average, gain-field corrected and registered to the Talairach (711-2C) atlas, 176×208×176 at 1 mm, without and with a brain mask. `t4_files/` holds the 4dfp transforms (scan → first scan, scan → atlas).
  - `FSL_SEG/…_t88_masked_gfc_fseg.{hdr,img}`: FSL FAST tissue segmentation of the masked atlas image (0 background, 1 CSF, 2 GM, 3 WM), with its log.
  - `OAS1_xxxx_MRy.xml` / `.txt`: session metadata (demographics, clinical scores, scan parameters, scan quality).

`source/oasis2/` (43 GB):

- `oasis_longitudinal_demographics.xlsx`: one row per session: Subject ID, MRI ID, Group, Visit, MR Delay (days since the first visit), M/F, Hand, Age, EDUC (years), SES (1–5), MMSE, CDR, eTIV, nWBV, ASF.
- `OAS2_RAW_PART1/` (OAS2_0001–0099, 25 GB) and `OAS2_RAW_PART2/` (OAS2_0100–0186, 19 GB): one folder per session, `OAS2_xxxx_MRy/RAW/mpr-<n>.nifti.{hdr,img}`: the individual MPRAGE scans (2: 1 session, 3: 123, 4: 249; 1,367 in total), NIfTI-1 pairs, 256×256×128 at 1×1×1.25 mm, sagittal, float32, with a valid qform. Not defaced.
- `OLD/` in 181 sessions: leftover 4dfp atlas transforms (`mpr-<n>_to_711-2C_t4`). No processed images are released for OASIS-2.

`tables/` (derived from `source/`):

- `images.tsv`: one T1w per participant, `RAW/…mpr-1…`. The other repetitions and OASIS-1's processed and segmented images are not listed.
- `samples.tsv` + `samples.json`: age, sex, CDR, MMSE, education (OASIS-1 code and OASIS-2 years), SES, eTIV, nWBV and ASF from the image's release; the release and session IDs, including the OASIS-2 session of each shared OASIS-1 session; and, for the 149 participants in OASIS-2, the OASIS-2 group and the worst CDR at their later visits.
- `splits.tsv`: split, rank and complete per participant.

## Excluded

- The OASIS-1 fact sheet, the reliability spreadsheet (the `_MR2` rescans, not used) and the web pages.
- The OASIS-1 FreeSurfer release (`oasis_cs_freesurfer_disc{1..11}.tar.gz`, 108.5 GB): recon-all outputs per session, not needed for the T1w tasks.

## Notes

- **Orientation:** OASIS-1's images are big-endian Analyze 7.5 without a usable orientation, and nibabel loads them with a default axial affine. That is wrong for the sagittal `RAW` and `SUBJ_111` images (axes permuted). `brainmarks_smri/datasets.py` and `scripts/preprocess.py` give them the affine of an OASIS-2 scan instead, right after `nib.load` (`ANALYZE_REFERENCES`). Both releases share the voxel layout (the shared sessions are voxel-identical) and all 373 OASIS-2 scans have the same qform: voxel axes (A, S, L), +i anterior, +j superior, +k left. The atlas-space `T88_111` and `FSL_SEG` images look right with nibabel's default (transverse), but left–right is not verified.
- **Overlap:** 115 OASIS-2 participants (59 Nondemented, 46 Demented, 10 Converted) each have one session that is also in OASIS-1, under an `OAS1_xxxx_MR1` ID (visit 1: 61, 2: 46, 3: 8). That is 115 of the 198 OASIS-1 participants over 60. The raw scans are voxel-identical (r > 0.999; OASIS-1's are defaced), and age, sex, SES, MMSE and CDR agree. `build_tables.py` finds them from the images (pairs with the same sex, age, CDR and MMSE) and keeps each person once; every other pair has r < 0.71. OAS1_0082_MR1 and OAS2_0116_MR2 match on every spreadsheet field and on eTIV but not voxel-wise (r = 0.70), so they may be the same person scanned twice; both are kept.