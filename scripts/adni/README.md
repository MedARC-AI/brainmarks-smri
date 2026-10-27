# ADNI

Alzheimer's Disease Neuroimaging Initiative: a longitudinal study of older adults who are cognitively normal, have mild cognitive impairment or have Alzheimer's dementia, at about 70 sites in the US and Canada (ADNI1, GO, 2, 3 and 4, 2004 onward). We use T1w and FLAIR MRI at baseline, 2 and 4 years, with diagnosis, cognitive scores, amyloid and tau PET, CSF and plasma biomarkers, and FreeSurfer volumes as targets.

- **Homepage:** <https://adni.loni.usc.edu/>
- **Source:** LONI Image and Data Archive (IDA, <https://ida.loni.usc.edu/>), after an approved ADNI data access application. Images are downloaded as DICOM zips of "Original" series and converted to NIfTI with `dcm2niix` v1.0.20260724 (`-z y -b y -ba y`). Study data is the ADNIMERGE2 R package (Study Data, ADNIMERGE2).
- **Version:** ADNIMERGE2 0.1.1 (data pulled 2025-11-12). The images are pinned by their LONI image IDs, listed in `source/selected_images.tsv` (IDA download 2026-06).
- **DOI:** none.
- **License:** ADNI Data Use Agreement: research use only, no redistribution, and the ADNI acknowledgement and publication policy apply.
- **Citation:** Petersen, R. C., Aisen, P. S., Beckett, L. A., et al. (2010). Alzheimer's Disease Neuroimaging Initiative (ADNI): clinical characterization. *Neurology*, 74(3), 201-209. Publications must also include the ADNI acknowledgement from the data use agreement.
- **Code:** [`scripts/adni/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/adni) chooses the images, converts them and builds `tables/`.

## Samples

One sample per participant and session: y0 (the first visit with a usable T1w), y2 and y4 (the visits nearest 2 and 4 years later, within 6 months). There is no official split: 60/20/20 by participant, stratified by y0 diagnosis and enrolling phase. Complete = y0 T1w and diagnosis.

| split | participants | samples | complete | age | female | sites | session_id | dx | amyloid_status |
|---|---|---|---|---|---|---|---|---|---|
| train | 1841 | 3061 | 1811 | 73.6 ± 7.7 | 50% | 71 | y0 1841 / y2 815 / y4 405 | CN 1254 / Dementia 528 / MCI 1198 | 0 987 / 1 834 |
| val | 615 | 1031 | 604 | 73.8 ± 7.4 | 48% | 68 | y0 615 / y2 276 / y4 140 | CN 408 / Dementia 189 / MCI 403 | 0 289 / 1 300 |
| test | 614 | 996 | 603 | 73.5 ± 7.2 | 51% | 70 | y0 614 / y2 259 / y4 123 | CN 401 / Dementia 172 / MCI 394 | 0 326 / 1 263 |
| total | 3070 | 5088 | 3018 | 73.6 ± 7.6 | 50% | 72 | y0 3070 / y2 1350 / y4 668 | CN 2063 / Dementia 889 / MCI 1995 | 0 1602 / 1 1397 |

## Contents

`source/` (65 GB):

- `sub-<PTID>/ses-<y0|y2|y4>/anat/`: per session one T1w, an `acq-3T` T1w at ADNI1 visits that also have a 3 T scan (240), and FLAIR (ADNIGO onward), each with its `dcm2niix` JSON sidecar. 8,582 images.
- `selected_images.tsv`: the LONI image ID, series description and field strength of every image. `image_dates.tsv`: the acquisition date of each image.
- `ADNIMERGE2/`: the extracted R package; the tables are `data/*.rda` (readable with `pyreadr`).

`tables/` (derived from `source/`):

- `images.tsv`: T1w, T1w `acq-3T` and FLAIR per session.
- `samples.tsv` + `samples.json`: demographics, APOE, diagnosis, years to dementia and diagnostic follow-up, cognitive scores, amyloid PET (status, Centiloid), tau PET, CSF, plasma, WMH and FreeSurfer 7 volumes and thickness. Each is matched to the scan by the nearest record within a window (180 days for diagnosis, 90 days for cognition, 1 year for PET, CSF and plasma, 60 days for MRI measures).
- `splits.tsv`: split, rank and complete per participant.

## Image choice

Many visits have several T1w scans. Per visit, we drop scans that ADNI's Mayo QC rates unusable, distortion-uncorrected copies (`_ND`), scans covering under 140 mm, and one series that `dcm2niix` cannot convert (I820315). Then we prefer, in order: 1.5 T (the ADNI1 protocol; later phases are 3 T), the series Mayo selected or ranked best (MAYOADIRL_MRI_QUALITY_ADNI3, MRIMPRANK), accelerated scans (as in ADNI3 and 4), the protocol scan over its repeat, the later series (a rescan), and the lowest image ID. FLAIR: the series nearest the visit, preferring Mayo's selection.

## Excluded

- PET images, other MRI series (T2w, T2*, DWI, ASL, localizers) and all visits other than y0, y2 and y4.

## Notes

- ADNI1 is 1.5 T and has no FLAIR or amyloid PET. ADNI4 is almost all y0.
- A few y4 scans were acquired under the next phase's protocol (ADNI1 to ADNI2, ADNI2 to 3, ADNI3 to 4).
- Most FreeSurfer 7 rows are not yet QC'd by UCSF (`fs_status` partial).
- For 5 series with uneven slice spacing, `dcm2niix` writes a resampled copy (`Eq_1`), which we keep.
- `dcm2niix` output depends on its version: `source.sha256` matches only with v1.0.20260724.
