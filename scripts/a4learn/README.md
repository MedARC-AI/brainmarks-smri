# A4/LEARN

The Anti-Amyloid Treatment in Asymptomatic Alzheimer's (A4) study and its companion LEARN study: cognitively normal adults aged 65-85 at 67 sites in the US, Canada, Australia and Japan (2014-2023). A4 randomized 1,169 people with elevated brain amyloid to solanezumab or placebo for 4.5 years; LEARN followed 538 people without elevated amyloid. We use T1w MRI at screening, week 84 and week 240, with amyloid and tau PET, plasma pTau217, cognition (PACC) and treatment arm as targets.

- **Homepage:** <https://www.a4studydata.org/>
- **Source:** LONI Image and Data Archive (IDA, <https://ida.loni.usc.edu/>), after an approved A4 data access application. The T1 images come as de-faced NIfTI with JSON sidecars in zips (`ImageData - T1 - NN of 16.zip`), the clinical data as `Clinical.zip`.
- **Version:** final study data (tranche 2; images released fall 2024), downloaded 2026-06-18. IDA does not version the release: `source.sha256` pins the files.
- **DOI:** none.
- **License:** A4 Data Use Agreement: research use only, no redistribution.
- **Citation:** Sperling, R. A., Donohue, M. C., Raman, R., et al. (2023). Trial of Solanezumab in Preclinical Alzheimer's Disease. *New England Journal of Medicine*, 389(12), 1096-1107. Data sharing: Raman, R., et al. (2024). Maximizing the utility of Alzheimer's disease trial data: sharing of baseline A4 and LEARN data. *Journal of Prevention of Alzheimer's Disease*, 11(4), 889-894.
- **Code:** [`scripts/a4learn/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/a4learn) chooses the images, extracts them and builds `tables/`.

## Samples

One sample per participant and session: bl (the first visit with a T1w: screening visit 004, or 006 for most of LEARN), wk84 (visit 027) and wk240 (visit 066, end of the double-blind phase). There is no official split: 60/20/20 by participant, stratified by group (A4 placebo, A4 solanezumab, LEARN, screen-fail). Complete = bl T1w and amyloid PET SUVR.

| split | participants | samples | complete | age | female | session_id | group | screening_amyloid_read |
|---|---|---|---|---|---|---|---|---|
| train | 1065 | 2336 | 1065 | 73.6 ± 5.0 | 59% | bl 1065 / wk240 681 / wk84 590 | A4 1783 / LEARN 501 / SF 52 | 0 505 / 1 1831 |
| val | 354 | 747 | 353 | 73.4 ± 4.9 | 60% | bl 354 / wk240 204 / wk84 189 | A4 567 / LEARN 163 / SF 17 | 0 163 / 1 584 |
| test | 354 | 753 | 354 | 73.7 ± 5.1 | 59% | bl 354 / wk240 212 / wk84 187 | A4 578 / LEARN 158 / SF 17 | 0 159 / 1 594 |
| total | 1773 | 3836 | 1772 | 73.5 ± 5.0 | 60% | bl 1773 / wk240 1097 / wk84 966 | A4 2928 / LEARN 822 / SF 86 | 0 827 / 1 3009 |

## Contents

`source/` (43 GB):

- `ImageData/<A4|LEARN|SF>_MR_T1_<BID>_<VISCODE>.nii.gz` + `.json`: the T1w of each session (3,836), as distributed (3 T sagittal accelerated MPRAGE, 1.2 mm).
- `selected_images.tsv`: participant, session, BID, visit code and file of every image.
- `Clinical/Derived Data/`: participant info, visits, PACC, cognitive scores (ADQS), disposition. `Clinical/External Data/`: amyloid and tau PET SUVRs, PET visual reads, plasma biomarkers, MRI reads, WMH and regional volumes.

`tables/` (derived from `source/`):

- `images.tsv`: one T1w per session.
- `samples.tsv` + `samples.json`: group and treatment, demographics, APOE, screening amyloid (Centiloid, SUVR, visual read), cognition (PACC, MMSE, CFI, ADL, CDR), PACC change at week 240, amyloid and tau PET SUVR, plasma pTau217, WMH, MRI reads (microhemorrhages, siderosis) and regional volumes. Cognition is matched within 90 days of the MRI, PET and plasma within 1 year; MRI measures come from the same visit.
- `splits.tsv`: split, rank and complete per participant.

## Excluded

- Visits other than bl, wk84 and wk240; tau PET images; the FLAIR, T2* and T2 TSE series (not downloaded); the safety ECG and lab tables (`clrm_ecg.csv`, `clrm_lab.csv`, 270 MB).

## Notes

- 36 images are left out as unusable: 35 whose intensity scaling puts the background far below zero (about -60,000 to -160,000), and one blank image. Baseline comes from visits 004, 006 or 009 only. 13 participants have no usable image left.
- A4 participants all have a positive screening amyloid read, LEARN participants a negative one, so `group` is also the amyloid eligibility label.
- 178 people were rescreened under a new BID (`PREVBID`); none has images under both.
- Dates are days from consent; 5 LEARN wk240 sessions have no date in any table (age n/a).
- Sex, race and ethnicity codes are mapped with the A4 data dictionary (`derived_datadic.csv`, in the A4 documentation on IDA). Site IDs are not shared.
- Most participants have pTau217, and about 400 have tau PET.
