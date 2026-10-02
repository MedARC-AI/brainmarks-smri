# UCSF-PDGM

Preoperative 3T MRI of adult diffuse glioma (WHO grade 2–4), 495 patients / 501 exams, single center (UCSF, GE Discovery 750). We use it for IDH, MGMT, 1p/19q and grade classification, OS survival, and tumor segmentation.

- **Homepage:** <https://www.cancerimagingarchive.net/collection/ucsf-pdgm/>
- **Source:** TCIA collection [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/). The clinical CSVs are plain HTTPS downloads from the collection page. The images are only distributed as an Aspera Faspex package (id 1065, "UCSF-PDGM Version 5", UUID `e6a98a44-bc23-4ef1-8380-9a64c2150b8a`, released 2025-05-29, 12,030 files, 142 GB, top folder `UCSF-PDGM-v5/`). There is no HTTPS route: the package isn't in NBIA or IDC, and the Faspex server has no HTTP gateway. We fetch the exam folders over FASP (TCP/UDP 33001 to `144.30.235.113`) with `ascp`, using the public link from the collection page and `ascp` exclude patterns for the file types we leave out. The server sometimes stalls a session or refuses its connection partway through, so the download restarts `ascp` when it stops making progress, and `ascp` resumes.
- **Version:** collection version 5 (2025-05-30).
- **DOI:** [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37)
- **License:** CC BY 4.0. Use must follow the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/), and the dataset must be cited.
- **Citation:** Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37. Paper: Calabrese, E., et al. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset. *Radiology: Artificial Intelligence*, 4(6), e220058.
- **Code:** [`scripts/ucsf_pdgm/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/ucsf_pdgm) re-downloads `source/` and rebuilds `tables/`.

## Samples

One sample per exam: 501 exams of 495 patients (6 follow-up exams are extra sessions of their patient). Split 60/20/20 by patient, stratified by IDH status × WHO grade; there is no official split. Complete = all 13 images, IDH status and overall survival.

| split | participants | samples | complete | age | female | idh | who_grade | mgmt | os_event |
|---|---|---|---|---|---|---|---|---|---|
| train | 297 | 299 | 296 | 56.5 ± 15.4 | 41% | mutant 62 / wildtype 237 | 2 34 / 3 25 / 4 240 | negative 68 / positive 176 | 0 148 / 1 151 |
| val | 100 | 103 | 100 | 57.1 ± 15.2 | 42% | mutant 21 / wildtype 82 | 2 11 / 3 9 / 4 83 | negative 22 / positive 65 | 0 51 / 1 52 |
| test | 98 | 99 | 98 | 57.8 ± 13.9 | 36% | mutant 20 / wildtype 79 | 2 11 / 3 9 / 4 79 | negative 24 / positive 61 | 0 51 / 1 48 |
| total | 495 | 501 | 494 | 56.9 ± 15.0 | 40% | mutant 103 / wildtype 398 | 2 56 / 3 43 / 4 402 | negative 114 / positive 302 | 0 250 / 1 251 |

## Contents

`source/` (6,515 files, 15.9 GB):

- `UCSF-PDGM-metadata_v5.csv`: one row per exam: demographics, WHO grade and diagnosis, IDH, MGMT, 1p/19q, OS, extent of resection, and the BraTS 2021 ID and cohort.
- `UCSF-PDGM-metadata_glossary.csv`: column definitions.
- `UCSF-PDGM-v5/UCSF-PDGM-NNNN_nifti/UCSF-PDGM-NNNN_<type>.nii.gz`: 501 exam folders × 13 files, as in the package tree. All volumes are skull-stripped and co-registered by the source, on the same 240×240×155 1 mm grid as BraTS 2021. No DICOM is available.

Each package folder holds 24 files (25 for the six follow-up exams, which add `ASL_M0`):

| Group | Files | Kept | Size kept |
|---|---|---|---|
| structural | `T1`, `T1c`, `T2`, `FLAIR` | yes | 6.2 GB |
| structural, bias-corrected | `T1_bias`, `T1c_bias`, `T2_bias`, `FLAIR_bias` | yes | 6.3 GB |
| diffusion | `DWI` (trace), `ADC` | yes | 3.1 GB |
| diffusion | `DWI_bias` | no | |
| DTI | fits `DTI_eddy_{FA,MD,L1,L2,L3}`; raw 4D `DTI_eddy_noreg` + `DTI_eddy.eddy_rotated_bvecs` | no | |
| other | `SWI`, `SWI_bias`, `ASL` (+ `ASL_M0` in 6 exams) | no | |
| targets | `tumor_segmentation` (BraTS-style labels 1/2/4, radiologist-corrected), `brain_segmentation`, `brain_parenchyma_segmentation` | yes | 0.36 GB |

`tables/` (derived from `source/`):

- `images.tsv`: the 13 images per exam. Modalities T1w, T1c, T2w, FLAIR (desc `bias` for the bias-corrected copies), DWI, ADC, and masks `tumor`, `brain`, `parenchyma`.
- `samples.tsv` + `samples.json`: one row per exam with cleaned metadata. Targets: `idh` (with `idh_variant`), `mgmt`, `codeletion_1p19q`, `who_grade`, `os_days` + `os_event`. IDs use 4 digits (`UCSF-PDGM-0004`); follow-ups are session `FU007d` etc. of their patient.
- `splits.tsv`: split, rank and complete per patient.

## Excluded

- About 126 GB of the 142 GB package (the API reports no per-file sizes):
  - `DWI_bias`: the trace DWI is kept, so this bias-corrected copy is redundant.
  - All `DTI_eddy_*` (model fits, raw 4D series, bvecs): DTI fits are not benchmark targets, and the raw 4D series are not used.
  - `SWI`, `SWI_bias`, `ASL`, `ASL_M0`: large, and not used as inputs by the common benchmarks (IDH/MGMT/grade classification, survival, tumor segmentation).
- The DTI bval/bvec zips on the collection page, which describe the raw 4D DTI.

## Notes

- Integrity: the package has no checksum file. FASP verifies each transfer, and all 6,513 files pass `gzip -t`.
- In `source/`, the image folders use 4-digit IDs (`UCSF-PDGM-0004`) but `metadata_v5.csv` uses 3 (`UCSF-PDGM-004`), and the six follow-up exams have their own IDs (renamed in v3, e.g. `UCSF-PDGM-0315` → `UCSF-PDGM-0433_FU007d`). The tables normalize both.
- `mgmt` (clinical interpretation) and `mgmt_index` disagree for 5 exams in the source.
- Overlap: 298 exams are also in BraTS 2021 (`brats21_id`). Benchmarks are evaluated within each dataset, so this only correlates scores across the two.
- Version history: v2 fixed segmentation rounding errors and added BraTS IDs; v3 renamed the follow-up exams; v5 fixed headers in `DTI_eddy_noreg` and added rotated bvecs.
