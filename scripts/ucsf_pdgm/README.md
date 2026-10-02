# UCSF-PDGM

Preoperative 3T MRI of adult diffuse glioma (WHO grade 2–4), 495 patients / 501 exams, single center (UCSF, GE Discovery 750). We use it for IDH, MGMT, 1p/19q and grade classification, OS survival, and tumor segmentation.

- **Source:** TCIA collection [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/). The clinical CSVs are plain HTTPS downloads from the collection page. The images are only distributed as an Aspera Faspex package (id 1065, "UCSF-PDGM Version 5", UUID `e6a98a44-bc23-4ef1-8380-9a64c2150b8a`, released 2025-05-29, 12,030 files, 142 GB, top folder `UCSF-PDGM-v5/`). There is no HTTPS route: the package isn't in NBIA or IDC, and the Faspex server has no HTTP gateway. `download.sh` fetches the exam folders over FASP (TCP/UDP 33001 to `144.30.235.113`) with `scripts/tcia_faspex.py`, using the public link from the collection page and `ascp` exclude patterns for the file types we leave out. The server sometimes stalls a session or refuses its connection partway through; `tcia_faspex.py` restarts `ascp` when it stops making progress, and `ascp` resumes.
- **Version:** collection version 5 (2025-05-30).
- **DOI:** [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37)
- **License:** CC BY 4.0. Use must follow the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/), and the dataset must be cited.
- **Citation:** Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37. Paper: Calabrese, E., et al. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset. *Radiology: Artificial Intelligence*, 4(6), e220058.

## Contents

6,515 files, 15.9 GB:

- `UCSF-PDGM-metadata_v5.csv`: 501 rows, one per exam. Sex, age at MRI, WHO CNS grade (2: 56, 3: 43, 4: 402), WHO 2021 diagnosis, MGMT status/index, 1p/19q, IDH (wildtype 398 / mutant 103), vital status + OS (days), extent of resection, prior biopsy, and **BraTS 2021 ID and cohort** (262 BraTS21 segmentation-training cases, 36 validation).
- `UCSF-PDGM-metadata_glossary.csv`: column definitions.
- `UCSF-PDGM-v5/UCSF-PDGM-NNNN_nifti/UCSF-PDGM-NNNN_<type>.nii.gz`: 501 exam folders × 13 files = 6,513 NIfTI files, as in the package tree. All volumes were skull-stripped and co-registered to 1 mm FLAIR space by the source. No DICOM is available. Every exam has all 13 types.

Each package folder holds 24 files (25 for the six follow-up exams, which add `ASL_M0`):

| Group | Files | Kept | Size kept |
|---|---|---|---|
| structural | `T1`, `T1c`, `T2`, `FLAIR` | yes | 6.2 GB |
| structural, bias-corrected | `T1_bias`, `T1c_bias`, `T2_bias`, `FLAIR_bias` | yes | 6.3 GB |
| diffusion | `DWI` (trace), `ADC` | yes | 3.1 GB |
| diffusion | `DWI_bias` | no | |
| DTI | fits `DTI_eddy_{FA,MD,L1,L2,L3}`; raw 4D `DTI_eddy_noreg` + `DTI_eddy.eddy_rotated_bvecs` | no | |
| other | `SWI`, `SWI_bias`, `ASL` (+ `ASL_M0` in 6 exams) | no | |
| targets | `tumor_segmentation` (BraTS-style labels, radiologist-corrected), `brain_segmentation`, `brain_parenchyma_segmentation` | yes | 0.36 GB |

## Excluded

- About 126 GB of the 142 GB package (the API reports no per-file sizes):
  - `DWI_bias`: the trace DWI is kept, so this bias-corrected copy is redundant.
  - All `DTI_eddy_*` (model fits, raw 4D series, bvecs): DTI fits are not benchmark targets, and the raw 4D series are not used.
  - `SWI`, `SWI_bias`, `ASL`, `ASL_M0`: large, and not used as inputs by the common benchmarks (IDH/MGMT/grade classification, survival, tumor segmentation).
- The DTI bval/bvec zips on the collection page, which describe the raw 4D DTI.

## Notes

- Integrity: the package has no checksum file. FASP verifies each transfer, and all 6,513 files pass `gzip -t`.
- IDs: the image folders use 4 digits (`UCSF-PDGM-0004`), but `metadata_v5.csv` uses 3 (`UCSF-PDGM-004`). Normalize before joining.
- 501 exams from 495 patients. Six IDs are follow-up exams of other patients, renamed in v3 (e.g. `UCSF-PDGM-0315` → `UCSF-PDGM-0433_FU007d`). Split by patient.
- Overlap: 298 exams are in BraTS 2021 (`BraTS21 ID` column). Use it to avoid train/test leakage when using both datasets.
- Version history: v2 fixed segmentation rounding errors and added BraTS IDs; v3 renamed the follow-up exams; v5 fixed headers in `DTI_eddy_noreg` and added rotated bvecs.

## Usage

Requires `ascp` at `~/.aspera/sdk/` (install with `ascli config ascp install`) and outbound FASP to TCIA.

```sh
bash scripts/ucsf_pdgm/download.sh   # 15.9 GB, ~25 min; resumable (a complete re-run transfers nothing but takes ~12 min)
```
