# UCSF-PDGM

Preoperative 3T MRI of adult diffuse glioma (WHO grade 2–4), 495 patients / 501 exams, single center (UCSF, GE Discovery 750). We use it for IDH, MGMT, 1p/19q and grade classification, OS survival, and tumor segmentation.

- **Source:** TCIA collection [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/). The clinical CSVs are plain HTTPS downloads from the collection page. The images are an Aspera Faspex package: id 1065, "UCSF-PDGM Version 5", UUID `e6a98a44-bc23-4ef1-8380-9a64c2150b8a`, released 2025-05-29, 12,030 files, top folder `UCSF-PDGM-v5/`.
- **Version:** collection version 5 (2025-05-30).
- **DOI:** [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37)
- **License:** CC BY 4.0. Use must follow the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/), and the dataset must be cited.
- **Citation:** Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37. Paper: Calabrese, E., et al. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset. *Radiology: Artificial Intelligence*, 4(6), e220058.

## Status

**Partial: metadata only.** The images (142 GB) are only distributed as the Aspera Faspex package. There is no HTTPS route: not in NBIA or IDC, and the Faspex server has no HTTP gateway (`http_gateway_url: null`). The devcontainer firewall now allows FASP to TCIA's transfer node (TCP/UDP 33001 to `144.30.235.113`), so once the container is rebuilt, `download.sh` can be extended to fetch the images with `ascli`.

## Contents

Downloaded (2 files, 60 KB):

- `UCSF-PDGM-metadata_v5.csv`: 501 rows, one per exam. Sex, age at MRI, WHO CNS grade (2: 56, 3: 43, 4: 402), WHO 2021 diagnosis, MGMT status/index, 1p/19q, IDH (wildtype 398 / mutant 103), vital status + OS (days), extent of resection, prior biopsy, and **BraTS 2021 ID and cohort** (262 BraTS21 segmentation-training cases, 36 validation).
- `UCSF-PDGM-metadata_glossary.csv`: column definitions.

Image package (not yet downloaded): one folder per exam (`UCSF-PDGM-NNNN_nifti/`), all volumes skull-stripped and co-registered to 1 mm FLAIR space by the source (no DICOM is available). Each folder holds 24 files:

| Group | Files | Plan |
|---|---|---|
| structural | `T1`, `T1c`, `T2`, `FLAIR` (+ `*_bias` bias-corrected copies) | keep (`*_bias`: undecided) |
| diffusion | `DWI`, `DWI_bias`, `ADC` | keep `DWI`, `ADC` |
| DTI | fits `DTI_eddy_{FA,MD,L1,L2,L3}`; raw 4D `DTI_eddy_noreg` + `DTI_eddy.eddy_rotated_bvecs` | exclude |
| other | `SWI`, `SWI_bias`, `ASL` | exclude |
| targets | `tumor_segmentation` (BraTS-style labels, radiologist-corrected), `brain_segmentation`, `brain_parenchyma_segmentation` | keep |

## Excluded

- The DTI bval/bvec zips on the collection page (they describe the raw 4D DTI, which we don't plan to fetch).
- For now, all images (see Status).

## Notes

- IDs: the image folders use 4 digits (`UCSF-PDGM-0004`), but `metadata_v5.csv` uses 3 (`UCSF-PDGM-004`). Normalize before joining.
- 501 exams from 495 patients. Six IDs are follow-up exams of other patients, renamed in v3 (e.g. `UCSF-PDGM-0315` → `UCSF-PDGM-0433_FU007d`). Split by patient.
- Overlap: 298 exams are in BraTS 2021 (`BraTS21 ID` column). Use it to avoid train/test leakage when using both datasets.
- Version history: v2 fixed segmentation rounding errors and added BraTS IDs; v3 renamed the follow-up exams; v5 fixed headers in `DTI_eddy_noreg` and added rotated bvecs.

## Usage

```sh
bash scripts/ucsf_pdgm/download.sh   # 60 KB (metadata only); resumable
```
