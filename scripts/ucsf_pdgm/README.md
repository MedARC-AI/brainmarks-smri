# UCSF-PDGM

Preoperative 3T MRI of adult diffuse glioma (WHO grade 2–4), 495 patients / 501 exams, single center (UCSF, GE Discovery 750). We use it for IDH, MGMT, 1p/19q and grade classification, OS survival, and tumor segmentation.

**Status: partial.** Only the clinical metadata is downloaded. TCIA distributes the images (142 GB) only as an Aspera Faspex package, and Aspera is blocked by our firewall. See "Getting the images" below.

- **Source:** TCIA collection [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/), **version 5** (2025-05-30). The clinical CSVs are plain HTTPS downloads from the collection page. The images are Faspex package id 1065, "UCSF-PDGM Version 5" (package UUID `e6a98a44-bc23-4ef1-8380-9a64c2150b8a`, released 2025-05-29, 12,030 files, top folder `UCSF-PDGM-v5/`).
- **DOI:** [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37)
- **License:** CC BY 4.0. You must follow the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/) and cite the dataset.
- **Citation:**
  - Data: Calabrese, E., Villanueva-Meyer, J., Rudie, J., Rauschecker, A., Baid, U., Bakas, S., Cha, S., Mongan, J., Hess, C. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) (Version 5) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.bdgf-8v37
  - Paper: Calabrese, E., et al. (2022). The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset. *Radiology: Artificial Intelligence*, 4(6), e220058. https://doi.org/10.1148/ryai.220058
- **Contents (downloaded):**
  - `UCSF-PDGM-metadata_v5.csv`: 501 rows, one per exam. Columns: sex, age at MRI, WHO CNS grade (2: 56, 3: 43, 4: 402), WHO 2021 diagnosis, MGMT status/index, 1p/19q, IDH (wildtype 398 / mutant 103), vital status + OS (days), extent of resection, prior biopsy, plus **BraTS 2021 ID and cohort** (262 BraTS21 segmentation-training cases, 36 validation).
  - `UCSF-PDGM-metadata_glossary.csv`: column definitions.
- **Contents of the image package (not downloaded):** one folder per exam (`UCSF-PDGM-NNNN_nifti/`), with all volumes skull-stripped and co-registered to 1 mm FLAIR space (preprocessed by the source; no DICOM is available). Each folder holds 24 files:
  - structural: `T1`, `T1c`, `T2`, `FLAIR`, each also as `*_bias` (bias-corrected)
  - diffusion: `DWI`, `DWI_bias`, `ADC`; DTI fits `DTI_eddy_{FA,MD,L1,L2,L3}`; raw 4D HARDI `DTI_eddy_noreg` (native space) + `DTI_eddy.eddy_rotated_bvecs`
  - other: `SWI`, `SWI_bias`, `ASL`
  - targets: `tumor_segmentation` (BraTS-style enhancing / necrotic / edema labels, radiologist-corrected), `brain_segmentation`, `brain_parenchyma_segmentation`

## Getting the images

The collection page offers only the Faspex link. We checked these HTTPS alternatives:
- **NBIA REST API:** UCSF-PDGM is not in `getCollectionValues` (it's not a DICOM collection).
- **IDC:** not in the IDC v3 collection list.
- **Faspex HTTP gateway:** the public link can be opened over HTTPS (OAuth `authorize_public_link`, then `/api/v5`), and the API lists files. But `http_gateway_url` is `null` in `/api/v5/configuration`. Every transfer spec, including `transfer_type=http_gateway`, points at FASP/SSH on `144.30.235.113:33001`, and our firewall refuses that port.

So the images need a machine with Aspera access, or a TCIA-provided HTTPS route. Aspera can select individual files, so the planned subset is: `T1`, `T1c`, `T2`, `FLAIR`, `DWI`, `ADC`, `tumor_segmentation`, `brain_segmentation`, `brain_parenchyma_segmentation` (the `*_bias` variants are still undecided). It would exclude DTI fits, raw 4D DTI + bvecs, SWI and ASL. Per-file sizes could not be measured: the API lists the package files as symlinks with no size. The whole package (142 GB) is under the 200 GB limit anyway, so taking all of it is also an option.

## Notes

- IDs: the image folders use 4 digits (`UCSF-PDGM-0004`), but `metadata_v5.csv` uses 3 (`UCSF-PDGM-004`). Normalize before joining.
- 501 exams from 495 patients. Six IDs are follow-up exams of other patients and were renamed in v3, e.g. `UCSF-PDGM-0315` → `UCSF-PDGM-0433_FU007d`. Split by patient.
- Overlap: 298 exams are in BraTS 2021 (`BraTS21 ID` column). Use this column to avoid train/test leakage when using both datasets.
- Version history: v2 fixed segmentation rounding errors and added BraTS IDs; v3 renamed the follow-up exams; v5 fixed headers in `DTI_eddy_noreg` and added rotated bvecs.

```sh
bash scripts/ucsf_pdgm/download.sh   # 60 KB metadata only; resumable
```
