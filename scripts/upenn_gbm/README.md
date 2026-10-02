# UPENN-GBM

630 de novo glioblastoma patients from the University of Pennsylvania (671 scans: 611 pre-operative baselines + 60 follow-ups), with T1, T1Gd, T2, FLAIR, DTI and DSC MRI. We use it for overall-survival, IDH1 and MGMT prediction and for tumor segmentation.

- **Source:** TCIA collection [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/). The supporting files are plain HTTPS links under `https://www.cancerimagingarchive.net/wp-content/uploads/`, fetched with `curl`. The NIfTI images and segmentations are only distributed as an Aspera Faspex package: id 604, "UPENN-GBM-NIfTI", UUID `3bc22992-1885-449c-a360-8eff82e5e515`, released 2023-11-09. We fetch it over FASP with `brain_datasets.tcia_faspex` (`src/brain_datasets/tcia_faspex.py`), using the public download link from the collection page (no account needed). The Faspex server has no HTTP gateway, so this needs outbound TCP/UDP 33001 to TCIA's transfer node. The collection page also links a histopathology package (Faspex 938, NDPI whole-slide images, 149 GB), which we don't fetch.
- **Version:** collection Version 2 (2022/10/24; files listed as updated 2023/11/20, clinical v2.1 2023/12/05). The supporting files are not versioned on the server, so `manifest.sha256` is the pin. The NIfTI release is Faspex package 604 (2023-11-09), checked against its `UPENN-GBM.sums`.
- **DOI:** [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49)
- **License:** CC BY 4.0. Use must follow the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/). No login or DUA.
- **Citation:** Bakas, S., Sako, C., Akbari, H., et al. (2021). Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM) (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.709X-DN49. Paper: Bakas, S., et al. (2022). The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, & radiomics. *Scientific Data*, 9, 453.

## Contents

Downloaded: 6,152 files, 25.3 GB (25,263,534,793 bytes) under `source/`.

Supporting files at the top of `source/` (9 files, 17 MB):

| File | What |
|---|---|
| `UPENN-GBM_clinical_info_v2.1.csv` | 671 rows, one per scan ID (`UPENN-GBM-NNNNN_11` baseline / `_21` follow-up): sex, age, `Survival_from_surgery_days_UPDATED`, `Survival_Status`, `Survival_Censor`, `IDH1` (546 wildtype / 19 mutated / 106 NOS), `MGMT` (121 methylated / 170 unmethylated / 32 indeterminate / 348 n/a), KPS (75), `GTR_over90percent`, `PsP_TP_score` (60 follow-ups) |
| `UPENN-GBM_data_availability.csv` | per-scan availability of every modality, segmentation and label |
| `UPENN-GBM_acquisition.csv` | scanner and sequence parameters per scan |
| `radiology_mapping.csv` | radiology ID to histopathology ID mapping |
| `radiomic_features_CaPTk.zip`, `UPENN-GBM_CaPTk_radiomic_features_list.csv`, `UPENN-GBM_CaPTk_fe_params.csv` | CaPTk radiomic features (automated and corrected segmentations × modality × sub-region) |
| `UPENN-GBM_DownloadManifest20221129.tcia`, `…-nbia-digest.xlsx` | NBIA manifest (3680 series) and per-series digest for the DICOM release |

From the NIfTI package, with the package tree kept (6,143 files, 25.2 GB):

- `UPENN-GBM.sums`: the package's md5 list (`md5 relpath`, 10,646 entries for the full package). `download.sh` checks every kept file against it.
- `UPENN-GBM/NIfTI-files/`: all images are co-registered to the SRI-24 atlas (240×240×155, 1 mm isotropic). The segmentations align with these images, not with the DICOMs.

| Folder | Scans | Files | Size |
|---|---|---|---|
| `images_structural/<ID>/<ID>_{T1,T1GD,T2,FLAIR}.nii.gz` (skull-stripped) | 671 | 2,692 | 5.9 GB |
| `images_structural_unstripped/<ID>/<ID>_{T1,T1GD,T2,FLAIR}_unstripped.nii.gz` | 671 | 2,692 | 19.3 GB |
| `automated_segm/<ID>_automated_approx_segm.nii.gz` | 611 (baselines) | 611 | 16 MB |
| `images_segm/<ID>_segm.nii.gz` (expert-corrected) | 147 (baselines) | 147 | 5 MB |

Targets: tumor sub-region masks in `automated_segm/` and `images_segm/`, with BraTS labels (1 necrotic core, 2 edema, 4 enhancing tumor). Survival, IDH1 and MGMT are in `UPENN-GBM_clinical_info_v2.1.csv`.

## Excluded

- The Version 1-only supporting files (`clinical_info_v1.0`, `manifest_20210923`, `captk_parameter_file_ispy1`).
- The DICOM release via NBIA (139.4 GB, 3680 series; about 45.9 GB structural, 93.5 GB DTI/DWI and DSC). It is reachable over HTTPS but has no tumor segmentations, so the NIfTI release is preferred.
- The NCI Imaging Data Commons mirror of the same DICOM. Its segmentations are third-party (BAMF AIMI annotations, [10.5281/zenodo.8345959](https://doi.org/10.5281/zenodo.8345959)), not UPenn's.
- `images_DTI/` (DTI fits AD/FA/RD/TR, 592 scans, 2,368 files) and `images_DSC/` (DSC perfusion series and derived ap-rCBV/PH/PSR maps, 534 scans, 2,136 files) from the NIfTI package. They are not structural inputs, and they make up about 44 GB of the 69 GB package (the API gives no per-file sizes).

## Notes

- **Corrected segmentations:** the NIfTI package has 147 expert-corrected masks, but `UPENN-GBM_data_availability.csv` marks 232 scans as "Corrected Tumor Segmentation: available". All 147 are among the 232.
- **`old/` subfolders:** two baselines (`UPENN-GBM-00260_11`, `UPENN-GBM-00509_11`) also have an `old/` subfolder in both structural folders, holding a different (earlier, by its name) version of their 4 images; the md5s differ from the top-level files. So each folder has 2,692 files: 671 × 4 + 8. Use the files at the top level of each scan folder.
- 630 patients have 671 scans (60 follow-ups, 41 of them with a linked baseline). Segmentations exist for baselines only.
- Label coverage: overall survival 452 of 671 scans, IDH1 565, MGMT 291.
- Overlap: BraTS 2021 includes 447 UPenn cases (preprocessed differently). Use the BraTS crosswalk (`scripts/brats2021/README.md`) to deduplicate.

## Usage

```sh
bash scripts/upenn_gbm/download.sh   # 25.3 GB; resumable (needs FASP, TCP/UDP 33001, to TCIA)
```
