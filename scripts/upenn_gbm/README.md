# UPENN-GBM

630 de novo glioblastoma patients from the University of Pennsylvania (671 scans: 611 pre-operative baselines + 60 follow-ups), with T1, T1Gd, T2, FLAIR, DTI and DSC MRI. We use it for overall-survival, IDH1 and MGMT prediction and for tumor segmentation.

- **Source:** TCIA collection [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/). The supporting files are plain HTTPS links under `https://www.cancerimagingarchive.net/wp-content/uploads/`, fetched with `curl`. The NIfTI images and segmentations are an Aspera Faspex package: id 604, "UPENN-GBM-NIfTI", UUID `3bc22992-1885-449c-a360-8eff82e5e515`, released 2023-11-09, with a top-level `UPENN-GBM.sums`.
- **Version:** collection Version 2 (2022/10/24; files listed as updated 2023/11/20, clinical v2.1 2023/12/05). The supporting files are not versioned on the server, so `manifest.sha256` is the pin.
- **DOI:** [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49)
- **License:** CC BY 4.0. Use must follow the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/). No login or DUA.
- **Citation:** Bakas, S., Sako, C., Akbari, H., et al. (2021). Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM) (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.709X-DN49. Paper: Bakas, S., et al. (2022). The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, & radiomics. *Scientific Data*, 9, 453.

## Status

**Partial: supporting tables only.** The official NIfTI release (69 GB), which is the only source of the UPenn tumor segmentations, is only distributed as the Aspera Faspex package. The Faspex server has no HTTP gateway. The devcontainer firewall now allows FASP to TCIA's transfer node (TCP/UDP 33001 to `144.30.235.113`), so once the container is rebuilt, `download.sh` can be extended to fetch the "keep" folders below with `ascli`.

## Contents

Downloaded (9 files, 17 MB):

| File | What |
|---|---|
| `UPENN-GBM_clinical_info_v2.1.csv` | 671 rows, one per scan ID (`UPENN-GBM-NNNNN_11` baseline / `_21` follow-up): sex, age, `Survival_from_surgery_days_UPDATED`, `Survival_Status`, `Survival_Censor`, `IDH1` (546 wildtype / 19 mutated / 106 NOS), `MGMT` (121 methylated / 170 unmethylated / 32 indeterminate / 348 n/a), KPS (75), `GTR_over90percent`, `PsP_TP_score` (60 follow-ups) |
| `UPENN-GBM_data_availability.csv` | per-scan availability of every modality, segmentation and label |
| `UPENN-GBM_acquisition.csv` | scanner and sequence parameters per scan |
| `radiology_mapping.csv` | radiology ID to histopathology ID mapping |
| `radiomic_features_CaPTk.zip`, `UPENN-GBM_CaPTk_radiomic_features_list.csv`, `UPENN-GBM_CaPTk_fe_params.csv` | CaPTk radiomic features (automated and corrected segmentations × modality × sub-region) |
| `UPENN-GBM_DownloadManifest20221129.tcia`, `…-nbia-digest.xlsx` | NBIA manifest (3680 series) and per-series digest for the DICOM release |

NIfTI package (not yet downloaded). All images are skull-stripped and co-registered to the SRI atlas; the segmentations align with them, not with the DICOMs. The API lists the files without sizes:

| Folder | Count | Plan |
|---|---|---|
| `images_structural/<ID>/{T1,T1GD,T2,FLAIR}.nii.gz` | 671 scans | keep |
| `images_structural_unstripped/<ID>/*_unstripped.nii.gz` | 671 scans | keep (less processed) |
| `automated_segm/<ID>_automated_approx_segm.nii.gz` | 611 (baselines) | keep |
| `images_segm/<ID>_segm.nii.gz` (expert-corrected) | 147 (baselines) | keep |
| `images_DTI/images_DTI/<ID>/DTI_{AD,FA,RD,TR}` | 592 | exclude (DTI fits) |
| `images_DSC/<ID>/DSC`, `DSC_{ap-rCBV,PH,PSR}` | 534 | exclude (perfusion) |

## Excluded

- The Version 1-only supporting files (`clinical_info_v1.0`, `manifest_20210923`, `captk_parameter_file_ispy1`).
- The DICOM release via NBIA (139.4 GB, 3680 series; about 45.9 GB structural, 93.5 GB DTI/DWI and DSC). It is reachable over HTTPS but has no tumor segmentations, so the NIfTI release is preferred.
- The NCI Imaging Data Commons mirror of the same DICOM. Its segmentations are third-party (BAMF AIMI annotations, [10.5281/zenodo.8345959](https://doi.org/10.5281/zenodo.8345959)), not UPenn's.
- For now, all images (see Status).

## Notes

- **Corrected segmentations:** the NIfTI package has 147 expert-corrected masks, but `UPENN-GBM_data_availability.csv` marks 232 scans as "Corrected Tumor Segmentation: available". All 147 are among the 232.
- 630 patients have 671 scans (60 follow-ups, 41 of them with a linked baseline). Segmentations exist for baselines only.
- Label coverage: overall survival 452 of 671 scans, IDH1 565, MGMT 291.
- Overlap: BraTS 2021 includes 447 UPenn cases (preprocessed differently). Use the BraTS crosswalk (`scripts/brats2021/README.md`) to deduplicate.

## Usage

```sh
bash scripts/upenn_gbm/download.sh   # 17 MB (supporting tables only); resumable
```
