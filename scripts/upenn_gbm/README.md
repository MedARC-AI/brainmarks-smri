# UPENN-GBM

630 de novo glioblastoma patients from the University of Pennsylvania (671 scans: 611 pre-operative baselines + 60 follow-ups), with T1, T1Gd, T2, FLAIR, DTI and DSC MRI. We use it for overall-survival, IDH1 and MGMT prediction and for tumor segmentation.

**Status: partial.** Only the clinical, molecular and supporting tables are downloaded. The imaging is set aside because the NIfTI release, which holds the segmentations, is Aspera-only (see *Imaging: set aside* below).

- **Source:** TCIA collection [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/), **Version 2** (2022/10/24; the page lists the files as updated 2023/11/20, and clinical v2.1 as updated 2023/12/05). The supporting files are plain HTTPS links under `https://www.cancerimagingarchive.net/wp-content/uploads/`, fetched with `curl`. These files are not versioned on the server, so `manifest.sha256` is the pin.
- **DOI:** [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49)
- **License:** CC BY 4.0. Attribution required, and use must follow the [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/). No login or DUA is needed.
- **Citation (data):** Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G., Rudie, J. D., Flores Santamaria, N., Fathi Kazerooni, A., Pati, S., Rathore, S., Mamourian, E., Ha, S. M., Parker, W., Doshi, J., Baid, U., Bergman, M., Binder, Z. A., Verma, R., … Davatzikos, C. (2021). Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM) (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.709X-DN49
- **Citation (paper):** Bakas, S., et al. (2022). The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, & radiomics. *Scientific Data*, 9, 453. https://doi.org/10.1038/s41597-022-01560-7

## Contents (downloaded, 17 MB)

| File | What |
|---|---|
| `UPENN-GBM_clinical_info_v2.1.csv` | 671 rows (one per scan ID `UPENN-GBM-NNNNN_11` baseline / `_21` follow-up): sex, age, `Survival_from_surgery_days_UPDATED`, `Survival_Status`, `Survival_Censor`, `IDH1` (546 wildtype / 19 mutated / 106 NOS), `MGMT` (121 methylated / 170 unmethylated / 32 indeterminate / 348 n/a), KPS (75), `GTR_over90percent`, `PsP_TP_score` (60 follow-ups) |
| `UPENN-GBM_data_availability.csv` | Per-scan availability of every modality, segmentation and label |
| `UPENN-GBM_acquisition.csv` | Scanner and sequence parameters per scan |
| `radiology_mapping.csv` | Radiology ID to histopathology ID mapping |
| `radiomic_features_CaPTk.zip`, `UPENN-GBM_CaPTk_radiomic_features_list.csv`, `UPENN-GBM_CaPTk_fe_params.csv` | CaPTk radiomic features (automated and corrected segmentations × modality × sub-region) |
| `UPENN-GBM_DownloadManifest20221129.tcia`, `…-nbia-digest.xlsx` | NBIA manifest (3680 series UIDs) and per-series digest for the DICOM release |

Left out: the Version 1-only files (`clinical_info_v1.0`, `manifest_20210923`, `captk_parameter_file_ispy1`).

## Imaging: set aside

TCIA distributes the imaging in three forms. None of them meets our requirements (official segmentations, plain HTTPS):

1. **NIfTI release (69 GB, official; the one we want).** It is the only source of the UPenn structural NIfTIs and the tumor segmentations. All images are skull-stripped and co-registered to the SRI atlas, and the segmentations align with them but not with the DICOMs. It is published only as an IBM Aspera Faspex public package: package 604 "UPENN-GBM-NIfTI", released 2023-11-09, uuid `3bc22992-1885-449c-a360-8eff82e5e515`, with a top-level `UPENN-GBM.sums`. The Faspex API returns only a FASP transfer spec (port 33001) and `http_gateway_url` is null, so there is no HTTPS download. Listed through the public-link API:

   | Folder | Count | Keep? |
   |---|---|---|
   | `images_structural/<ID>/` `{T1,T1GD,T2,FLAIR}.nii.gz` | 671 scans | yes |
   | `images_structural_unstripped/<ID>/` `*_unstripped.nii.gz` | 671 scans | yes (less processed) |
   | `automated_segm/<ID>_automated_approx_segm.nii.gz` | 611 (baselines only) | yes |
   | `images_segm/<ID>_segm.nii.gz` (expert-corrected) | 147 (baselines only) | yes |
   | `images_DTI/images_DTI/<ID>/` `DTI_{AD,FA,RD,TR}` | 592 | no: DTI model fits only |
   | `images_DSC/<ID>/` `DSC`, `DSC_{ap-rCBV,PH,PSR}` | 534 | no: DSC and its derivatives |

   The API lists the files as symlinks with no sizes, so the per-folder sizes are unknown (69 GB in total).
2. **DICOM via NBIA (139.4 GB, 3680 series).** This is reachable over HTTPS through the public NBIA REST API (no login). By series description it splits into about 2610 structural series (`…: Processed_CaPTk`, **45.9 GB**) and about 1070 DTI/DWI and DSC perfusion series (93.5 GB). It contains **no tumor segmentations**.
3. **NCI Imaging Data Commons** (`idc-index` v24, `s3://idc-open-data`, HTTPS). It mirrors the same DICOM and adds third-party DICOM-SEG from `bamf_aimi_annotations` (2164 AI plus 220 radiologist-corrected segmentations for 541 patients, [10.5281/zenodo.8345959](https://doi.org/10.5281/zenodo.8345959)). These are not the UPenn segmentations.

**User decision needed.** The options are:
- (a) Fetch the Faspex package with Aspera (e.g. `ascli faspex5`, using the public link from the collection page) from a machine that allows FASP, keeping only the four "yes" folders above. Place them under `datasets/upenn_gbm/source/NIfTI-files/` and re-run `write_manifest`.
- (b) Ask the TCIA helpdesk for an HTTPS route to the NIfTI package.
- (c) Accept the DICOM structural subset (about 46 GB, NBIA API). This gives survival/IDH/MGMT inputs but no segmentation targets.

```sh
bash scripts/upenn_gbm/download.sh   # 17 MB (supporting tables only); resumable
```

## Notes

- **Corrected segmentations:** the NIfTI package has **147** expert-corrected masks (`images_segm/`), but `UPENN-GBM_data_availability.csv` marks **232** scans as "Corrected Tumor Segmentation: available". All 147 files are among those 232. The corrected-segmentation radiomics in `radiomic_features_CaPTk.zip` (`*_segm_*`) may cover the larger set.
- `datasets.md` row checks out: 630 patients, 3301 studies, 3680 series in DICOM, CC BY 4.0. The 630 patients have 671 NIfTI scans (60 follow-ups, 41 of them with a linked baseline).
- Overall survival is available for 452 of 671 scans, IDH1 for 565, and MGMT for 291.
- **Overlap:** BraTS 2021 includes UPenn cases (preprocessed differently). Record IDs when building benchmarks.
