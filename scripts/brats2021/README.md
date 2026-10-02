# BraTS 2021

Adult glioma pre-operative mpMRI (T1, T1Gd, T2, FLAIR) from the RSNA-ASNR-MICCAI BraTS 2021 challenge, skull-stripped and co-registered as distributed. We use it for tumor segmentation (Task 1) and MGMT promoter methylation classification (Task 2).

- **Source:** TCIA analysis result [RSNA-ASNR-MICCAI-BraTS-2021](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/). The metadata is 28 plain HTTPS files linked from the page, fetched with `curl`. The imaging is an Aspera Faspex package: id 636, "RSNA-ASNR-MICCAI-BraTS-2021", released 2023-11-11, 407,245 files, plus `RSNA-ASNR-MICCAI-BraTS-2021.sums` (checksum list, 2024-03-25). `list_package.py` lists the package over HTTPS (public-link OAuth + Faspex v5 API).
- **Version:** Version 1 (updated 2023/08/25). The metadata files are not versioned on the server, so `manifest.sha256` is the pin.
- **DOI:** [10.7937/jc8x-9874](https://doi.org/10.7937/jc8x-9874)
- **License:** CC BY 4.0 for the challenge data and the crosswalk. The *original* DICOMs of TCGA-GBM/LGG, CPTAC-GBM, IvyGAP and ACRIN-FMISO-Brain (which the `.tcia` manifests point to) fall under the NIH Controlled Data Access Policy; we don't fetch them.
- **Citation:** Baid, U., et al. (2023). RSNA-ASNR-MICCAI-BraTS-2021 Dataset. The Cancer Imaging Archive. https://doi.org/10.7937/jc8x-9874. Also: Baid, U., et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification. arXiv:2107.02314; Menze et al. 2015 (*IEEE TMI*); Bakas et al. 2017 (*Scientific Data*); and the source collections (full list on the TCIA page). TCGA acknowledgement required.

## Status

**Partial: metadata only.** The imaging (142 GB) is only distributed as the Aspera Faspex package. It is not in NBIA, and the Faspex server has no HTTP gateway (`http_gateway_url: null`). The devcontainer firewall now allows FASP to TCIA's transfer node (TCP/UDP 33001 to `144.30.235.113`), so once the container is rebuilt, `download.sh` can be extended to fetch the Task 1 NIfTI with `ascli`.

## Contents

`source/metadata/` (downloaded, 28 files, 2.1 MB):

- `BraTS2021_MappingToTCIA.xlsx`: the **ID crosswalk and label table**, 1,479 rows. Columns: source collection, site ID, TCIA PatientID, BraTS2021 ID, Task 1 cohort, Task 2 cohort, **MGMT value**.
  - Task 1: 1,251 Training, 219 Validation, plus 9 Task-2-only rows.
  - Task 2: 585 Training and 87 Validation (the Kaggle split). All 672 have an MGMT label (Training 307 methylated / 278 unmethylated; Validation 43 / 44), so TCIA ships the Task 2 labels **including validation**. Another 23 non-Task-2 cases also have an MGMT value (695 in total).
- `NotPreviouslyInTCIA.csv`: the 1,967 package files with no TCIA DICOM equivalent ("new" institutional cases).
- `BraTS2021_<source>_<Seg|Class>-Task-<Training|Validation>.tcia` (23 files), plus `RSNA-ASNR-MICCAI-BraTS-2021_UPENN-GBM_manifest.tcia` and its `...nbia-digest-1.xlsx`: NBIA manifests of the original DICOM series behind each split. The TCIA page calls the series-to-volume link a "best effort" reconstruction.
- `GC_manifest_RSNA-ASNR-MICCAI-BRATS-2021_sources.csv`: CRDC DRS IDs (`dg.4DFC/...`) to SeriesInstanceUID for the original series.

Faspex package 636 (not yet downloaded; listed with `list_package.py` on 2026-10-01):

| Folder | Cases | Per case | Plan |
|---|---|---|---|
| `BraTS2021_TrainingSet/<source>/BraTS2021_NNNNN/` | 1,251 | `_t1`, `_t1ce`, `_t2`, `_flair`, **`_seg`** `.nii.gz` (240×240×155) | keep (~12 GB with validation) |
| `BraTS2021_ValidationSet/<source>/...` | 219 | the 4 modalities, **no `_seg`** | keep |
| `BraTS2021_TrainingSet_dcm/<source>/NNNNN/{FLAIR,T1w,T1wCE,T2w}/` | 585 | Task 2 DICOMs (Kaggle format; converted NIfTI→DCM, not strictly standard) | exclude (~128 GB with validation) |
| `BraTS2021_ValidationSet_dcm/...` | 87 | the same | exclude |

Source folders: UPENN-GBM, UCSF-PDGM, TCGA-GBM, TCGA-LGG, CPTAC-GBM, IvyGAP, ACRIN-FMISO-Brain, new-not-previously-in-TCIA.

## Excluded

- The Task 2 `_dcm` folders (~128 GB): re-encodings of the same kind of scans, only needed to reproduce the Kaggle Task 2 setup. MGMT can be predicted from the Task 1 NIfTI with the crosswalk labels.
- The original DICOMs behind the `.tcia` manifests (controlled access, or duplicates of UPENN-GBM).
- The challenge test set (sequestered on Synapse, syn25829067).
- The UCSF-PDGM v1 excerpt for BraTS (Faspex package 679, 3 GB): covered by `ucsf_pdgm`.
- For now, all images (see Status).

## Notes

- **Case counts:** `datasets.md` and the TCIA page say 1,480. The package has 1,251 + 219 = 1,470 Task 1 cases, plus 9 Task-2-only cases (8 train, 1 val), so **1,479**, matching the crosswalk.
- **Validation segmentations are not included.** We checked all 53 "new" validation cases and one case per source folder. Only the 1,251 training cases have labels.
- **Subject overlap** (from the crosswalk): UPENN-GBM 447 (TCIA IDs like `UPENN-GBM-00011_11`), UCSF-PDGM 299 (PatientID holds only the number, e.g. `57`), TCGA-GBM 135, TCGA-LGG 108, CPTAC-GBM 39, IvyGAP 34, ACRIN-FMISO-Brain 4. 413 are new institutional cases with no TCIA equivalent: `UPENN-GBM_Additional` 115, `UCSF-PDGM_Additional` 139, and anonymized "Collection N" sites 159. The `_Additional` rows are probably patients *not* in the UPENN-GBM or UCSF-PDGM collections, but the crosswalk can't confirm it. Use the crosswalk to deduplicate against `upenn_gbm` and `ucsf_pdgm`.
- The 9 Task-2-only cases use bare IDs (`00169`) where the others use `BraTS2021_NNNNN`.

## Usage

```sh
bash scripts/brats2021/download.sh                                # 2.1 MB (metadata only); resumable
uv run --with requests python scripts/brats2021/list_package.py   # list the Faspex package (HTTPS)
```
