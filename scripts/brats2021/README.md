# BraTS 2021

Adult glioma pre-operative mpMRI (T1, T1Gd, T2, FLAIR) from the RSNA-ASNR-MICCAI BraTS 2021 challenge, skull-stripped and co-registered as distributed. We use it for tumor segmentation (Task 1) and MGMT promoter methylation classification (Task 2).

## Status: incomplete, imaging not downloaded

Only the metadata (2.1 MB) is downloaded. The imaging (142 GB) is published only as an IBM Aspera **Faspex** package. Faspex can list the package over HTTPS, but downloads go over FASP to `144.30.235.113:33001`, and the devcontainer firewall blocks that. The server has no Aspera HTTP Gateway (`/api/v5/configuration` returns `"http_gateway_url": null`). The imaging is not in NBIA either, so the NBIA REST API and Data Retriever can't fetch it. To finish, someone must run an Aspera transfer of package 636 from a host where Aspera is allowed (Aspera Connect from the TCIA page, or `ascli faspex5 packages receive` with the public link). Then extend `download.sh`.

## Source

- **Source:** TCIA analysis result [RSNA-ASNR-MICCAI-BraTS-2021](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/), **Version 1** (updated 2023/08/25). The imaging is in Faspex package 636, "RSNA-ASNR-MICCAI-BraTS-2021", released 2023-11-11, 407,245 files, plus `RSNA-ASNR-MICCAI-BraTS-2021.sums` (50 MB checksum list, dated 2024-03-25). The UCSF-PDGM v1 excerpt (298 to 299 cases, 3 GB) is a separate package, 679. The page notes (April 2026) that updates to the source-data manifests don't affect the challenge downloads.
- **Downloaded:** `download.sh` fetches the 28 plain files linked from the page into `source/metadata/` with `curl`. `list_package.py` lists the Faspex package over HTTPS (public-link OAuth plus the v5 API) and was used for the counts below.
- **DOI:** [10.7937/jc8x-9874](https://doi.org/10.7937/jc8x-9874)
- **License:** CC BY 4.0 for the challenge data and the crosswalk. The *original* DICOMs of TCGA-GBM/LGG, CPTAC-GBM, IvyGAP and ACRIN-FMISO-Brain, which the `.tcia` manifests point to, fall under the NIH Controlled Data Access Policy. We don't fetch them.
- **Citation:** Baid, U., et al. (2023). *RSNA-ASNR-MICCAI-BraTS-2021 Dataset*. The Cancer Imaging Archive. https://doi.org/10.7937/jc8x-9874. Also cite Baid et al. (2021), *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification*, arXiv:2107.02314, plus Menze et al. 2015 (TMI), Bakas et al. 2017 (Sci Data), and the source collections. The full list is on the TCIA page. TCGA acknowledgement required.

## Contents

`source/metadata/` (downloaded):
- `BraTS2021_MappingToTCIA.xlsx`: the **ID crosswalk and label table**, 1,479 rows. Columns: source collection, site ID, TCIA PatientID, BraTS2021 ID, Task 1 cohort, Task 2 cohort, **MGMT value**.
  - Task 1: 1,251 Training, 219 Validation, and 9 rows that are Task-2-only.
  - Task 2: 585 Training and 87 Validation (the Kaggle split). All 672 have an MGMT label (Training 307 methylated / 278 unmethylated; Validation 43 / 44). So TCIA ships the **Task 2 labels, including validation**. Another 23 non-Task-2 cases also carry an MGMT value (695 labels in total).
- `NotPreviouslyInTCIA.csv`: the 1,967 package files with no TCIA DICOM equivalent ("new" institutional cases).
- `BraTS2021_<source>_<Seg|Class>-Task-<Training|Validation>.tcia` (23 files) and `RSNA-ASNR-MICCAI-BraTS-2021_UPENN-GBM_manifest.tcia` with its `...nbia-digest-1.xlsx`: NBIA manifests of the original DICOM series behind each split. The TCIA page calls the series-to-volume link a "best effort" reconstruction.
- `GC_manifest_RSNA-ASNR-MICCAI-BRATS-2021_sources.csv`: CRDC DRS IDs (`dg.4DFC/...`) to SeriesInstanceUID for the original series.

Faspex package 636 (not downloaded; from `list_package.py`, 2026-10-01):

| Folder | Cases | Per case |
|---|---|---|
| `BraTS2021_TrainingSet/<source>/BraTS2021_NNNNN/` | 1,251 | `_t1`, `_t1ce`, `_t2`, `_flair`, **`_seg`** `.nii.gz` (240×240×155) |
| `BraTS2021_ValidationSet/<source>/...` | 219 | the 4 modalities, **no `_seg`** |
| `BraTS2021_TrainingSet_dcm/<source>/NNNNN/{FLAIR,T1w,T1wCE,T2w}/` | 585 | Task 2 DICOMs (Kaggle format; converted NIfTI→DCM, not strictly standard) |
| `BraTS2021_ValidationSet_dcm/...` | 87 | the same |

Source folders: UPENN-GBM, UCSF-PDGM, TCGA-GBM, TCGA-LGG, CPTAC-GBM, IvyGAP, ACRIN-FMISO-Brain, new-not-previously-in-TCIA. The page gives the sizes: Task 1 NIfTI is about 12 GB and Task 2 NIfTI+DCM about 128 GB.

## Excluded

- All imaging (142 GB), because Aspera is blocked (see Status). When it becomes possible, the Task 1 NIfTI (~12 GB) is the priority. The Task 2 `_dcm` folders (~128 GB) are re-encodings of the same kind of scans at original orientation and resolution. They are only needed to reproduce the Kaggle Task 2 setup, since MGMT can also be predicted from the Task 1 NIfTI.
- The original DICOMs behind the `.tcia` manifests (controlled access, or duplicates of UPENN-GBM).
- The challenge test set (sequestered on Synapse, syn25829067).

## Notes

- **Case counts:** `datasets.md` and the TCIA page say 1,480 subjects. The package has 1,251 + 219 = **1,470** Task 1 cases, plus 9 cases that appear only in Task 2 (8 train, 1 val), so **1,479** in total. That matches the package description and the crosswalk.
- **Validation segmentations are not included.** We checked all 53 "new" validation cases via `NotPreviouslyInTCIA.csv` and one case per source folder via the listing. Only the 1,251 training cases have labels.
- **Subject overlap** (from the crosswalk, by Task 1 + Task-2-only rows): UPENN-GBM 447 (TCIA IDs like `UPENN-GBM-00011_11`) and UCSF-PDGM 299 (the PatientID column holds only the number, e.g. `57`). TCGA-GBM 135 and TCGA-LGG 108 (the BraTS-TCGA-GBM/LGG cases), CPTAC-GBM 39, IvyGAP 34 and ACRIN-FMISO-Brain 4 are also in TCIA. 413 cases are new institutional data with no TCIA equivalent: `UPENN-GBM_Additional` 115, `UCSF-PDGM_Additional` 139, and anonymized "Collection N" sites 159. The `_Additional` rows are probably further patients from those institutions who are *not* in the UPENN-GBM or UCSF-PDGM collections, but that can't be checked from the crosswalk. Use the crosswalk to deduplicate against `upenn_gbm` and `ucsf_pdgm`.
- In the crosswalk, the 9 Task-2-only cases use bare IDs (`00169`) where the other cases use `BraTS2021_NNNNN`.

```sh
bash scripts/brats2021/download.sh                         # 2.1 MB metadata only; resumable
uv run --with requests python scripts/brats2021/list_package.py   # list the Faspex package (HTTPS)
```
