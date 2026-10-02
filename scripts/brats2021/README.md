# BraTS 2021

Adult glioma pre-operative mpMRI (T1, T1Gd, T2, FLAIR) from the RSNA-ASNR-MICCAI BraTS 2021 challenge, skull-stripped and co-registered as distributed. We use it for tumor segmentation (Task 1) and MGMT promoter methylation classification (Task 2).

- **Homepage:** <https://www.med.upenn.edu/cbica/brats2021/>
- **Source:** TCIA analysis result [RSNA-ASNR-MICCAI-BraTS-2021](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/). The metadata is 28 plain HTTPS files linked from the page, fetched with `curl`. The imaging is an Aspera Faspex package: id 636, "RSNA-ASNR-MICCAI-BraTS-2021", released 2023-11-11, 407,245 files, plus `RSNA-ASNR-MICCAI-BraTS-2021.sums` (checksum list, 2024-03-25). The Task 1 NIfTI and the `.sums` file are fetched over FASP with `ascp` (public-link OAuth + Faspex v5 transfer spec; needs TCP/UDP 33001 to TCIA's transfer node `144.30.235.113`), using the page's "Challenge data both tasks" public link. The server has no HTTP gateway, and `ascli faspex5 packages receive` fails against it.
- **Version:** Version 1 (updated 2023/08/25). The metadata files are not versioned on the server, so `manifest.sha256` is the pin.
- **DOI:** [10.7937/jc8x-9874](https://doi.org/10.7937/jc8x-9874)
- **License:** CC BY 4.0 for the challenge data and the crosswalk. The *original* DICOMs of TCGA-GBM/LGG, CPTAC-GBM, IvyGAP and ACRIN-FMISO-Brain (which the `.tcia` manifests point to) fall under the NIH Controlled Data Access Policy; we don't fetch them.
- **Citation:** Baid, U., et al. (2023). RSNA-ASNR-MICCAI-BraTS-2021 Dataset. The Cancer Imaging Archive. https://doi.org/10.7937/jc8x-9874. Also: Baid, U., et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification. arXiv:2107.02314; Menze et al. 2015 (*IEEE TMI*); Bakas et al. 2017 (*Scientific Data*); and the source collections (full list on the TCIA page). TCGA acknowledgement required.
- **Code:** [`scripts/brats2021/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/brats2021) re-downloads `source/` and rebuilds `tables/`.

## Samples

One sample per BraTS case: 1,479 cases of 1,477 patients (two UCSF-PDGM follow-up cases belong to patients who also have a baseline case; they are sessions of one participant). The official Task 1 validation set has no masks, so all cases are split 60/20/20 by patient, stratified by has mask × MGMT label, giving a held-out test set with masks. `official_split` keeps the Task 1 cohort and `mgmt_cohort` the Task 2 cohort. Complete = the 4 images and the tumor mask. The source has no age or sex.

| split | participants | samples | complete | age | female | sites | mgmt |
|---|---|---|---|---|---|---|---|
| train | 885 | 886 | 749 | n/a | n/a | 25 | methylated 218 / unmethylated 198 |
| val | 294 | 295 | 249 | n/a | n/a | 23 | methylated 73 / unmethylated 66 |
| test | 298 | 298 | 251 | n/a | n/a | 23 | methylated 73 / unmethylated 67 |
| total | 1477 | 1479 | 1249 | n/a | n/a | 26 | methylated 364 / unmethylated 331 |

## Contents

`source/`: 7,160 files, 15.8 GB.

`source/metadata/` (28 files, 2.1 MB), from the TCIA page:

- `BraTS2021_MappingToTCIA.xlsx`: the **ID crosswalk and label table**, 1,479 rows. Columns: source collection, site ID, TCIA PatientID, BraTS2021 ID, Task 1 cohort, Task 2 cohort, **MGMT value**.
  - Task 1: 1,251 Training, 219 Validation, plus 9 Task-2-only rows.
  - Task 2: 585 Training and 87 Validation (the Kaggle split). All 672 have an MGMT label (Training 307 methylated / 278 unmethylated; Validation 43 / 44), so TCIA ships the Task 2 labels **including validation**. Another 23 non-Task-2 cases also have an MGMT value (695 in total).
- `NotPreviouslyInTCIA.csv`: the 1,967 package files with no TCIA DICOM equivalent ("new" institutional cases).
- `BraTS2021_<source>_<Seg|Class>-Task-<Training|Validation>.tcia` (23 files), plus `RSNA-ASNR-MICCAI-BraTS-2021_UPENN-GBM_manifest.tcia` and its `...nbia-digest-1.xlsx`: NBIA manifests of the original DICOM series behind each split. The TCIA page calls the series-to-volume link a "best effort" reconstruction.
- `GC_manifest_RSNA-ASNR-MICCAI-BRATS-2021_sources.csv`: CRDC DRS IDs (`dg.4DFC/...`) to SeriesInstanceUID for the original series.

`source/RSNA-ASNR-MICCAI-BraTS-2021.sums` (50 MB): the package's md5 list (`md5 relpath`, 407,245 lines, paths relative to `source/`). Every downloaded NIfTI was checked against it.

`source/RSNA-ASNR-MICCAI-BraTS-2021/` (7,131 `.nii.gz`, 15.8 GB), the Task 1 NIfTI from Faspex package 636, in the package's tree:

| Folder | Cases | Files | Size | Per case |
|---|---|---|---|---|
| `BraTS2021_TrainingSet/<source>/BraTS2021_NNNNN/` | 1,251 | 6,255 | 13.4 GB | `_t1`, `_t1ce`, `_t2`, `_flair`, **`_seg`** `.nii.gz` (240×240×155) |
| `BraTS2021_ValidationSet/<source>/BraTS2021_NNNNN/` | 219 | 876 | 2.4 GB | the 4 modalities, **no `_seg`** |

Cases per source folder:

| Source folder | Training | Validation |
|---|---|---|
| UPENN-GBM | 403 | 44 |
| UCSF-PDGM | 263 | 36 |
| new-not-previously-in-TCIA | 351 | 53 |
| TCGA-GBM | 102 | 33 |
| TCGA-LGG | 65 | 43 |
| CPTAC-GBM | 33 | 6 |
| IvyGAP | 30 | 4 |
| ACRIN-FMISO-Brain | 4 | 0 |

Targets: the `_seg` masks (training only; labels 1 = necrotic core, 2 = edema, 4 = enhancing tumor) for Task 1, and the MGMT column of `BraTS2021_MappingToTCIA.xlsx` for Task 2.

`tables/` (derived from `source/`):

- `images.tsv`: T1w, T1c (`t1ce`), T2w, FLAIR and the tumor mask (desc `tumor`) per case; session_id is the BraTS case ID.
- `samples.tsv` + `samples.json`: MGMT (target), Task 2 cohort, site (BraTS institution code), source collection, TCIA PatientID (to link cases to `upenn_gbm` / `ucsf_pdgm`) and study date (de-identified, shifted by TCIA).
- `splits.tsv`: split, official Task 1 cohort, rank and complete per patient.

## Excluded

- The Task 2 `_dcm` folders, `BraTS2021_TrainingSet_dcm/` (585 cases) and `BraTS2021_ValidationSet_dcm/` (87 cases), ~127 GB, 400,114 files: the Task 2 scans converted NIfTI→DICOM (Kaggle format, not strictly standard DICOM), including the 9 Task-2-only cases. Only needed to reproduce the Kaggle Task 2 setup. MGMT can be predicted from the Task 1 NIfTI with the crosswalk labels.
- The original DICOMs behind the `.tcia` manifests (controlled access, or duplicates of UPENN-GBM).
- The challenge test set (sequestered on Synapse, syn25829067).
- The UCSF-PDGM v1 excerpt for BraTS (Faspex package 679, 3 GB): covered by `ucsf_pdgm`.

## Notes

- **Case counts:** the TCIA page says 1,480. The package has 1,251 + 219 = 1,470 Task 1 cases, plus 9 Task-2-only cases (8 train, 1 val), so **1,479**, matching the crosswalk.
- **Validation segmentations are not included.** We checked all 53 "new" validation cases and one case per source folder. Only the 1,251 training cases have labels.
- **Subject overlap** (from the crosswalk): UPENN-GBM 447 (TCIA IDs like `UPENN-GBM-00011_11`), UCSF-PDGM 299 (PatientID holds only the number, e.g. `57`), TCGA-GBM 135, TCGA-LGG 108, CPTAC-GBM 39, IvyGAP 34, ACRIN-FMISO-Brain 4. 413 are new institutional cases with no TCIA equivalent: `UPENN-GBM_Additional` 115, `UCSF-PDGM_Additional` 139, and anonymized "Collection N" sites 159. The `_Additional` rows are probably patients *not* in the UPENN-GBM or UCSF-PDGM collections, but the crosswalk can't confirm it. Benchmarks are evaluated within each dataset, so this overlap only correlates scores across datasets.
- **Same-patient cases:** the crosswalk lists two UCSF-PDGM follow-up exams under their old PatientIDs (138, 315), so it doesn't show that they belong to patients 429 and 433, who also have baseline cases. UCSF-PDGM v5 renamed them (`UCSF-PDGM-0429_FU003d`, `UCSF-PDGM-0433_FU007d`). The tables group them by patient. The anonymized `_Additional` and "Collection N" cases can't be checked for this.
- 8 of the 9 Task-2-only cases use bare IDs (`00169`) in the crosswalk where the others use `BraTS2021_NNNNN`; the tables normalize them.
- The crosswalk's study dates mix Excel dates and m/d/yyyy text; the tables write YYYY-MM-DD.
