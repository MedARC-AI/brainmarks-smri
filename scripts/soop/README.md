# SOOP

Stroke Outcome Optimization Project: clinical MRI of 1715 patients admitted to a South Carolina comprehensive stroke center (Prisma Health-Upstate). We use it for stroke lesion segmentation (DWI TRACE + ADC), discharge mRS prediction and NIHSS regression.

- **Source:** OpenNeuro [ds004889](https://openneuro.org/datasets/ds004889/versions/1.1.2). **Not** downloaded with `openneuro-py`: for this dataset OpenNeuro hands out S3 URLs presigned for GET only (15,153 of 16,838 files), and `openneuro-py` 2026.9.1 sends a HEAD request first, which S3 rejects with 403. Instead, `snapshot_files.py` lists the snapshot through the OpenNeuro GraphQL API, and curl fetches each file's pinned S3 object version (`?versionId=...`, public, unsigned). Every file is then checked against the snapshot's own checksums (the SHA256 in the git-annex key, or the git blob SHA1).
- **Version:** snapshot 1.1.2 (2024-05-08), the latest as of 2026-10-01.
- **DOI:** [10.18112/openneuro.ds004889.v1.1.2](https://doi.org/10.18112/openneuro.ds004889.v1.1.2)
- **License:** CC0
- **Citation:** Absher, J., Goncher, S., Newman-Norlund, R., Perkins, N., Yourganov, G., Vargas, J., & Rorden, C. (2024). The stroke outcome optimization project: Acute ischemic strokes from a comprehensive stroke center. *Scientific Data*, 11, 839. [doi:10.1038/s41597-024-03667-5](https://doi.org/10.1038/s41597-024-03667-5). (`dataset_description.json` has no `HowToAcknowledge`.)

## Contents

The complete snapshot (16,838 files, 72 GB):

- 1715 subjects, each with `anat/` T1w + FLAIR and `dwi/` `rec-TRACE_dwi` (b1000 trace) + `rec-ADC_dwi`, all with JSON sidecars. The top-level `dwi.bval`/`dwi.bvec` are BIDS-inherited placeholders. There is no raw multi-direction DWI.
- `derivatives/lesion_masks/`: segmentation targets in TRACE (DWI) space. `desc-lesion_mask` (combined) covers 1455 subjects, `desc-lesionAcute_mask` 1451 and `desc-lesionChronic_mask` 203.
- `participants.tsv` (+ `participants.json`): sex, age (clamped to 89), race, `acuteischaemicstroke`, `priorstroke`, `bmi`, `nihss`, `gs_rankin_6isdeath` (**discharge** mRS, 0–6 where 6 is death) and `etiology` (TOAST class, added in 1.1.1/1.1.2).
- `README.md`, `CHANGES`, `dataset_description.json`.

## Excluded

- Nothing. The snapshot has no fMRI, other modalities or non-target derivatives.

## Notes

- **Label coverage:** `participants.tsv` has 1505 rows for 1715 imaged subjects, and all 1505 are imaged. 399 rows are n/a for every clinical field. The other 1106 are all `acuteischaemicstroke=1` (there are no 0s), and these 1106 have NIHSS. Discharge mRS is labeled for 738 (0:112, 1:167, 2:106, 3:109, 4:147, 5:54, 6:43). Etiology is labeled for 1080. Of the 1455 subjects with a combined lesion mask, 1270 are in `participants.tsv`.
- The paper reports 1461 acute ischemic strokes among the 1715, but the table flags only 1106 (the rows with demographics). Build subject lists from `participants.tsv`, not the subject directories.
- The mRS is at **discharge**, not 90 days.
- `participants.json` describes a `lesion_size_cc` column that is not in `participants.tsv`. It does not describe the `etiology` column.
- The ARC dataset (OpenNeuro ds004512) is a companion chronic-stroke cohort from the same group.

## Usage

```sh
bash scripts/soop/download.sh   # 72 GB; resumable
```
