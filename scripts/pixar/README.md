# Pixar

Children (3–12 y, n=122) and adults (n=33) scanned while watching a short animated film. We use only the T1w scans, for brain-age prediction.

- **Source:** OpenNeuro [ds000228](https://openneuro.org/datasets/ds000228/versions/1.1.1). Downloaded with `openneuro-py`. The public S3 mirror (`s3://openneuro.org/ds000228`) is stale for this snapshot: its metadata files are still from 1.1.0.
- **Version:** snapshot 1.1.1 (2023-09-27), the latest.
- **DOI:** [10.18112/openneuro.ds000228.v1.1.1](https://doi.org/10.18112/openneuro.ds000228.v1.1.1)
- **License:** CC0
- **Citation:** Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018). Development of the social brain from age three to twelve years. *Nature Communications*, 9, 1027.

## Usage

```sh
bash scripts/pixar/download.sh                 # 1.0 GB; resumable
uv run python scripts/pixar/build_tables.py    # tables/; prints the summary below
```

## Samples

One sample per participant (one T1w each). Split 60/20/20, stratified by age group (3yo/4yo/5yo/7yo/8-12yo/Adult); there is no official split. Complete = T1w and age (all 155).

| split | participants | complete | age | female | child_adult |
|---|---|---|---|---|---|
| train | 92 | 92 | 10.8 ± 8.4 | 53% | adult 20 / child 72 |
| val | 31 | 31 | 10.1 ± 7.5 | 55% | adult 6 / child 25 |
| test | 32 | 32 | 10.3 ± 7.7 | 56% | adult 7 / child 25 |
| total | 155 | 155 | 10.6 ± 8.1 | 54% | adult 33 / child 122 |

## Contents

`source/`: the structural part of the raw BIDS release (315 files, 1.0 GB):

- `sub-*/anat/`: T1w + JSON sidecar for 155 subjects.
- `participants.tsv` (+ `participants.json`): age, sex, handedness, ToM and IQ scores, scanner info.
- `dataset_description.json`, `README`, `CHANGES`.

`tables/` (built by `build_tables.py`; layout in `src/brain_datasets/tables.py`):

- `images.tsv`: one T1w per participant.
- `samples.tsv` + `samples.json`: age (the brain-age target), sex, age group, handedness, ToM and nonverbal IQ scores, scanner and coil. The fMRI scan-log columns (voxel size, slice gap) are left out.
- `splits.tsv`: split, rank and complete per participant.

## Excluded

- fMRI: `sub-*/func/` and `task-pixar_bold.json` (4.3 GB).
- `derivatives/`: fMRI preprocessing and MRIQC outputs (17 GB).
