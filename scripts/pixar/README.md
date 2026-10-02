# Pixar

Children (3–12 y, n=122) and adults (n=33) scanned while watching a short animated film. We use only the T1w scans, for brain-age prediction.

- **Source:** OpenNeuro [ds000228](https://openneuro.org/datasets/ds000228/versions/1.1.1). Downloaded with `openneuro-py`. The public S3 mirror (`s3://openneuro.org/ds000228`) is stale for this snapshot: its metadata files are still from 1.1.0.
- **Version:** snapshot 1.1.1 (2023-09-27), the latest.
- **DOI:** [10.18112/openneuro.ds000228.v1.1.1](https://doi.org/10.18112/openneuro.ds000228.v1.1.1)
- **License:** CC0
- **Citation:** Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018). Development of the social brain from age three to twelve years. *Nature Communications*, 9, 1027.

## Contents

The structural part of the raw BIDS release (315 files, 1.0 GB):

- `sub-*/anat/`: T1w + JSON sidecar for 155 subjects.
- `participants.tsv` (+ `participants.json`): age, sex, handedness, ToM and IQ scores, scanner info.
- `dataset_description.json`, `README`, `CHANGES`.

## Excluded

- fMRI: `sub-*/func/` and `task-pixar_bold.json` (4.3 GB).
- `derivatives/`: fMRI preprocessing and MRIQC outputs (17 GB).

## Notes

- The target for brain age is `Age` in `participants.tsv`. `AgeGroup` and `Child_Adult` are coarser bins.

## Usage

```sh
bash scripts/pixar/download.sh   # 1.0 GB; resumable
```
