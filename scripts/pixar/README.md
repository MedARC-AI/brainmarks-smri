# Pixar

Children (3–12 y, n=122) and adults (n=33) scanned while watching a short animated film. We use only the T1w scans, for brain-age prediction.

- **Source:** OpenNeuro [ds000228](https://openneuro.org/datasets/ds000228/versions/1.1.1), snapshot 1.1.1 (2023-09-27). Downloaded with `openneuro-py`. The public S3 mirror is stale for this snapshot.
- **DOI:** [10.18112/openneuro.ds000228.v1.1.1](https://doi.org/10.18112/openneuro.ds000228.v1.1.1)
- **License:** CC0
- **Citation:** Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018). Development of the social brain from age three to twelve years. *Nature Communications*, 9, 1027.
- **Contents:** the structural part of the raw BIDS release: 155 subjects with `anat/` T1w + JSON sidecars, `participants.tsv` (age, sex, handedness, ToM and IQ scores, scanner), and top-level metadata.
- **Excluded:** fMRI (`sub-*/func/`, `task-pixar_bold.json`; 4.3 GB) and `derivatives/` (fMRI preprocessing + MRIQC; 17 GB).

```sh
bash scripts/pixar/download.sh   # 1.0 GB; resumable
```
