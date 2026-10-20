# FOMO26 Task 4: trigeminal neuralgia

A few-shot segmentation task from the FOMO26 challenge: T2w MRIs of 40 patients with trigeminal neuralgia, with masks of the trigeminal nerve and the vessels around it.

- **Homepage:** <https://fomo26.github.io/>
- **Source:** the organizers' ERDA share, whose link was given to challenge participants (ask the organizers, fomo26@di.ku.dk). We download `Task_4.zip` from it with `curl`. The images are from OpenNeuro [ds005713](https://openneuro.org/datasets/ds005713) (CC0), reoriented to RAS by the organizers. The masks are the organizers'.
- **Version:** unversioned; `Task_4.zip` dated 2026-04-12, pinned by sha256.
- **DOI:** none for the challenge data. Images: [10.18112/openneuro.ds005713](https://doi.org/10.18112/openneuro.ds005713.v2.0.3).
- **License:** the images are CC0; the masks have no stated license. Not redistributable, so our mirror has only this README and the tables without `samples.tsv`. Each user downloads the data and builds the rest.
- **Citation:**
  - Munk, A., Cerri, S., Nersesjan, V., et al. (2026). Towards brain MRI foundation models for the clinic: Findings from the FOMO25 challenge. arXiv:2604.11679.
  - Filimonova, E., Pashkov, A., Tiutiunnik, A., et al. (2025). A large-scale dataset of pre- and postsurgical MRI data from patients with chronic trigeminal neuralgia. *Scientific Data*. <https://doi.org/10.1038/s41597-025-06311-y>
- **Code:** [`scripts/fomo26_task4/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/fomo26_task4) downloads `source/` and builds `tables/`, `derivatives/` and `shards/`.

## Samples

One sample per participant (40), each with a T2w and a mask. The set is tiny, so every participant is in `train`: evaluate with k-fold cross-validation. Complete = T2w and mask (all 40). The source has no age, sex or site.

| split | participants | complete |
|---|---|---|
| train | 40 | 40 |

## Contents

`source/`: the extracted `Task_4.zip` (80 files, 2.2 GB):

- `preprocessed/sub-<id>/ses-01/`: the T2w image (360×512×512, 0.5 mm, RAS).
- `labels/sub-<id>/ses-01/seg.nii.gz`: the mask on the T2w grid. Label 1 is the trigeminal nerve and label 2 the vessels (a few hundred to ~1,400 voxels).

`tables/` (derived from `source/`):

- `images.tsv`: T2w and mask (desc `trigeminal`).
- `samples.tsv` + `samples.json`: identifiers only. Not on our mirror.
- `splits.tsv`: split, rank and complete per participant.
