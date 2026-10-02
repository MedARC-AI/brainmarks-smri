# OpenBHB

Open Big Healthy Brains: T1w scans of healthy controls from 10 public cohorts (ABIDE I/II, IXI, CoRR, GSP, Localizer, MPI-Leipzig, NAR, NPC, RBP), ages 6–88. We use it for brain-age prediction with site debiasing (the OpenBHB challenge).

- **Homepage:** <https://baobablab.github.io/bhb/>
- **Source:** HuggingFace [`benoit-dufumier/openBHB`](https://huggingface.co/datasets/benoit-dufumier/openBHB), uploaded by the first author; not gated. Downloaded with `hf download` (huggingface_hub 2.1.1). The project site ([baobablab.github.io/bhb](https://baobablab.github.io/bhb/)) points to [IEEE DataPort](https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing), which requires an IEEE login, so we don't use it.
- **Version:** HF commit `8508cda68fea74f217926acbf46ee5863f8879d1` (2025-09-05).
- **DOI:** [10.21227/7jsg-jx57](https://doi.org/10.21227/7jsg-jx57) (the IEEE DataPort release). The HF repo has no DOI.
- **License:** CC BY-NC-SA 3.0, as stated in the dataset card text and on IEEE DataPort. The HF metadata tag says `apache-2.0`, which is wrong. The card also says: "By downloading this dataset, you also agree to the most restrictive Data Usage Agreement (DUA) of all cohorts". Per-cohort terms listed there: ABIDE I/II CC BY-NC-SA 3.0 + DUA; IXI, CoRR CC0 + DUA; GSP DUA only (Harvard Dataverse); NAR, MPI-Leipzig, NPC, RBP CC0; Localizer CC BY 3.0. The DUA text itself is only on IEEE DataPort, behind the login.
- **Citation:** Dufumier, B., Grigis, A., Victor, J., Ambroise, C., Frouin, V., & Duchesnay, E. (2022). OpenBHB: a Large-Scale Multi-Site Brain MRI Data-set for Age Prediction and Debiasing. *NeuroImage*, 263, 119637. Cite the source cohorts as well (list on the dataset card).
- **Code:** [`scripts/openbhb/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/openbhb) re-downloads `source/` and rebuilds `tables/`.

## Samples

One sample per participant (3984 healthy controls, one T1w each). The official split is kept: `internal_test` and `external_test` together are `test`, and the official train set is split 80/20 into train/val, stratified by site, as in Neuro-JEPA (`official_split` holds the original labels). The private test set is not public. Complete = T1w and age (all).

| split | participants | complete | age | female | sites |
|---|---|---|---|---|---|
| train | 2583 | 2583 | 25.2 ± 14.6 | 48% | 57 |
| val | 644 | 644 | 25.1 ± 14.5 | 48% | 57 |
| test | 757 | 757 | 23.8 ± 12.8 | 45% | 60 |
| total | 3984 | 3984 | 24.9 ± 14.3 | 48% | 62 |

## Contents

`source/`: 3984 subjects, one session each (4003 files, 32 GB):

- `{train,val}/rawdata/sub-*/ses-1/sub-*_T1w.nii.gz`: the original T1w scans (native space, not skull-stripped). 3227 train + 757 val.
- `participants.tsv`: `participant_id, study, sex, age, site, diagnosis` (all `control`), `tiv, csfv, gmv, wmv, magnetic_field_strength, acquisition_setting, siteXacq, split`. `split` is `train` (3227), `internal_test` (362) or `external_test` (395); `val/` holds both test splits. `study` (10 cohorts) and `site` (62 codes in the public part) are integer codes.
- `qc.tsv`: QC metrics (FreeSurfer Euler number, CAT12 NCR/IQR, quasi-raw correlation).
- `resource/`: MNI templates used for quasi-raw and CAT12 registration, the Neuromorphometrics atlas, ROI/channel names, and `resources.json` (array shapes).
- `{train,val}/derivatives/{cat12vbm_roi,freesurfer_roi}/*.csv`: ROI features (CAT12 GM volumes, FreeSurfer Desikan/Destrieux), all subjects concatenated per split.
- `README.md`: the dataset card.

`tables/` (derived from `source/`):

- `images.tsv`: one original T1w (`rawdata/`) per participant.
- `samples.tsv` + `samples.json`: age (target), sex, site and study codes, field strength, acquisition setting, CAT12 tissue volumes, and the `qc.tsv` metrics. participant_id gets the BIDS `sub-` prefix.
- `splits.tsv`: split, official split, rank and complete per participant.

## Excluded

The per-subject preprocessed arrays in `{train,val}/derivatives/sub-*/` (float32 `.npy`, uncompressed; whole repo 372 GB):

- quasi-raw `*_preproc-quasiraw_T1w.npy` (skull-stripped, bias-corrected, MNI-affine; the challenge's main input): 230 GB. It can be recomputed from `rawdata/` with [brainprep](https://github.com/neurospin-deepinsight/brainprep).
- CAT12 VBM grey matter `*_preproc-cat12vbm_desc-gm_T1w.npy`: 68 GB.
- FreeSurfer xhemi surface features `*_preproc-freesurfer_desc-xhemi_T1w.npy`: 42 GB.
- Per-subject ROI arrays `*_ROI.npy` (0.06 GB): duplicates of the concatenated CSVs.

## Notes

- The HF release includes the original T1w (`rawdata/`), which the challenge description doesn't mention. It fits our "no preprocessing" rule better than quasi-raw.
- Counts: the public part has 3984 subjects (3227 / 362 / 395). The HF card says 664 private test subjects; the paper's 5330 total includes privateBHB. Site counts vary by source: "93 centers" on the HF card and 62 `site` codes (64 `siteXacq`) in the public `participants.tsv`. The external test set is defined by site × acquisition setting: 5 of its 6 sites are absent from train, but site 36 is also in train with another acquisition setting.
- Subject IDs are anonymized 12-digit numbers with no mapping back to the source cohort IDs, so overlap with our ABIDE I and IXI copies can't be resolved from these files.
- `hf download` writes timestamped metadata to `source/.cache/`; the script deletes it so the manifest is stable. A re-run re-hashes local files (about 3.5 min) instead of downloading.
