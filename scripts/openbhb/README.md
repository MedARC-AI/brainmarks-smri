# OpenBHB

Open Big Healthy Brains: T1w scans of healthy controls aggregated from 10 public cohorts (ABIDE I/II, IXI, CoRR, GSP, Localizer, MPI-Leipzig, NAR, NPC, RBP), ages 6–88. We use it for brain-age prediction with site debiasing (the OpenBHB challenge).

- **Source:** HuggingFace [`benoit-dufumier/openBHB`](https://huggingface.co/datasets/benoit-dufumier/openBHB), uploaded by the first author. Pinned to commit `8508cda68fea74f217926acbf46ee5863f8879d1` (2025-09-05). Downloaded with `hf download` (huggingface_hub 2.1.1); the repo is not gated. The project site ([baobablab.github.io/bhb](https://baobablab.github.io/bhb/)) points to [IEEE DataPort](https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing), which requires an IEEE login, so we don't use it.
- **DOI:** [10.21227/7jsg-jx57](https://doi.org/10.21227/7jsg-jx57) (IEEE DataPort release). The HF repo has no DOI.
- **License:** **CC BY-NC-SA 3.0** (non-commercial, share-alike), as stated in the body of the dataset card and on IEEE DataPort. The HF metadata tag says `apache-2.0`, which contradicts the card text and the source cohorts; treat the tag as wrong. The card also says: "By downloading this dataset, you also agree to the most restrictive Data Usage Agreement (DUA) of all cohorts". Per-cohort terms listed there: ABIDE I/II CC BY-NC-SA 3.0 + DUA; IXI, CoRR CC0 + DUA; GSP DUA only (Harvard Dataverse open-access terms); NAR, MPI-Leipzig, NPC, RBP CC0; Localizer CC BY 3.0. The DUA text itself (`DUA_openBHB` zip) is only on IEEE DataPort, behind the login.
- **Citation:** Dufumier, B., Grigis, A., Victor, J., Ambroise, C., Frouin, V., & Duchesnay, E. (2022). OpenBHB: a Large-Scale Multi-Site Brain MRI Data-set for Age Prediction and Debiasing. *NeuroImage*, 263, 119637. Cite the source cohorts as well (list on the dataset card).
- **Contents** (default download, 3984 subjects, one session each):
  - `{train,val}/rawdata/sub-*/ses-1/sub-*_T1w.nii.gz`: the original T1w scans (native space, not skull-stripped, int16). 3227 train + 757 val.
  - `participants.tsv`: `participant_id, study, sex, age, site, diagnosis` (all `control`), `tiv, csfv, gmv, wmv, magnetic_field_strength, acquisition_setting, siteXacq, split`. `split` is `train` (3227), `internal_test` (362) or `external_test` (395); `val/` holds both test splits. `study` (10 cohorts) and `site` (62 codes in the public part) are integer codes.
  - `qc.tsv`: QC metrics (FreeSurfer Euler number, CAT12 NCR/IQR, quasi-raw correlation).
  - `resource/`: MNI templates used for quasi-raw and CAT12 registration, the Neuromorphometrics atlas, ROI/channel names, and `resources.json` (array shapes).
  - `{train,val}/derivatives/{cat12vbm_roi,freesurfer_roi}/*.csv`: ROI features (CAT12 GM volumes, FreeSurfer Desikan/Destrieux), all subjects concatenated per split.
- **Excluded** (per-subject preprocessed arrays in `{train,val}/derivatives/sub-*/`, float32 `.npy`, uncompressed):
  - quasi-raw `*_preproc-quasiraw_T1w.npy` (1×182×218×182, MNI-affine, skull-stripped, bias-corrected): **230 GB**. This is the challenge's main image input. It is over the 200 GB check-with-user threshold, so it is opt-in: `OPENBHB_QUASIRAW=1 bash scripts/openbhb/download.sh`. It can also be recomputed from `rawdata/` with [brainprep](https://github.com/neurospin-deepinsight/brainprep).
  - CAT12 VBM grey matter `*_preproc-cat12vbm_desc-gm_T1w.npy`: 68 GB.
  - FreeSurfer xhemi surface features `*_preproc-freesurfer_desc-xhemi_T1w.npy`: 42 GB.
  - Per-subject ROI arrays `*_ROI.npy` (0.06 GB): duplicates of the concatenated CSVs.
  - Whole repo: 372 GB.
- **Notes:**
  - The HF release includes the original T1w (`rawdata/`), which the challenge description doesn't mention. It is the closest thing to raw and fits our "no preprocessing" rule better than quasi-raw.
  - Counts: the public part has 3984 subjects (3227 / 362 / 395), matching `datasets.md`. The HF card says 664 private test subjects; the paper's 5330 total includes privateBHB. Site counts vary by source: "93 centers" on the HF card, 71 sites in `datasets.md`, and 62 `site` codes (64 `siteXacq`) in the public `participants.tsv`. The external test set covers 6 sites that are absent from train.
  - Subject IDs are anonymized 12-digit numbers. There is no mapping back to the source cohort IDs, so overlap with our ABIDE I and IXI copies can't be resolved from these files.

```sh
bash scripts/openbhb/download.sh   # 32 GB; resumable (OPENBHB_QUASIRAW=1: +230 GB)
```
