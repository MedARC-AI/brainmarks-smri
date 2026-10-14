# ATLAS R2.1

Anatomical Tracings of Lesions After Stroke: T1w MRIs of chronic stroke patients from 33 sites, with manually traced lesion masks. We use it for stroke lesion segmentation.

- **Homepage:** <https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html>
- **Source:** INDI S3 (`s3://fcp-indi/data/Projects/INDI/ATLAS/R2.1/`), downloaded with `curl`. The archive is encrypted with `openssl`. To get the key, accept the terms of use and fill in the Google form on the homepage. Each user must apply for their own key.
- **Version:** R2.1, minimally processed (`atlas21_training_raw.tar.gz`, 2026-04-22). The same 955 scans as R2.0, now with the masks of the 300 test scans.
- **DOI:** [10.1038/s41597-022-01401-7](https://doi.org/10.1038/s41597-022-01401-7) (R2.0 data descriptor)
- **License:** ATLAS terms of use, accepted with the key request. Not redistributable, so it is not on our mirror; each user downloads it.
- **Citation:** Liew, S.-L., Lo, B. P., Donnelly, M. R., et al. (2022). A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms. *Scientific Data*, 9, 320.
- **Code:** [`scripts/atlas/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/atlas) downloads `source/` and builds `tables/`.
