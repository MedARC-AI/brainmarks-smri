# FOMO26 Task 5: polymicrogyria

A few-shot classification task from the FOMO26 challenge: T1w MRIs of 24 children with polymicrogyria and 24 controls (Children's Hospital of Eastern Ontario).

- **Homepage:** <https://fomo26.github.io/>
- **Source:** the organizers' ERDA share, whose link was given to challenge participants (ask the organizers, fomo26@di.ku.dk). We download it with `curl`. The organizers distribute the PPMR dataset ([uOttawa repository](https://ruor.uottawa.ca/items/298d789f-e390-43c7-a657-a823701b18b6): coronal JPEG slices) and their script `Task_5_extract.py`, which stacks the slices into NIfTI volumes and picks one control per control folder. We run the script, keep its output and drop the zip.
- **Version:** unversioned; files dated 2026-05-01, pinned by sha256.
- **DOI:** none for the challenge data. PPMR: [10.20381/ruor-28525](https://doi.org/10.20381/ruor-28525).
- **License:** PPMR is CC BY-NC-ND 4.0. Not redistributable, so it is not on our mirror; each user downloads it.
- **Citation:**
  - Munk, A., Cerri, S., Nersesjan, V., et al. (2026). Towards brain MRI foundation models for the clinic: Findings from the FOMO25 challenge. arXiv:2604.11679.
  - Zhang, L., Abdeen, N., & Lang, J. (2022). A novel center-based deep contrastive metric learning method for the detection of polymicrogyria in pediatric brain MRI. arXiv:2211.12565.
- **Code:** [`scripts/fomo26_task5/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/fomo26_task5) downloads `source/` and builds `tables/`.
