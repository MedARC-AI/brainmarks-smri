# FOMO26 Task 4: trigeminal neuralgia

A few-shot segmentation task from the FOMO26 challenge: T2w MRIs of 40 patients with trigeminal neuralgia, with masks of the trigeminal nerve and the vessels around it.

- **Homepage:** <https://fomo26.github.io/>
- **Source:** the organizers' ERDA share, whose link was given to challenge participants (ask the organizers, fomo26@di.ku.dk). We download `Task_4.zip` from it with `curl`. The images are from OpenNeuro [ds005713](https://openneuro.org/datasets/ds005713) (CC0), reoriented to RAS by the organizers. The masks are the organizers'.
- **Version:** unversioned; `Task_4.zip` dated 2026-04-12, pinned by sha256.
- **DOI:** none for the challenge data. Images: [10.18112/openneuro.ds005713](https://doi.org/10.18112/openneuro.ds005713.v2.0.3).
- **License:** the images are CC0; the masks have no stated license. Not redistributable, so it is not on our mirror; each user downloads it.
- **Citation:**
  - Munk, A., Cerri, S., Nersesjan, V., et al. (2026). Towards brain MRI foundation models for the clinic: Findings from the FOMO25 challenge. arXiv:2604.11679.
  - Filimonova, E., Pashkov, A., Tiutiunnik, A., et al. (2025). A large-scale dataset of pre- and postsurgical MRI data from patients with chronic trigeminal neuralgia. *Scientific Data*. <https://doi.org/10.1038/s41597-025-06311-y>
- **Code:** [`scripts/fomo26_task4/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/fomo26_task4) downloads `source/` and builds `tables/`.
