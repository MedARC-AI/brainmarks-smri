# OASIS-2

Open Access Series of Imaging Studies, longitudinal: 150 adults aged 60-96 (Washington University), scanned at 2-5 visits at least a year apart (373 sessions, 3-4 T1w MPRAGE scans each). 72 stayed nondemented, 64 were demented at the first visit, and 14 converted from nondemented to demented. We use it for dementia tasks.

- **Homepage:** <https://sites.wustl.edu/oasisbrains/home/oasis-2/>
- **Source:** two raw image archives from <https://download.nrg.wustl.edu/data/> and the demographics spreadsheet from the homepage, downloaded with `curl`. No login is needed, but users must accept the OASIS data use terms (<https://sites.wustl.edu/oasisbrains/home/access/>). The images are distributed as NIfTI pairs (`.nifti.hdr`/`.img`); we store each as one `.nii.gz` with the same header and voxels.
- **Version:** unversioned; archives dated 2009-03, spreadsheet 2024. Pinned by sha256.
- **DOI:** [10.1162/jocn.2009.21407](https://doi.org/10.1162/jocn.2009.21407) (paper)
- **License:** OASIS data use terms: non-commercial research only, no redistribution without written permission. So it is not on our mirror; each user downloads it.
- **Citation:** Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies: Longitudinal MRI data in nondemented and demented older adults. *Journal of Cognitive Neuroscience*, 22(12), 2677-2684.
- **Code:** [`scripts/oasis2/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/oasis2) downloads `source/` and builds `tables/`.
