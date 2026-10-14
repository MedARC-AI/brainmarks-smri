# OASIS-1

Open Access Series of Imaging Studies, cross-sectional: 416 adults aged 18-96 (Washington University), each with 3-4 T1w MPRAGE scans from one session. About 100 of those over 60 have very mild to moderate Alzheimer's disease (CDR 0.5-2). A reliability set of 20 nondemented subjects was rescanned within 90 days. We use it for dementia classification.

- **Homepage:** <https://sites.wustl.edu/oasisbrains/home/oasis-1/>
- **Source:** 12 raw image discs from <https://download.nrg.wustl.edu/data/> and the clinical spreadsheets from the homepage, downloaded with `curl`. No login is needed, but users must accept the OASIS data use terms (<https://sites.wustl.edu/oasisbrains/home/access/>). The images are distributed in Analyze 7.5 format (`.hdr`/`.img`), which has no affine. We store each as `.nii.gz` with the same voxels, and set the affine from the header's orientation code as FreeSurfer reads it (sagittal: ASL; transverse: LAS; the two `*.4dfp` images: PIR).
- **Version:** unversioned; discs dated 2007-06, spreadsheets 2024. Pinned by sha256.
- **DOI:** [10.1162/jocn.2007.19.9.1498](https://doi.org/10.1162/jocn.2007.19.9.1498) (paper)
- **License:** OASIS data use terms: non-commercial research only, no redistribution without written permission. So it is not on our mirror; each user downloads it.
- **Citation:** Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498-1507.
- **Code:** [`scripts/oasis1/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/oasis1) downloads `source/` and builds `tables/`.
