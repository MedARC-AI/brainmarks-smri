# IXI

Nearly 600 healthy adults (about 20–86 y) scanned at three London hospitals: Guy's (Philips 1.5T), Hammersmith (Philips 3T) and IOP (GE 1.5T). We use the T1/T2/PD scans for brain-age prediction.

- **Homepage:** <https://brain-development.org/ixi-dataset/>
- **Source:** [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/) (Imperial College London). Plain HTTPS tarballs from `https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/`, fetched with `curl -fL -C -`. The page links use `http://`, which redirects to `https://`.
- **Version:** no versioning and no published checksums. The files have been unchanged since 2014-10-07. Pinned by the server metadata below (retrieved 2026-10-01), the tarball sha256s in `download.sh` and `tables/source.sha256`.

  | File | Size (bytes) | Last-Modified | ETag |
  |---|---|---|---|
  | IXI-T1.tar | 4,840,816,640 | Tue, 07 Oct 2014 17:48:09 GMT | `"120890000-504d8cd6ce25a"` |
  | IXI-T2.tar | 3,853,445,120 | Tue, 07 Oct 2014 17:51:22 GMT | `"e5aee800-504d8d8ead982"` |
  | IXI-PD.tar | 4,068,966,400 | Tue, 07 Oct 2014 17:44:14 GMT | `"f2878000-504d8bf6541e7"` |
  | IXI.xls | 210,432 | Tue, 07 Oct 2014 17:51:22 GMT | `"33600-504d8d8ebb79c"` |
  | marital.xls | 1.2K | 2014-10-07 18:51 (dir listing) | |

- **DOI:** none.
- **License:** [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode). Attribution required; derivatives must be shared alike. No DUA or login.
- **Citation:** the source asks: "If you use the IXI data please acknowledge the source of the IXI data, e.g. this website" (https://brain-development.org/ixi-dataset/). Collected under the EPSRC project IXI – Information eXtraction from Images (GR/S21533/02).
- **Code:** [`scripts/ixi/`](https://github.com/MedARC-AI/brainmarks-smri/tree/main/scripts/ixi) re-downloads `source/` and rebuilds `tables/`.

## Samples

One sample per participant: everyone with at least one image (582; 1 has no T1). Split 60/20/20, stratified by site × age bin (<30, 30–45, 45–60, 60–75, 75+); there is no official split. Complete = T1, T2, PD and age (557); 561 T1 participants have an age.

| split | participants | complete | age | female | sites |
|---|---|---|---|---|---|
| train | 350 | 335 | 48.8 ± 16.8 | 54% | 3 |
| val | 115 | 110 | 48.3 ± 16.3 | 55% | 3 |
| test | 117 | 112 | 48.3 ± 16.1 | 58% | 3 |
| total | 582 | 557 | 48.6 ± 16.5 | 55% | 3 |

## Contents

`source/` (12.8 GB). Each tarball is checked against the sha256 pinned in `download.sh`, extracted to a folder of the same name, and deleted:

- `IXI-T1/`, `IXI-T2/`, `IXI-PD/`: one NIfTI (`.nii.gz`) per subject, named `IXI<id>-<Site>-<scan>-<mod>.nii.gz`, where the site is `Guys`/`HH`/`IOP`.
- `IXI.xls`: demographics. Sheet `Table` has IXI_ID, SEX_ID (1 = m, 2 = f), HEIGHT, WEIGHT, ETHNIC_ID, MARITAL_ID, OCCUPATION_ID, QUALIFICATION_ID, DOB, DATE_AVAILABLE, STUDY_DATE and **AGE** (the brain-age target). The other sheets are code lookup tables.
- `marital.xls`: the marital-status code lookup. It is in the server directory but not linked from the page.
- `docs/`: a one-time HTML snapshot of the dataset page (description + license) and the two Philips scanner-parameter pages. The GE/IOP parameters were never published.

Subject counts:

| | Guys | HH | IOP | Total |
|---|---|---|---|---|
| T1 | 322 | 185 | 74 | 581 |
| T2 | 319 | 185 | 74 | 578 |
| PD | 319 | 185 | 74 | 578 |

`tables/` (derived from `source/`):

- `images.tsv`: T1w, T2w and PD.
- `samples.tsv` + `samples.json`: age (the brain-age target), sex, site (from the file names), study date, height, weight and the decoded demographic codes. The spreadsheet's duplicate and conflicting rows are resolved (see `demographics_conflict`).
- `splits.tsv`: split, rank and complete per participant.

## Excluded

- `IXI-MRA.tar`: MR angiography (12.4 GB, 42% of the full release). Not an input to the brain-age benchmarks.
- `IXI-DTI.tar`, `bvals.txt`, `bvecs.txt`: raw DTI (4.3 GB, 400 subjects, one NIfTI per volume). No task uses it.

## Notes

- `IXI.xls` has 619 rows for 593 IDs: duplicate rows, mostly a full row plus a zeroed copy, and 4 IDs whose rows disagree. It also has entry errors (e.g. heights of 1520 cm) and an ethnicity code (2) that is missing from the lookup sheet. 15 of the 581 T1 subjects have no row. The tables handle all of these.
- OpenBHB includes IXI subjects, but its IDs are anonymized (see the [OpenBHB README](../openbhb/README.md)).
