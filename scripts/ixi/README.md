# IXI

Nearly 600 healthy adults (about 20–86 y) scanned at three London hospitals: Guy's (Philips 1.5T), Hammersmith (Philips 3T) and IOP (GE 1.5T). We use the T1/T2/PD scans for brain-age prediction.

- **Source:** [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/) (Imperial College London). Plain HTTPS tarballs from `https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/`, fetched with `curl -fL -C -`. The page links use `http://`, which redirects to `https://`.
- **Version:** no versioning and no published checksums. The files have been unchanged since 2014-10-07. Pinned by the server metadata below (retrieved 2026-10-01) and `manifest.sha256`.

  | File | Size (bytes) | Last-Modified | ETag |
  |---|---|---|---|
  | IXI-T1.tar | 4,840,816,640 | Tue, 07 Oct 2014 17:48:09 GMT | `"120890000-504d8cd6ce25a"` |
  | IXI-T2.tar | 3,853,445,120 | Tue, 07 Oct 2014 17:51:22 GMT | `"e5aee800-504d8d8ead982"` |
  | IXI-PD.tar | 4,068,966,400 | Tue, 07 Oct 2014 17:44:14 GMT | `"f2878000-504d8bf6541e7"` |
  | IXI-DTI.tar | 4,271,247,986 | Tue, 07 Oct 2014 17:28:27 GMT | `"fe961272-504d886f9bda0"` |
  | IXI.xls | 210,432 | Tue, 07 Oct 2014 17:51:22 GMT | `"33600-504d8d8ebb79c"` |
  | bvals.txt | 77 | Tue, 07 Oct 2014 17:51:22 GMT | `"4d-504d8d8ebc11e"` |
  | bvecs.txt | 284 | Tue, 07 Oct 2014 17:51:22 GMT | `"11c-504d8d8ebc88f"` |
  | marital.xls | 1.2K | 2014-10-07 18:51 (dir listing) | |

- **DOI:** none.
- **License:** [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode). Attribution required; derivatives must be shared alike. No DUA or login.
- **Citation:** the source asks: "If you use the IXI data please acknowledge the source of the IXI data, e.g. this website" (https://brain-development.org/ixi-dataset/). Collected under the EPSRC project IXI – Information eXtraction from Images (GR/S21533/02).

## Contents

`source/` (tarballs kept as distributed, not unpacked; 17 GB):

- `IXI-T1.tar`, `IXI-T2.tar`, `IXI-PD.tar`: one NIfTI (`.nii.gz`) per subject, named `IXI<id>-<Site>-<scan>-<mod>.nii.gz`, where the site is `Guys`/`HH`/`IOP`.
- `IXI-DTI.tar`: raw DWI, one NIfTI per volume (`IXI<id>-<Site>-<scan>-DTI-<nn>.nii.gz`). `bvals.txt`/`bvecs.txt` hold the gradient table.
- `IXI.xls`: demographics. Sheet `Table` has IXI_ID, SEX_ID (1 = m, 2 = f), HEIGHT, WEIGHT, ETHNIC_ID, MARITAL_ID, OCCUPATION_ID, QUALIFICATION_ID, DOB, DATE_AVAILABLE, STUDY_DATE and **AGE** (the brain-age target). The other sheets are code lookup tables.
- `marital.xls`: the marital-status code lookup. It is in the server directory but not linked from the page.
- `docs/`: a one-time HTML snapshot of the dataset page (description + license) and the two Philips scanner-parameter pages. The GE/IOP parameters were never published.

Subject counts (from `tar -tf`):

| | Guys | HH | IOP | Total |
|---|---|---|---|---|
| T1 | 322 | 185 | 74 | 581 |
| T2 | 319 | 185 | 74 | 578 |
| PD | 319 | 185 | 74 | 578 |
| DTI | 217 | 183 | – | 400 |

## Excluded

- `IXI-MRA.tar`: MR angiography (12.4 GB, 42% of the full release). Not an input to the brain-age benchmarks.

## Notes

- **Usable brain-age cohort: 563 T1 subjects with an age.** `IXI.xls` has 619 rows but only 593 unique IXI_IDs (duplicate rows), and 590 have AGE. 15 of the 581 T1 subjects are missing from the spreadsheet. Build subject lists from the tarballs and join on IXI_ID, after deduplicating.
- DTI: `bvals.txt`/`bvecs.txt` list **16** gradients (1 b0 + 15 at b=1000), but 397 of 400 subjects have **17** volumes (3 have 16). Check the volume order before using the gradient table. There is no DTI for IOP.
- OpenBHB includes IXI subjects, but its IDs are anonymized (see `scripts/openbhb/README.md`).

## Usage

```sh
bash scripts/ixi/download.sh   # 17 GB; resumable
```
