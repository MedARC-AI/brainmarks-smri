# brain_datasets

Benchmark datasets for evaluating structural MRI foundation models. The data is collected unmodified from the original sources, with pinned versions, checksums and provenance.

Each dataset has a self-contained folder `scripts/<name>/` with:
- `download.sh`: downloads the data into `datasets/<name>/source/`
- `README.md`: source, version, license, citation, and what is included or excluded
- `manifest.sha256`: checksums of the expected files

```sh
bash scripts/pixar/download.sh                              # download (resumable)
cd datasets/pixar && sha256sum -c --quiet manifest.sha256   # verify
```

## Benchmark tables

`scripts/<name>/build_tables.py` derives metadata tables from `datasets/<name>/source/` (which is never modified). They are written to `datasets/<name>/tables/`, with a tracked copy in `scripts/<name>/tables/`. Shared helpers are in the `brain_datasets` package (`src/brain_datasets/tables.py`).

| File | One row per | Columns |
|---|---|---|
| `images.tsv` | image file | `participant_id, session_id, modality, desc, path` (+ `member` for files inside a tar, IXI) |
| `samples.tsv` | sample (scan session) | `participant_id, session_id, age, sex, site`, then dataset-specific labels and covariates |
| `samples.json` | `samples.tsv` column | description, source column, levels, units |
| `splits.tsv` | participant | `participant_id, split, official_split, rank, complete` |

- `split` is train/val/test by participant. Official splits are kept where they exist (`official_split`); otherwise 60/20/20, stratified, with a fixed seed.
- `rank` orders participants within a split so that every prefix is balanced; `complete` marks participants with all core images and primary targets. `brain_datasets.tables.mini_split(splits, "train", 100)` gives a nested 100-participant subset.
- Paths in `images.tsv` are relative to `datasets/<name>/`.

```sh
uv sync                                          # dependencies + the brain_datasets package
uv run python scripts/ucsf_pdgm/build_tables.py  # rebuild one dataset's tables
```
