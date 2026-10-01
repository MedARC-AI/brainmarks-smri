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
