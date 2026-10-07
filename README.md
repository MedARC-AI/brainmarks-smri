# Brainmarks-sMRI

Benchmark datasets for evaluating structural MRI foundation models. The data is collected unmodified from the original sources, with pinned versions, checksums and provenance.

This repo holds the code that builds the collection. The data itself is mirrored on Hugging Face at [medarc/brainmarks-smri](https://huggingface.co/datasets/medarc/brainmarks-smri). The dataset index, licenses and table schemas are in the dataset card, [`README_hf.md`](README_hf.md).

## Layout

```
scripts/
  lib.sh                 # shared bash helpers
  upload.sh              # uploads datasets/ to the Hugging Face mirror
  preprocess.py          # brain masks + affines to MNI into datasets/<name>/derivatives/
  <name>/
    download.sh          # downloads the release into datasets/<name>/source/ (resumable)
    build_tables.py      # builds datasets/<name>/tables/ from source/
    README.md            # source, version, license, citation, contents, exclusions
    manifest.sha256      # checksums of source/
    derivatives.sha256   # checksums of derivatives/
    tables/              # tracked copy of the built tables
src/brainmarks_smri/     # package: datasets/ (dataset classes, table building, TCIA downloads), models/, tasks, probes, run
datasets/                # the data (gitignored); this folder is what gets mirrored
```

`download.sh` and `build_tables.py` also copy the dataset's README into `datasets/<name>/`, so each dataset folder describes itself.

## Reproducing

Requirements: [uv](https://docs.astral.sh/uv/), `curl`, the [AWS CLI](https://aws.amazon.com/cli/) (anonymous S3 for ABIDE I and ADHD-200), and for the TCIA datasets (UCSF-PDGM, UPENN-GBM, BraTS 2021) the Aspera `ascp` binary (`gem install aspera-cli && ascli config ascp install`) plus outbound TCP/UDP port 33001. The [devcontainer](.devcontainer/) sets all of this up.

```sh
uv sync                                         # dependencies + the brainmarks_smri package
bash scripts/pixar/download.sh                  # download into datasets/pixar/source/
(cd datasets/pixar && sha256sum -c --quiet manifest.sha256)  # verify
uv run python scripts/pixar/build_tables.py     # rebuild datasets/pixar/tables/
uv run --extra preprocess scripts/preprocess.py datasets/pixar  # rebuild datasets/pixar/derivatives/
```

- Each dataset is pinned to a fixed release. Re-running `download.sh` only fetches missing files. If the scripts reproduce the collection, `git diff` on the tracked manifest and tables shows no changes.
- Some sources (the INDI S3 buckets for ABIDE I and ADHD-200, and IXI) are not versioned. For those, the manifest detects changes but cannot restore old files; the Hugging Face mirror keeps the collected copy.
- The TCIA downloads are the slow ones: about 17 min for BraTS 2021 and 25 min for UCSF-PDGM. A complete re-run transfers nothing but still takes 12–17 min, because `ascp` checks every file against the server.

To publish to the mirror: `bash scripts/upload.sh`. It checks every dataset against its manifest, creates the repo if needed with an automatic access gate, and uploads (resumable).

## License

The code is under the [MIT License](LICENSE). Each dataset keeps its original license, listed in its README and in the dataset card. The tracked tables in `scripts/<name>/tables/` are derived from each dataset and follow its license.
