"""Shared helpers for the per-dataset `scripts/<name>/build_tables.py` scripts.

Each dataset gets a small set of derived tables in `datasets/<name>/tables/`, built only from
`datasets/<name>/source/` (which is never modified), plus a tracked copy in
`scripts/<name>/tables/` so that the splits are versioned:

    images.tsv     one row per image file: participant_id, session_id, modality, desc, path
                   (+ member, for images inside a tar archive)
    samples.tsv    one row per sample (= scan session): participant_id, session_id, age, sex,
                   site, then dataset-specific columns (labels/targets) with cleaned values
    samples.json   description of every samples.tsv column (BIDS sidecar style)
    splits.tsv     one row per participant: participant_id, split, official_split, rank, complete

`modality` uses one vocabulary across datasets: T1w, T1c (contrast-enhanced T1), T2w, FLAIR, PD,
DWI (trace / b1000 diffusion image), ADC, DTI (raw multi-direction diffusion series, 4D or one
file per volume), mask (segmentations; `desc` says which, e.g. tumor, lesion). `desc` marks
variants (e.g. bias-corrected, skull-stripped or not) and is n/a for the plain image.

IDs follow BIDS naming: `participant_id` is the person, `session_id` one scan session of that
person (often the only one). Splits are by participant, so all sessions of a person land in the
same split.

Splits are train/val/test. Datasets without an official split are split 60/20/20, stratified by
a dataset-specific key, with a fixed seed. Where the source defines a split, it is recorded in
`official_split` and our split refines it (e.g. official train -> our train + val), so the two
columns can differ.

`rank` orders the participants within each split, interleaved across strata, so that every
prefix is balanced. `complete` marks participants with all core images and primary targets.
Nested mini-splits for fast benchmarking: the N complete participants with the lowest rank
(e.g. N = 50, 100, 200, 400), see `mini_split`.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "datasets"))
SEED = 0
FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
IMAGE_COLUMNS = ["participant_id", "session_id", "modality", "desc", "path"]
COMMON = {
    "participant_id": {"Description": "Person identifier (as in the source, normalized where noted)."},
    "session_id": {"Description": "Scan session identifier within the participant."},
    "age": {"Description": "Age at scan.", "Units": "years"},
    "sex": {"Description": "Sex.", "Levels": {"M": "male", "F": "female"}},
    "site": {"Description": "Acquisition site, as coded by the source."},
}
TSV = dict(sep="\t", index=False, na_rep="n/a", lineterminator="\n")


def dataset_dir(name: str) -> Path:
    """`datasets/<name>/`; image paths in images.tsv are relative to it."""
    return DATA_ROOT / name


def bids_images(root: Path, base: Path) -> pd.DataFrame:
    """images.tsv rows for the NIfTI files of a BIDS tree `root` (sub-*/[ses-*/]<datatype>/).

    modality = the BIDS suffix (T1w, dwi, ...), desc = the other entities except sub/ses joined
    by '_' (e.g. 'rec-ADC'), session_id = the ses label or '1'. Paths are relative to `base`.
    """
    rows = []
    for path in sorted(root.glob("sub-*/**/*.nii.gz")):
        entities = path.name.removesuffix(".nii.gz").split("_")
        suffix = entities.pop()
        labels = dict(e.split("-", 1) for e in entities)
        desc = "_".join(e for e in entities if not e.startswith(("sub-", "ses-")))
        rows.append((f"sub-{labels['sub']}", labels.get("ses", "1"), suffix, desc or None,
                     str(path.relative_to(base))))
    return pd.DataFrame(rows, columns=IMAGE_COLUMNS)


def stratum_keys(strata: pd.Series) -> pd.Series:
    """Strata as strings, with missing values as their own 'n/a' stratum (groupby drops NA keys)."""
    return strata.astype(object).where(strata.notna(), "n/a").astype(str)


def stratified_split(strata: pd.Series, fractions: dict[str, float] = FRACTIONS, seed: int = SEED) -> pd.Series:
    """Assign each participant to a split.

    `strata` is indexed by participant_id and holds each participant's stratum key. Within each
    stratum the participants are shuffled and cut by `fractions`; the largest remainders get the
    leftover participants, so split sizes stay close to the target overall.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series(index=strata.index, dtype=object)
    assert strata.index.is_unique
    for _, group in strata.groupby(stratum_keys(strata)):  # groupby sorts the keys
        ids = list(rng.permutation(sorted(group.index)))
        exact = np.array(list(fractions.values())) * len(ids)
        counts = np.floor(exact).astype(int)
        leftover = len(ids) - counts.sum()
        # largest remainders first; equal remainders (e.g. val vs test) in random order
        order = rng.permutation(len(counts))
        by_remainder = order[np.argsort(-(exact - counts)[order], kind="stable")]
        for i in by_remainder[:leftover]:
            counts[i] += 1
        start = 0
        for name, count in zip(fractions, counts):
            split[ids[start:start + count]] = name
            start += count
    assert split.notna().all()
    return split


def interleaved_rank(split: pd.Series, strata: pd.Series, seed: int = SEED) -> pd.Series:
    """Rank the participants within each split so that every prefix is balanced across strata.

    The n participants of a (split, stratum) group get positions (k + u) / n for k = 0..n-1, in
    random order, with a random offset u. Sorting a split by position interleaves its strata in
    proportion to their sizes.
    """
    rng = np.random.default_rng(seed + 1)
    position = pd.Series(np.nan, index=split.index)
    groups = pd.DataFrame({"split": split, "stratum": stratum_keys(strata)}).groupby(["split", "stratum"])
    for _, group in groups:
        ids = rng.permutation(sorted(group.index))
        position[ids] = (np.arange(len(ids)) + rng.uniform()) / len(ids)
    rank = pd.Series(0, index=split.index, dtype=int)
    for name in split.unique():
        ids = position[split == name].sort_values(kind="stable").index
        rank[ids] = np.arange(len(ids))
    return rank


def make_splits(strata: pd.Series, complete: pd.Series, official: pd.Series | None = None,
                split: pd.Series | None = None) -> pd.DataFrame:
    """splits.tsv rows from per-participant strata and completeness (both indexed by participant_id).

    By default the split is a stratified 60/20/20; pass `split` to use a dataset-specific one.
    """
    if split is None:
        split = stratified_split(strata)
    return pd.DataFrame({
        "participant_id": split.index,
        "split": split.values,
        "official_split": official.reindex(split.index).values if official is not None else pd.NA,
        "rank": interleaved_rank(split, strata).values,
        "complete": complete.reindex(split.index).values,
    })


def mini_split(splits: pd.DataFrame, split: str, n: int) -> pd.DataFrame:
    """The n complete participants of `split` with the lowest rank (nested in n, balanced)."""
    rows = splits[(splits.split == split) & splits.complete]
    return rows.nsmallest(n, "rank")


def write(name: str, images: pd.DataFrame, samples: pd.DataFrame, columns: dict[str, dict],
          splits: pd.DataFrame) -> None:
    """Validate the tables and write them to datasets/<name>/tables/ and scripts/<name>/tables/.

    `columns` documents the dataset-specific samples.tsv columns (the common ones are added).
    """
    image_columns = IMAGE_COLUMNS + (["member"] if "member" in images else [])
    images = images[image_columns]
    columns = {**{c: COMMON[c] for c in COMMON if c in samples}, **columns}

    assert not samples.duplicated(["participant_id", "session_id"]).any(), "duplicate sessions"
    assert not splits.participant_id.duplicated().any(), "duplicate participants in splits"
    assert set(samples.participant_id) == set(splits.participant_id), "samples/splits mismatch"
    keys = ["participant_id", "session_id"]
    unknown = set(images[keys].itertuples(index=False)) - set(samples[keys].itertuples(index=False))
    assert not unknown, f"{len(unknown)} image sessions without a samples row, e.g. {sorted(unknown)[:3]}"
    assert splits.split.isin(list(FRACTIONS)).all(), "unknown split"
    assert splits.complete.dtype == bool, "complete must be boolean"
    assert list(columns) == list(samples.columns), \
        f"undocumented or missing columns: {set(columns) ^ set(samples.columns)}"
    missing = [p for p in images.path if not (dataset_dir(name) / p).exists()]
    assert not missing, f"{len(missing)} image paths missing, e.g. {missing[:3]}"

    out = dataset_dir(name) / "tables"
    shutil.rmtree(out, ignore_errors=True)  # no stale files from earlier builds
    out.mkdir()
    images.sort_values(["participant_id", "session_id", "modality", "path"]).to_csv(out / "images.tsv", **TSV)
    samples.sort_values(["participant_id", "session_id"]).to_csv(out / "samples.tsv", **TSV)
    splits.sort_values("participant_id").to_csv(out / "splits.tsv", **TSV)
    (out / "samples.json").write_text(json.dumps(columns, indent=2) + "\n")

    tracked = REPO / "scripts" / name / "tables"
    shutil.rmtree(tracked, ignore_errors=True)
    shutil.copytree(out, tracked)
    print_summary(splits)


def print_summary(splits: pd.DataFrame) -> None:
    for name in FRACTIONS:
        rows = splits[splits.split == name]
        print(f"{name:5s} {len(rows):5d} participants, {rows.complete.sum():5d} complete")
