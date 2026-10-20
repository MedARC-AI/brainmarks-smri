"""Helpers for the `scripts/<name>/build_tables.py` scripts. Table layout: `README_hf.md`.

`modality` vocabulary: T1w, T1c (contrast-enhanced T1), T2w, FLAIR, PD, DWI (trace / b1000), ADC,
mask (segmentations; `desc` says which). `desc` marks variants; the main image has none.
"""

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "datasets"))
SEED = 0
FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
IMAGE_COLUMNS = ["participant_id", "session_id", "modality", "desc", "path"]
COMMON_COLUMNS = {
    "participant_id": {
        "Description": "Person identifier (as in the source, normalized where noted)."
    },
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
        labels = dict(entity.split("-", 1) for entity in entities)
        desc = "_".join(entity for entity in entities if not entity.startswith(("sub-", "ses-")))
        participant_id = f"sub-{labels['sub']}"
        session_id = labels.get("ses", "1")
        rows.append((participant_id, session_id, suffix, desc or None, str(path.relative_to(base))))
    return pd.DataFrame(rows, columns=IMAGE_COLUMNS)


def stratum_keys(strata: pd.Series) -> pd.Series:
    """Strata as strings, with missing values as their own 'n/a' stratum (groupby drops NA keys)."""
    return strata.astype(object).where(strata.notna(), "n/a").astype(str)


def stratified_split(
    strata: pd.Series, fractions: dict[str, float] = FRACTIONS, seed: int = SEED
) -> pd.Series:
    """Assign each participant to a split.

    `strata` is indexed by participant_id and holds each participant's stratum key. Within each
    stratum the participants are shuffled and cut by `fractions`; the largest remainders get the
    leftover participants, so split sizes stay close to the target overall.
    """
    assert strata.index.is_unique
    rng = np.random.default_rng(seed)
    split = pd.Series(index=strata.index, dtype=object)
    for _, group in strata.groupby(stratum_keys(strata)):  # groupby sorts the keys
        ids = list(rng.permutation(sorted(group.index)))
        exact = np.array(list(fractions.values())) * len(ids)
        counts = np.floor(exact).astype(int)
        leftover = len(ids) - counts.sum()
        # largest remainders first; equal remainders (e.g. val vs test) in random order
        order = rng.permutation(len(counts))
        by_remainder = order[np.argsort(-(exact - counts)[order], kind="stable")]
        for index in by_remainder[:leftover]:
            counts[index] += 1
        start = 0
        for name, count in zip(fractions, counts):
            split[ids[start : start + count]] = name
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
    groups = pd.DataFrame({"split": split, "stratum": stratum_keys(strata)})
    for _, group in groups.groupby(["split", "stratum"]):
        ids = rng.permutation(sorted(group.index))
        position[ids] = (np.arange(len(ids)) + rng.uniform()) / len(ids)
    rank = pd.Series(0, index=split.index, dtype=int)
    for name in split.unique():
        ids = position[split == name].sort_values(kind="stable").index
        rank[ids] = np.arange(len(ids))
    return rank


def make_splits(
    strata: pd.Series,
    complete: pd.Series,
    official: pd.Series | None = None,
    split: pd.Series | None = None,
) -> pd.DataFrame:
    """splits.tsv rows from per-participant strata and completeness (both indexed by participant_id).

    By default the split is a stratified 60/20/20; pass `split` to use a dataset-specific one.
    The strata also balance the rank.
    """
    if split is None:
        split = stratified_split(strata)
    if official is None:
        official = pd.Series(pd.NA, index=split.index)
    return pd.DataFrame(
        {
            "participant_id": split.index,
            "split": split.values,
            "official_split": official.reindex(split.index).values,
            "rank": interleaved_rank(split, strata).values,
            "complete": complete.reindex(split.index).values,
        }
    )


def check_tables(
    name: str,
    images: pd.DataFrame,
    samples: pd.DataFrame,
    columns: dict[str, dict],
    splits: pd.DataFrame,
) -> None:
    """Assert that the tables are consistent with each other and with the files on disk."""
    keys = ["participant_id", "session_id"]
    assert not samples.duplicated(keys).any(), "duplicate sessions"
    assert not splits.participant_id.duplicated().any(), "duplicate participants in splits"
    assert set(samples.participant_id) == set(splits.participant_id), "samples/splits mismatch"
    image_sessions = set(images[keys].itertuples(index=False))
    sample_sessions = set(samples[keys].itertuples(index=False))
    unknown = sorted(image_sessions - sample_sessions)
    assert not unknown, f"{len(unknown)} image sessions without a samples row, e.g. {unknown[:3]}"
    assert splits.split.isin(list(FRACTIONS)).all(), "unknown split"
    assert splits.complete.dtype == bool, "complete must be boolean"
    undocumented = set(columns) ^ set(samples.columns)
    assert list(columns) == list(samples.columns), f"undocumented or missing: {undocumented}"
    missing = [path for path in images.path if not (dataset_dir(name) / path).exists()]
    assert not missing, f"{len(missing)} image paths missing, e.g. {missing[:3]}"


def write(
    name: str,
    images: pd.DataFrame,
    samples: pd.DataFrame,
    columns: dict[str, dict],
    splits: pd.DataFrame,
    track_samples: bool = True,
) -> None:
    """Check the tables and write them to datasets/<name>/tables/ and scripts/<name>/tables/.

    `columns` documents the dataset-specific samples.tsv columns (the common ones are added).
    `track_samples=False` keeps samples.tsv out of scripts/<name>/tables/ (manual datasets).
    """
    images = images[IMAGE_COLUMNS]
    common = {column: COMMON_COLUMNS[column] for column in COMMON_COLUMNS if column in samples}
    columns = {**common, **columns}
    check_tables(name, images, samples, columns, splits)

    out_dir = dataset_dir(name) / "tables"
    out_dir.mkdir(exist_ok=True)
    images = images.sort_values(["participant_id", "session_id", "modality", "path"])
    images.to_csv(out_dir / "images.tsv", **TSV)
    samples.sort_values(["participant_id", "session_id"]).to_csv(out_dir / "samples.tsv", **TSV)
    splits.sort_values("participant_id").to_csv(out_dir / "splits.tsv", **TSV)
    (out_dir / "samples.json").write_text(json.dumps(columns, indent=2) + "\n")

    tracked_dir = REPO / "scripts" / name / "tables"
    tracked_dir.mkdir(exist_ok=True)
    tracked_files = ["images.tsv", "splits.tsv", "samples.json"]
    if track_samples:
        tracked_files.append("samples.tsv")
    for file_name in tracked_files:
        shutil.copy(out_dir / file_name, tracked_dir / file_name)


def describe(values: pd.Series) -> str:
    """Compact summary of one column: mean ± sd for continuous values, level counts otherwise."""
    values = values.dropna()
    if values.empty:
        return "n/a"
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if numeric and values.nunique() > 10:
        return f"{values.mean():.1f} ± {values.std():.1f}"
    counts = values.astype(str).value_counts().sort_index()
    return " / ".join(f"{level} {count}" for level, count in counts.items())


def summary_table(samples: pd.DataFrame, splits: pd.DataFrame, targets: Sequence[str] = ()) -> str:
    """Markdown table of the samples by split (rows) for the dataset README."""
    merged = samples.merge(splits, on="participant_id", validate="many_to_one")
    multi_session = len(samples) > samples.participant_id.nunique()
    multi_site = merged.site.nunique() > 1
    header = ["split", "participants"]
    if multi_session:
        header.append("samples")
    header += ["complete", "age", "female"]
    if multi_site:
        header.append("sites")
    header += list(targets)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in [*FRACTIONS, "total"]:
        part = merged if name == "total" else merged[merged.split == name]
        people = part.drop_duplicates("participant_id")
        sex = part.sex.dropna()
        row = [name, str(len(people))]
        if multi_session:
            row.append(str(len(part)))
        row.append(str(int(people.complete.sum())))
        row.append(describe(part.age))
        row.append(f"{(sex == 'F').mean():.0%}" if len(sex) else "n/a")
        if multi_site:
            row.append(str(part.site.nunique()))
        row += [describe(part[target]) for target in targets]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
