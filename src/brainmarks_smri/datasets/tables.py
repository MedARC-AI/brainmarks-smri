"""Helpers for the per-dataset `scripts/<name>/build_tables.py` scripts.

Tables go to `datasets/<name>/tables/` with a tracked copy in `scripts/<name>/tables/`, and are
built only from `source/`. Layout and split conventions: `README_hf.md`.

`modality` vocabulary: T1w, T1c (contrast-enhanced T1), T2w, FLAIR, PD, DWI (trace / b1000), ADC,
DTI (raw multi-direction series), mask (segmentations; `desc` says which). `desc` marks variants.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "datasets"))
SEED = 0
FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
IMAGE_COLUMNS = ["participant_id", "session_id", "modality", "desc", "path"]
COMMON = {
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
        labels = dict(e.split("-", 1) for e in entities)
        desc = "_".join(e for e in entities if not e.startswith(("sub-", "ses-")))
        rows.append(
            (
                f"sub-{labels['sub']}",
                labels.get("ses", "1"),
                suffix,
                desc or None,
                str(path.relative_to(base)),
            )
        )
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
    groups = pd.DataFrame({"split": split, "stratum": stratum_keys(strata)}).groupby(
        ["split", "stratum"]
    )
    for _, group in groups:
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
    """
    if split is None:
        split = stratified_split(strata)
    return pd.DataFrame(
        {
            "participant_id": split.index,
            "split": split.values,
            "official_split": official.reindex(split.index).values
            if official is not None
            else pd.NA,
            "rank": interleaved_rank(split, strata).values,
            "complete": complete.reindex(split.index).values,
        }
    )


def mini_split(splits: pd.DataFrame, split: str, n: int) -> pd.DataFrame:
    """The n complete participants of `split` with the lowest rank (nested in n, balanced)."""
    rows = splits[(splits.split == split) & splits.complete]
    return rows.nsmallest(n, "rank")


def write(
    name: str,
    images: pd.DataFrame,
    samples: pd.DataFrame,
    columns: dict[str, dict],
    splits: pd.DataFrame,
    summary: list[str] = (),
) -> None:
    """Validate the tables and write them to datasets/<name>/tables/ and scripts/<name>/tables/.

    Also refreshes the README copy in datasets/<name>/ and prints a markdown summary table
    (counts, age, sex, site and the `summary` target columns by split) for the README.

    `columns` documents the dataset-specific samples.tsv columns (the common ones are added).
    """
    images = images[IMAGE_COLUMNS]
    columns = {**{c: COMMON[c] for c in COMMON if c in samples}, **columns}

    assert not samples.duplicated(["participant_id", "session_id"]).any(), "duplicate sessions"
    assert not splits.participant_id.duplicated().any(), "duplicate participants in splits"
    assert set(samples.participant_id) == set(splits.participant_id), "samples/splits mismatch"
    keys = ["participant_id", "session_id"]
    unknown = set(images[keys].itertuples(index=False)) - set(samples[keys].itertuples(index=False))
    assert not unknown, (
        f"{len(unknown)} image sessions without a samples row, e.g. {sorted(unknown)[:3]}"
    )
    assert splits.split.isin(list(FRACTIONS)).all(), "unknown split"
    assert splits.complete.dtype == bool, "complete must be boolean"
    assert list(columns) == list(samples.columns), (
        f"undocumented or missing columns: {set(columns) ^ set(samples.columns)}"
    )
    missing = [p for p in images.path if not (dataset_dir(name) / p).exists()]
    assert not missing, f"{len(missing)} image paths missing, e.g. {missing[:3]}"

    out = dataset_dir(name) / "tables"
    shutil.rmtree(out, ignore_errors=True)  # no stale files from earlier builds
    out.mkdir()
    images.sort_values(["participant_id", "session_id", "modality", "path"]).to_csv(
        out / "images.tsv", **TSV
    )
    samples.sort_values(["participant_id", "session_id"]).to_csv(out / "samples.tsv", **TSV)
    splits.sort_values("participant_id").to_csv(out / "splits.tsv", **TSV)
    (out / "samples.json").write_text(json.dumps(columns, indent=2) + "\n")

    tracked = REPO / "scripts" / name / "tables"
    shutil.rmtree(tracked, ignore_errors=True)
    shutil.copytree(out, tracked)
    shutil.copy(REPO / "scripts" / name / "README.md", dataset_dir(name) / "README.md")
    print(summary_table(samples, splits, list(summary)))


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


def summary_table(samples: pd.DataFrame, splits: pd.DataFrame, targets: list[str]) -> str:
    """Markdown table of the samples by split (rows) for the dataset README."""
    merged = samples.merge(splits, on="participant_id", validate="many_to_one")
    multi_session = len(samples) > samples.participant_id.nunique()
    multi_site = merged.site.nunique() > 1
    header = [
        "split",
        "participants",
        *(["samples"] if multi_session else []),
        "complete",
        "age",
        "female",
        *(["sites"] if multi_site else []),
        *targets,
    ]
    rows = []
    for name in [*FRACTIONS, "total"]:
        part = merged if name == "total" else merged[merged.split == name]
        people = part.drop_duplicates("participant_id")
        sex = part.sex.dropna()
        row = [
            name,
            str(len(people)),
            *([str(len(part))] if multi_session else []),
            str(int(people.complete.sum())),
            describe(part.age),
            f"{(sex == 'F').mean():.0%}" if len(sex) else "n/a",
            *([str(part.site.nunique())] if multi_site else []),
            *(describe(part[t]) for t in targets),
        ]
        rows.append(row)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
