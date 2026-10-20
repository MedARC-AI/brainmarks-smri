"""Summary figure of the FOMO300K images table for the README.

uv run scripts/fomo300k/plot_summary.py datasets/fomo300k scripts/fomo300k/summary.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP = 15


def barh(ax: plt.Axes, counts: pd.Series, title: str) -> None:
    ax.barh(counts.index[::-1].astype(str), counts.values[::-1])
    ax.set_title(title)
    ax.locator_params(axis="x", nbins=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    images = pd.read_parquet(args.dataset_dir / "tables" / "images.parquet")
    images["sequence"] = images["suffix"].where(
        images["modality"] != "dwi", "dwi " + images["dwi_type"].fillna("other")
    )
    images["session_key"] = images["collection"] + "/" + images["subject"] + "/" + images["session"]
    sessions = images.drop_duplicates("session_key")
    subjects = images.drop_duplicates(["collection", "subject"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    sequences = images["sequence"].value_counts()
    sequences = pd.concat([sequences[:TOP], pd.Series({"other": sequences[TOP:].sum()})])
    barh(axes[0, 0], sequences, "Images by sequence")

    collections = subjects["collection"].str.removeprefix("PT030_OpenNeuro/").value_counts()
    collections.index = [name if len(name) <= 16 else name[:15] + "…" for name in collections.index]
    barh(axes[0, 1], collections[:TOP], f"Subjects, top {TOP} of {len(collections)} collections")

    axes[0, 2].hist(sessions["age"].dropna(), bins=np.arange(0, 101, 4))
    axes[0, 2].set_title("Age at session")
    axes[0, 2].set_xlabel("years")

    field = pd.cut(
        images["field_strength"],
        [0, 1, 2, 3.5, 5, 8, 20],
        labels=["<1 T", "1.5 T", "3 T", "4 T", "7 T", "9.4 T+"],
    )
    field = field.cat.add_categories("unknown").fillna("unknown").value_counts(sort=False)
    barh(axes[1, 0], field, "Images by field strength")

    spacing = np.array(images["spacing"].tolist(), dtype=float)
    axes[1, 1].hist(
        [spacing.min(axis=1), spacing.max(axis=1)],
        bins=np.arange(0, 8.01, 0.25),
        label=["finest axis", "coarsest axis"],
        histtype="step",
    )
    axes[1, 1].set_title("Voxel spacing")
    axes[1, 1].set_xlabel("mm")
    axes[1, 1].legend()

    manufacturer = images["manufacturer"].str.lower().str.split().str[0]
    manufacturer = manufacturer.replace({"general": "ge"}).fillna("unknown").value_counts()
    manufacturer = pd.concat([manufacturer[:5], pd.Series({"other": manufacturer[5:].sum()})])
    barh(axes[1, 2], manufacturer, "Images by manufacturer")

    fig.tight_layout()
    fig.savefig(args.out, dpi=100)


if __name__ == "__main__":
    main()
